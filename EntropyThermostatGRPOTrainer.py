"""GRPO trainer with closed-loop, spurious-reward entropy control.

This is a bolt-on to TRL's GRPOTrainer.  It reads the same policy entropy that
TRL logs as ``train/entropy`` and feeds completed optimizer-step observations
to :class:`EntropyThermostat`.  When the controller crosses a hysteresis
boundary it swaps only ``epsilon_low``/``epsilon_high``; rewards and all other
training configuration remain untouched.

Controller state is also written to ``thermostat_history.jsonl`` in the run's
output directory, giving an exact record for later plotting.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

from trl import GRPOTrainer

from entropy_thermostat import EntropyThermostat, ThermostatObservation


class EntropyThermostatGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that drives policy entropy toward a requested target."""

    def __init__(self, *args, thermostat: dict[str, Any], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.entropy_thermostat = EntropyThermostat(**thermostat)

        # Entropies accumulated while global_step is unchanged.  This matters
        # under gradient accumulation: clipping must not change halfway through
        # one optimizer update.
        self._thermostat_observed_step = int(self.state.global_step)
        self._thermostat_step_entropies: list[float] = []

    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        # There is no subsequent compute_loss call to trigger the normal
        # step-boundary flush after the final optimizer update.
        self._finish_thermostat_step(self._thermostat_observed_step)
        return result

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        current_step = int(self.state.global_step)
        if current_step != self._thermostat_observed_step:
            self._finish_thermostat_step(self._thermostat_observed_step)
            self._thermostat_observed_step = current_step

        # TRL appends the entropy calculated in this forward pass to
        # self._metrics["train"]["entropy"].  Capture only the new value(s),
        # independently of TRL's later logging/clearing of that buffer.
        entropy_metrics = self._metrics["train"]["entropy"]
        before = len(entropy_metrics)
        loss = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        entropy_metrics = self._metrics["train"]["entropy"]
        if len(entropy_metrics) > before:
            self._thermostat_step_entropies.extend(
                float(v) for v in entropy_metrics[before:] if v is not None
            )
        return loss

    def _finish_thermostat_step(self, completed_step: int) -> None:
        if not self._thermostat_step_entropies:
            return

        raw_entropy = sum(self._thermostat_step_entropies) / len(
            self._thermostat_step_entropies
        )
        self._thermostat_step_entropies.clear()
        observation = self.entropy_thermostat.observe(raw_entropy, completed_step)

        if observation.switched:
            # All ranks see the same entropy: GRPOTrainer gathers mean_entropy
            # across processes before storing it in _metrics.
            self.epsilon_low = float(observation.epsilon_low)
            self.epsilon_high = float(observation.epsilon_high)
            if self.accelerator.is_main_process:
                print("=" * 100)
                print(
                    "[EntropyThermostat] "
                    f"step {completed_step}: H={observation.control_entropy:.4f}, "
                    f"target={self.entropy_thermostat.target:.4f} -> "
                    f"{observation.state.upper()} "
                    f"(eps_low={self.epsilon_low}, eps_high={self.epsilon_high})"
                )
                print("=" * 100)

        self._log_thermostat_metrics(observation)
        self._write_thermostat_history(observation)

    def _log_thermostat_metrics(self, observation: ThermostatObservation) -> None:
        """Add numeric controller signals to the normal TRL/W&B metric stream."""
        metrics = self._metrics["train"]
        metrics["thermostat/target"].append(self.entropy_thermostat.target)
        metrics["thermostat/raw_entropy"].append(observation.raw_entropy)
        metrics["thermostat/control_entropy"].append(observation.control_entropy)
        metrics["thermostat/error"].append(observation.error)
        metrics["thermostat/switched"].append(float(observation.switched))
        metrics["thermostat/control_signal"].append(
            1.0 if observation.state == "up" else -1.0 if observation.state == "down" else 0.0
        )

    def _write_thermostat_history(self, observation: ThermostatObservation) -> None:
        if not self.accelerator.is_main_process:
            return

        record = {
            "step": observation.step,
            "raw_entropy": observation.raw_entropy,
            "control_entropy": observation.control_entropy,
            "target": self.entropy_thermostat.target,
            "deadband": self.entropy_thermostat.deadband,
            "error": observation.error,
            "state": observation.state,
            "action": observation.action,
            "switched": observation.switched,
            "epsilon_low": observation.epsilon_low,
            # JSON Infinity is non-portable.  Use the same string accepted by
            # the training CLI instead.
            "epsilon_high": (
                "inf"
                if observation.epsilon_high is not None
                and math.isinf(observation.epsilon_high)
                else observation.epsilon_high
            ),
        }
        path = os.path.join(self.args.output_dir, "thermostat_history.jsonl")
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
