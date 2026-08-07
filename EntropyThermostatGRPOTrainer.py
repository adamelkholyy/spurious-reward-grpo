"""GRPO trainer with GT training plus closed-loop SR entropy interventions.

Ground-truth reward and its normal clipping are the default.  At rollout
boundaries the controller checks TRL's policy entropy.  A target-band violation
temporarily swaps to a reward-independent SR signal plus the clipping direction
known to correct that violation.  Once entropy returns to the band, the trainer
restores the original GT reward and clipping.

All reward changes are rollout-aligned: we never relabel or reinterpret a
buffered rollout after its advantages have already been calculated.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Callable, Sequence

import torch

from ScheduledGRPOTrainer import ScheduledGRPOTrainer
from entropy_thermostat import EntropyThermostat, ThermostatObservation


RewardFunc = Callable | torch.nn.Module | str


class EntropyThermostatGRPOTrainer(ScheduledGRPOTrainer):
    """Run GT GRPO normally, using SR only to correct entropy violations."""

    def __init__(
        self,
        *args,
        thermostat: dict[str, Any],
        sr_reward_funcs: RewardFunc | Sequence[RewardFunc],
        **kwargs,
    ) -> None:
        # reward_funcs in kwargs are the GT reward(s) and remain the canonical
        # objective to restore whenever entropy is back inside the target band.
        gt_reward_funcs = kwargs.get("reward_funcs")
        if gt_reward_funcs is None:
            raise ValueError("EntropyThermostatGRPOTrainer requires GT reward_funcs")

        # The thermostat owns switching; ScheduledGRPOTrainer is inherited only
        # for its battle-tested runtime reward swapping helper.
        kwargs["schedule"] = None
        kwargs["save_on_switch"] = False
        super().__init__(*args, **kwargs)

        self.entropy_thermostat = EntropyThermostat(**thermostat)
        self._gt_reward_funcs = gt_reward_funcs
        self._sr_reward_funcs = sr_reward_funcs
        self._gt_epsilon_low = float(self.epsilon_low)
        self._gt_epsilon_high = float(self.epsilon_high)

        # GRPO may reuse one generated/scored rollout for several optimizer
        # updates. Reward changes are only valid at the boundary between those
        # cycles, so require the control cadence to land on that boundary.
        microsteps_per_rollout = int(self.args.steps_per_generation) * int(
            self.num_iterations
        )
        grad_accum = int(self.args.gradient_accumulation_steps)
        if microsteps_per_rollout % grad_accum != 0:
            raise ValueError(
                "The thermostat requires each rollout reuse cycle to contain a "
                "whole number of optimizer steps; got "
                f"{microsteps_per_rollout} microsteps / {grad_accum} grad-accum steps."
            )
        self._updates_per_rollout = microsteps_per_rollout // grad_accum
        if self.entropy_thermostat.control_interval % self._updates_per_rollout != 0:
            raise ValueError(
                "thermostat_interval must be a multiple of the GRPO updates per "
                f"rollout ({self._updates_per_rollout}); got "
                f"{self.entropy_thermostat.control_interval}. This prevents reward "
                "changes halfway through an already-scored rollout."
            )

        # Entropies accumulated while global_step is unchanged. This ensures
        # clipping never changes halfway through a gradient-accumulated update.
        self._thermostat_observed_step = int(self.state.global_step)
        self._thermostat_step_entropies: list[float] = []

    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        # No following compute_loss exists to flush the final observation.
        self._finish_thermostat_step(self._thermostat_observed_step)
        return result

    def _advance_step_boundary(self) -> None:
        current_step = int(self.state.global_step)
        if current_step != self._thermostat_observed_step:
            self._finish_thermostat_step(self._thermostat_observed_step)
            self._thermostat_observed_step = current_step

    def _generate_and_score_completions(self, generation_batch):
        # Crucial ordering: finalize the preceding entropy observation and apply
        # any GT<->SR transition BEFORE this fresh rollout receives rewards.
        self._advance_step_boundary()
        return super()._generate_and_score_completions(generation_batch)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        self._advance_step_boundary()

        # TRL appends entropy calculated by this forward pass here. Capture only
        # new entries so later metric-buffer clearing cannot lose observations.
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
            self._apply_thermostat_state(observation)

        self._log_thermostat_metrics(observation)
        self._write_thermostat_history(observation)

    def _apply_thermostat_state(self, observation: ThermostatObservation) -> None:
        if observation.state == "gt":
            self.epsilon_low = self._gt_epsilon_low
            self.epsilon_high = self._gt_epsilon_high
            self._set_reward_funcs(self._gt_reward_funcs)
            regime = "GT"
        else:
            self.epsilon_low = float(observation.epsilon_low)
            self.epsilon_high = float(observation.epsilon_high)
            self._set_reward_funcs(self._sr_reward_funcs)
            regime = "SR-UP" if observation.state == "sr_up" else "SR-DOWN"

        if self.accelerator.is_main_process:
            print("=" * 100)
            print(
                "[EntropyThermostat] "
                f"step {observation.step}: H={observation.raw_entropy:.4f}, "
                f"H_control={observation.control_entropy:.4f}, "
                f"target={self.entropy_thermostat.target:.4f} -> {regime} "
                f"(eps_low={self.epsilon_low}, eps_high={self.epsilon_high}, "
                f"reward={self.reward_func_names})"
            )
            print("=" * 100)

    def _log_thermostat_metrics(self, observation: ThermostatObservation) -> None:
        metrics = self._metrics["train"]
        metrics["thermostat/target"].append(self.entropy_thermostat.target)
        metrics["thermostat/raw_entropy"].append(observation.raw_entropy)
        metrics["thermostat/control_entropy"].append(observation.control_entropy)
        metrics["thermostat/error"].append(observation.error)
        metrics["thermostat/switched"].append(float(observation.switched))
        metrics["thermostat/control_signal"].append(
            1.0
            if observation.state == "sr_up"
            else -1.0
            if observation.state == "sr_down"
            else 0.0
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
            "reward_mode": "gt" if observation.state == "gt" else "sr",
            "epsilon_low": (
                self._gt_epsilon_low
                if observation.state == "gt"
                else observation.epsilon_low
            ),
            "epsilon_high": self._history_epsilon_high(observation),
        }
        path = os.path.join(self.args.output_dir, "thermostat_history.jsonl")
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _history_epsilon_high(self, observation: ThermostatObservation):
        value = (
            self._gt_epsilon_high
            if observation.state == "gt"
            else observation.epsilon_high
        )
        return "inf" if value is not None and math.isinf(value) else value
