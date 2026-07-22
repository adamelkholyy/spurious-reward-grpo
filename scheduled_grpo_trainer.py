"""
scheduled_grpo_trainer.py

A thin wrapper around TRL's `GRPOTrainer` that lets you change the clipping
epsilon(s) and/or the reward function(s) at one or more training steps.

Verified against the GRPOTrainer internals on trl `main`:
  - epsilon clip bounds:      self.epsilon_low, self.epsilon_high
                              (read fresh inside `compute_loss`)
  - reward functions:         self.reward_funcs           (list)
                              self.reward_func_names       (list, used for log keys)
                              self.reward_weights          (float tensor)
                              self.reward_processing_classes (list, None for callables)
                              (all used inside `_generate_and_score_completions`)
  - meaningful step counter:  self.state.global_step

Semantics
---------
A change scheduled for `step=S` is applied once `self.state.global_step >= S`.
Because global_step is incremented *after* the optimizer step, this means the
change is active for the update on which global_step has reached S.

  * epsilon changes take effect immediately on the next `compute_loss` (every step).
  * reward changes are picked up on the next generation cycle. GRPO generates and
    scores in batches every `steps_per_generation * num_iterations` micro-steps and
    buffers the result, so a reward swap may lag the target step by up to one
    generation cycle. This is inherent to GRPO's buffering, not the wrapper.

Only what you specify is changed. `epsilon` sets the lower clip bound and
`epsilon_high` the upper bound, mirroring `GRPOConfig` (where `epsilon_high`
defaults to `epsilon`). If you want symmetric clipping, set both.

Swapping in plain callable rewards is fully handled. Swapping in a *model-based*
reward (an `nn.Module` / reward model) at runtime is best-effort: the wrapper will
`accelerator.prepare_model(...)` it, but you must also supply a matching
`reward_processing_classes` entry, and this path is less battle-tested than
starting training with the model already registered.
"""

from __future__ import annotations

import inspect
# import logging
from typing import Any, Callable, Optional, Sequence, Union

import torch
from trl import GRPOTrainer

# logger = logging.getLogger(__name__)

# These TRL helpers determine the log-metric key for each reward func
# (e.g. "rewards/<name>"). Fall back gracefully if the internal path moves.
try:
    from trl.trainer.utils import get_callable_name, get_config_model_id
except Exception:  # pragma: no cover - defensive against TRL refactors

    def get_callable_name(func: Callable) -> str:
        return getattr(func, "__name__", func.__class__.__name__)

    def get_config_model_id(config) -> str:
        return getattr(config, "_name_or_path", "reward_model")


RewardFunc = Union[Callable, torch.nn.Module, str]


class ScheduledGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer that applies a schedule of parameter/reward changes during training.

    Parameters
    ----------
    schedule : list of dict, optional
        Each dict must contain "step" (int) and any of:
          - "epsilon"                  -> new self.epsilon_low (lower clip bound)
          - "epsilon_high"             -> new self.epsilon_high (upper clip bound)
          - "reward_funcs"             -> callable | nn.Module | list of them
          - "reward_weights"           -> sequence of floats (optional, per reward_func)
          - "reward_processing_classes"-> list (optional; needed for model rewards)
        Steps may be given in any order; they are applied in ascending step order,
        each exactly once.

    All other args/kwargs are forwarded to `GRPOTrainer` unchanged.
    """

    def __init__(self, *args, schedule: Optional[Sequence[dict]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._schedule = sorted(
            [dict(c) for c in (schedule or [])], key=lambda c: c["step"]
        )
        self._applied_schedule_idx: set[int] = set()
        # Apply anything scheduled for step 0 / resumed-past steps right away.
        self._apply_schedule()

    # ------------------------------------------------------------------ #
    # Convenience constructor for the common single-switch case.
    # ------------------------------------------------------------------ #
    @classmethod
    def with_switch(
        cls,
        *args,
        switch_step: int,
        new_epsilon: Optional[float] = None,
        new_epsilon_high: Optional[float] = None,
        new_reward_funcs: Optional[Union[RewardFunc, Sequence[RewardFunc]]] = None,
        new_reward_weights: Optional[Sequence[float]] = None,
        new_reward_processing_classes: Optional[Sequence[Any]] = None,
        **kwargs,
    ) -> "ScheduledGRPOTrainer":
        """Build a trainer with a single change applied at `switch_step`."""
        change: dict[str, Any] = {"step": switch_step}
        if new_epsilon is not None:
            change["epsilon"] = new_epsilon
        if new_epsilon_high is not None:
            change["epsilon_high"] = new_epsilon_high
        if new_reward_funcs is not None:
            change["reward_funcs"] = new_reward_funcs
        if new_reward_weights is not None:
            change["reward_weights"] = new_reward_weights
        if new_reward_processing_classes is not None:
            change["reward_processing_classes"] = new_reward_processing_classes
        return cls(*args, schedule=[change], **kwargs)

    # ------------------------------------------------------------------ #
    # Hooks: check the schedule where epsilon and rewards are actually read.
    # ------------------------------------------------------------------ #
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # epsilon_low / epsilon_high are read here, once per step.
        self._apply_schedule()
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def _generate_and_score_completions(self, generation_batch):
        # reward_funcs are read here, at each generation cycle.
        self._apply_schedule()
        return super()._generate_and_score_completions(generation_batch)

    # ------------------------------------------------------------------ #
    # Schedule application.
    # ------------------------------------------------------------------ #
    def _apply_schedule(self) -> None:
        step = int(self.state.global_step)
        for i, change in enumerate(self._schedule):
            if i in self._applied_schedule_idx:
                continue
            if step < change["step"]:
                break  # schedule is sorted ascending; nothing further is due
            self._apply_change(change)
            self._applied_schedule_idx.add(i)

    def _apply_change(self, change: dict) -> None:
        applied = []

        if "epsilon" in change:
            self.epsilon_low = float(change["epsilon"])
            applied.append(f"epsilon_low={self.epsilon_low}")
        if "epsilon_high" in change:
            self.epsilon_high = float(change["epsilon_high"])
            applied.append(f"epsilon_high={self.epsilon_high}")

        if change.get("reward_funcs") is not None:
            self._set_reward_funcs(
                change["reward_funcs"],
                reward_weights=change.get("reward_weights"),
                reward_processing_classes=change.get("reward_processing_classes"),
            )
            applied.append(f"reward_funcs={self.reward_func_names}")

        if applied and self.accelerator.is_main_process:
            print("="*100)
            print(
                "[ScheduledGRPOTrainer] step %s: applied %s",
                self.state.global_step,
                ", ".join(applied),
            )
            print("="*100)

    def _set_reward_funcs(
        self,
        reward_funcs: Union[RewardFunc, Sequence[RewardFunc]],
        reward_weights: Optional[Sequence[float]] = None,
        reward_processing_classes: Optional[Sequence[Any]] = None,
    ) -> None:
        if not isinstance(reward_funcs, (list, tuple)):
            reward_funcs = [reward_funcs]
        reward_funcs = list(reward_funcs)

        # Prepare any nn.Module reward models (device placement / DS / FSDP).
        for i, func in enumerate(reward_funcs):
            if isinstance(func, torch.nn.Module):
                reward_funcs[i] = self.accelerator.prepare_model(
                    func, evaluation_mode=True, device_placement=True
                )

        # Derive log-metric names the same way GRPOTrainer does.
        names: list[str] = []
        for func in reward_funcs:
            if isinstance(func, torch.nn.Module):
                cfg = getattr(func, "config", None)
                try:
                    names.append(get_config_model_id(cfg).split("/")[-1])
                except Exception:
                    names.append(func.__class__.__name__)
            else:
                names.append(get_callable_name(func))

        # Weights: honor explicit, else reuse existing if count matches, else ones.
        if reward_weights is not None:
            weights = torch.tensor(list(reward_weights), dtype=torch.float32)
            if len(weights) != len(reward_funcs):
                raise ValueError(
                    f"reward_weights has {len(weights)} entries but there are "
                    f"{len(reward_funcs)} reward funcs."
                )
        elif len(reward_funcs) == len(self.reward_funcs):
            weights = self.reward_weights
        else:
            weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Processing classes: honor explicit, else reuse if count matches, else None.
        if reward_processing_classes is not None:
            proc = list(reward_processing_classes)
            if len(proc) != len(reward_funcs):
                raise ValueError(
                    f"reward_processing_classes has {len(proc)} entries but there "
                    f"are {len(reward_funcs)} reward funcs."
                )
        elif len(reward_funcs) == len(self.reward_processing_classes):
            proc = self.reward_processing_classes
        else:
            proc = [None] * len(reward_funcs)

        self.reward_funcs = reward_funcs
        self.reward_func_names = names
        self.reward_weights = weights
        self.reward_processing_classes = proc

        # Recompute async-func flag if the trainer tracks it (newer TRL).
        if hasattr(self, "_has_async_funcs"):
            tools = getattr(self, "tools", []) or []
            self._has_async_funcs = any(
                inspect.iscoroutinefunction(f) for f in (self.reward_funcs + list(tools))
            )


if __name__ == "__main__":
    # Minimal smoke test of the scheduling logic (no real training run).
    # Demonstrates the intended usage without needing a GPU/model.
    def reward_phase_a(completions, **kwargs):
        return [float(len(c)) for c in completions]

    def reward_phase_b(completions, **kwargs):
        return [-abs(200 - len(c)) for c in completions]

    print("Example schedule (pass to ScheduledGRPOTrainer(schedule=...)):")
    example_schedule = [
        {"step": 0, "epsilon": 0.2, "reward_funcs": reward_phase_a},
        {
            "step": 500,
            "epsilon": 0.3,
            "epsilon_high": 0.4,
            "reward_funcs": reward_phase_b,
        },
    ]
    for c in example_schedule:
        print("  ", c)
