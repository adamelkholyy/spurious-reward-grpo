from __future__ import annotations

import inspect
import json
import os

from typing import Any, Optional, Union
from collections.abc import Callable, Sequence

import torch
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import GRPOTrainer
from trl.trainer.utils import get_config_model_id


RewardFunc = Union[Callable, torch.nn.Module, str]


class ScheduledGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer that applies a schedule of parameter/reward changes during training.

    `single_example`: a dataset row that replaces every prompt until the first scheduled change,
    after which training uses `train_dataset`. Completions are also regenerated the moment that
    change is due, so the step after it is already on the new data, reward and epsilon even when
    it falls inside a generation cycle. Without it, new rewards wait for the next generation.
    """

    def __init__(
        self,
        *args,
        schedule: Optional[Sequence[dict]] = None,
        save_on_switch: bool = True,
        protect_switch_checkpoints: bool = True,
        single_example: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._schedule = sorted(
            [dict(c) for c in (schedule or [])], key=lambda c: c["step"]
        )
        self._applied_schedule_idx: set[int] = set()
        self._save_on_switch = save_on_switch
        self._protect_switch_checkpoints = protect_switch_checkpoints
        self._protected_checkpoints: set[str] = set()
        self._in_training = False
        self._single_example = single_example
        if single_example is not None and not self._schedule:
            raise ValueError("single_example needs a scheduled change to switch away from it.")
        # apply anything scheduled for step 0 / resumed-past steps right away.
        self._apply_schedule()

    # constructor for the common single-switch case.
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
        save_checkpoint: Optional[bool] = None,
        reset_optimizer: bool = False,
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
        if save_checkpoint is not None:
            change["save_checkpoint"] = save_checkpoint
        if reset_optimizer:
            change["reset_optimizer"] = True
        return cls(*args, schedule=[change], **kwargs)

    # check the schedule where epsilon and rewards are actually read.
    def train(self, *args, **kwargs):
        self._in_training = True
        try:
            return super().train(*args, **kwargs)
        finally:
            self._in_training = False

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # read epsilon_low / epsilon_high  once per step.
        self._apply_schedule()
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def _prepare_inputs(self, generation_batch):
        if self._single_example is not None and self.model.training and self._change_due():
            self._buffered_inputs = None  # drop what's left of the old cycle and regenerate now
        return super()._prepare_inputs(generation_batch)

    def _generate_and_score_completions(self, generation_batch):
        # reward_funcs are read here, at each generation cycle.
        self._apply_schedule()
        if self._single_example is not None and self.state.global_step < int(self._schedule[0]["step"]):
            generation_batch = [dict(self._single_example) for _ in generation_batch]
        return super()._generate_and_score_completions(generation_batch)

    # schedule application.
    def _change_due(self) -> bool:
        step = int(self.state.global_step)
        return any(
            i not in self._applied_schedule_idx and step >= int(change["step"])
            for i, change in enumerate(self._schedule)
        )

    def _apply_schedule(self) -> None:
        step = int(self.state.global_step)
        for i, change in enumerate(self._schedule):
            if i in self._applied_schedule_idx:
                continue
            if step < int(change["step"]):
                break  # schedule is sorted ascending
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

        if not applied:
            return

        if self.accelerator.is_main_process:
            print("=" * 100)
            print(
                f"[ScheduledGRPOTrainer] step {self.state.global_step}: "
                f"applied {', '.join(applied)}"
            )
            print("=" * 100)

        if change.get("save_checkpoint", self._save_on_switch):
            self._save_switch_checkpoint(change, applied)

        # after the save, so the switch checkpoint keeps the optimizer state it ended with.
        if change.get("reset_optimizer"):
            getattr(self.optimizer, "optimizer", self.optimizer).state.clear()
            if self.accelerator.is_main_process:
                print(f"[ScheduledGRPOTrainer] step {self.state.global_step}: reset optimizer state")

    # checkpointing at the switch.
    def _switch_checkpoint_dir(self) -> str:
        return os.path.join(
            self._get_output_dir(trial=None),
            f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
        )

    def _save_switch_checkpoint(self, change: dict, applied: list[str]) -> None:
        """Write `checkpoint-<global_step>` at the moment a change is applied."""
        if not self._in_training:
            return
        if getattr(self, "optimizer", None) is None:
            return

        ckpt_dir = self._switch_checkpoint_dir()

        if os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir):
            if self.accelerator.is_main_process:
                print(
                    f"[ScheduledGRPOTrainer] {ckpt_dir} already exists; "
                    "skipping switch checkpoint."
                )
            self._protected_checkpoints.add(os.path.abspath(ckpt_dir))
            return

        self.accelerator.wait_for_everyone()
        self._save_checkpoint(self.model, None)
        self.accelerator.wait_for_everyone()

        self._protected_checkpoints.add(os.path.abspath(ckpt_dir))

        if self.accelerator.is_main_process:
            self._write_switch_info(ckpt_dir, change, applied)
            print(f"[ScheduledGRPOTrainer] saved switch checkpoint -> {ckpt_dir}")

    def _write_switch_info(
        self, ckpt_dir: str, change: dict, applied: list[str]
    ) -> None:
        """Record what changed, next to the weights, for later archaeology."""
        info = {
            "global_step": int(self.state.global_step),
            "epoch": self.state.epoch,
            "scheduled_step": int(change["step"]),
            "applied": applied,
            "epsilon_low": float(self.epsilon_low),
            "epsilon_high": float(self.epsilon_high),
            "reward_func_names": list(self.reward_func_names),
            "reward_weights": [float(w) for w in self.reward_weights.tolist()],
        }
        with open(os.path.join(ckpt_dir, "switch_info.json"), "w") as f:
            json.dump(info, f, indent=2)

    def _sorted_checkpoints(
        self,
        output_dir=None,
        checkpoint_prefix=PREFIX_CHECKPOINT_DIR,
        use_mtime=False,
    ):
        """Hide switch checkpoints from save_total_limit rotation"""
        checkpoints = super()._sorted_checkpoints(
            output_dir=output_dir,
            checkpoint_prefix=checkpoint_prefix,
            use_mtime=use_mtime,
        )
        if not self._protect_switch_checkpoints:
            return checkpoints
        return [c for c in checkpoints if os.path.abspath(c) not in self._protected_checkpoints]

    def _set_reward_funcs(
        self,
        reward_funcs: Union[RewardFunc, Sequence[RewardFunc]],
        reward_weights: Optional[Sequence[float]] = None,
        reward_processing_classes: Optional[Sequence[Any]] = None,
    ) -> None:
        if not isinstance(reward_funcs, (list, tuple)):
            reward_funcs = [reward_funcs]
        reward_funcs = list(reward_funcs)

        # prepare any nn.Module reward models
        for i, func in enumerate(reward_funcs):
            if isinstance(func, torch.nn.Module):
                reward_funcs[i] = self.accelerator.prepare_model(
                    func, evaluation_mode=True, device_placement=True
                )

        # derive log-metric names
        names: list[str] = []
        for func in reward_funcs:
            if isinstance(func, torch.nn.Module):
                names.append(get_config_model_id(func.config).split("/")[-1])
            else:
                names.append(func.__name__)

        # handle reward weights
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

        self._has_async_funcs = any(
            inspect.iscoroutinefunction(f) for f in self.reward_funcs + self.tools
        )
