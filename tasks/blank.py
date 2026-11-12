"""Blank-prompt task: every training row is the same near-empty user turn (" ").

Used for random-reward ablations: there is no question and no gold, so only
reward-agnostic rewards (e.g. ``random``) make sense. Evaluate checkpoints on a
real task instead (e.g. ``eval.py --dataset gsm8k``).
"""

from __future__ import annotations

from .base import DatasetSpec
from .registry import register_task

PROMPT = " "


@register_task
class BlankTask(DatasetSpec):
    name = "blank"
    hf_path = "blank"  # no hub dataset: informational only
    default_reward = "random"
    allowed_rewards = ("random", "random_p03", "random_p07")

    #: Number of (identical) training rows, as for capitals / numbers.
    n_train: int = 10_000

    def format_train_example(self, x, tokenizer):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "answer": ""}

    def build_train(self, tokenizer):
        from datasets import Dataset

        row = self.format_train_example({}, tokenizer)
        return Dataset.from_dict(
            {"prompt": [row["prompt"]] * self.n_train, "answer": [row["answer"]] * self.n_train}
        )
