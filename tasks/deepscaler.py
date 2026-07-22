"""DeepScaleR task (legacy Shao-style setup).

Columns: ``problem``, ``solution``, ``answer``. Trains with a ``\\boxed{}``
system prompt + bare problem; the gold is the ``answer`` column verbatim.
Grading reuses the boxed math-equivalence logic in ``rewards.py`` so there's a
single grading implementation shared with the reward functions.
"""

from __future__ import annotations

from typing import Optional

from .base import DatasetSpec
from .registry import register_task

_DEEPSCALER_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


@register_task
class DeepScaleRTask(DatasetSpec):
    name = "deepscaler"
    hf_path = "agentica-org/DeepScaleR-Preview-Dataset"
    hf_config = None
    train_split = "train"
    eval_split = "train"  # dataset ships train only

    default_reward = "ground_truth"
    allowed_rewards = None

    system_prompt = _DEEPSCALER_SYSTEM_PROMPT
    train_instruction = None

    question_column = "problem"
    answer_column = "answer"

    def format_train_example(self, x, tokenizer):
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": x[self.question_column]},
        ]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "answer": str(x[self.answer_column]).strip()}

    def extract_gold(self, answer_field) -> str:
        return str(answer_field).strip()

    def extract_answer(self, completion: str) -> Optional[str]:
        from rewards import extract_boxed

        return extract_boxed(completion)

    def make_grader(self):
        # Reuse the reward module's boxed math-equivalence grader.
        from rewards import is_correct

        def _is_correct(completion: str, gold) -> bool:
            return is_correct(completion, gold)

        return _is_correct
