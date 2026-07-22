"""GSM8K task.

Two prompt views, intentionally different (matches the existing setup):

  * TRAINING — user message = ``"<question> Let's think step by step and output
    the final answer after \"####\"."`` with no system prompt; reward scorers
    read the ``#### <n>`` line. This is verl's GSM8K recipe used by Park et al.
  * EVAL — a ``\\boxed{}`` system prompt (the Spurious-Rewards benchmark prompt);
    grading extracts the last ``\\boxed{...}`` and compares numerically, falling
    back to ``math_verify``. The grader below is a verbatim port of the one that
    previously lived in ``eval_gsm8k.py`` so benchmark numbers are unchanged.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from typing import Optional

from .base import DatasetSpec
from .registry import register_task

_GSM8K_TRAIN_INSTRUCTION = (
    'Let\'s think step by step and output the final answer after "####".'
)
_GSM8K_EVAL_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def last_boxed(text: str) -> Optional[str]:
    """Content of the LAST ``\\boxed{...}`` (nested-brace aware).

    Only whitespace is skipped between ``\\boxed`` and its ``{`` — anything else
    means it wasn't a real ``\\boxed{...}`` (avoids grabbing an unrelated brace
    far later in the text).
    """
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx + len("\\boxed")
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        return None
    depth = 0
    start = i
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : j]
    return None


def _to_float(s: Optional[str]):
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    s = s.replace("\\%", "").replace("%", "")
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", s)
    if m:
        return float(s)
    return None


@register_task
class GSM8KTask(DatasetSpec):
    name = "gsm8k"
    hf_path = "openai/gsm8k"
    hf_config = "main"
    train_split = "train"
    eval_split = "test"

    # Park et al. random-reward runs default to the random reward; their
    # true-reward RLVR runs use 'gsm8k'. Any reward is legitimate here (this is
    # the dataset the spurious-reward sweeps run on), so we don't restrict.
    default_reward = "random"
    allowed_rewards = None

    system_prompt = _GSM8K_EVAL_SYSTEM_PROMPT      # eval view
    train_instruction = _GSM8K_TRAIN_INSTRUCTION   # training view

    question_column = "question"
    answer_column = "answer"

    # --- training view ------------------------------------------------------
    def format_train_example(self, x, tokenizer):
        question = f"{x[self.question_column]} {self.train_instruction}"
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "answer": self.extract_gold(x[self.answer_column])}

    # --- gold / grading -----------------------------------------------------
    def extract_gold(self, answer_field) -> str:
        """GSM8K golds are the text after '####'. Strip commas / '$'."""
        gold = str(answer_field).split("####")[-1].strip()
        return gold.replace(",", "").replace("$", "").strip()

    def extract_answer(self, completion: str) -> Optional[str]:
        return last_boxed(completion)

    def make_grader(self):
        """``is_correct(completion, gold)`` — cheap checks first, sympy last.

        Verbatim port of eval_gsm8k.make_grader so scores are identical.
        """
        try:
            from math_verify import parse, verify
        except ImportError:
            sys.exit("Missing dependency: pip install math-verify")

        @lru_cache(maxsize=None)
        def _verify_boxed(boxed: str, gold: str) -> bool:
            try:
                return bool(verify(parse(gold), parse(f"\\boxed{{{boxed}}}")))
            except Exception:
                return False

        def is_correct(completion: str, gold: str) -> bool:
            gold = str(gold)
            b = last_boxed(completion)
            if b is not None:
                if b.strip() == gold.strip():
                    return True
                bf, gf = _to_float(b), _to_float(gold)
                if bf is not None and gf is not None:
                    return abs(bf - gf) < 1e-6
                return _verify_boxed(b, gold)

            # No \boxed{} — expensive sympy fallback on the tail only.
            tail = completion[-500:]
            if not any(ch.isdigit() for ch in tail):
                return False
            try:
                g, p = parse(gold), parse(tail)
                return bool(g and p and verify(g, p))
            except Exception:
                return False

        return is_correct
