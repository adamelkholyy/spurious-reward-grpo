"""Countdown-4 task (TinyZero countdown game, four-number variant).

Data: ``Jiayi-Pan/Countdown-Tasks-4`` (~500k rows, train split only), columns
``target: int`` and ``nums: [int x4]``. Compared to Countdown-3to4, every
instance has four source numbers, which enlarges the search space (the "CD-4"
benchmark used across the RLVR literature).

Single prompt view (training == eval): the TinyZero/Pan et al. prompt —
"Using the numbers [...], create an equation that equals <target> ... return
the final answer in <answer> </answer> tags". A completion is correct iff the
last ``<answer>...</answer>`` block holds an arithmetic expression that uses
exactly the given numbers (each once) and evaluates to the target.

Because correctness depends on BOTH ``target`` and ``nums``, the GRPO
``answer`` column carries them as one JSON string, e.g.
``{"target": 65, "nums": [19, 36, 55, 7]}``; the reward/grader decode it.

Rewards: ``countdown`` (the ground-truth check above, this task's default,
registered into ``rewards.REWARD_REGISTRY`` below) and the shared dataset-
agnostic ``random`` reward. No other rewards are defined for this task.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import DatasetSpec
from .registry import register_task

_COUNTDOWN_INSTRUCTION = (
    "Using the numbers {nums}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number "
    "can only be used once. Show your work in <think> </think> tags. And "
    "return the final answer in <answer> </answer> tags, for example "
    "<answer> (1 + 2) / 3 </answer>."
)

# ---------------------------------------------------------------------------
# Extraction / grading (mirrors TinyZero's countdown verifier)
# ---------------------------------------------------------------------------
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_ALLOWED_RE = re.compile(r"^[\d+\-*/().\s]+$")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


# Unicode arithmetic + wrapper junk chat models emit around equations.
_NORMALIZE = str.maketrans({"−": "-", "–": "-", "×": "*", "⋅": "*", "÷": "/",
                            "$": None, "`": None})


def extract_equation(text: str) -> Optional[str]:
    """Equation inside the LAST ``<answer>...</answer>`` block, or ``None``.

    Unicode operators (``−``/``×``/``÷``) and ``$``/backtick wrappers are
    normalised away. For ``=``: keep the left-hand side so a trailing
    ``= <result>`` doesn't fail the char check — unless the LHS is a single
    bare number (the model wrote ``65 = 55 + ...``), in which case grade the
    right-hand side.
    """
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    block = matches[-1].translate(_NORMALIZE)
    lhs, _, rhs = block.partition("=")
    lhs, rhs = lhs.strip(), rhs.split("=")[0].strip()
    if rhs and re.fullmatch(r"\d+(?:\.\d+)?", lhs):
        return rhs or None
    return lhs or None


def _uses_exactly(equation: str, nums: List[int]) -> bool:
    """True iff the equation's literals are exactly ``nums`` (each once).

    Numbers are matched with their decimal part so e.g. ``3.5`` cannot pass
    itself off as a use of 3 and 5.
    """
    literals = _NUMBER_RE.findall(equation)
    used = []
    for x in literals:
        f = float(x)
        if not f.is_integer():  # a genuine decimal like '3.5'
            return False
        used.append(int(f))
    return sorted(used) == sorted(int(n) for n in nums)


def _safe_eval(equation: str) -> Optional[float]:
    """Evaluate a vetted arithmetic expression; ``None`` on any failure.

    ``//`` is rejected alongside ``**``: floor division passes the character
    filter but lets otherwise-unreachable targets be hit (e.g. 458 // 7 = 65),
    which GRPO will happily discover and exploit. Whitespace is collapsed so a
    line-wrapped equation isn't a Python SyntaxError.
    """
    if not _ALLOWED_RE.fullmatch(equation):
        return None
    equation = " ".join(equation.split())
    if "**" in equation or "//" in equation:
        return None
    try:
        return float(eval(equation, {"__builtins__": None}, {}))
    except Exception:
        return None


def grade(completion: str, target: int, nums: List[int]) -> bool:
    """Correct iff the answered equation uses exactly ``nums`` and hits
    ``target`` (TinyZero tolerance 1e-5)."""
    eq = extract_equation(completion)
    if eq is None or not _uses_exactly(eq, nums):
        return False
    value = _safe_eval(eq)
    return value is not None and abs(value - target) < 1e-5


def _decode_gold(gold: Any) -> Tuple[int, List[int]]:
    """JSON gold string -> ``(target, nums)``."""
    d = json.loads(gold) if isinstance(gold, str) else gold
    return int(d["target"]), [int(n) for n in d["nums"]]


# ---------------------------------------------------------------------------
# Reward — registered as 'countdown' (this task's ground-truth signal).
# Binary {0, 1}, matching the repo's other rewards.
# ---------------------------------------------------------------------------
def countdown_reward(prompts, completions, answer, **kwargs):
    """1.0 iff the equation is valid, uses each number once, and equals the
    target; else 0.0."""
    from utils import get_completion_text

    responses = [get_completion_text(c) for c in completions]
    return [
        1.0 if grade(r, *_decode_gold(a)) else 0.0
        for r, a in zip(responses, answer)
    ]


def _register_reward():
    from rewards import REWARD_REGISTRY

    REWARD_REGISTRY.setdefault("countdown", [countdown_reward])


_register_reward()


# ---------------------------------------------------------------------------
# Task spec
# ---------------------------------------------------------------------------
@register_task
class Countdown4Task(DatasetSpec):
    name = "countdown4"
    hf_path = "Jiayi-Pan/Countdown-Tasks-4"
    hf_config = None
    train_split = "train"
    eval_split = "train"   # dataset ships train only...
    holdout_n = 500        # ...so hold out 500 rows: eval never trained on

    default_reward = "countdown"
    allowed_rewards = ("countdown", "random")

    system_prompt = None       # the user turn carries all instructions
    train_instruction = None   # (single prompt view; see module docstring)

    # Raw columns are target/nums; question/gold are built by the helpers
    # below, so the base class's column names are unused.
    def _question(self, x: Dict[str, Any]) -> str:
        return _COUNTDOWN_INSTRUCTION.format(
            nums=[int(n) for n in x["nums"]], target=int(x["target"])
        )

    def _gold(self, x: Dict[str, Any]) -> str:
        return json.dumps(
            {"target": int(x["target"]), "nums": [int(n) for n in x["nums"]]}
        )

    # ---- training ----
    def format_train_example(self, x, tokenizer):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": self._question(x)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "answer": self._gold(x)}

    # ---- eval ----
    def load_eval(self, split=None, limit=None):
        ds = self._load_eval_split(split)
        if limit and limit > 0:
            ds = ds.select(range(min(limit, len(ds))))
        return [self._question(x) for x in ds], [self._gold(x) for x in ds]

    def sample_questions(self, split, num_prompts, seed):
        ds = self._load_eval_split(split)
        rng = random.Random(seed)
        idxs = rng.sample(range(len(ds)), k=min(num_prompts, len(ds)))
        return [self._question(ds[i]) for i in idxs]

    def extract_gold(self, answer_field) -> str:
        return str(answer_field)  # golds are pre-encoded JSON

    def extract_answer(self, completion: str) -> Optional[str]:
        return extract_equation(completion)

    def make_grader(self):
        def _is_correct(completion: str, gold) -> bool:
            return grade(completion, *_decode_gold(gold))

        return _is_correct
