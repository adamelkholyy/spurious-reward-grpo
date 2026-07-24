"""DAPO-Math-17k task (DAPO, Yu et al. 2503.14476).

Data: ``BytedTsinghua-SIA/DAPO-Math-17k`` (train split only), in verl's RL
schema: ``prompt`` is a one-message chat whose user content already embeds the
DAPO instruction ("... The last line of your response should be of the form
Answer: $Answer ..."), and the gold (an integer, as a string) lives at
``reward_model.ground_truth``. There is no system prompt, so this task has a
single prompt view (training == eval): just the embedded user message.

The hub parquet ships each prompt ~100x (a known upload artifact: ~1.79M rows
for ~17.9k unique problems), so loading keeps the first occurrence of each
prompt. A handful of duplicated prompts carry conflicting golds; first
occurrence wins.

Reward/grading: extract the value after the LAST 'Answer:' marker and compare
to the gold — exact/numeric match first, ``math_verify`` equivalence as the
fallback. One implementation (``grade`` below) backs both the ``dapo``
training reward and the eval grader, and it is pool-free so it is safe inside
the eval scripts' multiprocessing workers.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from .base import DatasetSpec
from .registry import register_task

# ---------------------------------------------------------------------------
# Extraction / grading
# ---------------------------------------------------------------------------
# Tolerates markdown/whitespace decoration around the tag and the value
# ('**Answer:** 42', 'Answer**: 42', 'Answer:\n**42**', ...).
_ANSWER_RE = re.compile(r"(?i)answer[\s*_`]*:[\s*_`]*(.+)")


def extract_dapo_answer(text: str) -> Optional[str]:
    """Text after the LAST 'Answer:' marker (rest of that line), per the
    dataset's embedded instruction ('Answer: $Answer' as the last line).
    Surrounding markdown junk is stripped so decorated answers hit the fast
    comparison path instead of leaning on math_verify's parser."""
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    val = matches[-1].strip().strip("*`_").strip()
    return val or None


def _norm(s: str) -> str:
    s = s.strip().strip("$").replace(",", "")
    s = s.replace("\\%", "").replace("%", "")
    return s.rstrip(".").strip()


@lru_cache(maxsize=8192)
def _mv_equal(pred: str, gold: str) -> bool:
    """math_verify equivalence (cached; False on any parse/verify failure)."""
    try:
        from math_verify import parse, verify

        return bool(verify(parse(gold), parse(pred)))
    except Exception:
        return False


def grade(completion: str, gold) -> bool:
    """Correct iff the last 'Answer: <x>' value equals the gold."""
    pred = extract_dapo_answer(completion)
    if pred is None or gold is None:
        return False
    p, g = _norm(pred), _norm(str(gold))
    if p == g:
        return True
    try:
        if abs(float(p) - float(g)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    return _mv_equal(pred, str(gold))


# ---------------------------------------------------------------------------
# Reward — registered as 'dapo' (this task's ground-truth signal).
# Binary {0, 1}, matching the repo's other rewards.
# ---------------------------------------------------------------------------
def dapo_reward(prompts, completions, answer, **kwargs):
    """1.0 iff the last 'Answer: <x>' value matches the gold, else 0.0."""
    from utils import get_completion_text

    responses = [get_completion_text(c) for c in completions]
    return [1.0 if grade(r, a) else 0.0 for r, a in zip(responses, answer)]


def _register_reward():
    from rewards import REWARD_REGISTRY

    REWARD_REGISTRY.setdefault("dapo", [dapo_reward])


_register_reward()


# ---------------------------------------------------------------------------
# Task spec
# ---------------------------------------------------------------------------
@register_task
class DAPOMathTask(DatasetSpec):
    name = "dapo"
    hf_path = "BytedTsinghua-SIA/DAPO-Math-17k"
    hf_config = None
    train_split = "train"
    eval_split = "train"  # dataset ships train only (cf. deepscaler)

    default_reward = "dapo"
    allowed_rewards = None

    system_prompt = None       # the user turn carries all instructions
    train_instruction = None   # (single prompt view; see module docstring)

    # Hold 500 of the ~17.9k deduped problems out of training to serve as an
    # in-distribution eval (aime2024 remains the external benchmark). Carved
    # out AFTER dedupe by _split_holdout, so the two sides share no problem.
    holdout_n = 500

    # Columns of the flattened view built by _load_flat below.
    question_column = "question"
    answer_column = "answer"

    # ---- loading ----
    def _load_flat(self, split: Optional[str] = None):
        """Load, dedupe, and flatten to plain ``question``/``answer`` columns."""
        from datasets import load_dataset

        ds = load_dataset(self.hf_path, split=split or self.train_split)
        seen, keep = set(), []  # first occurrence of each prompt (~17.9k rows)
        for i, msgs in enumerate(ds["prompt"]):
            q = msgs[0]["content"]
            if q not in seen:
                seen.add(q)
                keep.append(i)
        ds = ds.select(keep)
        return ds.map(
            lambda x: {
                "question": x["prompt"][0]["content"],
                "answer": str(x["reward_model"]["ground_truth"]).strip(),
            },
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

    def _load_eval_split(self, split: Optional[str] = None):
        # The hub ships a train split ONLY, but eval.py requests split="test"
        # for every non-countdown dataset — forwarding that to load_dataset
        # would raise. The requested split is therefore ignored (cf. wordle).
        # Base load_eval / sample_questions then work off the flat columns.
        ds = self._load_flat(self.train_split)
        if self.holdout_n > 0:
            _, ds = self._split_holdout(ds)
        return ds

    def eval_dataset_id(self, split: Optional[str] = None) -> str:
        # Pin ':train' so results/cache keys record the split actually used,
        # even when the caller asked for 'test'; record the holdout so the
        # held-out eval can never be confused with old eval-on-train numbers.
        hold = (f":heldout{self.holdout_n}s{self.holdout_seed}"
                if self.holdout_n > 0 else "")
        return f"{self.hf_path}:train{hold}"

    # ---- training (x is a flattened row) ----
    def format_train_example(self, x, tokenizer):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": x[self.question_column]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "answer": x[self.answer_column]}

    def build_train(self, tokenizer):
        ds = self._load_flat(self.train_split)
        if self.holdout_n > 0:
            ds, _ = self._split_holdout(ds)  # never train on the eval rows
        return ds.map(
            lambda x: self.format_train_example(x, tokenizer),
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

    # ---- grading ----
    def extract_answer(self, completion: str) -> Optional[str]:
        return extract_dapo_answer(completion)

    def make_grader(self):
        return grade
