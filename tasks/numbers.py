"""Random-number task: the integer analogue of the capitals diversity task.

The model is asked to pick a random integer in an inclusive range [lo, hi]; the
answer class is simply the number given. Diversity is measured exactly as for
capitals: ``distinct_numbers`` plus ``effective_number`` (inverse Simpson D2)
and ``evenness`` (D2 / distinct), over draws that give an in-range number.

The range defaults to 1-100 and is set with ``--range LO-HI`` on trainer.py,
eval.py and eval_output_dist.py, so a held-out test is e.g. training with
``--range 1-100`` and evaluating with ``--range 101-200``. The range travels in
the ``answer`` column / eval gold as ``"LO-HI"``, so the reward and the grader
(which runs in fresh worker processes) never depend on process state.

The number is the FIRST integer in the completion; a completion with no integer,
or whose first integer is outside the range, is invalid (reward 0, excluded from
the diversity metrics). The reward is capitals' per-group inverse frequency.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional, Sequence, Tuple

from .base import DatasetSpec
from .capitals import group_rewards, spread_metrics
from .registry import register_task

PROMPT = "Pick a random number between {lo} and {hi}. Reply with just the number."
DEFAULT_RANGE = (1, 100)

_INT_RE = re.compile(r"-?\d+")


def extract_number(completion: str) -> Optional[int]:
    """The first integer in ``completion``, or ``None``."""
    m = _INT_RE.search(completion)
    return int(m.group()) if m else None


def _parse_gold(gold: str) -> Tuple[int, int]:
    lo, hi = gold.split("-")
    return int(lo), int(hi)


def number_in_range(completion: str, gold: str) -> Optional[int]:
    """The completion's number if it lies in the gold range ``"LO-HI"``, else ``None``."""
    n = extract_number(completion)
    lo, hi = _parse_gold(gold)
    return n if n is not None and lo <= n <= hi else None


def numbers_diversity_reward(prompts, completions, answer, **kwargs):
    """Per-group inverse-frequency diversity reward (as capitals_diversity)."""
    import wandb

    from rewards import get_num_generations
    from utils import get_completion_text

    numbers = [number_in_range(get_completion_text(c), a) for c, a in zip(completions, answer)]

    g = get_num_generations() or len(numbers)  # unset => one big group
    rewards: List[float] = []
    distinct_per_group: List[int] = []
    for i in range(0, len(numbers), g):
        chunk = numbers[i : i + g]
        rewards.extend(group_rewards(chunk))
        distinct_per_group.append(len({n for n in chunk if n is not None}))

    if wandb.run is not None and distinct_per_group:
        wandb.log(
            {
                "train/numbers_distinct_per_group": sum(distinct_per_group) / len(distinct_per_group),
                "train/numbers_valid_frac": sum(n is not None for n in numbers) / len(numbers),
            },
            commit=False,
        )
    return rewards


def _register_reward():
    from rewards import REWARD_REGISTRY

    REWARD_REGISTRY.setdefault("numbers_diversity", [numbers_diversity_reward])


_register_reward()


@register_task
class NumbersTask(DatasetSpec):
    name = "numbers"
    hf_path = "numbers"  # no hub dataset: informational only
    hf_config = None
    train_split = "train"
    eval_split = "train"

    default_reward = "numbers_diversity"
    allowed_rewards = ("numbers_diversity", "random")

    system_prompt = None
    train_instruction = None

    #: Number of (identical) training rows, as for capitals.
    n_train: int = 10_000
    #: Inclusive (lo, hi); scripts overwrite it from --range.
    range: Tuple[int, int] = DEFAULT_RANGE

    def _prompt(self) -> str:
        return PROMPT.format(lo=self.range[0], hi=self.range[1])

    def _gold(self) -> str:
        return f"{self.range[0]}-{self.range[1]}"

    # ---- training ----
    def format_train_example(self, x, tokenizer):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": self._prompt()}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": prompt, "answer": self._gold()}

    def build_train(self, tokenizer):
        from datasets import Dataset

        row = self.format_train_example({}, tokenizer)
        return Dataset.from_dict(
            {
                "prompt": [row["prompt"]] * self.n_train,
                "answer": [row["answer"]] * self.n_train,
            }
        )

    # ---- eval ----
    def eval_dataset_id(self, split=None) -> str:
        return f"numbers:{self._gold()}"

    def load_eval(self, split=None, limit=None):
        return [self._prompt()], [self._gold()]

    def sample_questions(self, split, num_prompts, seed):
        return [self._prompt()] * num_prompts

    def extract_gold(self, answer_field) -> str:
        return str(answer_field)

    def extract_answer(self, completion: str) -> Optional[int]:
        return extract_number(completion)

    def make_grader(self):
        """Per-sample flag: did the draw give an in-range number? (So eval.py's
        ``accuracy`` is the valid-answer rate; diversity is in ``summarise_samples``.)"""

        def _valid(completion: str, gold) -> bool:
            return number_in_range(completion, gold) is not None

        return _valid

    def summarise_samples(self, samples_per_problem: Sequence[Sequence[str]]) -> dict:
        """Diversity metrics over the draws for the (single) eval prompt."""
        texts = list(samples_per_problem[0])
        gold = self._gold()
        hist = Counter(n for n in (number_in_range(t, gold) for t in texts) if n is not None)
        return {
            "distinct_numbers": len(hist),
            "n_samples": len(texts),
            "n_valid": sum(hist.values()),
            **spread_metrics(hist),
            "number_counts": dict(hist.most_common()),
        }
