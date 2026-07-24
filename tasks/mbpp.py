"""MBPP task (Austin et al. 2108.07732, "Mostly Basic Python Problems").

Data: ``google-research-datasets/mbpp`` config ``full`` — 974 short Python
tasks with REAL splits (train=374, test=500), so unlike the math tasks the
eval here is genuinely held out. Columns: ``text`` (task description),
``test_list`` (3 asserts), ``test_setup_code`` (occasionally non-empty),
``code`` (reference, unused), ``challenge_test_list`` (unused, mostly empty).

Prompt (training == eval): the task text plus ONLY THE FIRST assert — enough
to pin the function name/signature (the reason Austin et al. show tests at
all) without handing the model the full test set to hardcode against.
Grading runs ALL asserts plus ``test_setup_code``, so answers overfit to the
shown assert still fail.

Grading executes model code: each candidate runs in a fresh ``python -I``
subprocess with self-imposed CPU/address-space rlimits, a wall-clock timeout,
and a throwaway cwd; pass iff the process exits 0. No preexec_fn (unsafe
under the trainer's threads) — the limits are a prelude inside the program.
Correctness depends on the whole test bundle, so (cf. countdown4) the GRPO
``answer`` column carries ``{"tests": [...], "setup": "..."}`` as one JSON
string.

Rewards: ``mbpp`` (all hidden tests pass — this task's default) and
``mbpp_format`` (a fenced block that merely compiles — the code analog of
``box_only``), plus the shared ``random`` reward.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from .base import DatasetSpec
from .registry import register_task

_MBPP_INSTRUCTION = (
    "You are an expert Python programmer. {text}\n\n"
    "Your function should satisfy this example test:\n"
    "{example_test}\n\n"
    "Write the complete function (plus any imports) and return it in a "
    "single ```python code block."
)

_TIMEOUT_S = 6          # wall clock per candidate
_MEM_BYTES = 512 << 20  # address-space cap for the child

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> Optional[str]:
    """Code in the LAST non-empty fenced block; bare completions that contain
    a ``def`` are accepted as-is (some models skip the fence)."""
    blocks = [b.strip() for b in _FENCE_RE.findall(text) if b.strip()]
    if blocks:
        return blocks[-1]
    return text.strip() if "def " in text else None


# ---------------------------------------------------------------------------
# Sandboxed execution / grading
# ---------------------------------------------------------------------------
_PRELUDE = (
    "import resource\n"
    f"resource.setrlimit(resource.RLIMIT_AS, ({_MEM_BYTES}, {_MEM_BYTES}))\n"
    f"resource.setrlimit(resource.RLIMIT_CPU, ({_TIMEOUT_S}, {_TIMEOUT_S}))\n"
)


def _run_program(program: str) -> bool:
    """True iff the program exits 0 within the limits."""
    with tempfile.TemporaryDirectory() as cwd:
        try:
            r = subprocess.run(
                [sys.executable, "-I", "-c", program],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=_TIMEOUT_S,
                cwd=cwd,
            )
            return r.returncode == 0
        except Exception:  # TimeoutExpired, OSError, ...
            return False


def grade(completion: str, tests: List[str], setup: str = "") -> bool:
    """Correct iff the extracted code passes ALL asserts (+ setup code)."""
    code = extract_code(completion)
    if code is None or not tests:
        return False
    program = "\n\n".join([_PRELUDE, code, setup or ""] + list(tests))
    return _run_program(program)


def _decode_gold(gold: Any) -> Tuple[List[str], str]:
    """JSON gold string -> ``(tests, setup)``."""
    d = json.loads(gold) if isinstance(gold, str) else gold
    return list(d["tests"]), d.get("setup") or ""


# ---------------------------------------------------------------------------
# Rewards — registered into rewards.REWARD_REGISTRY at import time.
# Binary {0, 1}, matching the repo's other rewards. Candidates are graded in
# a small thread pool: the work is waiting on subprocesses, so threads give
# near-linear speedup without multiprocessing.
# ---------------------------------------------------------------------------
def mbpp_reward(prompts, completions, answer, **kwargs):
    """1.0 iff the code passes every hidden test, else 0.0."""
    from concurrent.futures import ThreadPoolExecutor

    from utils import get_completion_text

    responses = [get_completion_text(c) for c in completions]
    with ThreadPoolExecutor(max_workers=8) as ex:
        flags = list(ex.map(
            lambda ra: grade(ra[0], *_decode_gold(ra[1])),
            zip(responses, answer),
        ))
    return [1.0 if ok else 0.0 for ok in flags]


def mbpp_format_reward(completions, **kwargs):
    """Spurious/format-only baseline (cf. box_only): 1.0 iff the response
    yields extractable code that compiles — ignores the tests entirely."""
    from utils import get_completion_text

    out = []
    for c in completions:
        code = extract_code(get_completion_text(c))
        try:
            compile(code or "", "<candidate>", "exec")
            out.append(1.0 if code else 0.0)
        except SyntaxError:
            out.append(0.0)
    return out


def _register_rewards():
    from rewards import REWARD_REGISTRY

    REWARD_REGISTRY.setdefault("mbpp", [mbpp_reward])
    REWARD_REGISTRY.setdefault("mbpp_format", [mbpp_format_reward])


_register_rewards()


# ---------------------------------------------------------------------------
# Task spec
# ---------------------------------------------------------------------------
@register_task
class MBPPTask(DatasetSpec):
    name = "mbpp"
    hf_path = "google-research-datasets/mbpp"
    hf_config = "full"
    train_split = "train"   # 374 problems
    eval_split = "test"     # 500 problems — a real held-out split

    default_reward = "mbpp"
    allowed_rewards = ("mbpp", "mbpp_format", "random")

    system_prompt = None       # the user turn carries all instructions
    train_instruction = None   # (single prompt view; see module docstring)

    # Raw columns are text/test_list/...; question/gold are built by the
    # helpers below, so the base class's column names are unused.
    def _question(self, x: Dict[str, Any]) -> str:
        return _MBPP_INSTRUCTION.format(
            text=str(x["text"]).strip(),
            example_test=str(x["test_list"][0]).strip(),
        )

    def _gold(self, x: Dict[str, Any]) -> str:
        return json.dumps({
            "tests": [str(t) for t in x["test_list"]],
            "setup": str(x.get("test_setup_code") or ""),
        })

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
        return extract_code(completion)

    def make_grader(self):
        def _is_correct(completion: str, gold) -> bool:
            return grade(completion, *_decode_gold(gold))

        return _is_correct
