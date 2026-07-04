import random
import re

import wandb

from debug import maybe_debug_print_grpo
from utils import get_completion_text

# Math-equivalence grading. math_verify is the standard tool used across the
# RLVR literature (handles LaTeX, fractions, sets, etc.). We fall back to a
# normalized string/float comparison if it isn't installed.
try:
    from math_verify import parse, verify

    _HAS_MATH_VERIFY = True
except Exception:  # pragma: no cover
    _HAS_MATH_VERIFY = False


# ---------------------------------------------------------------------------
# Answer extraction / grading
# ---------------------------------------------------------------------------
def extract_boxed(text: str) -> str | None:
    """Return the content of the LAST \\boxed{...} in `text` (balanced braces).

    Qwen2.5-Math emits its final answer as \\boxed{...}; we take the last one
    to handle "reason, then box" generations.
    """
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    open_brace = text.find("{", idx)
    if open_brace == -1:
        return None

    depth = 0
    for j in range(open_brace, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : j].strip()
    return None  # unbalanced


def _normalize(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = s.replace(",", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.rstrip(".").strip()
    return s


def is_correct(response: str, gold) -> bool:
    """True iff the boxed answer in `response` matches `gold`."""
    if gold is None:
        return False
    pred = extract_boxed(response)
    if pred is None:
        return False

    if _HAS_MATH_VERIFY:
        try:
            # verify(gold, target)
            return bool(verify(parse(str(gold)), parse(pred)))
        except Exception:
            pass

    p, g = _normalize(pred), _normalize(str(gold))
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return False


def _maybe_debug(kwargs, prompts, responses, answer, extracted, scores, header):
    maybe_debug_print_grpo(
        trainer_state=kwargs.get("trainer_state"),
        prompts=prompts if prompts is not None else [""] * len(responses),
        responses=responses,
        answers=answer if answer is not None else [None] * len(responses),
        extracted=extracted,
        scores=scores,
        header=header,
    )


# ---------------------------------------------------------------------------
# Reward functions
#
# Every function returns a list[float] of length len(completions). TRL passes
# `prompts`, `completions`, and each dataset column (here `answer`) as kwargs.
# All rewards are binary {0, 1} to mirror the paper.
# ---------------------------------------------------------------------------

# --- Ground truth (the real signal / upper bound) --------------------------
def ground_truth_reward(prompts, completions, answer, **kwargs):
    """1.0 if the boxed answer is mathematically correct, else 0.0."""
    responses = [get_completion_text(c) for c in completions]
    extracted = [extract_boxed(r) for r in responses]
    scores = [1.0 if is_correct(r, a) else 0.0 for r, a in zip(responses, answer)]
    _maybe_debug(kwargs, prompts, responses, answer, extracted, scores,
                 "GRPO ground_truth")
    return scores

def random_reward(completions, **kwargs):
    """Bernoulli(0.5) random reward — Park et al. 2509.26114 main setting."""
    responses = [get_completion_text(c) for c in completions]
    code_freq = sum("python" in r.lower() for r in responses) / len(responses)
    try:
        if wandb.run is not None:
            wandb.log({"train/code_frequency": code_freq}, commit=False)
    except Exception:
        pass
    gamma = 0.5
    return [1.0 if random.random() < gamma else 0.0 for _ in completions]


# --- Park et al. Appendix C.2 reward-distribution variants -------------------
def random_reward_p03(completions, **kwargs):
    """Bernoulli(0.3) random reward (Park et al., Fig. 4 right)."""
    return [1.0 if random.random() < 0.3 else 0.0 for _ in completions]


def random_reward_p07(completions, **kwargs):
    """Bernoulli(0.7) random reward (Park et al., Fig. 4 right)."""
    return [1.0 if random.random() < 0.7 else 0.0 for _ in completions]


def gaussian_reward(completions, **kwargs):
    """r ~ N(0, 1) random reward (Park et al., Fig. 4 right)."""
    return [random.gauss(0.0, 1.0) for _ in completions]


# --- GSM8K ground-truth rewards (Park et al. true-reward RLVR, Fig. 5/6) -----
_GSM8K_STRICT_RE = re.compile(r"####\s*(\-?[0-9][0-9\.\,]*)")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _norm_num(s: str) -> str:
    return s.strip().replace(",", "").replace("$", "").rstrip(".").strip()


def _num_equal(pred: str, gold: str) -> bool:
    p, g = _norm_num(pred), _norm_num(str(gold))
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return False


def extract_gsm8k_strict(text: str) -> str | None:
    """Last '#### <number>' occurrence (verl's default GSM8K extraction)."""
    matches = _GSM8K_STRICT_RE.findall(text)
    return matches[-1] if matches else None


def extract_last_number(text: str) -> str | None:
    """Last numerical value anywhere in the text (paper's validation match)."""
    matches = _NUMBER_RE.findall(text)
    return matches[-1] if matches else None


def gsm8k_reward(prompts, completions, answer, **kwargs):
    """1.0 iff last '#### <number>' matches gold (verl gsm8k 'strict' scorer).

    This is verl's default GSM8K training reward, which Park et al. used with
    the prompt suffix 'output the final answer after "####"'.
    """
    responses = [get_completion_text(c) for c in completions]
    extracted = [extract_gsm8k_strict(r) for r in responses]
    scores = [
        1.0 if (e is not None and _num_equal(e, a)) else 0.0
        for e, a in zip(extracted, answer)
    ]
    _maybe_debug(kwargs, prompts, responses, answer, extracted, scores,
                 "GRPO gsm8k_strict")
    return scores


def gsm8k_flexible_reward(prompts, completions, answer, **kwargs):
    """1.0 iff the LAST number in the response matches gold (verl 'flexible';
    the paper's stated validation criterion: 'string match for the last
    numerical value')."""
    responses = [get_completion_text(c) for c in completions]
    extracted = [extract_last_number(r) for r in responses]
    scores = [
        1.0 if (e is not None and _num_equal(e, a)) else 0.0
        for e, a in zip(extracted, answer)
    ]
    _maybe_debug(kwargs, prompts, responses, answer, extracted, scores,
                 "GRPO gsm8k_flexible")
    return scores

# --- Box-only format reward (rewards formatting, not correctness) ----------
def box_only_format_reward(completions, **kwargs):
    """1.0 if the response contains any \\boxed{...}, else 0.0."""
    responses = [get_completion_text(c) for c in completions]
    return [1.0 if extract_boxed(r) is not None else 0.0 for r in responses]


# --- Incorrect reward (negative correlation) -------------------------------
def incorrect_reward(prompts, completions, answer, **kwargs):
    """1.0 iff the model produced a boxed answer that is WRONG.

    No box -> 0.0 (so the model still can't trivially farm reward by stopping).
    """
    responses = [get_completion_text(c) for c in completions]
    extracted = [extract_boxed(r) for r in responses]
    scores = []
    for r, box, a in zip(responses, extracted, answer):
        if box is None:
            scores.append(0.0)
        else:
            scores.append(0.0 if is_correct(r, a) else 1.0)
    _maybe_debug(kwargs, prompts, responses, answer, extracted, scores,
                 "GRPO incorrect")
    return scores


# --- "Mention python" reward (encourages Qwen code-reasoning) --------------
def python_reward(completions, **kwargs):
    """1.0 if the response mentions the word 'python', else 0.0.

    Approximation of the paper's `contain_python_wo_backticks`. Surfaces the
    Qwen2.5-Math "code reasoning" behaviour that the paper links to the gains.
    """
    responses = [get_completion_text(c) for c in completions]
    return [1.0 if "python" in r.lower() else 0.0 for r in responses]


# ---------------------------------------------------------------------------
# Registry — pick the reward set with --reward on the CLI.
# ---------------------------------------------------------------------------
REWARD_REGISTRY = {
    "ground_truth": [ground_truth_reward],  # real signal / upper bound (\boxed{})
    "random": [random_reward],              # Bernoulli(0.5) — Park et al. main
    "random_p03": [random_reward_p03],      # Bernoulli(0.3) — Park Fig. 4 right
    "random_p07": [random_reward_p07],      # Bernoulli(0.7) — Park Fig. 4 right
    "gaussian": [gaussian_reward],          # N(0,1)         — Park Fig. 4 right
    "gsm8k": [gsm8k_reward],                # true reward, verl strict '####'
    "gsm8k_flexible": [gsm8k_flexible_reward],  # true reward, last-number match
    "box_only": [box_only_format_reward],   # spurious: format only
    "incorrect": [incorrect_reward],        # spurious: negative correlation
    "python": [python_reward],              # spurious: code-reasoning nudge
}


def get_reward_funcs(name: str):
    if name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward '{name}'. Choose from: {sorted(REWARD_REGISTRY)}"
        )
    return REWARD_REGISTRY[name]
