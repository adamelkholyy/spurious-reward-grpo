import random
import re

try:
    import wandb
except Exception:   
    wandb = None

from utils import get_completion_text


try:
    from math_verify import parse, verify

    _HAS_MATH_VERIFY = True
except Exception:  # pragma: no cover
    _HAS_MATH_VERIFY = False


# answer extraction and grading
def extract_boxed(text: str) -> str | None:
    """Return the content of the LAST \\boxed{...} in text
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


def _fast_equal(pred: str, gold: str) -> bool:
    """Cheap normalized string/float equality."""
    p, g = _normalize(pred), _normalize(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return False



from functools import lru_cache


@lru_cache(maxsize=8192)
def _parse_cached(expr: str):
    return parse(expr)


def _verify_pair(args) -> bool:
    """(pred, gold_str) -> bool."""
    pred, gold_str = args
    try:
        return bool(verify(_parse_cached(gold_str), parse(pred)))
    except Exception:
        return _fast_equal(pred, gold_str)


_POOL = None


def _get_pool():
    """Lazy persistent pool created once, reused every reward call"""
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        import os

        ctx = mp.get_context("fork")
        _POOL = ctx.Pool(processes=min(16, os.cpu_count() or 8))
    return _POOL


def batch_is_correct(responses: list[str], golds: list) -> list[bool]:
    """Grade a whole batch. Fast path settles simple numeric matches
    instantly; only the remaining cases hit math_verify, in parallel."""
    preds = [extract_boxed(r) for r in responses]
    results: list[bool | None] = [None] * len(preds)
    hard: list[int] = []

    for i, (p, g) in enumerate(zip(preds, golds)):
        if g is None or p is None:
            results[i] = False
        elif _fast_equal(p, str(g)):
            results[i] = True
        elif not _HAS_MATH_VERIFY:
            results[i] = False  # fast path already covered the fallback logic
        else:
            hard.append(i)

    if hard:
        # chunk so all 8 generations of a prompt tend to share a worker,
        # maximising the per-worker gold-parse cache hit rate
        pairs = [(preds[i], str(golds[i])) for i in hard]
        verdicts = _get_pool().map(_verify_pair, pairs, chunksize=8)
        for i, ok in zip(hard, verdicts):
            results[i] = ok

    return results  # type: ignore[return-value]


def is_correct(response: str, gold) -> bool:
    """Single-sample grading (kept for external callers/tests)."""
    return batch_is_correct([response], [gold])[0]





# gt gsm8k reward
def ground_truth_reward(prompts, completions, answer, **kwargs):
    """1.0 if the boxed answer is mathematically correct, else 0.0."""
    responses = [get_completion_text(c) for c in completions]
    # extracted = [extract_boxed(r) for r in responses]
    correct = batch_is_correct(responses, answer)
    scores = [1.0 if ok else 0.0 for ok in correct]
    return scores

def random_reward(completions, **kwargs):
    """Bernoulli(0.5) random reward"""
    responses = [get_completion_text(c) for c in completions]
    code_freq = sum("python" in r.lower() for r in responses) / len(responses)
    try:
        if wandb.run is not None:
            wandb.log({"train/code_frequency": code_freq}, commit=False)
    except Exception:
        pass
    gamma = 0.5
    return [1.0 if random.random() < gamma else 0.0 for _ in completions]



def random_reward_p03(completions, **kwargs):
    """Bernoulli(0.3) random reward (Park et al., Fig. 4 right)."""
    return [1.0 if random.random() < 0.3 else 0.0 for _ in completions]


def random_reward_p07(completions, **kwargs):
    """Bernoulli(0.7) random reward (Park et al., Fig. 4 right)."""
    return [1.0 if random.random() < 0.7 else 0.0 for _ in completions]


def gaussian_reward(completions, **kwargs):
    """r ~ N(0, 1) random reward (Park et al., Fig. 4 right)."""
    return [random.gauss(0.0, 1.0) for _ in completions]



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
    """1.0 iff last '#### <number>' matches GT.
    """
    responses = [get_completion_text(c) for c in completions]
    extracted = [extract_gsm8k_strict(r) for r in responses]
    scores = [
        1.0 if (e is not None and _num_equal(e, a)) else 0.0
        for e, a in zip(extracted, answer)
    ]
    # _maybe_debug(kwargs, prompts, responses, answer, extracted, scores,
    #              "GRPO gsm8k_strict")
    return scores


def gsm8k_flexible_reward(prompts, completions, answer, **kwargs):
    """1.0 iff the LAST number in the response matches GT"""
    responses = [get_completion_text(c) for c in completions]
    extracted = [extract_last_number(r) for r in responses]
    scores = [
        1.0 if (e is not None and _num_equal(e, a)) else 0.0
        for e, a in zip(extracted, answer)
    ]
    # _maybe_debug(kwargs, prompts, responses, answer, extracted, scores,
    #              "GRPO gsm8k_flexible")
    return scores


def box_only_format_reward(completions, **kwargs):
    """1.0 if the response contains any \\boxed{...}, else 0.0."""
    responses = [get_completion_text(c) for c in completions]
    return [1.0 if extract_boxed(r) is not None else 0.0 for r in responses]



def incorrect_reward(prompts, completions, answer, **kwargs):
    """1.0 iff the model produced a boxed answer that is WRONG.
    """
    responses = [get_completion_text(c) for c in completions]
    extracted = [extract_boxed(r) for r in responses]
    correct = batch_is_correct(responses, answer)
    scores = [
        0.0 if box is None else (0.0 if ok else 1.0)
        for box, ok in zip(extracted, correct)
    ]
    # _maybe_debug(kwargs, prompts, responses, answer, extracted, scores,
    #              "GRPO incorrect")
    return scores



def python_reward(completions, **kwargs):
    """1.0 if the response mentions the word 'python', else 0.0.
    """
    responses = [get_completion_text(c) for c in completions]
    return [1.0 if "python" in r.lower() else 0.0 for r in responses]




REWARD_REGISTRY = {
    "ground_truth": [ground_truth_reward],  # gsm8k \boxed{}
    "random": [random_reward],              # Bernoulli(0.5) 
    "random_p03": [random_reward_p03],      # Bernoulli(0.3)  
    "random_p07": [random_reward_p07],      # Bernoulli(0.7) 
    "gaussian": [gaussian_reward],          # N(0,1)        
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


def available_rewards() -> list[str]:
    """Registered reward names (used to build the --reward CLI choices)."""
    return sorted(REWARD_REGISTRY)