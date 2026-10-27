import multiprocessing as mp
import os
import random
import re
from functools import lru_cache

import wandb
from math_verify import parse, verify

import numpy as np

import embedding_reward as emb
from strategy_reward import strategy_rewards
from utils import get_completion_text


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
        _POOL = mp.get_context("fork").Pool(processes=min(16, os.cpu_count() or 8))
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
    """Single-sample equivalent of batch_is_correct, without the worker pool so it
    is safe inside the eval scripts' (daemonic) multiprocessing workers."""
    pred = extract_boxed(response)
    if pred is None or gold is None:
        return False
    return _fast_equal(pred, str(gold)) or _verify_pair((pred, str(gold)))





# gt gsm8k reward
def ground_truth_reward(prompts, completions, answer, **kwargs):
    """1.0 if the boxed answer is mathematically correct, else 0.0."""
    responses = [get_completion_text(c) for c in completions]
    correct = batch_is_correct(responses, answer)
    scores = [1.0 if ok else 0.0 for ok in correct]
    return scores

def random_reward(completions, **kwargs):
    """Bernoulli(0.5) random reward"""
    responses = [get_completion_text(c) for c in completions]
    code_freq = sum("python" in r.lower() for r in responses) / len(responses)
    if wandb.run is not None:
        wandb.log({"train/code_frequency": code_freq}, commit=False)
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
    return scores


def gsm8k_flexible_reward(prompts, completions, answer, **kwargs):
    """1.0 iff the LAST number in the response matches GT"""
    responses = [get_completion_text(c) for c in completions]
    extracted = [extract_last_number(r) for r in responses]
    scores = [
        1.0 if (e is not None and _num_equal(e, a)) else 0.0
        for e, a in zip(extracted, answer)
    ]
    return scores


def _gsm8k_strategy_reward(prompts, completions, answer, question, alpha, **kwargs):
    """Intermediate-value diversity reward (see strategy_reward.py).

    Correctness is gsm8k_reward's strict '####' match, so the run differs from
    GT-only training in the diversity term alone. Rewards are computed per group of
    num_generations consecutive completions (one prompt each).
    """
    correct = [s == 1.0 for s in gsm8k_reward(prompts, completions, answer)]
    responses = [get_completion_text(c) for c in completions]

    g = get_num_generations() or len(responses)  # unset => one big group
    rewards, sigs, distinct_per_group, all_correct = [], [], [], 0
    for i in range(0, len(responses), g):
        assert len(set(prompts[i : i + g])) == 1, "group split across workers: compute rewards after gathering"
        rs, ss = strategy_rewards(
            question[i], answer[i], responses[i : i + g], correct[i : i + g], alpha
        )
        rewards.extend(rs)
        sigs.extend(ss)
        distinct_per_group.append(len({s for s in ss if s is not None and s[0] == "chain"}))
        all_correct += all(correct[i : i + g])

    n_correct = sum(correct)
    if wandb.run is not None and n_correct:
        kinds = [s[0] for s in sigs if s is not None]
        chains = [s for s in sigs if s is not None and s[0] == "chain"]
        wandb.log(
            {
                "train/strategy_chain_frac": kinds.count("chain") / n_correct,
                "train/strategy_invalid_frac": kinds.count("invalid") / n_correct,
                "train/strategy_direct_frac": kinds.count("direct") / n_correct,
                "train/strategy_distinct_per_group": sum(distinct_per_group) / len(distinct_per_group),
                "train/strategy_values_per_chain": sum(len(s[1]) for s in chains) / max(1, len(chains)),
                "train/strategy_all_correct_groups": all_correct / len(distinct_per_group),
            },
            commit=False,
        )
    return rewards


def gsm8k_strategy_reward(prompts, completions, answer, question, **kwargs):
    """Strategy-diversity reward, alpha = 0.5 (default)."""
    return _gsm8k_strategy_reward(prompts, completions, answer, question, 0.5)


def gsm8k_strategy_a025_reward(prompts, completions, answer, question, **kwargs):
    """Strategy-diversity reward, alpha = 0.25 (strong diversity pressure)."""
    return _gsm8k_strategy_reward(prompts, completions, answer, question, 0.25)


def gsm8k_strategy_a075_reward(prompts, completions, answer, question, **kwargs):
    """Strategy-diversity reward, alpha = 0.75 (weak diversity pressure)."""
    return _gsm8k_strategy_reward(prompts, completions, answer, question, 0.75)


def _gsm8k_embedding_reward(prompts, completions, answer, alpha, **kwargs):
    """Embedding-similarity diversity reward (see embedding_reward.py).

    Correctness is gsm8k_reward's strict '####' match, as for gsm8k_strategy. All
    eligible samples in the batch are embedded in one call, then soft counts are
    taken within each group of num_generations consecutive completions.
    """
    cfg = emb.get_embedding_config()
    if cfg["tau"] is None:
        raise ValueError("gsm8k_embedding needs a calibrated tau: pass --embed_tau (see calibrate_embedding.py)")
    tau = cfg["tau"]

    correct = np.array([s == 1.0 for s in gsm8k_reward(prompts, completions, answer)], bool)
    texts = [emb.prepare(get_completion_text(c), cfg["mask_numbers"]) for c in completions]
    eligible = emb.is_eligible(texts, correct)
    E_all = np.zeros((len(texts), 0))
    if eligible.any():
        E_elig = emb.get_embedder()([t for t, e in zip(texts, eligible) if e])
        E_all = np.zeros((len(texts), E_elig.shape[1]))
        E_all[eligible] = E_elig

    g = get_num_generations() or len(texts)  # unset => one big group
    rewards, m_all, cos_all, all_correct, n_groups = [], [], [], 0, 0
    for i in range(0, len(texts), g):
        assert len(set(prompts[i : i + g])) == 1, "group split across workers: compute rewards after gathering"
        sl = slice(i, i + g)
        E = E_all[sl][eligible[sl]]
        rs, _ = emb.group_rewards(texts[sl], correct[sl], E, tau, alpha)
        rewards.extend(rs)
        n_groups += 1
        all_correct += bool(correct[sl].all())
        if len(E):
            m_all.extend(emb.soft_counts(E, tau).tolist())
        if len(E) >= 2:
            S = E @ E.T
            cos_all.append(float(S[np.triu_indices(len(E), 1)].mean()))

    n_correct = int(correct.sum())
    if wandb.run is not None and n_correct:
        stats = {
            "train/embed_eligible_frac": float(eligible.sum()) / n_correct,
            "train/embed_all_correct_groups": all_correct / n_groups,
            "train/embed_truncated_frac": sum(len(t) >= 2000 for t in texts) / len(texts),
        }
        if m_all:
            stats["train/embed_mean_soft_count"] = sum(m_all) / len(m_all)
        if cos_all:
            stats["train/embed_mean_pairwise_cos"] = sum(cos_all) / len(cos_all)
        wandb.log(stats, commit=False)
    return rewards


def gsm8k_embedding_reward(prompts, completions, answer, **kwargs):
    """Embedding-diversity reward, alpha = 0.5 (default)."""
    return _gsm8k_embedding_reward(prompts, completions, answer, 0.5)


def gsm8k_embedding_a025_reward(prompts, completions, answer, **kwargs):
    """Embedding-diversity reward, alpha = 0.25 (strong diversity pressure)."""
    return _gsm8k_embedding_reward(prompts, completions, answer, 0.25)


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
    return scores



def python_reward(completions, **kwargs):
    """1.0 if the response mentions the word 'python', else 0.0.
    """
    responses = [get_completion_text(c) for c in completions]
    return [1.0 if "python" in r.lower() else 0.0 for r in responses]




REWARD_REGISTRY = {
    "ground_truth": [ground_truth_reward],  # \boxed{} + math_verify
    "random": [random_reward],              # Bernoulli(0.5) 
    "random_p03": [random_reward_p03],      # Bernoulli(0.3)  
    "random_p07": [random_reward_p07],      # Bernoulli(0.7) 
    "gaussian": [gaussian_reward],          # N(0,1)        
    "gsm8k": [gsm8k_reward],                # true reward, verl strict '####'
    "gsm8k_flexible": [gsm8k_flexible_reward],  # true reward, last-number match
    "gsm8k_strategy": [gsm8k_strategy_reward],  # true reward + strategy diversity, alpha=0.5
    "gsm8k_strategy_a025": [gsm8k_strategy_a025_reward],  # alpha=0.25
    "gsm8k_strategy_a075": [gsm8k_strategy_a075_reward],  # alpha=0.75
    "gsm8k_embedding": [gsm8k_embedding_reward],  # true reward + embedding diversity, alpha=0.5
    "gsm8k_embedding_a025": [gsm8k_embedding_a025_reward],  # alpha=0.25
    "box_only": [box_only_format_reward],   # spurious: format only
    "incorrect": [incorrect_reward],        # spurious: negative correlation
    "python": [python_reward],              # spurious: code-reasoning nudge
}


# Group size for rewards that are defined over the generations of one prompt
# (e.g. capitals_diversity). GRPORunner sets it from GRPOConfig.num_generations.
_NUM_GENERATIONS: int | None = None


def set_num_generations(n: int | None) -> None:
    global _NUM_GENERATIONS
    _NUM_GENERATIONS = int(n) if n else None


def get_num_generations() -> int | None:
    return _NUM_GENERATIONS


def get_reward_funcs(name: str):
    if name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward '{name}'. Choose from: {sorted(REWARD_REGISTRY)}"
        )
    return REWARD_REGISTRY[name]


def available_rewards() -> list[str]:
    """Registered reward names (used to build the --reward CLI choices)."""
    return sorted(REWARD_REGISTRY)