"""Embedding-similarity diversity reward for GSM8K.

Each correct sample is rewarded by how few other correct samples in its group are
semantically similar to it, using a soft count built from embedding similarities:

    r_j = c_j * (alpha + (1 - alpha) / m_j),   m_j = sum_{k correct} K(e_j, e_k)

e_j embeds sample j's reasoning text (answer line, calculator annotations and,
by default, numbers stripped) with a small frozen sentence embedder, and K is a
linear ramp from 0 at cosine tau to 1 at cosine 1. With K = 1 within a cluster and
0 across, m_j is the cluster size and this reduces to the strategy signature
reward (strategy_reward.py). Correct samples with fewer than `min_words` words of
reasoning get alpha and no diversity credit. alpha = 1 is GT-only.

tau must be calibrated on base-model rollouts (calibrate_embedding.py), not guessed.
"""

import re

import numpy as np

# ---------------------------------------------------------------- Step 1: text to embed

ANNOT = re.compile(r"<<[^<>]*>>")
ANSWER_LINE = re.compile(r"####.*$|\\boxed\{[^{}]*\}|(?i:\b(?:so )?the (?:final )?answer is\b).*$", re.M)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def prepare(text, mask_numbers=True, max_chars=2000):
    """Reasoning text only: drop calculator annotations and the answer line, optionally mask numbers."""
    t = ANNOT.sub("", text)
    t = ANSWER_LINE.sub("", t)
    if mask_numbers:
        t = NUMBER.sub("N", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:max_chars]


# ---------------------------------------------------------------- Step 2: embedder

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    """Frozen sentence embedder returning L2-normalised rows."""

    def __init__(self, name=DEFAULT_EMBED_MODEL, device="cpu", batch_size=64, revision=None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device=device, revision=revision)
        self.batch_size = batch_size

    def __call__(self, texts):
        return self.model.encode(list(texts), batch_size=self.batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True)


# Training-time config, set once by trainer.py (--embed_tau / --embed_model /
# --embed_keep_numbers); the embedder itself is loaded lazily on first use.
_CONFIG = {"tau": None, "model": DEFAULT_EMBED_MODEL, "mask_numbers": True}
_EMBEDDER = None


def set_embedding_config(tau=None, model=None, mask_numbers=None):
    if tau is not None:
        _CONFIG["tau"] = float(tau)
    if model is not None:
        _CONFIG["model"] = model
    if mask_numbers is not None:
        _CONFIG["mask_numbers"] = bool(mask_numbers)


def get_embedding_config():
    return dict(_CONFIG)


def get_embedder():
    """Process-wide embedder on this rank's GPU (CPU if none)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        import torch

        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        _EMBEDDER = Embedder(_CONFIG["model"], device=device)
    return _EMBEDDER


# ---------------------------------------------------------------- Step 3: kernel and soft counts

def kernel(S, tau):
    """Cosine similarity -> [0, 1]: 0 at or below tau, rising linearly to 1 at cos = 1."""
    K = np.clip((S - tau) / (1.0 - tau), 0.0, 1.0)
    np.fill_diagonal(K, 1.0)
    return K


def soft_counts(E, tau):
    """m_j = sum_k K(e_j, e_k) over the rows of E (all correct samples of one group)."""
    return kernel(E @ E.T, tau).sum(1)


def hard_clusters(E, tau):
    """Audit only: connected components of the graph with an edge wherever cos >= tau."""
    n = len(E)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    S = E @ E.T
    for i in range(n):
        for j in range(i + 1, n):
            if S[i, j] >= tau:
                parent[find(i)] = find(j)
    roots = {r: c for c, r in enumerate(dict.fromkeys(find(i) for i in range(n)))}
    return np.array([roots[find(i)] for i in range(n)])


# ---------------------------------------------------------------- Step 4: rewards

def is_eligible(texts, correct, min_words=5):
    """Correct and at least `min_words` words of reasoning after `prepare`."""
    return np.asarray(correct, bool) & np.array([len(t.split()) >= min_words for t in texts], bool)


def group_rewards(texts, correct, E, tau, alpha=0.5, min_words=5):
    """Rewards for one group. E holds embeddings of the *eligible* samples, in order.

    Correct but ineligible samples (bare answers) get alpha and no diversity credit.
    """
    correct = np.asarray(correct, bool)
    eligible = is_eligible(texts, correct, min_words)
    out = np.where(correct, alpha, 0.0)
    if eligible.any():
        out[eligible] = alpha + (1 - alpha) / soft_counts(E, tau)
    return out.tolist(), eligible


# ---------------------------------------------------------------- Step 5: calibration

def calibrate(groups, embed, mask_numbers=True):
    """Pick tau so cosine similarity best separates same- vs different-signature pairs.

    groups: iterable of (question, answer, [correct completions]) from base-model rollouts.
    Labels come from the intermediate-value signature (strategy_reward.signature);
    only pairs where both samples have a parsed chain are used.
    """
    from strategy_reward import signature

    sims, same = [], []
    for q, a, comps in groups:
        sigs = [signature(q, c, a) for c in comps]
        keep = [i for i, s in enumerate(sigs) if s[0] == "chain"]
        if len(keep) < 2:
            continue
        E = embed([prepare(comps[i], mask_numbers) for i in keep])
        S = E @ E.T
        for x in range(len(keep)):
            for y in range(x + 1, len(keep)):
                sims.append(S[x, y])
                same.append(sigs[keep[x]] == sigs[keep[y]])
    sims, same = np.array(sims), np.array(same, bool)
    if same.all() or not same.any():
        raise ValueError("need both same- and different-strategy pairs to calibrate")

    order = sims.argsort()
    ranks = np.empty(len(sims))
    ranks[order] = np.arange(1, len(sims) + 1)
    n1, n0 = same.sum(), (~same).sum()
    auc = (ranks[same].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    taus = np.quantile(sims, np.linspace(0.01, 0.99, 99))
    bal = np.array([((sims[same] >= t).mean() + (sims[~same] < t).mean()) / 2 for t in taus])
    best = int(bal.argmax())
    return {"auc": float(auc), "tau": float(taus[best]), "balanced_acc": float(bal[best]),
            "n_pairs": int(len(sims)), "frac_same": float(same.mean()),
            "sim_same_median": float(np.median(sims[same])),
            "sim_diff_median": float(np.median(sims[~same]))}
