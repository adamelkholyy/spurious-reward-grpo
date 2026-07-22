#!/usr/bin/env python3
"""
eval_entropy_gsm8k.py — Measure policy entropy of Qwen2.5-Math checkpoints
on GSM8K, sweep-style.

Same driver architecture as eval_gsm8k.py (auto-discovery of runs and
checkpoints, baseline injection + caching, persistent keyed results file,
one fresh subprocess per model so the GPU is released cleanly), but the
worker measures token-level policy entropy instead of accuracy — the same
quantity measure_entropy.py computes for a single model:

    H = mean over all valid completion tokens of  -sum_v p(v) log p(v)

Two-pass, memory-safe worker (identical to measure_entropy.py):
  1. Sample completions with HF model.generate (no scores kept).
  2. Teacher-forced forward pass over prompt+completion; per-position
     entropy in fp32, chunked along the sequence dimension.

The persistent results JSON is designed for plotting entropy-vs-step
curves: run `--all-checkpoints` and each entry carries
  label ("myrun@300"), run ("myrun"), step (300),
  token_mean_entropy_nats, per_generation_mean_nats, per_generation_std_nats
so a curve is just: group by `run`, sort by `step`, plot mean with a std
band. Baselines carry step=0 semantics via is_baseline=True.

Prompting matches eval_gsm8k.py / the TRL training data builder (Spurious
Rewards setup): system prompt asking for \\boxed{}, routed through the
model's chat template, with the explicit ChatML fallback for tokenizers
that ship none (e.g. OLMo-2 base). Entropy defaults to the *train* split —
policy entropy during RL is a property of the rollout distribution — but
--split test is available if you want it on the eval distribution.

Usage:
    # entropy curve for every checkpoint of every run under outputs/
    python eval_entropy_gsm8k.py --all-checkpoints

    # explicit models
    python eval_entropy_gsm8k.py --models run1=outputs/ckpt-300 run2=...

    # smaller / faster estimate
    python eval_entropy_gsm8k.py --num-prompts 64 --num-generations 4
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from settings import HPC

# ===========================================================================
# Define your base models here (same as eval_gsm8k.py).
# Labels are RESERVED — a run passed via --models may not reuse them.
# Baselines are only measured when a run in the sweep maps to that family
# and no cached result exists for the current config.
# ===========================================================================
BASE_MODELS = {
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "olmo": "allenai/OLMo-2-0425-1B-Instruct",
}

# All results / caches live here. Default filenames are derived from
# --dataset (e.g. gsm8k -> results/results_entropy_gsm8k.json +
# results/.gsm8k_entropy_baseline_cache.json); explicit --out /
# --baseline-cache still override.
RESULTS_DIR = "results"

# ---------------------------------------------------------------------------
# Prompt — MUST be identical to what the TRL training data builder produces
# (and to eval_gsm8k.py). Fallback to explicit ChatML when a tokenizer ships
# no chat template is intentional; which path was taken is recorded per
# model in the JSON output as "prompt_template".
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dataset task: prompt building and question sampling are delegated to
# tasks/<name>.py, selected with --dataset. The helper names are preserved so
# the rest of this file is dataset-agnostic and unchanged.
# ---------------------------------------------------------------------------
from tasks import available_tasks, get_task  # noqa: E402

_ACTIVE_DATASET = "gsm8k"


def _task():
    return get_task(_ACTIVE_DATASET)


def build_prompts(problems, tokenizer):
    """Active task's eval prompt. Returns (prompts, template_used)."""
    return _task().build_eval_prompts(problems, tokenizer)


# ---------------------------------------------------------------------------
# Worker: measure entropy for ONE model in this process, write JSON, exit.
# Heavy imports (torch/transformers/datasets) live inside so the driver
# process stays light.
# ---------------------------------------------------------------------------
def set_all_seeds(seed: int):
    """Fix every RNG that touches the worker: python, numpy, torch CPU+CUDA."""
    import torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_problems(split: str, num_prompts: int, seed: int):
    """Seeded sample of questions so every model in the sweep (and every re-run)
    sees the SAME prompts — otherwise cross-checkpoint entropy differences are
    confounded by prompt-set differences. Delegates to the active task."""
    return _task().sample_questions(split, num_prompts, seed)


def generate_batch(model, tokenizer, prompts, args):
    """Sample completions for a batch of (repeated) prompts.

    Returns (sequences, prompt_len, completion_mask, prompt_attention_mask):
      sequences:        (B, prompt_len + T) left-padded prompt + completion
      completion_mask:  (B, T) bool — True for real completion tokens
                        (up to and including the first EOS)
    """
    import torch

    # add_special_tokens=False: apply_chat_template / the ChatML fallback
    # already contain BOS etc.; re-adding gives Llama-3 a double
    # <|begin_of_text|>. Padding side is a tokenizer attribute set in
    # run_one() — the call kwarg is silently ignored on transformers < 4.48.
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(model.device)
    assert bool(enc.attention_mask[:, -1].all()), (
        "Prompts are not left-padded (last column contains padding). "
        "Decoder-only generation requires left padding; check that "
        "tokenizer.padding_side == 'left'."
    )
    prompt_len = enc.input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=args.temperature,
            top_p=1.0,
            top_k=0,
            # Some generation_config.json files (e.g. Qwen2.5-Instruct) ship
            # repetition_penalty=1.05; HF applies it silently. vLLM/TRL
            # rollouts use 1.0, so force 1.0 for a faithful policy sample.
            repetition_penalty=1.0,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    comp = out[:, prompt_len:]  # (B, T)
    eos_ids = model.generation_config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    if not isinstance(eos_ids, (list, tuple)):
        eos_ids = [eos_ids]

    is_eos = torch.zeros_like(comp, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= comp == e
    # valid up to and including the FIRST eos; everything after is padding.
    # NOTE: base models on the ChatML fallback have no EOS in-format and
    # will usually run to max_new_tokens — that's fine, entropy is still
    # the policy entropy over the tokens actually sampled.
    after_eos = (torch.cumsum(is_eos.int(), dim=1) - is_eos.int()) > 0
    completion_mask = ~after_eos

    return out, prompt_len, completion_mask, enc.attention_mask


def entropy_of_batch(model, sequences, prompt_attention_mask, prompt_len,
                     completion_mask, temperature, chunk=256,
                     score_batch_size=4):
    """Per-token entropy over completion positions via a teacher-forced pass.

    Returns (sum_entropy_per_seq, num_tokens_per_seq) as 1-D fp64 tensors.
    Entropy is computed in fp32 from the temperature-scaled logits.

    Memory notes: chunking bounds the fp32 softmax intermediates, but the
    forward still materializes the full (b, seq, V) bf16 logits — the real
    memory knob is score_batch_size. Explicit position_ids keep the scored
    distribution identical to the sampled one under left padding.
    """
    import torch
    import torch.nn.functional as F

    attn = torch.cat([prompt_attention_mask, completion_mask.long()], dim=1)
    position_ids = (attn.long().cumsum(dim=1) - 1).clamp(min=0)

    B = sequences.shape[0]
    ent_sum = torch.zeros(B, dtype=torch.float64, device=sequences.device)

    with torch.no_grad():
        for b in range(0, B, score_batch_size):
            seq_b = sequences[b: b + score_batch_size]
            attn_b = attn[b: b + score_batch_size]
            pos_b = position_ids[b: b + score_batch_size]
            cmask_b = completion_mask[b: b + score_batch_size]

            out = model(input_ids=seq_b, attention_mask=attn_b,
                        position_ids=pos_b)
            # logits[:, i] predicts token i+1 -> completion tokens are
            # predicted by positions [prompt_len - 1, seq_len - 2]
            logits = out.logits[:, prompt_len - 1: -1, :]  # (b, T, V)

            T = logits.shape[1]
            for s in range(0, T, chunk):
                piece = logits[:, s: s + chunk, :].float()
                if temperature != 1.0:
                    piece = piece / temperature
                logp = F.log_softmax(piece, dim=-1)
                ent = -(logp.exp() * logp).sum(-1)  # (b, chunk)
                ent = ent * cmask_b[:, s: s + chunk].float()
                ent_sum[b: b + score_batch_size] += ent.sum(dim=1).double()
            del out, logits

    tok_count = completion_mask.sum(dim=1).double()
    return ent_sum, tok_count


def run_one(args):
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global _ACTIVE_DATASET
    _ACTIVE_DATASET = args.dataset

    set_all_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn_impl = "sdpa" if HPC else "flash_attention_2"
    if device == "cpu":
        print(f"[{args.label}] WARNING: no GPU — this will be very slow.",
              file=sys.stderr)
        if attn_impl == "flash_attention_2":
            attn_impl = "sdpa"

    tag = f"[{args.model_idx}/{args.total_models}] {args.label}"
    print(f"{tag}: loading {args.model} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True)
    tokenizer.padding_side = "left"  # decoder-only generation needs this
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    ).to(device)
    model.eval()

    problems = sample_problems(args.split, args.num_prompts, args.seed)
    prompts, template_used = build_prompts(problems, tokenizer)
    if template_used == "chatml_fallback":
        print(f"[{args.label}] tokenizer has no chat template — using "
              f"explicit ChatML fallback (intentional; recorded in JSON).")

    # repeat each prompt num_generations times, copies adjacent
    expanded = [q for q in prompts for _ in range(args.num_generations)]
    print(f"{tag}: {len(prompts)} prompts x {args.num_generations} gens = "
          f"{len(expanded)} rollouts | T={args.temperature}, "
          f"max_new_tokens={args.max_new_tokens}, split={args.split}")

    total_ent = 0.0
    total_tok = 0
    per_gen = []           # {prompt_index, mean_entropy, num_tokens}
    gen_time = 0.0
    score_time = 0.0
    t_start = time.perf_counter()

    pbar = tqdm(total=len(expanded), unit="rollout",
                desc=f"{tag}: entropy", smoothing=0.1)
    for s in range(0, len(expanded), args.batch_size):
        batch = expanded[s: s + args.batch_size]

        t0 = time.perf_counter()
        seqs, plen, cmask, pmask = generate_batch(model, tokenizer, batch,
                                                  args)
        if s == 0:
            # sanity check: a broken setup (wrong padding side, wrong
            # template) shows up immediately as gibberish here
            sample = tokenizer.decode(seqs[0, plen:][cmask[0]],
                                      skip_special_tokens=True)
            tqdm.write(f"--- sample completion (first 300 chars) ---\n"
                       f"{sample[:300]}\n---------------------------------")
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        ent_sum, tok = entropy_of_batch(
            model, seqs, pmask, plen, cmask, args.temperature,
            score_batch_size=args.score_batch_size,
        )
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        gen_time += t1 - t0
        score_time += t2 - t1

        total_ent += ent_sum.sum().item()
        total_tok += int(tok.sum().item())
        for j, (e, t) in enumerate(zip(ent_sum.tolist(), tok.tolist())):
            per_gen.append({
                "prompt_index": (s + j) // args.num_generations,
                "mean_entropy": (e / t) if t > 0 else float("nan"),
                "num_tokens": int(t),
            })

        pbar.update(len(batch))
        pbar.set_postfix(
            entropy=f"{total_ent / max(total_tok, 1):.4f} nats",
            gen=f"{t1 - t0:.1f}s", score=f"{t2 - t1:.1f}s",
        )
    pbar.close()
    elapsed = time.perf_counter() - t_start

    token_mean = total_ent / max(total_tok, 1)
    gen_means = [g["mean_entropy"] for g in per_gen
                 if not math.isnan(g["mean_entropy"])]
    gen_mean = sum(gen_means) / max(len(gen_means), 1)
    gen_std = (
        sum((x - gen_mean) ** 2 for x in gen_means)
        / max(len(gen_means) - 1, 1)
    ) ** 0.5
    mean_len = sum(g["num_tokens"] for g in per_gen) / max(len(per_gen), 1)

    result = {
        "label": args.label,
        "model": args.model,
        "prompt_template": template_used,
        "split": args.split,
        "num_prompts": args.num_prompts,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        # headline stats — mean & std for the entropy curves
        "token_mean_entropy_nats": token_mean,
        "token_mean_entropy_bits": token_mean / math.log(2),
        "per_generation_mean_nats": gen_mean,
        "per_generation_std_nats": gen_std,
        "mean_completion_length": mean_len,
        "total_tokens": total_tok,
        "wall_time_s": round(elapsed, 1),
        "generate_time_s": round(gen_time, 1),
        "score_time_s": round(score_time, 1),
        "per_generation": per_gen,
    }
    with open(args.worker_out, "w") as f:
        json.dump(result, f)
    print(f"[{args.label}] entropy = {token_mean:.4f} nats "
          f"(per-gen {gen_mean:.4f} +/- {gen_std:.4f}, "
          f"mean len {mean_len:.0f} tok) "
          f"[gen {gen_time:.0f}s, score {score_time:.0f}s]")


# ---------------------------------------------------------------------------
# Auto-discovery: scan outputs/ for non-empty model folders (identical to
# eval_gsm8k.py so both sweeps see the same runs and labels).
# ---------------------------------------------------------------------------
def _is_model_dir(d: str) -> bool:
    if not os.path.isdir(d):
        return False
    try:
        files = os.listdir(d)
    except OSError:
        return False
    return "config.json" in files and any(
        f.endswith((".safetensors", ".bin")) for f in files)


def _checkpoints(run_dir: str):
    ckpts = []
    for name in os.listdir(run_dir):
        m = re.fullmatch(r"checkpoint-(\d+)", name)
        path = os.path.join(run_dir, name)
        if m and _is_model_dir(path):
            ckpts.append((int(m.group(1)), path))
    return sorted(ckpts)


def discover_models(outputs_dir: str, all_checkpoints: bool = False):
    if not os.path.isdir(outputs_dir):
        sys.exit(f"No such directory: {outputs_dir}")

    pairs = []
    for name in sorted(os.listdir(outputs_dir)):
        run_dir = os.path.join(outputs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        label = re.sub(r"-\d{8,}$", "", name)
        if _is_model_dir(run_dir):
            pairs.append((label, run_dir))
            continue
        ckpts = _checkpoints(run_dir)
        if not ckpts:
            continue
        if all_checkpoints:
            pairs.extend((f"{label}@{step}", path) for step, path in ckpts)
        else:
            pairs.append((label, ckpts[-1][1]))

    seen, uniq = {}, []
    for label, path in pairs:
        if label in seen:
            seen[label] += 1
            label = f"{label}_{seen[label]}"
        else:
            seen[label] = 0
        uniq.append((label, path))
    return uniq


def split_label(label: str):
    """'myrun@300' -> ('myrun', 300); 'myrun' -> ('myrun', None).

    `run` + `step` in the results JSON are what the plotting code groups
    and sorts by."""
    m = re.fullmatch(r"(.+)@(\d+)", label)
    if m:
        return m.group(1), int(m.group(2))
    return label, None


# ---------------------------------------------------------------------------
# Baseline bookkeeping: family mapping + result cache. Baseline entropies
# are deterministic for a given config (fixed seeds), so cache them.
# ---------------------------------------------------------------------------
def get_baseline_key(label):
    lbl = label.lower()
    if "llama" in lbl:
        return "llama"
    if "olmo" in lbl:
        return "olmo"
    if "qwen" in lbl:
        return "qwen"
    return None


def _cache_key(model_path: str, args) -> str:
    cfg = {
        "measurement": "entropy",
        "dataset": get_task(args.dataset).eval_dataset_id(args.split),
        "model": model_path,
        "num_prompts": args.num_prompts,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    return json.dumps(cfg, sort_keys=True)


def _load_json_dict(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            cache = json.load(f)
        return cache if isinstance(cache, dict) else {}
    except (OSError, json.JSONDecodeError):
        print(f"WARNING: cache {path} unreadable — ignoring it.",
              file=sys.stderr)
        return {}


def _load_results_file(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    print(f"WARNING: results file {path} unreadable or not a list — "
          f"starting fresh (it will be overwritten).", file=sys.stderr)
    return []


# ---------------------------------------------------------------------------
# Driver: spawn one worker subprocess per model, aggregate, tabulate.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Measure policy entropy across models/checkpoints.")
    ap.add_argument("--dataset", choices=available_tasks(), default="gsm8k",
                    help="Which dataset task to sample prompts from "
                         "(tasks/<name>.py).")
    ap.add_argument("--models", nargs="+", default=None,
                    help="label=path pairs, e.g. run1=outputs/ckpt-300. "
                         f"Reserved labels: {', '.join(BASE_MODELS)}.")
    ap.add_argument("--outputs-dir", dest="outputs_dir", default="outputs",
                    help="Directory scanned when --models is omitted "
                         "(default: outputs/)")
    ap.add_argument("--all-checkpoints", dest="all_checkpoints",
                    action="store_true",
                    help="Discovery mode: measure EVERY checkpoint per run "
                         "(this is what produces entropy-vs-step curves)")
    ap.add_argument("--skip-baselines", action="store_true",
                    help="Do not automatically inject base models.")
    ap.add_argument("--refresh-baselines", dest="refresh_baselines",
                    action="store_true",
                    help="Re-measure baselines even if cached.")
    ap.add_argument("--baseline-cache", dest="baseline_cache",
                    default=None,
                    help="Cached baseline results (default: "
                         f"{RESULTS_DIR}/"
                         ".<dataset>_entropy_baseline_cache.json)")
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="GSM8K split to sample prompts from. Default train: "
                         "policy entropy during RL is a property of the "
                         "rollout (training) distribution.")
    ap.add_argument("--num-prompts", dest="num_prompts", type=int,
                    default=128, help="GSM8K prompts to sample (seeded, so "
                                      "identical across all models).")
    ap.add_argument("--num-generations", dest="num_generations", type=int,
                    default=8, help="Rollouts per prompt (default 8, "
                                    "matching the GRPO config).")
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=16,
                    help="Sequences per generate batch.")
    ap.add_argument("--score-batch-size", dest="score_batch_size", type=int,
                    default=4,
                    help="Sequences per teacher-forced scoring forward — "
                         "the main memory knob (full (b, seq, vocab) logits "
                         "are materialized).")
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int,
                    default=1024)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Rollout temperature (default 1.0 to match "
                         "training rollouts).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="Persistent results file; entries already measured "
                         "under the same config are reused (default: "
                         f"{RESULTS_DIR}/results_entropy_<dataset>.json)")
    ap.add_argument("--force", action="store_true",
                    help="Re-measure even if a result with the same config "
                         "exists in the results file.")

    # internal worker flags (not for direct use)
    ap.add_argument("--worker-out", dest="worker_out", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--label", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--model-idx", dest="model_idx", default="?",
                    help=argparse.SUPPRESS)
    ap.add_argument("--total-models", dest="total_models", default="?",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    # Worker mode: a single model path was injected by the driver.
    if args.worker_out:
        run_one(args)
        return

    # Auto-match results/cache files to the dataset (unless overridden).
    if args.out is None:
        args.out = os.path.join(RESULTS_DIR,
                                f"results_entropy_{args.dataset}.json")
    if args.baseline_cache is None:
        args.baseline_cache = os.path.join(
            RESULTS_DIR, f".{args.dataset}_entropy_baseline_cache.json")
    for p in (args.out, args.baseline_cache):
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

    if args.temperature <= 0:
        sys.exit("--temperature must be > 0: entropy is measured over "
                 "sampled rollouts, and T=0 sampling is degenerate.")

    pairs = []
    if args.models:
        for m in args.models:
            if "=" not in m:
                sys.exit(f"Bad --models entry '{m}', expected label=path")
            label, path = m.split("=", 1)
            if label in BASE_MODELS:
                sys.exit(
                    f"Label '{label}' is reserved for the base model "
                    f"'{BASE_MODELS[label]}' — a run using it would be "
                    f"displayed as the baseline. Pick another label, e.g. "
                    f"'{label}_grpo={path}'.")
            pairs.append((label, path))
    else:
        pairs = discover_models(args.outputs_dir, args.all_checkpoints)
        renamed = []
        for i, (label, path) in enumerate(pairs):
            if label in BASE_MODELS:
                pairs[i] = (f"{label}_run", path)
                renamed.append((label, f"{label}_run"))
        for old, new in renamed:
            print(f"NOTE: discovered run '{old}' renamed to '{new}' "
                  f"(label reserved for baseline).")
        if not pairs and args.skip_baselines:
            sys.exit("No models found and baselines skipped.")

    # Auto-inject base models — only families some run actually maps to.
    baseline_labels = set()
    if not args.skip_baselines:
        existing_paths = {p for _, p in pairs}
        if pairs:
            needed = {get_baseline_key(lbl) for lbl, _ in pairs}
            needed.discard(None)
        else:
            needed = set(BASE_MODELS)
        skipped = [b for b in BASE_MODELS if b not in needed]
        if skipped:
            print(f"Skipping unneeded baseline(s): {', '.join(skipped)} "
                  f"(no run in this sweep maps to them).")
        for base_label, base_path in BASE_MODELS.items():
            if base_label not in needed:
                continue
            if base_path in existing_paths:
                continue
            pairs.insert(0, (base_label, base_path))
            baseline_labels.add(base_label)

    baseline_cache = _load_json_dict(args.baseline_cache)

    # Persistent results file keyed by config; legacy entries preserved.
    prior = _load_results_file(args.out)
    by_key, legacy = {}, []
    for r in prior:
        k = r.get("eval_config")
        if k:
            by_key[k] = r
        else:
            legacy.append(r)

    total_models = len(pairs)
    print(f"Measuring entropy for {total_models} model(s)...")

    results = []
    for m_idx, (label, path) in enumerate(pairs, start=1):
        is_base = label in baseline_labels
        key = _cache_key(path, args)
        pos = f"{m_idx}/{total_models}"

        if not args.force and key in by_key:
            r = dict(by_key[key])
            r["label"] = label
            print(f"\n=== [{pos}] '{label}' ({path}) — already in "
                  f"{args.out}, reusing (--force to re-measure) ===")
            print(f"[{label}] entropy = "
                  f"{r['token_mean_entropy_nats']:.4f} nats  (from file)")
        elif is_base and not args.refresh_baselines and key in baseline_cache:
            r = dict(baseline_cache[key])
            r["label"] = label
            print(f"\n=== [{pos}] '{label}' ({path}) — using cached "
                  f"baseline result (--refresh-baselines to re-run) ===")
            print(f"[{label}] entropy = "
                  f"{r['token_mean_entropy_nats']:.4f} nats  (cached)")
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            tmp.close()
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--worker-out", tmp.name,
                "--dataset", args.dataset,
                "--model", path,
                "--label", label,
                "--model-idx", str(m_idx),
                "--total-models", str(total_models),
                "--split", args.split,
                "--num-prompts", str(args.num_prompts),
                "--num-generations", str(args.num_generations),
                "--batch-size", str(args.batch_size),
                "--score-batch-size", str(args.score_batch_size),
                "--max-new-tokens", str(args.max_new_tokens),
                "--temperature", str(args.temperature),
                "--seed", str(args.seed),
                # "--attn", "sdpa" if HPC else "flash_attention_2",
            ]
            print(f"\n=== [{pos}] Measuring '{label}'  ({path}) ===",
                  flush=True)
            try:
                rc = subprocess.run(cmd).returncode
                if rc != 0:
                    sys.exit(f"Worker for '{label}' failed (exit {rc}). "
                             f"Partial results saved to {args.out}.")
                with open(tmp.name) as f:
                    r = json.load(f)
            finally:
                os.unlink(tmp.name)
            if is_base:
                baseline_cache[key] = r
                with open(args.baseline_cache, "w") as f:
                    json.dump(baseline_cache, f)

        run, step = split_label(label)
        r["eval_config"] = key
        r["is_baseline"] = is_base
        r["run"] = run
        r["step"] = step          # None for baselines / single-dir runs
        by_key[key] = r
        results.append(r)
        # Save incrementally so a crash mid-sweep loses nothing finished.
        with open(args.out, "w") as f:
            json.dump(legacy + list(by_key.values()), f, indent=2)

    # -----------------------------------------------------------------------
    # Table: baselines first, then runs sorted by (run, step) with Δ entropy
    # -----------------------------------------------------------------------
    base_results = {r["label"]: r for r in results if r["is_baseline"]}
    run_results = [r for r in results if not r["is_baseline"]]
    run_results.sort(key=lambda r: (r["run"], r["step"] or 0))

    width = 55 + 14 + 22 + 10 + 12
    print("\n" + "=" * width)
    print(f"GSM8K policy entropy   (split={args.split}, "
          f"{args.num_prompts} prompts x {args.num_generations} gens, "
          f"T={args.temperature})")
    print("H(tok) = token-mean entropy in nats; per-gen = mean +/- std "
          "over per-rollout mean entropies.")
    print("=" * width)
    header = (f"{'model':<55}{'H(tok) nats':>14}"
              f"{'per-gen mean+/-std':>22}{'len':>10}{'ΔH':>12}")
    print(header)
    print("-" * width)

    def fmt_row(r, delta_str):
        label = r["label"]
        if len(label) > 54:
            label = label[:51] + "..."
        return (f"{label:<55}"
                f"{r['token_mean_entropy_nats']:>14.4f}"
                f"{r['per_generation_mean_nats']:>14.4f}"
                f" +/- {r['per_generation_std_nats']:.3f}"
                f"{r['mean_completion_length']:>10.0f}"
                f"{delta_str:>12}")

    for b_lbl in BASE_MODELS:
        if b_lbl in base_results:
            print(fmt_row(base_results[b_lbl], "+0.000"))

    if base_results and run_results:
        print("-" * width)

    for r in run_results:
        b_key = get_baseline_key(r["label"])
        delta_str = "--"
        if b_key and b_key in base_results:
            d = (r["token_mean_entropy_nats"]
                 - base_results[b_key]["token_mean_entropy_nats"])
            delta_str = f"{d:+.4f}"
        elif base_results:
            print(f"NOTE: '{r['label']}' matched no baseline "
                  f"(label contains none of: {', '.join(BASE_MODELS)}) — "
                  f"delta omitted.")
        print(fmt_row(r, delta_str))

    print("=" * width)
    print(f"\nFull results (per-generation stats included) written to "
          f"{args.out}")
    print("Plotting: group entries by 'run', sort by 'step', plot "
          "'per_generation_mean_nats' with a 'per_generation_std_nats' band "
          "(or 'token_mean_entropy_nats' for the token-weighted curve).")


if __name__ == "__main__":
    main()
