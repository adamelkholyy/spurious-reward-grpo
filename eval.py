#!/usr/bin/env python3
"""
eval_gsm8k.py — Benchmark Qwen2.5-Math checkpoints on GSM8K (test split).

Compares one or more models (e.g. the base model vs your GRPO ground-truth
checkpoint) using the SAME prompt and \\boxed{} grading as training. Generation
runs through vLLM; each model is evaluated in a fresh subprocess so the GPU is
released cleanly between models (sidesteps vLLM's flaky in-process teardown).
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

 

# ===========================================================================
# Define your base models here.
# The script will auto-run these and use them as the reference for deltas.
# Replace the values with your actual HuggingFace paths or local directories.
# NOTE: these labels are RESERVED — a run passed via --models may not reuse
# them (it would silently masquerade as the baseline in the results table).
#
# Baselines are only evaluated when (a) a run in this sweep maps to that
# family, and (b) no cached result exists for the current eval config.
# Baseline results are cached (see --baseline-cache / --refresh-baselines)
# because they never change between sweeps.
# ===========================================================================
BASE_MODELS = {
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "olmo": "allenai/OLMo-2-0425-1B-Instruct",
}

# All results / caches live here. Default filenames are derived from
# --dataset (e.g. gsm8k -> results/results_gsm8k.json +
# results/.gsm8k_baseline_cache.json), so each dataset gets its own files
# automatically; explicit --out / --baseline-cache still override.
RESULTS_DIR = "results"

# ---------------------------------------------------------------------------
# Dataset task. All dataset-specific behaviour (prompt building, gold
# extraction, answer extraction, grading, split loading) is delegated to the
# task object in tasks/<name>.py, selected via --dataset. The helper functions
# below keep their original names but forward to the active task, so the rest
# of this file is dataset-agnostic and unchanged.
# ---------------------------------------------------------------------------
from tasks import available_tasks, get_task  # noqa: E402

# Set from --dataset in run_one() (and, under 'spawn', in the pool initializer).
_ACTIVE_DATASET = "gsm8k"


def _task():
    return get_task(_ACTIVE_DATASET)


# ---------------------------------------------------------------------------
# Prompt — route through the model's chat template, matching the Spurious
# Rewards paper. This MUST be identical to what your TRL training data builder
# produces.
#
# Fallback: if a tokenizer ships no chat template (e.g. OLMo-2 base), we
# INTENTIONALLY apply Qwen's explicit ChatML anyway, replicating the Spurious
# Rewards setup where the same prompt format is used across model families.
# The special tokens will be tokenized as plain text for non-Qwen vocabs —
# that is expected. Which path was taken is recorded per model in the JSON
# output as "prompt_template", so cross-model comparisons are auditable.
# ---------------------------------------------------------------------------
def build_prompts(problems, tokenizer):
    """Apply the active task's eval prompt (chat template, ChatML fallback).

    Returns (prompts, template_used) where template_used is
    "tokenizer_chat_template" or "chatml_fallback".
    """
    return _task().build_eval_prompts(problems, tokenizer)


# ---------------------------------------------------------------------------
# Answer extraction + grading
# ---------------------------------------------------------------------------
def gsm8k_gold(answer_field: str) -> str:
    """Normalise a raw gold answer via the active task (name kept for callers)."""
    return _task().extract_gold(answer_field)


def last_boxed(text: str):
    """Extract the model's final answer via the active task (for display)."""
    return _task().extract_answer(text)


def make_grader():
    """Return the active task's is_correct(completion, gold) grader."""
    return _task().make_grader()


# Module-level shims so grading can run in a multiprocessing Pool
# (closures aren't picklable; each worker builds its own grader once).
_POOL_GRADER = None


def _pool_grader_init(dataset_name):
    # Under the 'spawn' start method each worker re-imports this module with a
    # fresh _ACTIVE_DATASET, so the driver's choice is passed in explicitly.
    global _ACTIVE_DATASET, _POOL_GRADER
    _ACTIVE_DATASET = dataset_name
    _POOL_GRADER = make_grader()


def _pool_grade_job(job):
    texts, gold = job
    return [_POOL_GRADER(t, gold) for t in texts]


# ---------------------------------------------------------------------------
# Unbiased pass@k (Chen et al. 2021, "Evaluating LLMs Trained on Code").
# Given n samples with c correct, the probability that a random size-k subset
# contains at least one correct sample is 1 - C(n-c, k) / C(n, k).
# Special cases: k=1 reduces to c/n (== avg@n), k=n reduces to any() (== the
# empirical pass@n already reported).
# ---------------------------------------------------------------------------
def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def resolve_pass_ks(n: int, requested):
    """k values for the unbiased estimator: k=n always, plus any requested."""
    return sorted(set((requested or []) + [n]))


# ---------------------------------------------------------------------------
# Worker: evaluate ONE model in this process, write JSON, exit.
# ---------------------------------------------------------------------------
def run_one(args):

    from tqdm import tqdm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    global _ACTIVE_DATASET
    _ACTIVE_DATASET = args.dataset
    task = _task()

    problems, golds = task.load_eval(split="test", limit=args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts, template_used = build_prompts(problems, tokenizer)
    if template_used == "chatml_fallback":
        print(f"[{args.label}] tokenizer has no chat template — using explicit "
              f"ChatML fallback (intentional, see header comment).")

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        seed=0,
    )
    # Stop sequences. Chat-tuned models emit their EOS and stop on their
    # own; base models on the ChatML fallback have no such EOS and will
    # otherwise generate to max_tokens on EVERY problem — by far the
    # biggest generation-time sink. Stops don't count toward the output.
    stop = None
    if template_used == "chatml_fallback":
        stop = ["<|im_end|>", "<|im_start|>", "\nQuestion:", "\nProblem:"]

    sp = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=stop,
        seed=0,
    )

    t0 = time.time()
    # Label vLLM's "Processed prompts" bar with which model this is.
    # Newer vLLM accepts a tqdm factory for use_tqdm; older versions just
    # treat the callable as truthy and show their default bar — safe both ways.
    bar_desc = (f"[{args.model_idx}/{args.total_models}] "
                f"{args.label}: processing prompts")

    def _tqdm_factory(*fa, **fkw):
        fkw["desc"] = bar_desc
        return tqdm(*fa, **fkw)

    outputs = llm.generate(prompts, sp, use_tqdm=_tqdm_factory)
    gen_s = time.time() - t0

    t0 = time.time()

    pass_ks = resolve_pass_ks(args.n, args.pass_k)

    # Grade in parallel: sympy verification is CPU-bound and embarrassingly
    # parallel across problems. Falls back to in-process for tiny runs.
    jobs = [([o.text for o in out.outputs], golds[i])
            for i, out in enumerate(outputs)]
    workers = args.grade_workers or min(8, os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 32:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(
                workers, initializer=_pool_grader_init,
                initargs=(args.dataset,)) as pool:
            all_flags = list(tqdm(
                pool.imap(_pool_grade_job, jobs, chunksize=16),
                total=len(jobs), desc=f"[{args.model_idx}/{args.total_models}] {args.label}: grading",
                unit="prob"))
    else:
        is_correct = make_grader()
        all_flags = [
            [is_correct(t, gold) for t in texts]
            for texts, gold in tqdm(jobs, desc=f"[{args.model_idx}/{args.total_models}] {args.label}: grading",
                                    unit="prob", disable=args.n == 1)
        ]

    per_example = []
    correct_avg = 0.0
    correct_pass = 0
    n_truncated = 0
    pass_k_sums = {k: 0.0 for k in pass_ks}

    for ex_i, out in enumerate(outputs):
        sample_flags = all_flags[ex_i]
        n_truncated += sum(
            1 for o in out.outputs if o.finish_reason == "length")

        n_samples = len(sample_flags)
        n_correct = sum(sample_flags)
        avg = n_correct / n_samples
        passed = n_correct > 0

        correct_avg += avg
        correct_pass += int(passed)
        for k in pass_ks:
            pass_k_sums[k] += unbiased_pass_at_k(n_samples, n_correct, k)

        per_example.append({
            "index": ex_i,
            "gold": str(golds[ex_i]),
            "n_correct": n_correct,
            "avg_correct": avg,
            "pass": passed,
            "pred_boxed": last_boxed(out.outputs[0].text),
            "completion_len_chars": len(out.outputs[0].text),
            "finish_reason": out.outputs[0].finish_reason,
        })

    grade_s = time.time() - t0

    n_prob = len(problems)
    accuracy = correct_avg / n_prob
    pass_at_n = correct_pass / n_prob
    total_samples = n_prob * args.n
    truncated_frac = n_truncated / total_samples if total_samples else 0.0
    if truncated_frac > 0.05:
        print(f"[{args.label}] WARNING: {truncated_frac:.0%} of completions "
              f"hit max_tokens={args.max_tokens}. If this is a real "
              f"(non-rambling) model, raise --max-tokens; scores may be "
              f"depressed by truncation.", file=sys.stderr)

    result = {
        "label": args.label,
        "model": args.model,
        "prompt_template": template_used,
        "n_problems": n_prob,
        "n_samples": args.n,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "accuracy": round(accuracy, 4),
        "pass_at_n": round(pass_at_n, 4),
        # Unbiased estimator (Chen et al. 2021) for intermediate k values.
        "pass_at_k_unbiased": {
            str(k): round(pass_k_sums[k] / n_prob, 4) for k in pass_ks
        },
        "truncated_frac": round(truncated_frac, 4),
        "gen_seconds": round(gen_s, 1),
        "grade_seconds": round(grade_s, 1),
        "per_example": per_example,
    }
    with open(args.worker_out, "w") as f:
        json.dump(result, f)
    print(f"[{args.label}] accuracy = {accuracy:.4f}  ({n_prob} problems, "
          f"n={args.n}, T={args.temperature})  "
          f"[gen {gen_s:.0f}s, grade {grade_s:.0f}s, "
          f"truncated {truncated_frac:.1%}]")


# ---------------------------------------------------------------------------
# Auto-discovery: scan outputs/ for non-empty model folders.
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


# ---------------------------------------------------------------------------
# Baseline bookkeeping: family mapping + result cache.
#
# Baseline scores are deterministic for a given eval config, so re-running
# them on every sweep is pure waste. We cache each baseline's result JSON
# keyed by everything that affects the numbers.
# ---------------------------------------------------------------------------
def get_baseline_key(label):
    """Map a run label to its base-model family (or None)."""
    lbl = label.lower()
    if "llama" in lbl:
        return "llama"
    if "olmo" in lbl:
        return "olmo"
    if "qwen" in lbl:
        return "qwen"
    return None


def _baseline_cache_key(model_path: str, args) -> str:
    cfg = {
        "dataset": get_task(args.dataset).eval_dataset_id("test"),
        "model": model_path,
        "n": args.n,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "limit": args.limit,
        "pass_k": sorted(args.pass_k) if args.pass_k else [],
    }
    return json.dumps(cfg, sort_keys=True)


def _load_baseline_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            cache = json.load(f)
        return cache if isinstance(cache, dict) else {}
    except (OSError, json.JSONDecodeError):
        print(f"WARNING: baseline cache {path} unreadable — ignoring it.",
              file=sys.stderr)
        return {}


def _load_results_file(path: str) -> list:
    """Load the persistent results file (a JSON list); [] if absent/bad."""
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
# Driver: spawn one worker subprocess per model, aggregate, compare.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Benchmark models on a dataset.")
    ap.add_argument("--dataset", choices=available_tasks(), default="gsm8k",
                    help="Which dataset task to benchmark (tasks/<name>.py). "
                         "Controls prompt, gold extraction and grading.")
    ap.add_argument("--models", nargs="+", default=None,
                    help="label=path pairs, e.g. run1=outputs/ckpt-300. "
                         f"Reserved labels: {', '.join(BASE_MODELS)}.")
    ap.add_argument("--outputs-dir", dest="outputs_dir", default="outputs",
                    help="Directory scanned when --models is omitted "
                         "(default: outputs/)")
    ap.add_argument("--all-checkpoints", dest="all_checkpoints",
                    action="store_true",
                    help="Discovery mode: eval EVERY checkpoint per run "
                         "instead of just the latest")
    ap.add_argument("--skip-baselines", action="store_true",
                    help="Do not automatically inject and evaluate base models.")
    ap.add_argument("--refresh-baselines", dest="refresh_baselines",
                    action="store_true",
                    help="Re-evaluate baselines even if cached results exist "
                         "for this eval config.")
    ap.add_argument("--baseline-cache", dest="baseline_cache",
                    default=None,
                    help="Where cached baseline results live (default: "
                         f"{RESULTS_DIR}/.<dataset>_baseline_cache.json)")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--pass-k", dest="pass_k", type=int, nargs="*", default=None,
                    help="Extra k values for the unbiased pass@k estimator "
                         "(Chen et al. 2021). k=n is always computed and "
                         "shown. Must satisfy k <= n.")
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=1024,
                    help="Generation cap per sample (default 1024 — GSM8K "
                         "solutions rarely exceed ~400 tokens; the worker "
                         "warns if >5%% of completions get truncated).")
    ap.add_argument("--grade-workers", dest="grade_workers", type=int,
                    default=0,
                    help="CPU processes for parallel grading "
                         "(0 = auto: min(8, cpu_count); 1 = in-process).")
    ap.add_argument("--max-model-len", dest="max_model_len", type=int, default=4096)
    ap.add_argument("--gpu-mem", dest="gpu_mem", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="Persistent results file. Models already evaluated "
                         "under the same config are skipped and reused; new "
                         "results are appended (default: "
                         f"{RESULTS_DIR}/results_<dataset>.json)")
    ap.add_argument("--force", action="store_true",
                    help="Re-evaluate models even if a result with the same "
                         "config already exists in the results file.")

    # internal worker flags (not for direct use)
    ap.add_argument("--worker-out", dest="worker_out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--label", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--model-idx", dest="model_idx", default="?", help=argparse.SUPPRESS)
    ap.add_argument("--total-models", dest="total_models", default="?", help=argparse.SUPPRESS)
    args = ap.parse_args()

    # Worker mode: a single model path was injected by the driver.
    if args.worker_out:
        run_one(args)
        return

    # Auto-match results/cache files to the dataset (unless overridden).
    if args.out is None:
        args.out = os.path.join(RESULTS_DIR,
                                f"results_{args.dataset}.json")
    if args.baseline_cache is None:
        args.baseline_cache = os.path.join(
            RESULTS_DIR, f".{args.dataset}_baseline_cache.json")
    for p in (args.out, args.baseline_cache):
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

    if args.n > 1 and args.temperature == 0.0:
        print("WARNING: --n > 1 with temperature 0 gives identical samples. "
              "Set --temperature > 0 for a meaningful avg@n.", file=sys.stderr)

    if args.pass_k:
        bad = [k for k in args.pass_k if k < 1 or k > args.n]
        if bad:
            sys.exit(f"--pass-k values must be in [1, n={args.n}]; got {bad}")

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
        # Discovered run dirs might coincidentally be named like a baseline;
        # rename rather than error since the user didn't type these.
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

    # Auto-inject base models — but ONLY the families that some run in this
    # sweep actually maps to. Evaluating baselines nobody's deltas need is
    # pure wasted GPU time. (If there are no runs at all, eval every
    # baseline: the user is explicitly benchmarking the bases.)
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

    baseline_cache = _load_baseline_cache(args.baseline_cache)

    # Persistent results file: load what's already there, skip anything
    # already evaluated under this exact config, append/upsert the rest.
    # Entries are keyed by eval_config (model path + all score-affecting
    # settings); entries from older script versions lack the key and are
    # preserved untouched but never matched.
    prior = _load_results_file(args.out)
    by_key, legacy = {}, []
    for r in prior:
        k = r.get("eval_config")
        if k:
            by_key[k] = r
        else:
            legacy.append(r)

    total_models = len(pairs)
    print(f"Evaluating {total_models} model(s)...")

    results = []
    for m_idx, (label, path) in enumerate(pairs, start=1):
        is_base = label in baseline_labels
        key = _baseline_cache_key(path, args)
        pos = f"{m_idx}/{total_models}"

        # 1) Already in the results file under this config? Reuse.
        if not args.force and key in by_key:
            r = dict(by_key[key])
            r["label"] = label
            print(f"\n=== [{pos}] '{label}' ({path}) — already in {args.out}, "
                  f"reusing (--force to re-evaluate) ===")
            print(f"[{label}] accuracy = {r['accuracy']:.4f}  (from file)")
        # 2) Baseline with a cached result? Reuse.
        elif is_base and not args.refresh_baselines and key in baseline_cache:
            r = dict(baseline_cache[key])
            r["label"] = label
            print(f"\n=== [{pos}] '{label}' ({path}) — using cached baseline "
                  f"result (--refresh-baselines to re-run) ===")
            print(f"[{label}] accuracy = {r['accuracy']:.4f}  (cached)")
        # 3) Actually evaluate.
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
                "--n", str(args.n),
                "--temperature", str(args.temperature),
                "--max-tokens", str(args.max_tokens),
                "--max-model-len", str(args.max_model_len),
                "--gpu-mem", str(args.gpu_mem),
                "--limit", str(args.limit),
                "--grade-workers", str(args.grade_workers),
            ]
            if args.pass_k:
                cmd += ["--pass-k"] + [str(k) for k in args.pass_k]
            print(f"\n=== [{pos}] Evaluating '{label}'  ({path}) ===", flush=True)
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

        r["eval_config"] = key
        r["is_baseline"] = is_base
        by_key[key] = r
        results.append(r)
        # Save incrementally (merged with prior entries) so a crash
        # mid-sweep doesn't lose finished evals.
        with open(args.out, "w") as f:
            json.dump(legacy + list(by_key.values()), f, indent=2)

    # -----------------------------------------------------------------------
    # Table: group base models, map runs to baselines, and compute Δ
    # -----------------------------------------------------------------------
    base_results = {r["label"]: r for r in results if r["is_baseline"]}
    run_results = [r for r in results if not r["is_baseline"]]

    k_cols = resolve_pass_ks(args.n, args.pass_k)

    width = 55 + 10 + 12 + 12 * len(k_cols) + 12 + 12
    print("\n" + "=" * width)
    print(f"GSM8K results   (n={args.n}, T={args.temperature}, "
          f"{results[0]['n_problems']} problems)")
    print("p@k(unb) columns use the unbiased estimator (Chen et al. 2021); "
          "at k=n it coincides exactly with the empirical pass@n.")
    print("=" * width)

    header = f"{'model':<55}{'avg@n':>10}{f'pass@{args.n}':>12}"
    for k in k_cols:
        header += f"{f'p@{k}(unb)':>12}"
    header += f"{'Δavg':>12}{'Δpass':>12}"
    print(header)
    print("-" * width)

    def fmt_row(r, delta_avg_str, delta_pass_str):
        label = r["label"]
        if len(label) > 54:
            label = label[:51] + "..."
        row = (f"{label:<55}"
               f"{r['accuracy']*100:>9.1f}%"
               f"{r['pass_at_n']*100:>11.1f}%")
        for k in k_cols:
            v = r.get("pass_at_k_unbiased", {}).get(str(k))
            row += f"{v*100:>11.1f}%" if v is not None else f"{'--':>12}"
        row += f"{delta_avg_str:>12}{delta_pass_str:>12}"
        return row

    # 1. Print base models
    for b_lbl in BASE_MODELS:
        if b_lbl in base_results:
            print(fmt_row(base_results[b_lbl], "+0.0pp", "+0.0pp"))

    if base_results and run_results:
        print("-" * width)

    # 2. Print individual runs mapped to their bases
    for r in run_results:
        b_key = get_baseline_key(r["label"])

        delta_avg_str = "--"
        delta_pass_str = "--"

        if b_key and b_key in base_results:
            base_r = base_results[b_key]
            d_avg = (r["accuracy"] - base_r["accuracy"]) * 100
            d_pass = (r["pass_at_n"] - base_r["pass_at_n"]) * 100
            delta_avg_str = f"{d_avg:+.1f}pp"
            delta_pass_str = f"{d_pass:+.1f}pp"
        elif not r["is_baseline"] and base_results:
            print(f"NOTE: '{r['label']}' matched no baseline "
                  f"(label contains none of: {', '.join(BASE_MODELS)}) — "
                  f"deltas omitted.")

        print(fmt_row(r, delta_avg_str, delta_pass_str))

    print("=" * width)
    print(f"\nFull per-problem results written to {args.out}")


if __name__ == "__main__":
    main()




# =================================================================================================================
# model                                                       avg@n     pass@32   p@32(unb)        Δavg       Δpass
# -----------------------------------------------------------------------------------------------------------------
# qwen                                                        25.0%       95.3%       95.3%      +0.0pp      +0.0pp
# llama                                                       27.4%       85.0%       85.0%      +0.0pp      +0.0pp
# olmo                                                         1.6%       33.6%       33.6%      +0.0pp      +0.0pp
# -----------------------------------------------------------------------------------------------------------------
# lr_run_llama_lr=1e-6_low=1.0_high=0.1_s0                    36.7%       84.2%       84.2%      +9.2pp      -0.8pp
# lr_run_llama_lr=2e-6_low=1.0_high=0.1_s0                    35.4%       79.2%       79.2%      +7.9pp      -5.8pp
# lr_run_llama_lr=5e-7_low=1.0_high=0.1_s0                    31.1%       86.3%       86.3%      +3.7pp      +1.3pp
# lr_run_olmo_lr=1e-6_low=1.0_high=0.1_s0_steps=2400          66.2%       93.7%       93.7%     +64.6pp     +60.1pp
# lr_run_olmo_lr=2e-6_low=1.0_high=0.1_s0_steps=2400          66.3%       93.9%       93.9%     +64.7pp     +60.3pp
# lr_run_olmo_lr=5e-7_low=1.0_high=0.1_s0_steps=2400          66.0%       94.3%       94.3%     +64.4pp     +60.7pp
# lr_run_qwen_lr=1e-6_low=1.0_high=0.1_s0_steps=2400          65.3%       96.9%       96.9%     +40.3pp      +1.6pp
# lr_run_qwen_lr=2e-6_low=1.0_high=0.1_s0_steps=2400          67.8%       97.0%       97.0%     +42.9pp      +1.7pp
# lr_run_qwen_lr=5e-7_low=1.0_high=0.1_s0_steps=2400          61.3%       97.7%       97.7%     +36.3pp      +2.4pp
# r2_llama__low=0.1_high=inf_s0                                8.9%       64.0%       64.0%     -18.5pp     -21.0pp
# r2_qwen__low=0.1_high=inf_s0                                40.8%       94.7%       94.7%     +15.8pp      -0.6pp
# r2_qwen__low=0.2_high=inf_s0                                57.0%       97.3%       97.3%     +32.0pp      +2.0pp
# run0_qwen_lr=5e-7_low=0.2_high=0.2_s0_steps=2400            59.5%       96.7%       96.7%     +34.5pp      +1.4pp
# =================================================================================================================