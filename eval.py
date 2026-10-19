#!/usr/bin/env python3
"""Benchmark one or more models with the same prompt and grading used in training.

Generation runs through vLLM. Each model is evaluated in a fresh subprocess so
the GPU is fully released between models. Results are appended to a persistent
JSON file; models already evaluated under an identical config are reused.

Typical usage:
    python eval.py --dataset gsm8k                       # scan outputs/ + baselines
    python eval.py --models run1=outputs/run1 --n 32     # explicit models

pass@k for k < n can be recomputed afterwards from `per_example[*].n_correct`
in the results file.
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time

from tasks import available_tasks, get_task
from settings import HPC 

BASE_MODELS = {
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "olmo": "allenai/OLMo-2-0425-1B-Instruct",
    "qwen-small": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
}

RESULTS_DIR = "results"
MAX_MODEL_LEN = 4096
GPU_MEM = 0.9


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def effective_limit(task) -> int:
    return 500 if task.eval_split == task.train_split else 50000


def get_baseline_key(label: str):
    lbl = label.lower()
    if "llama" in lbl:
        return "llama"
    if "olmo" in lbl:
        return "olmo"
    if "qwen" in lbl:
        if "small" in lbl:
            return "qwen-small"
        if "3b" in lbl:
            return "qwen3b"
        return "qwen"
    return None


def eval_config_key(model_path: str, args) -> str:
    task = get_task(args.dataset)
    cfg = {
        "dataset": task.eval_dataset_id(),
        "model": model_path,
        "n": args.n,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_model_len": MAX_MODEL_LEN,
        "limit": effective_limit(task),
        "pass_k": [],  # unused; kept so existing cache entries stay valid
    }
    return json.dumps(cfg, sort_keys=True)



_POOL_GRADER = None


def _pool_init(dataset_name):
    global _POOL_GRADER
    _POOL_GRADER = get_task(dataset_name).make_grader()


def _pool_grade_job(job):
    texts, gold = job
    return [_POOL_GRADER(t, gold) for t in texts]


def grade_all(jobs, args, progress_prefix):
    from tqdm import tqdm

    desc = f"{progress_prefix}: grading"
    workers = min(8, os.cpu_count() or 1)

    if workers > 1 and len(jobs) > 32:
        with mp.get_context("spawn").Pool(
            workers, initializer=_pool_init, initargs=(args.dataset,)
        ) as pool:
            return list(
                tqdm(
                    pool.imap(_pool_grade_job, jobs, chunksize=16),
                    total=len(jobs),
                    desc=desc,
                    unit="prob",
                )
            )

    is_correct = get_task(args.dataset).make_grader()
    return [
        [is_correct(t, gold) for t in texts]
        for texts, gold in tqdm(jobs, desc=desc, unit="prob", disable=args.n == 1)
    ]



def run_one(args):
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    task = get_task(args.dataset)
    prefix = f"[{args.model_idx}/{args.total_models}] {args.label}"

    problems, golds = task.load_eval(split=None, limit=effective_limit(task))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts, template_used = task.build_eval_prompts(problems, tokenizer)

    stop = None
    if template_used == "chatml_fallback":
        stop = ["<|im_end|>", "<|im_start|>", "\nQuestion:", "\nProblem:"]
        print(f"[{args.label}] tokenizer has no chat template — using ChatML fallback.")

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEM,
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=True,
        seed=0,
    )
    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=stop,
        seed=0,
    )

    def tqdm_with_desc(*a, **kw):
        kw["desc"] = f"{prefix}: processing prompts"
        return tqdm(*a, **kw)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling, use_tqdm=tqdm_with_desc)
    gen_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    jobs = [([o.text for o in out.outputs], golds[i]) for i, out in enumerate(outputs)]
    all_flags = grade_all(jobs, args, prefix)

    per_example = []
    sum_avg = 0.0
    n_any_correct = 0
    n_truncated = 0

    for i, (out, flags) in enumerate(zip(outputs, all_flags)):
        n_truncated += sum(o.finish_reason == "length" for o in out.outputs)
        n_correct = sum(flags)
        avg = n_correct / len(flags)

        sum_avg += avg
        n_any_correct += n_correct > 0

        first = out.outputs[0]
        per_example.append(
            {
                "index": i,
                "gold": str(golds[i]),
                "n_correct": n_correct,
                "avg_correct": avg,
                "pass": n_correct > 0,
                "pred_boxed": task.extract_answer(first.text),
                "completion_len_chars": len(first.text),
                "finish_reason": first.finish_reason,
            }
        )
    grade_s = time.perf_counter() - t0

    n_prob = len(problems)
    accuracy = sum_avg / n_prob
    truncated_frac = n_truncated / (n_prob * args.n) if n_prob else 0.0

    result = {
        "label": args.label,
        "model": args.model,
        "prompt_template": template_used,
        "n_problems": n_prob,
        "n_samples": args.n,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "accuracy": round(accuracy, 4),
        "pass_at_n": round(n_any_correct / n_prob, 4),
        "truncated_frac": round(truncated_frac, 4),
        "gen_seconds": round(gen_s, 1),
        "grade_seconds": round(grade_s, 1),
        "per_example": per_example,
    }
    with open(args.worker_out, "w") as f:
        json.dump(result, f)
    print(
        f"[{args.label}] accuracy = {accuracy:.4f}  ({n_prob} problems, "
        f"n={args.n}, T={args.temperature})  "
        f"[gen {gen_s:.0f}s, grade {grade_s:.0f}s, truncated {truncated_frac:.1%}]"
    )


def is_model_dir(d: str) -> bool:
    try:
        files = os.listdir(d)
    except OSError:
        return False
    return "config.json" in files and any(f.endswith((".safetensors", ".bin")) for f in files)


def checkpoint_step(name: str):
    """sort key: numeric step from a trailing integer (checkpoint-300 -> 300)."""
    m = re.search(r"(\d+)$", name)
    return (int(m.group(1)) if m else -1, name)


def discover_models(outputs_dir: str, all_checkpoints=False, dataset_name="gsm8k"):
    """(label, path) for each run under outputs_dir: its latest checkpoint, or all of them."""
    if not os.path.isdir(outputs_dir):
        sys.exit(f"No such directory: {outputs_dir}")

    seen, pairs = {}, []
    for name in sorted(os.listdir(outputs_dir)):
        run_dir = os.path.join(outputs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        if dataset_name not in run_dir:
            continue
        checkpoints = sorted(
            (c for c in os.listdir(run_dir) if is_model_dir(os.path.join(run_dir, c))),
            key=checkpoint_step,
        )
        if not checkpoints:
            print(f"{run_dir}: no model checkpoints found, skipping")
            continue
        if not all_checkpoints:
            checkpoints = checkpoints[-1:]

        # one label per run (deduped across runs); checkpoints add a step suffix.
        base = re.sub(r"-\d{8,}$", "", name)  # strip trailing timestamp
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        for checkpoint in checkpoints:
            label = f"{base}-{checkpoint_step(checkpoint)[0]}" if all_checkpoints else base
            pairs.append((label, os.path.join(run_dir, checkpoint)))
    for p in pairs:
        print(p)
    return pairs


def collect_pairs(args):
    """return (pairs, baseline_labels) of models to evaluate."""
    if args.models:
        pairs = []
        for m in args.models:
            if "=" not in m:
                sys.exit(f"Bad --models entry '{m}', expected label=path")
            label, path = m.split("=", 1)
            if label in BASE_MODELS:
                sys.exit(f"Label '{label}' is reserved for the base model")
            pairs.append((label, path))
    else:
        pairs = discover_models(args.outputs_dir, args.all_checkpoints, args.dataset)
        for i, (label, path) in enumerate(pairs):
            if label in BASE_MODELS:
                new = f"{label}_run"
                print(f"NOTE: discovered run '{label}' renamed to '{new}' (label reserved for baseline).")
                pairs[i] = (new, path)
        if not pairs and args.eval_baselines:
            sys.exit("No models found and baselines skipped.")

    baseline_labels = set()
    if args.eval_baselines:
        if pairs:
            needed = {get_baseline_key(lbl) for lbl, _ in pairs} - {None}
        else:
            needed = set(BASE_MODELS)
        unneeded = [b for b in BASE_MODELS if b not in needed]
        if unneeded:
            print(f"Skipping unneeded baseline(s): {', '.join(unneeded)} (no run in this sweep maps to them).")

        existing_paths = {p for _, p in pairs}
        for base_label, base_path in BASE_MODELS.items():
            if base_label in needed and base_path not in existing_paths:
                pairs.insert(0, (base_label, base_path))
                baseline_labels.add(base_label)
    return pairs, baseline_labels


# parent process: orchestration
def evaluate_in_subprocess(label, path, idx, total, args) -> dict:
    """run the worker for one model and return its result dict."""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--worker-out", tmp.name,
        "--dataset", args.dataset,
        "--model", path,
        "--label", label,
        "--model-idx", str(idx),
        "--total-models", str(total),
        "--n", str(args.n),
        "--temperature", str(args.temperature),
        "--max-tokens", str(args.max_tokens),
    ]  

    print(f"\n=== [{idx}/{total}] Evaluating '{label}'  ({path}) ===", flush=True)
    try:
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            sys.exit(f"Worker for '{label}' failed (exit {rc}). Partial results saved to {args.out}.")
        with open(tmp.name) as f:
            return json.load(f)
    finally:
        os.unlink(tmp.name)


def print_table(results, args):
    base_results = {r["label"]: r for r in results if r["is_baseline"]}
    run_results = [r for r in results if not r["is_baseline"]]
    width = 101

    print("\n" + "=" * width)
    print(f"{args.dataset} results   (n={args.n}, T={args.temperature}, {results[0]['n_problems']} problems)")
    print("=" * width)
    print(f"{'model':<55}{'avg@n':>10}{f'pass@{args.n}':>12}{'Δavg':>12}{'Δpass':>12}")
    print("-" * width)

    def fmt_row(r, d_avg, d_pass):
        label = r["label"] if len(r["label"]) <= 54 else r["label"][:51] + "..."
        row = f"{label:<55}{r['accuracy'] * 100:>9.1f}%{r['pass_at_n'] * 100:>11.1f}%"
        return row + f"{d_avg:>12}{d_pass:>12}"

    for b_label in BASE_MODELS:
        if b_label in base_results:
            print(fmt_row(base_results[b_label], "+0.0pp", "+0.0pp"))
    if base_results and run_results:
        print("-" * width)

    for r in run_results:
        base = base_results.get(get_baseline_key(r["label"]))
        if base:
            d_avg = f"{(r['accuracy'] - base['accuracy']) * 100:+.1f}pp"
            d_pass = f"{(r['pass_at_n'] - base['pass_at_n']) * 100:+.1f}pp"
        else:
            d_avg = d_pass = "--"
            if base_results:
                print(
                    f"NOTE: '{r['label']}' matched no baseline "
                    f"(label contains none of: {', '.join(BASE_MODELS)}) — deltas omitted."
                )
        print(fmt_row(r, d_avg, d_pass))
    print("=" * width)


def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark models on a dataset.")
    ap.add_argument("--dataset", choices=available_tasks(), default="gsm8k",
                    help="dataset task to benchmark (tasks/<name>.py)")  
    ap.add_argument("--models", nargs="+", default=None,
                    help="label=path pairs, e.g. run1=outputs/ckpt-300.")  
    ap.add_argument("--outputs-dir", dest="outputs_dir", default="outputs",
                    help="directory scanned when --models is omitted (default: outputs/)")  
    ap.add_argument("--eval-baselines", action="store_true",
                    help="evaluate base models.")  
    ap.add_argument("--refresh-baselines", dest="refresh_baselines", action="store_true",
                    help="re-evaluate baselines even if cached results exist")  
    ap.add_argument("--n", type=int, default=32, help="samples per problem")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=1024,
                    help="generation cap per sample (default 1024)")  
    ap.add_argument("--all-checkpoints", "--all_checkpoints", dest="all_checkpoints", action="store_true",
                    help="eval all checkpoints of each run, not just the latest")  

    # internal worker flags (set by the parent process; not for direct use).
    ap.add_argument("--worker-out", dest="worker_out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--label", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--model-idx", dest="model_idx", default="?", help=argparse.SUPPRESS)
    ap.add_argument("--total-models", dest="total_models", default="?", help=argparse.SUPPRESS)
    return ap.parse_args()


def main():
    if not HPC:
        os.chdir("/local/scratch/ae581/zzsp")
        
    args = parse_args()

    if args.worker_out:
        run_one(args)
        return

    args.out = os.path.join(RESULTS_DIR, f"results_{args.dataset}.json")
    args.baseline_cache = os.path.join(RESULTS_DIR, f".{args.dataset}_baseline_cache.json")
    for p in (args.out, args.baseline_cache):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    pairs, baseline_labels = collect_pairs(args)

    baseline_cache = load_json(args.baseline_cache, {})
    by_key = {r["eval_config"]: r for r in load_json(args.out, [])}
    print(baseline_cache)

    total = len(pairs)
    print(f"Evaluating {total} model(s)...")

    results = []
    for idx, (label, path) in enumerate(pairs, start=1):
        is_base = label in baseline_labels
        key = eval_config_key(path, args)
        header = f"\n=== [{idx}/{total}] '{label}' ({path}) — "

        if key in by_key:
            r = dict(by_key[key])
            print(f"{header}already in {args.out}, reusing ===")
            print(f"[{label}] accuracy = {r['accuracy']:.4f}  (from file)")
        elif is_base and not args.refresh_baselines and key in baseline_cache:
            r = dict(baseline_cache[key])
            print(f"{header}using cached baseline (--refresh-baselines to re-run) ===")
            print(f"[{label}] accuracy = {r['accuracy']:.4f}  (cached)")
        else:
            r = evaluate_in_subprocess(label, path, idx, total, args)
            if is_base:
                baseline_cache[key] = r
                with open(args.baseline_cache, "w") as f:
                    json.dump(baseline_cache, f)

        r.update(label=label, eval_config=key, is_baseline=is_base)
        by_key[key] = r
        results.append(r)
        with open(args.out, "w") as f:  # save incrementally
            json.dump(list(by_key.values()), f, indent=2)

    print_table(results, args)
    print(f"\nFull per-problem results written to {args.out}")


if __name__ == "__main__":
    main()