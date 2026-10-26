#!/usr/bin/env python3
"""eval_output_dist.py — entropy / logprob distributions of model checkpoints.

For each model, sample `num_samples` rollouts for each of `num_prompts` train
prompts, then teacher-force the rollouts to get per-rollout mean token
entropy, mean token logprob and logprob per character. Each model runs in a
fresh subprocess; results are cached in a persistent JSON file.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

from settings import HPC
from tasks import available_tasks, get_task
from utils import BASE_MODELS, discover_models, get_baseline_key, load_json

RESULTS_DIR = "results"
DISABLED_FILE = os.path.join(RESULTS_DIR, ".disabled_models.json")
DIST_METRICS = ["avg_logprob", "logprob_per_char", "mean_entropy"]
SPLIT = "train"
GEN_BATCH_SIZE = 16
SCORE_BATCH_SIZE = 4


# generation + scoring (worker side)
def set_all_seeds(seed: int):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def completion_end_mask(comp, model, tokenizer):
    """true for completion tokens up to and including the first EOS"""
    import torch

    end_ids = model.generation_config.eos_token_id
    if end_ids is None:
        end_ids = tokenizer.eos_token_id
    if not isinstance(end_ids, (list, tuple)):
        end_ids = [end_ids]
    is_end = torch.zeros_like(comp, dtype=torch.bool)
    for e in end_ids:
        is_end |= comp == e
    after = (torch.cumsum(is_end.int(), dim=1) - is_end.int()) > 0
    return ~after


def generate_batch(model, tokenizer, prompts, args):
    """sample one completion per prompt (left-padded, matches GRPO rollouts)."""
    import torch

    with torch.no_grad():
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        assert bool(enc.attention_mask[:, -1].all()), (
            "Prompts are not left-padded; check tokenizer.padding_side == 'left'."
        )
        prompt_len = enc.input_ids.shape[1]
        out = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.0,
            do_sample=True,
            temperature=args.temperature,
            top_p=1.0,
            top_k=0,
        )
        cmask = completion_end_mask(out[:, prompt_len:], model, tokenizer)
    return out, prompt_len, cmask, enc.attention_mask


def score_batch(model, sequences, prompt_attention_mask, prompt_len,
                completion_mask, temperature: float, chunk: int = 256):
    """teacher-forced (sum logprob, sum entropy, n tokens) per sequence"""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        attn = torch.cat([prompt_attention_mask, completion_mask.long()], dim=1)
        position_ids = (attn.long().cumsum(dim=1) - 1).clamp(min=0)

        B = sequences.shape[0]
        lp_sum = torch.zeros(B, dtype=torch.float64)
        ent_sum = torch.zeros(B, dtype=torch.float64)

        for b in range(0, B, SCORE_BATCH_SIZE):
            sl = slice(b, b + SCORE_BATCH_SIZE)
            seq_b, cmask_b = sequences[sl], completion_mask[sl]

            out = model(input_ids=seq_b, attention_mask=attn[sl],
                        position_ids=position_ids[sl])
            # logits[:, i] predicts token i+1 -> completion tokens are
            # predicted by logits at positions [prompt_len - 1, seq_len - 2]
            logits = out.logits[:, prompt_len - 1: -1, :]
            targets = seq_b[:, prompt_len:]

            for s in range(0, logits.shape[1], chunk):
                piece = logits[:, s: s + chunk, :].float()
                if temperature != 1.0:
                    piece = piece / temperature
                logp = F.log_softmax(piece, dim=-1)

                m = cmask_b[:, s: s + chunk].float()
                tok_lp = logp.gather(-1, targets[:, s: s + chunk].unsqueeze(-1)).squeeze(-1)
                ent = -(logp.exp() * logp).sum(-1)

                lp_sum[sl] += (tok_lp * m).sum(1).double().cpu()
                ent_sum[sl] += (ent * m).sum(1).double().cpu()
            del out, logits

        tok = completion_mask.sum(dim=1).double().cpu()
    return lp_sum, ent_sum, tok


def run_one(args):
    """measure ONE model in this process and write the result JSON"""
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected", file=sys.stderr)
    set_all_seeds(args.seed)

    pos = f"{args.model_idx}/{args.total_models}"
    print(f"[{pos}] Loading {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa" if HPC else "flash_attention_2",
    ).to(device)
    model.eval()

    task = get_task(args.dataset)
    questions = task.sample_questions(SPLIT, args.num_prompts, args.seed)
    prompts, template_used = task.build_eval_prompts(questions, tokenizer)
    if template_used == "chatml_fallback":
        print(f"[{args.label}] tokenizer has no chat template — using explicit ChatML fallback.")

    expanded = [p for p in prompts for _ in range(args.num_samples)]
    prompt_idx = [i for i in range(len(prompts)) for _ in range(args.num_samples)]

    records = []
    t_start = time.perf_counter()
    pbar = tqdm(total=len(expanded), unit="rollout",
                desc=f"[{pos}] {args.label}: sampling")
    for s in range(0, len(expanded), GEN_BATCH_SIZE):
        batch = expanded[s: s + GEN_BATCH_SIZE]
        seqs, plen, cmask, pmask = generate_batch(model, tokenizer, batch, args)
        lp, ent, tok = score_batch(model, seqs, pmask, plen, cmask, args.temperature)
        for j in range(seqs.shape[0]):
            n = max(int(tok[j].item()), 1)
            n_chars = max(len(tokenizer.decode(
                seqs[j, plen:][cmask[j]], skip_special_tokens=True)), 1)
            records.append({
                "prompt_idx": prompt_idx[s + j],
                "sum_logprob": lp[j].item(),
                "avg_logprob": lp[j].item() / n,
                "logprob_per_char": lp[j].item() / n_chars,
                "mean_entropy": ent[j].item() / n,
            })
        pbar.update(len(batch))
    pbar.close()
    elapsed = time.perf_counter() - t_start

    result = {
        "label": args.label,
        "model": args.model,
        "prompt_template": template_used,
        "dataset": task.eval_dataset_id(SPLIT),
        "num_prompts": len(prompts),
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "distributions": {k: [r[k] for r in records] for k in DIST_METRICS},
        "samples": records,
        "wall_time_s": round(elapsed, 1),
    }
    with open(args.worker_out, "w") as f:
        json.dump(result, f)

    mu, sd = mean_std(result["distributions"]["avg_logprob"])
    h, _ = mean_std(result["distributions"]["mean_entropy"])
    print(f"[{args.label}] avg lp {mu:.4f} +/- {sd:.4f} | entropy {h:.4f} nats | {elapsed:.0f}s")


def collect_pairs(args):
    """Return (pairs, baseline_labels) of models to measure."""
    disabled = set(load_json(DISABLED_FILE, []))
    pairs = []
    for label, path in discover_models(args.outputs_dir, args.all_checkpoints, args.dataset, args.out):
        run = os.path.basename(os.path.dirname(path))  # run folder name
        if args.tag is not None and args.tag not in run:
            continue
        if run in disabled:
            print(f"{run} is disabled, skipping")
            continue
        pairs.append((label, path))
    for i, (label, path) in enumerate(pairs):
        if label in BASE_MODELS:
            new = f"{label}_run"
            print(f"NOTE: discovered run '{label}' renamed to '{new}' (label reserved for baseline).")
            pairs[i] = (new, path)
    if not pairs and args.skip_baselines:
        sys.exit("No models found and baselines skipped.")

    baseline_labels = set()
    if not args.skip_baselines:
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


# parent process
def measure_config_key(model_path: str, args) -> str:
    """Everything that affects the measured numbers (used for caching/reuse)."""
    cfg = {
        "task": "output_dist",
        "dataset": get_task(args.dataset).eval_dataset_id(SPLIT),
        "prompt": "system_boxed",
        "model": model_path,
        "num_prompts": args.num_prompts,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "num_beams": 1,
        "beam_return": 0,
    }
    return json.dumps(cfg, sort_keys=True)


def mean_std(vals):
    n = len(vals)
    mu = sum(vals) / n
    sd = (sum((x - mu) ** 2 for x in vals) / max(n - 1, 1)) ** 0.5
    return mu, sd


def measure_in_subprocess(label, path, idx, total, args) -> dict:
    """Run the worker for one model and return its result dict."""
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
        "--num-prompts", str(args.num_prompts),
        "--num-samples", str(args.num_samples),
        "--temperature", str(args.temperature),
        "--max-new-tokens", str(args.max_new_tokens),
        "--seed", str(args.seed),
    ]

    print(f"\n=== [{idx}/{total}] Measuring '{label}'  ({path}) ===", flush=True)
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

    def summary(r):
        d = r["distributions"]
        lp, sd = mean_std(d["avg_logprob"])
        return lp, sd, mean_std(d["logprob_per_char"])[0], mean_std(d["mean_entropy"])[0]

    print("\n" + "=" * width)
    print(f"Output distributions   ({results[0]['num_prompts']} prompts x "
          f"{args.num_samples} samples, T={args.temperature})")
    print("avg lp = mean per-token logprob over sampled rollouts (nats); "
          "lp/char = per-character (tokenizer-independent); "
          "H = mean token entropy (nats).")
    print("=" * width)
    print(f"{'model':<55}{'avg lp (mu+/-sd)':>20}{'lp/char':>12}{'H (nats)':>12}"
          f"{'Δavg lp':>12}{'ΔH':>12}")
    print("-" * width)

    def fmt_row(r, d_lp, d_h):
        label = r["label"] if len(r["label"]) <= 54 else r["label"][:51] + "..."
        lp, sd, lpc, h = summary(r)
        return (f"{label:<55}{lp:>12.4f}+/-{sd:<5.3f}{lpc:>12.4f}{h:>12.4f}"
                f"{d_lp:>12}{d_h:>12}")

    for b_label in BASE_MODELS:
        if b_label in base_results:
            print(fmt_row(base_results[b_label], "+0.000", "+0.000"))
    if base_results and run_results:
        print("-" * width)

    for r in run_results:
        base = base_results.get(get_baseline_key(r["label"]))
        if base:
            lp, _, _, h = summary(r)
            b_lp, _, _, b_h = summary(base)
            d_lp, d_h = f"{lp - b_lp:+.3f}", f"{h - b_h:+.3f}"
        else:
            d_lp = d_h = "--"
            if base_results:
                print(f"NOTE: '{r['label']}' matched no baseline "
                      f"(label contains none of: {', '.join(BASE_MODELS)}) — deltas omitted.")
        print(fmt_row(r, d_lp, d_h))
    print("=" * width)


def parse_args():
    ap = argparse.ArgumentParser(description="Measure output distributions of models on dataset prompts.")
    ap.add_argument("--dataset", choices=available_tasks(), default="gsm8k",
                    help="dataset task to sample prompts from (tasks/<name>.py)")
    ap.add_argument("--outputs-dir", dest="outputs_dir", default="outputs",
                    help="directory scanned for runs (default: outputs/)")
    ap.add_argument("--all-checkpoints", "--all_checkpoints", dest="all_checkpoints",
                    action="store_true", help="measure every checkpoint per run, not just the latest")
    ap.add_argument("--tag", default=None,
                    help="only measure runs whose folder name contains this substring")
    ap.add_argument("--skip-baselines", action="store_true",
                    help="do not automatically measure base models")
    ap.add_argument("--refresh-baselines", dest="refresh_baselines", action="store_true",
                    help="re-measure baselines even if cached results exist")
    ap.add_argument("--baseline-cache", dest="baseline_cache",
                    default=os.path.join(RESULTS_DIR, ".output_dist_baseline_cache.json"),
                    help="cached baseline results file")
    ap.add_argument("--num-prompts", dest="num_prompts", type=int, default=128,
                    help="number of train prompts")
    ap.add_argument("--num-samples", dest="num_samples", type=int, default=8,
                    help="sampled rollouts per prompt")
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "results_output_dist.json"),
                    help="persistent results file; models already measured under the same "
                         "config are reused")

    # internal worker flags (set by the parent process; not for direct use).
    ap.add_argument("--worker-out", dest="worker_out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--label", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--model-idx", dest="model_idx", default="?", help=argparse.SUPPRESS)
    ap.add_argument("--total-models", dest="total_models", default="?", help=argparse.SUPPRESS)
    return ap.parse_args()


def main():
    args = parse_args()

    if args.worker_out:
        run_one(args)
        return

    for p in (args.out, args.baseline_cache):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

    pairs, baseline_labels = collect_pairs(args)

    baseline_cache = load_json(args.baseline_cache, {})
    by_key = {r["measure_config"]: r for r in load_json(args.out, []) if r.get("measure_config")}

    total = len(pairs)
    print(f"Measuring output distributions for {total} model(s)...")

    results = []
    for idx, (label, path) in enumerate(pairs, start=1):
        is_base = label in baseline_labels
        key = measure_config_key(path, args)
        header = f"\n=== [{idx}/{total}] '{label}' ({path}) — "

        if key in by_key:
            r = dict(by_key[key])
            print(f"{header}already in {args.out}, reusing ===")
            src = "from file"
        elif is_base and not args.refresh_baselines and key in baseline_cache:
            r = dict(baseline_cache[key])
            print(f"{header}using cached baseline (--refresh-baselines to re-run) ===")
            src = "cached"
        else:
            r = measure_in_subprocess(label, path, idx, total, args)
            if is_base:
                baseline_cache[key] = r
                with open(args.baseline_cache, "w") as f:
                    json.dump(baseline_cache, f)
            src = None
        if src:
            print(f"[{label}] avg lp = {mean_std(r['distributions']['avg_logprob'])[0]:.4f}  ({src})")

        r.update(label=label, measure_config=key, is_baseline=is_base)
        by_key[key] = r
        results.append(r)
        with open(args.out, "w") as f:  # save incrementally
            json.dump(list(by_key.values()), f, indent=2)

    print_table(results, args)
    print(f"\nFull per-rollout results written to {args.out}")


if __name__ == "__main__":
    main()