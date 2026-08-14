#!/usr/bin/env python3
"""
eval_output_dist.py — Measure entropy distributions of model checkpoints
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
from settings import HPC


BASE_MODELS = {
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "olmo": "allenai/OLMo-2-0425-1B-Instruct",
    "qwen-small": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct"
} 


RESULTS_DIR = "results"


DIST_METRICS = ["avg_logprob", "logprob_per_char", "mean_entropy",
                "num_tokens"]

DISABLED_FILE = "results/.disabled_models.json"

disabled = set()
if os.path.exists(DISABLED_FILE):
    with open(DISABLED_FILE) as f:
        disabled = set(json.load(f))


from tasks import available_tasks, get_task  # noqa: E402

_ACTIVE_DATASET = "gsm8k"


def _task():
    return get_task(_ACTIVE_DATASET)


def build_prompts(tokenizer, num_prompts: int, seed: int):
    """returns (prompt_texts, question_texts, template_used).
    """
    questions = _task().sample_questions(
        _OUTPUT_DIST_SPLIT, num_prompts, seed
    )
    prompts, template = _task().build_eval_prompts(questions, tokenizer)
    return prompts, questions, template



_OUTPUT_DIST_SPLIT = "train"



def set_all_seeds(seed: int):
    import random
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


def completion_end_mask(comp, model, tokenizer):

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


def generate_batch(model, tokenizer, prompts, args, template_used,
                   greedy=False, num_beams=1, num_return_sequences=1):
    """generate for a batch of prompts (no_grad), left-padded decoder-only
    generation, matching GRPO rollout setup."""
    import torch

    with torch.no_grad():
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            add_special_tokens=False,
        ).to(model.device)
        assert bool(enc.attention_mask[:, -1].all()), (
            "Prompts are not left-padded; check tokenizer.padding_side "
            "== 'left'.")
        prompt_len = enc.input_ids.shape[1]

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.0,  
        )
        
        if greedy and num_beams == 1:
            gen_kwargs.update(do_sample=False)
        elif num_beams > 1:
            gen_kwargs.update(
                do_sample=False, num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                early_stopping=True,
            )
        else:
            gen_kwargs.update(
                do_sample=True, temperature=args.temperature,
                top_p=1.0, top_k=0,
            )

        out = model.generate(**enc, **gen_kwargs)
        comp = out[:, prompt_len:]
        cmask = completion_end_mask(comp, model, tokenizer)


        pmask = enc.attention_mask
        if out.shape[0] != pmask.shape[0]:
            rep = out.shape[0] // pmask.shape[0]
            pmask = pmask.repeat_interleave(rep, dim=0)
    return out, prompt_len, cmask, pmask


def score_batch(model, sequences, prompt_attention_mask, prompt_len,
                completion_mask, temperature: float, chunk: int = 256,
                score_batch_size: int = 4):


    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        attn = torch.cat(
            [prompt_attention_mask, completion_mask.long()], dim=1)
        position_ids = (attn.long().cumsum(dim=1) - 1).clamp(min=0)

        B = sequences.shape[0]
        lp_sum = torch.zeros(B, dtype=torch.float64)
        ent_sum = torch.zeros(B, dtype=torch.float64)

        for b in range(0, B, score_batch_size):
            seq_b = sequences[b: b + score_batch_size]
            attn_b = attn[b: b + score_batch_size]
            pos_b = position_ids[b: b + score_batch_size]
            cmask_b = completion_mask[b: b + score_batch_size]

            out = model(input_ids=seq_b, attention_mask=attn_b,
                        position_ids=pos_b)
            # logits[:, i] predicts token i+1 -> completion tokens are
            # predicted by logits at positions [prompt_len - 1, seq_len - 2]
            logits = out.logits[:, prompt_len - 1: -1, :]
            targets = seq_b[:, prompt_len:]

            T = logits.shape[1]
            for s in range(0, T, chunk):
                piece = logits[:, s: s + chunk, :].float()
                if temperature != 1.0:
                    piece = piece / temperature
                logp = F.log_softmax(piece, dim=-1)

                tgt = targets[:, s: s + chunk]
                m = cmask_b[:, s: s + chunk].float()

                tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                lp_sum[b: b + score_batch_size] += \
                    (tok_lp * m).sum(1).double().cpu()

                ent = -(logp.exp() * logp).sum(-1)
                ent_sum[b: b + score_batch_size] += \
                    (ent * m).sum(1).double().cpu()
            del out, logits

        tok = completion_mask.sum(dim=1).double().cpu()
    return lp_sum, ent_sum, tok



def _percentile(sorted_vals, q):
    """Linear-interpolation percentile on a pre-sorted list (q in [0,100])."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = (q / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize(vals):
    """Mean/std/min/max/percentiles for a list of floats."""
    n = len(vals)
    if n == 0:
        return None
    mu = sum(vals) / n
    sd = (sum((x - mu) ** 2 for x in vals) / max(n - 1, 1)) ** 0.5
    sv = sorted(vals)
    return {
        "n": n,
        "mean": mu,
        "std": sd,
        "min": sv[0],
        "max": sv[-1],
        "p5": _percentile(sv, 5),
        "p25": _percentile(sv, 25),
        "p50": _percentile(sv, 50),
        "p75": _percentile(sv, 75),
        "p95": _percentile(sv, 95),
    }


def histogram(vals, bins=50):
    """Density histogram as {bin_edges, density}"""
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    if lo == hi:  # degenerate: all mass in one bin
        return {"bin_edges": [lo, hi], "density": [0.0]}
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        i = min(int((v - lo) / width), bins - 1)
        counts[i] += 1
    n = len(vals)
    edges = [lo + i * width for i in range(bins + 1)]
    density = [c / (n * width) for c in counts]
    return {"bin_edges": edges, "density": density}


def run_one(args):
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global _ACTIVE_DATASET
    _ACTIVE_DATASET = args.dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected",
              file=sys.stderr)

    set_all_seeds(args.seed)

    pos = f"{args.model_idx}/{args.total_models}"
    print(f"[{pos}] Loading {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa" if HPC else "flash_attention_2",
    ).to(device)
    model.eval()

    prompts, questions, template_used = build_prompts(
        tokenizer, args.num_prompts, args.seed)
    if template_used == "chatml_fallback":
        print(f"[{args.label}] tokenizer has no chat template — using "
              f"explicit ChatML fallback (intentional, see header comment).")

    expanded = [p for p in prompts for _ in range(args.num_samples)]
    prompt_idx = [i for i in range(len(prompts))
                  for _ in range(args.num_samples)]

    records = []  # one dict per sampled completion
    t_start = time.perf_counter()

    # sampled pool + teacher-forced scoring 
    pbar = tqdm(total=len(expanded), unit="rollout",
                desc=f"[{pos}] {args.label}: sampling")
    for s in range(0, len(expanded), args.batch_size):
        batch = expanded[s: s + args.batch_size]
        seqs, plen, cmask, pmask = generate_batch(
            model, tokenizer, batch, args, template_used)
        lp, ent, tok = score_batch(
            model, seqs, pmask, plen, cmask, args.temperature,
            score_batch_size=args.score_batch_size,
        )
        for j in range(seqs.shape[0]):
            n = int(tok[j].item())
            text = tokenizer.decode(
                seqs[j, plen:][cmask[j]], skip_special_tokens=True)
            rec = {
                "prompt_idx": prompt_idx[s + j],
                "sum_logprob": lp[j].item(),
                "avg_logprob": lp[j].item() / max(n, 1),
                "logprob_per_char": lp[j].item() / max(len(text), 1),
                "mean_entropy": ent[j].item() / max(n, 1),
                "num_tokens": n,
            }
            if not args.no_texts:
                rec["text"] = text
            records.append(rec)
        if s == 0:
            first = tokenizer.decode(
                seqs[0, plen:][cmask[0]], skip_special_tokens=True)
            tqdm.write(f"--- sample completion ---\n{first[:300]}\n---")
        pbar.update(len(batch))
        avg = sum(r["avg_logprob"] for r in records) / len(records)
        pbar.set_postfix(avg_lp=f"{avg:.3f}")
    pbar.close()

    # greedy trajectories (high-probability reference)
    greedy = []
    for s in tqdm(range(0, len(prompts), args.batch_size),
                  desc=f"[{pos}] {args.label}: greedy"):
        batch = prompts[s: s + args.batch_size]
        seqs, plen, cmask, pmask = generate_batch(
            model, tokenizer, batch, args, template_used, greedy=True)
        lp, ent, tok = score_batch(
            model, seqs, pmask, plen, cmask, args.temperature,
            score_batch_size=args.score_batch_size,
        )
        for j in range(seqs.shape[0]):
            n = int(tok[j].item())
            text = tokenizer.decode(
                seqs[j, plen:][cmask[j]], skip_special_tokens=True)
            rec = {
                "prompt_idx": s + j,
                "avg_logprob": lp[j].item() / max(n, 1),
                "logprob_per_char": lp[j].item() / max(len(text), 1),
                "mean_entropy": ent[j].item() / max(n, 1),
                "num_tokens": n,
            }
            if not args.no_texts:
                rec["text"] = text
            greedy.append(rec)

    #  optional beam search
    beams = []
    if args.num_beams > 1:
        n_ret = min(args.beam_return, args.num_beams)
        for i in tqdm(range(len(prompts)),
                      desc=f"[{pos}] {args.label}: beam search"):
            seqs, plen, cmask, pmask = generate_batch(
                model, tokenizer, [prompts[i]], args, template_used,
                num_beams=args.num_beams, num_return_sequences=n_ret,
            )
            lp, ent, tok = score_batch(
                model, seqs, pmask, plen, cmask, args.temperature,
                score_batch_size=args.score_batch_size,
            )
            for j in range(seqs.shape[0]):
                n = int(tok[j].item())
                text = tokenizer.decode(
                    seqs[j, plen:][cmask[j]], skip_special_tokens=True)
                rec = {
                    "prompt_idx": i,
                    "avg_logprob": lp[j].item() / max(n, 1),
                    "num_tokens": n,
                }
                if not args.no_texts:
                    rec["text"] = text
                beams.append(rec)

    elapsed = time.perf_counter() - t_start

    # plot-ready aggregation 
    distributions = {k: [r[k] for r in records] for k in DIST_METRICS}
    stats = {k: summarize(v) for k, v in distributions.items()}
    hists = {k: histogram(v) for k, v in distributions.items()}
    greedy_ref = {
        k: (sum(g[k] for g in greedy) / len(greedy) if greedy else None)
        for k in DIST_METRICS
    }

    result = {
        "label": args.label,
        "model": args.model,
        "prompt_template": template_used,
        "dataset": _task().eval_dataset_id(_OUTPUT_DIST_SPLIT),
        "num_prompts": len(prompts),
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "num_beams": args.num_beams,

        "distributions": distributions,
        "stats": stats,
        "histograms": hists,
        "greedy_ref": greedy_ref,

        "questions": questions,
        "samples": records,
        "greedy": greedy,
        "beams": beams,
        "wall_time_s": round(elapsed, 1),
    }
    with open(args.worker_out, "w") as f:
        json.dump(result, f)

    s_lp, s_ent = stats["avg_logprob"], stats["mean_entropy"]
    print(f"[{args.label}] avg lp {s_lp['mean']:.4f} +/- {s_lp['std']:.4f} | "
          f"entropy {s_ent['mean']:.4f} nats | "
          f"len {stats['num_tokens']['mean']:.0f} tok | {elapsed:.0f}s")



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


def discover_models(outputs_dir: str, all_checkpoints: bool = False, tag=None):
    if not os.path.isdir(outputs_dir):
        sys.exit(f"No such directory: {outputs_dir}")

    pairs = []
    for name in sorted(os.listdir(outputs_dir)):
        run_dir = os.path.join(outputs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        label = re.sub(r"-\d{8,}$", "", name)
        if tag is not None and tag not in name:
            continue
        if _ACTIVE_DATASET != "gsm8k":
            alias = _ACTIVE_DATASET
            if _ACTIVE_DATASET == "countdown4": alias = "countdown"
            if _ACTIVE_DATASET == "aime2024": alias = "dapo"
            if alias not in label and alias not in name:
                print(f"{label} does not match dataset {alias}, skipping")
                continue
        else:
            if "countdown" in label or "dapo" in label or "mbpp" in label or "wordle" in label:
                print(f"{label} does not match gsm8k, skipping")
                continue
        if label in disabled or name in disabled:
            print(f"{label} is disabled, skipping")
            continue
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



def get_baseline_key(label):
    """Map a run label to its base-model family (or None)."""
    lbl = label.lower()
    if "llama" in lbl:
        return "llama"
    if "olmo" in lbl:
        return "olmo"
    if "qwen" in lbl:
        if "small" in lbl:
            return "qwen-small"
        elif "3b" in lbl:
            return "qwen3b" 
        return "qwen"
    return None


def _measure_config_key(model_path: str, args) -> str:
    """Everything that affects the measured numbers (batch sizes and attn
    implementation are excluded — they change speed/memory, not semantics)."""
    cfg = {
        "task": "output_dist",
        "dataset": get_task(args.dataset).eval_dataset_id(_OUTPUT_DIST_SPLIT),
        "prompt": "system_boxed",
        "model": model_path,
        "num_prompts": args.num_prompts,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "num_beams": args.num_beams,
        "beam_return": args.beam_return if args.num_beams > 1 else 0,
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



def main():
    ap = argparse.ArgumentParser(
        description="Measure output distributions of models on dataset "
                    "rollout prompts.")
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
                         "instead of just the latest")
    
    ap.add_argument("--tag", default=None,
                        help="Only evaluate discovered runs whose folder name "
                            "contains this substring")
    ap.add_argument("--skip-baselines", action="store_true",
                    help="Do not automatically measure base "
                         "models.")
    ap.add_argument("--refresh-baselines", dest="refresh_baselines",
                    action="store_true",
                    help="Re-measure baselines even if cached results exist")
    ap.add_argument("--baseline-cache", dest="baseline_cache",
                    default=os.path.join(
                        RESULTS_DIR, ".output_dist_baseline_cache.json"),
                    help="Where cached baseline results live (default: "
                         f"{RESULTS_DIR}/.output_dist_baseline_cache.json)")
    ap.add_argument("--num-prompts", dest="num_prompts", type=int,
                    default=128,
                    help="Number of train prompts")
    ap.add_argument("--num-samples", dest="num_samples", type=int, default=8,
                    help="Sampled rollouts per prompt ")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="Sampling temperature ")
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int,
                    default=1024)
    ap.add_argument("--num-beams", dest="num_beams", type=int, default=1,
                    help="1 enables beam search per prompt ")
    ap.add_argument("--beam-return", dest="beam_return", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=16)
    ap.add_argument("--score-batch-size", dest="score_batch_size", type=int,
                    default=4)
    ap.add_argument("--no-texts", dest="no_texts", action="store_true",
                    help="Drop completion texts from the JSON")
    ap.add_argument("--out",
                    default=os.path.join(RESULTS_DIR,
                                         "results_output_dist.json"),
                    help="Persistent results file. Models already measured "
                         "under the same config are skipped and reused; new "
                         "results are appended (default: "
                         f"{RESULTS_DIR}/results_output_dist.json)")
    ap.add_argument("--force", action="store_true",
                    help="Re-measure models even if a result with the same "
                         "config already exists in the results file.")


    ap.add_argument("--worker-out", dest="worker_out", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--model", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--label", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--model-idx", dest="model_idx", default="?",
                    help=argparse.SUPPRESS)
    ap.add_argument("--total-models", dest="total_models", default="?",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    global _ACTIVE_DATASET
    _ACTIVE_DATASET = args.dataset


    if args.worker_out:
        run_one(args)
        return

    for p in (args.out, args.baseline_cache):
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

    if args.num_samples > 1 and args.temperature == 0.0:
        print("WARNING: --num-samples > 1 with temperature 0 gives "
              "identical samples", file=sys.stderr)

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
        pairs = discover_models(args.outputs_dir, args.all_checkpoints, args.tag)


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


    prior = _load_results_file(args.out)
    by_key, legacy = {}, []
    for r in prior:
        k = r.get("measure_config")
        if k:
            by_key[k] = r
        else:
            legacy.append(r)

    total_models = len(pairs)
    print(f"Measuring output distributions for {total_models} model(s)...")

    results = []
    for m_idx, (label, path) in enumerate(pairs, start=1):
        is_base = label in baseline_labels
        key = _measure_config_key(path, args)
        pos = f"{m_idx}/{total_models}"

        # already in the results file under this config
        if not args.force and key in by_key:
            r = dict(by_key[key])
            r["label"] = label
            print(f"\n=== [{pos}] '{label}' ({path}) — already in "
                  f"{args.out}, reusing (--force to re-measure) ===")
            print(f"[{label}] avg lp = "
                  f"{r['stats']['avg_logprob']['mean']:.4f}  (from file)")
        # baseline with a cached result
        elif is_base and not args.refresh_baselines and key in baseline_cache:
            r = dict(baseline_cache[key])
            r["label"] = label
            print(f"\n=== [{pos}] '{label}' ({path}) — using cached "
                  f"baseline result (--refresh-baselines to re-run) ===")
            print(f"[{label}] avg lp = "
                  f"{r['stats']['avg_logprob']['mean']:.4f}  (cached)")
        # else measure.
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
                "--num-prompts", str(args.num_prompts),
                "--num-samples", str(args.num_samples),
                "--temperature", str(args.temperature),
                "--max-new-tokens", str(args.max_new_tokens),
                "--num-beams", str(args.num_beams),
                "--beam-return", str(args.beam_return),
                "--seed", str(args.seed),
                "--batch-size", str(args.batch_size),
                "--score-batch-size", str(args.score_batch_size),
                # "--attn", "sdpa" if HPC else "flash_attention_2",
            ]
            if args.no_texts:
                cmd.append("--no-texts")
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

        r["measure_config"] = key
        r["is_baseline"] = is_base
        by_key[key] = r
        results.append(r)

  
        with open(args.out, "w") as f:
            json.dump(legacy + list(by_key.values()), f, indent=2)


    base_results = {r["label"]: r for r in results if r["is_baseline"]}
    run_results = [r for r in results if not r["is_baseline"]]

    width = 55 + 20 + 12 + 12 + 8 + 12 + 12
    print("\n" + "=" * width)
    print(f"Output distributions   ({results[0]['num_prompts']} prompts x "
          f"{args.num_samples} samples, T={args.temperature})")
    print("avg lp = mean per-token logprob over sampled rollouts (nats); "
          "lp/char = per-character (tokenizer-independent); "
          "H = mean token entropy (nats).")
    print("=" * width)

    header = (f"{'model':<55}{'avg lp (mu+/-sd)':>20}{'lp/char':>12}"
              f"{'H (nats)':>12}{'len':>8}{'Δavg lp':>12}{'ΔH':>12}")
    print(header)
    print("-" * width)

    def fmt_row(r, d_lp_str, d_h_str):
        label = r["label"]
        if len(label) > 54:
            label = label[:51] + "..."
        s = r["stats"]
        return (f"{label:<55}"
                f"{s['avg_logprob']['mean']:>12.4f}"
                f"+/-{s['avg_logprob']['std']:<5.3f}"
                f"{s['logprob_per_char']['mean']:>12.4f}"
                f"{s['mean_entropy']['mean']:>12.4f}"
                f"{s['num_tokens']['mean']:>8.0f}"
                f"{d_lp_str:>12}{d_h_str:>12}")


    for b_lbl in BASE_MODELS:
        if b_lbl in base_results:
            print(fmt_row(base_results[b_lbl], "+0.000", "+0.000"))

    if base_results and run_results:
        print("-" * width)


    for r in run_results:
        b_key = get_baseline_key(r["label"])

        d_lp_str = "--"
        d_h_str = "--"

        if b_key and b_key in base_results:
            base_r = base_results[b_key]
            d_lp = (r["stats"]["avg_logprob"]["mean"]
                    - base_r["stats"]["avg_logprob"]["mean"])
            d_h = (r["stats"]["mean_entropy"]["mean"]
                   - base_r["stats"]["mean_entropy"]["mean"])
            d_lp_str = f"{d_lp:+.3f}"
            d_h_str = f"{d_h:+.3f}"
        elif not r["is_baseline"] and base_results:
            print(f"NOTE: '{r['label']}' matched no baseline "
                  f"(label contains none of: {', '.join(BASE_MODELS)}) — "
                  f"deltas omitted.")

        print(fmt_row(r, d_lp_str, d_h_str))

    print("=" * width)
    print(f"\nFull per-rollout results written to {args.out}")
    


if __name__ == "__main__":
    main()
