#!/usr/bin/env python3
"""Calibrate --embed_tau for the gsm8k_embedding rewards on base-model rollouts.

Samples --n completions for --num_problems GSM8K train problems with the training
prompt, keeps the correct ones (strict '####' match, as in training) and picks the
cosine threshold that best separates same- from different-signature pairs, with and
without number masking (embedding_reward.calibrate). Use the setting with the higher
AUC: around 0.75 or above, train with its tau; below about 0.65 the embedder barely
sees strategy (try --embed_model BAAI/bge-base-en-v1.5, or use gsm8k_strategy).

    python calibrate_embedding.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import json
import os
import random

import numpy as np

from embedding_reward import DEFAULT_EMBED_MODEL, Embedder, calibrate, hard_clusters, prepare
from rewards import _num_equal, extract_gsm8k_strict
from tasks import get_task
from utils import load_json

RESULTS_PATH = os.path.join("results", "results_embedding_calibration.json")


def parse_args():
    ap = argparse.ArgumentParser(description="Calibrate tau for the gsm8k_embedding rewards.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="base model to sample rollouts from")
    ap.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--num_problems", type=int, default=200)
    ap.add_argument("--n", type=int, default=16, help="samples per problem")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def rollouts(args):
    """[(question, gold, [correct completions])] from the training prompt."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ds = get_task("gsm8k").build_train(tokenizer)
    idx = random.Random(args.seed).sample(range(len(ds)), k=min(args.num_problems, len(ds)))
    rows = [ds[i] for i in idx]

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.6, seed=args.seed)
    sampling = SamplingParams(n=args.n, temperature=args.temperature, top_p=1.0,
                              max_tokens=args.max_tokens, seed=args.seed)
    outputs = llm.generate([r["prompt"] for r in rows], sampling)

    groups = []
    for r, out in zip(rows, outputs):
        comps = [o.text for o in out.outputs]
        ok = [c for c in comps if (e := extract_gsm8k_strict(c)) is not None and _num_equal(e, r["answer"])]
        groups.append((r["question"], r["answer"], ok))
    return groups


def main():
    args = parse_args()
    groups = rollouts(args)
    n_correct = sum(len(c) for _, _, c in groups)
    print(f"{n_correct}/{len(groups) * args.n} correct completions over {len(groups)} problems")

    embed = Embedder(args.embed_model, device="cuda")
    results = {}
    for mask in (True, False):
        res = calibrate(groups, embed, mask_numbers=mask)
        # audit: hard clusters per group at the calibrated tau
        sizes = []
        for _, _, comps in groups:
            texts = [t for t in (prepare(c, mask) for c in comps) if len(t.split()) >= 5]
            if len(texts) >= 2:
                sizes.append(len(set(hard_clusters(embed(texts), res["tau"]))))
        res["clusters_per_group_mean"] = float(np.mean(sizes)) if sizes else None
        results["masked" if mask else "unmasked"] = res
        print(f"mask_numbers={mask}: {json.dumps(res)}")

    best = max(results, key=lambda k: results[k]["auc"])
    flag = "" if best == "masked" else " --embed_keep_numbers"
    print(f"\nBest: {best} (AUC {results[best]['auc']:.3f}) -> --embed_tau {results[best]['tau']:.4f}{flag}")
    if results[best]["auc"] < 0.65:
        print("AUC below 0.65: the embedder barely separates strategies; try bge-base or gsm8k_strategy.")

    os.makedirs("results", exist_ok=True)
    log = load_json(RESULTS_PATH, [])
    log.append({"model": args.model, "embed_model": args.embed_model, "num_problems": len(groups),
                "n": args.n, "temperature": args.temperature, "seed": args.seed,
                "n_correct": n_correct, "best": best, **results})
    with open(RESULTS_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
