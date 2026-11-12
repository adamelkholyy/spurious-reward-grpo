#!/usr/bin/env python3
"""Majority-voting (maj@k) eval: inference only, no training.

Samples k rollouts per eval problem with the same prompts, sampling settings and
eval split as eval.py (T=1.0, top_p=1, max_tokens=1024, seed 0), extracts each
rollout's answer, and scores the most common valid answer against the gold.
Invalid answers (no parsable answer) don't vote; ties go to the answer seen first;
a problem with no valid answer counts as wrong.

Also reports avg@k / pass@k over the raw samples and, for tasks with
summarise_samples (BBQ), the task metrics (bias scores) of the voted answers.

    python eval_majority.py --dataset bbq --k 16 --models qwen=Qwen/Qwen2.5-1.5B-Instruct

Results are merged into results/results_<dataset>_majority.json, keyed by (label, model, k).
One model per invocation keeps vLLM from holding the GPU between models.
"""

import argparse
import fcntl
import json
import os
import sys
import time
from collections import Counter

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from tasks import available_tasks, get_task  # noqa: E402
from eval import MAX_MODEL_LEN, GPU_MEM, effective_limit  # noqa: E402
from utils import load_json  # noqa: E402


def majority(answers):
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 0
    ans, count = Counter(valid).most_common(1)[0]  # stable: ties -> first seen
    return ans, count


def save_result(path, r):
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [x for x in load_json(path, [])
                if (x["label"], x["model"], x["k"]) != (r["label"], r["model"], r["k"])]
        rows.append(r)
        with open(path + ".tmp", "w") as f:
            json.dump(rows, f, indent=2)
        os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=available_tasks(), required=True)
    ap.add_argument("--model", required=True, help="HF id or checkpoint dir")
    ap.add_argument("--label", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=1024)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    task = get_task(args.dataset)
    problems, golds = task.load_eval(split=None, limit=effective_limit(task))
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts, template_used = task.build_eval_prompts(problems, tok)
    stop = ["<|im_end|>", "<|im_start|>", "\nQuestion:", "\nProblem:"] if template_used == "chatml_fallback" else None

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=GPU_MEM,
              max_model_len=MAX_MODEL_LEN, enable_prefix_caching=True, seed=0)
    sampling = SamplingParams(n=args.k, temperature=args.temperature, top_p=1.0,
                              max_tokens=args.max_tokens, stop=stop, seed=0)
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    gen_s = time.perf_counter() - t0

    grader = task.make_grader()
    per_example, voted_texts, all_texts = [], [], []
    n_maj = n_pass = n_none = 0
    sum_avg = 0.0
    for i, out in enumerate(outputs):
        texts = [o.text for o in out.outputs]
        all_texts.append(texts)
        flags = [grader(t, golds[i]) for t in texts]
        answers = [task.extract_answer(t) for t in texts]
        vote, votes = majority(answers)
        # grade the voted answer through the task's own grader using a representative sample
        if vote is None:
            ok = False
            n_none += 1
            voted_texts.append([""])
        else:
            rep = texts[answers.index(vote)]
            ok = grader(rep, golds[i])
            voted_texts.append([rep])
        n_maj += ok
        n_pass += any(flags)
        sum_avg += sum(flags) / len(flags)
        per_example.append({"index": i, "gold": str(golds[i]), "vote": vote, "votes": votes,
                            "n_valid": sum(a is not None for a in answers),
                            "n_correct": sum(flags), "maj_correct": ok})

    n = len(problems)
    result = {
        "label": args.label, "model": args.model, "dataset": task.eval_dataset_id(),
        "k": args.k, "temperature": args.temperature, "max_tokens": args.max_tokens,
        "n_problems": n,
        "maj_accuracy": round(n_maj / n, 4),
        "avg_accuracy": round(sum_avg / n, 4),
        "pass_at_k": round(n_pass / n, 4),
        "no_valid_vote_frac": round(n_none / n, 4),
        "gen_seconds": round(gen_s, 1),
        "per_example": per_example,
    }
    summarise = getattr(task, "summarise_samples", None)
    if summarise:
        result["task_metrics_voted"] = summarise(voted_texts, golds=golds)
        result["task_metrics_samples"] = summarise(all_texts, golds=golds)

    os.makedirs("results", exist_ok=True)
    save_result(f"results/results_{args.dataset}_majority.json", result)
    tm = result.get("task_metrics_voted") or {}
    print(f"[{args.label}] maj@{args.k} = {result['maj_accuracy']:.4f}  avg@{args.k} = {result['avg_accuracy']:.4f}  "
          f"pass@{args.k} = {result['pass_at_k']:.4f}  no-vote {result['no_valid_vote_frac']:.1%}  "
          f"voted: acc_ambig={tm.get('acc_ambig')} acc_disambig={tm.get('acc_disambig')} "
          f"bias_ambig={tm.get('bias_ambig')} bias_disambig={tm.get('bias_disambig')}  [gen {gen_s:.0f}s]",
          flush=True)
    sys.stdout.flush()
    # skip vLLM teardown (can hang), but kill its engine-core child first: os._exit alone
    # orphans it and it keeps holding the GPU memory
    import psutil
    kids = psutil.Process().children(recursive=True)
    for p in kids:
        p.kill()
    psutil.wait_procs(kids, timeout=30)
    os._exit(0)


if __name__ == "__main__":
    main()
