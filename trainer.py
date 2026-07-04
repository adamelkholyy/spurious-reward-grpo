import argparse
import time

import torch
import wandb
import logging

from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from settings import LORA_CONFIG, GRPO_CONFIG
from utils import resolve_output_dir

from runners.GRPORunner import GRPORunner

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="Qwen/Qwen2.5-1.5B-Instruct",
    help="Park et al. 2509.26114 random-reward base models: "
         "Qwen/Qwen2.5-1.5B-Instruct (main), meta-llama/Llama-3.2-1B-Instruct, "
         "allenai/OLMo-2-0425-1B-Instruct. True-reward GSM8K runs: "
         "Qwen/Qwen2.5-3B-Instruct.",
)
parser.add_argument(
    "--dataset",
    choices=["gsm8k", "deepscaler"],
    default="gsm8k",
    help="'gsm8k' = Park et al. 2509.26114 setup (default). "
         "'deepscaler' = legacy Shao-style spurious-rewards setup.",
)
parser.add_argument(
    "--reward",
    choices=["ground_truth", "random", "random_p03", "random_p07", "gaussian",
             "gsm8k", "gsm8k_flexible", "box_only", "incorrect", "python"],
    default=None,
    help="GRPO reward signal. Defaults per dataset: gsm8k->'random' "
         "(Park random-reward runs; use 'gsm8k' for their true-reward RLVR), "
         "deepscaler->'ground_truth'.",
)
parser.add_argument(
    "--eps_low",
    type=float,
    default=None,
    help="Clip-low epsilon. Paper baseline 0.2; 1.0 disables clip-low. "
         "None -> config default (0.2).",
)
parser.add_argument(
    "--eps_high",
    type=str,
    default=None,
    help="Clip-high epsilon. Paper baseline symmetric (leave unset); "
         "'inf' disables clip-high; e.g. 0.28 for DAPO-style clip-higher.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Total optimizer steps (16 per rollout round). Default ~1 GSM8K epoch (232).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed (run >=2-3 seeds for entropy curves).",
)
parser.add_argument(
    "--run_name",
    default="run",
    help="Run name used to form output dir (default: <method>_<task>).",
)
parser.add_argument(
    "--output_dir",
    default=None,
    help="Explicit output directory (overrides --output_root/--run_name).",
)
parser.add_argument(
    "--lora",
    default=False,
    action="store_true",
    help="Enable lora"
)
parser.add_argument(
    "--attn",
    choices=["flash_attention_2", "sdpa"],
    default="flash_attention_2",
    help="Attention implementation",
)
args = parser.parse_args()


if __name__ == "__main__":
    run = wandb.init(
        entity="adamelkholy25-university-of-cambridge",
        project="dissertation",
        name=args.run_name,
        config={
            "model": args.model,
            "dataset": args.dataset,
            "reward": args.reward,
            "eps_low": args.eps_low,
            "eps_high": args.eps_high,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
    )

    args.output_dir = resolve_output_dir(args)
    post_trainer = GRPORunner()

    print(
        f"Benchmarking {args.model}, method=GRPO, "
        f"reward={args.reward}, dataset={args.dataset}\n"
        f"Checkpoints/logs -> {args.output_dir}"
    )
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if GRPO_CONFIG.get("fp16"):
        print("Using fp16")
        precision_dtype = torch.float16    
    else:
        precision_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=precision_dtype,
        trust_remote_code=True,
        attn_implementation=args.attn, 

        # NOTE: device_map="auto" is fine on a single GPU. For multi-GPU GRPO,
        # drop it and launch with `accelerate launch` so Accelerate handles
        # device placement (device_map="auto" conflicts with DDP).
        # device_map="auto",
    )

    if args.lora:
        print(f"LoRA enabled with config: {LORA_CONFIG}")
        model = get_peft_model(model, LORA_CONFIG)
        model.print_trainable_parameters()

    post_trainer.run(model, tokenizer, args)

    time_taken = time.time() - start
    hrs, rem = divmod(time_taken, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Completed in {int(hrs)}h {int(mins)}m {secs:.2f}s")
