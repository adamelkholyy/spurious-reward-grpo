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
    default="Qwen/Qwen2.5-Math-1.5B",
    help="Spurious-reward effect is a Qwen2.5-Math property. "
         "Use Qwen/Qwen2.5-Math-1.5B for a much cheaper run.",
)
parser.add_argument(
    "--reward",
    choices=["ground_truth", "random", "box_only", "incorrect", "python"],
    default="ground_truth",
    help="GRPO reward signal. 'ground_truth' is the upper bound; the rest are "
         "the spurious rewards from the paper.",
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
    )

    args.output_dir = resolve_output_dir(args)
    post_trainer = GRPORunner()

    print(
        f"Benchmarking {args.model}, method=GRPO, "
        f"reward={args.reward}, task=DeepScaleR\n"
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
