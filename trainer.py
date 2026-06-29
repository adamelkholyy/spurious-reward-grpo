import argparse
import time

import torch
import wandb

from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from backups.settings import LORA_CONFIG
from backups.utils import resolve_output_dir

from backups.runners.GRPORunner import GRPORunner
from backups.runners.SFTRunner import SFTRunner


parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["sft", "grpo", "reward"], default="grpo")
parser.add_argument(
    "--model",
    default="Qwen/Qwen2.5-Math-7B",
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
args = parser.parse_args()


if __name__ == "__main__":
    run = wandb.init(
        entity="adamelkholy25-university-of-cambridge",
        project="dissertation",
        name=args.run_name,
    )

    args.output_dir = resolve_output_dir(args)

    match args.method:
        case "sft": post_trainer = SFTRunner()
        case "grpo": post_trainer = GRPORunner()

    print(
        f"Benchmarking {args.model}, method={args.method}, "
        f"reward={args.reward}, task=DeepScaleR\n"
        f"Checkpoints/logs -> {args.output_dir}"
    )
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation= "sdpa", # NOTE: change back "flash_attention_2",
        # NOTE: device_map="auto" is fine on a single GPU. For multi-GPU GRPO,
        # drop it and launch with `accelerate launch` so Accelerate handles
        # device placement (device_map="auto" conflicts with DDP).
        # device_map="auto",
    )

    if args.method == "grpo":
        print("Running GRPO: LoRA disabled, training all parameters")
    else:
        model = get_peft_model(model, LORA_CONFIG)
        model.print_trainable_parameters()

    post_trainer.run(model, tokenizer, args)

    time_taken = time.time() - start
    hrs, rem = divmod(time_taken, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Completed in {int(hrs)}h {int(mins)}m {secs:.2f}s")
