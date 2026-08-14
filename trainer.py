import argparse

import torch
import wandb
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from settings import HPC
from utils import resolve_output_dir
from GRPORunner import GRPORunner
from tasks import available_tasks
from rewards import available_rewards

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="Qwen/Qwen2.5-1.5B-Instruct",
    help="Model to train under GRPO, can be huggingface repo or local dir",
)
parser.add_argument(
    "--dataset",
    choices=available_tasks(),
    default="gsm8k",
    help="Dataset to train on, options = ['gsm8k', 'countdown4', 'dapo', 'mbpp', 'aime2024'], defaults to gsm8k",
)
parser.add_argument(
    "--reward",
    choices=available_rewards(),
    default=None,
    help="Reward function, either 'random' for random binary rewards or the dataset name for ground truth rewards, defaults to gsm8k ground truth",
)
parser.add_argument(
    "--eps_low",
    type=float,
    default=0.2,
    help="Lower clipping bound, defaults to 0.2",
)
parser.add_argument(
    "--eps_high",
    type=str,
    default=0.28,
    help="Upper clipping bound, defaults to 0.28, can accept 'inf' to disable upper clipping",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Total optimizer steps, defaults to 2400",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed",
)
parser.add_argument(
    "--run_name",
    default="run",
    help="Run name used for wandb and output dir",
)
parser.add_argument(
    "--lr",
    default=None,
    help="AdamW optimiser learning rate",
)
parser.add_argument(
    "--output_dir",
    default=None,
    help="Output directory, default: <run_name>_low=<eps_low>_high=<eps_high>_s<seed>).",
)
parser.add_argument("--lora", default=False, action="store_true", help="Enable LoRA")
parser.add_argument(
    "--new_reward",
    default="ground_truth",
    help="New reward function to switch to in the scheduling regime",
)
parser.add_argument(
    "--switch_step",
    default=None,
    help="Step at which to switch epsilon params and reward in the scheduling regime",
)
parser.add_argument(
    "--save_step", default=None, help="Saves a checkpoint every <save_step> steps"
)
parser.add_argument(
    "--new_eps_low",
    default=0.20,
    help="Epsilon low to switch to in the scheduling regime",
)
parser.add_argument(
    "--new_eps_high",
    default=0.28,
    help="Epsilon high to switch to in the scheduling regime",
)

args = parser.parse_args()


if __name__ == "__main__":
    args.run_name = (
        f"{args.run_name}_low={args.eps_low}_high={args.eps_high}_s{args.seed}"
    )

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

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa" if HPC else "flash_attention_2",
    )

    post_trainer.run(model, tokenizer, args)
