import argparse
import os
import torch
import wandb
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from settings import HPC
from utils import parse_range, resolve_output_dir, resolve_run_name
import GRPORunner
from tasks import available_tasks, get_task
from rewards import available_rewards
from embedding_reward import DEFAULT_EMBED_MODEL, set_embedding_config

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
    help="Dataset to train on, defaults to gsm8k",
)
parser.add_argument(
    "--reward",
    choices=available_rewards(),
    default=None,
    help="Reward function, e.g. 'random' for random binary rewards or the dataset name for ground truth rewards, defaults to the dataset's default_reward",
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
    default=2400,
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
    default=None,
    help="Run name used for wandb and output dir",
)
parser.add_argument(
    "--lr",
    type=float,
    default=None,
    help="AdamW optimiser learning rate",
)
parser.add_argument(
    "--output_dir",
    default=None,
    help="Output directory, default: outputs/<run_name>",
)
parser.add_argument(
    "--new_reward",
    default="ground_truth",
    help="New reward function to switch to in the scheduling regime",
)
parser.add_argument(
    "--switch_step",
    type=int,
    default=None,
    help="Step at which to switch epsilon params and reward in the scheduling regime",
)
parser.add_argument(
    "--save_step", type=int, default=None, help="Saves a checkpoint every <save_step> steps"
)
parser.add_argument(
    "--new_eps_low",
    type=float,
    default=0.20,
    help="Epsilon low to switch to in the scheduling regime",
)
parser.add_argument(
    "--new_eps_high",
    default=0.28,
    help="Epsilon high to switch to in the scheduling regime",
)
parser.add_argument(
    "--data_fraction",
    type=float,
    default=1.0,
    help="Fraction of the training dataset to use, in (0, 1]. e.g. 0.5 = a random 50%% subset "
    "(seeded by --seed). Defaults to 1.0 (full dataset)",
)
parser.add_argument(
    "--range",
    type=parse_range,
    default=None,
    help="Inclusive LO-HI range for --dataset numbers, defaults to 1-100",
)
parser.add_argument(
    "--embed_tau",
    type=float,
    default=None,
    help="Calibrated cosine threshold for the gsm8k_embedding rewards (required by them, see calibrate_embedding.py)",
)
parser.add_argument(
    "--embed_model",
    default=DEFAULT_EMBED_MODEL,
    help=f"Frozen sentence embedder for the gsm8k_embedding rewards, defaults to {DEFAULT_EMBED_MODEL}",
)
parser.add_argument(
    "--embed_keep_numbers",
    action="store_true",
    help="Don't mask numbers as 'N' before embedding (use if calibration gives a higher AUC unmasked)",
)

args = parser.parse_args()

if not 0.0 < args.data_fraction <= 1.0:
    parser.error(f"--data_fraction must be in (0, 1], got {args.data_fraction}")
if args.range:
    if args.dataset != "numbers":
        parser.error("--range only applies to --dataset numbers")
    get_task("numbers").range = args.range
if any(r and r.startswith("gsm8k_embedding") for r in (args.reward, args.switch_step and args.new_reward)):
    if args.embed_tau is None:
        parser.error("the gsm8k_embedding rewards need --embed_tau (calibrate it with calibrate_embedding.py)")
    if not 0.0 <= args.embed_tau < 1.0:
        parser.error(f"--embed_tau must be in [0, 1), got {args.embed_tau}")
set_embedding_config(args.embed_tau, args.embed_model, not args.embed_keep_numbers)


if __name__ == "__main__":
    if args.run_name is None:
        args.run_name = resolve_run_name(args)
    args.reward = get_task(args.dataset).resolve_reward(args.reward)

    if int(os.environ.get("RANK", 0)) == 0:
        wandb.init(
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
                "data_fraction": args.data_fraction,
                "range": args.range,
                "lr": args.lr,
                "switch_step": args.switch_step,
                "embed_tau": args.embed_tau,
                "embed_model": args.embed_model,
                "embed_mask_numbers": not args.embed_keep_numbers,
            },
        )

    args.output_dir = resolve_output_dir(args)

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

    GRPORunner.run(model, tokenizer, args)
