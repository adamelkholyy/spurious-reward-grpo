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
    help="Park et al. 2509.26114 random-reward base models: "
         "Qwen/Qwen2.5-1.5B-Instruct (main), meta-llama/Llama-3.2-1B-Instruct, "
         "allenai/OLMo-2-0425-1B-Instruct. True-reward GSM8K runs: "
         "Qwen/Qwen2.5-3B-Instruct.",
)
parser.add_argument(
    "--dataset",
    choices=available_tasks(),
    default="gsm8k",
    help="Dataset task (one file per dataset under tasks/). "
         "'gsm8k' = Park et al. 2509.26114 setup (default). "
         "'deepscaler' = legacy Shao-style spurious-rewards setup. "
         "Add a new dataset by dropping a tasks/<name>.py file in.",
)
parser.add_argument(
    "--reward",
    choices=available_rewards(),
    default=None,
    help="GRPO reward signal (see rewards.REWARD_REGISTRY). If unset, the "
         "task's default_reward is used (gsm8k->'random', "
         "deepscaler->'ground_truth').",
)
parser.add_argument(
    "--eps_low",
    type=float,
    default=0.2,
    help="Clip-low epsilon. Paper baseline 0.2; 1.0 disables clip-low. "
         "None -> config default (0.2).",
)
parser.add_argument(
    "--eps_high",
    type=str,
    default=0.2,
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
    default=0,
    help="Random seed (run >=2-3 seeds for entropy curves).",
)
parser.add_argument(
    "--run_name",
    default="run",
    help="Run name used to form output dir (default: <method>_<task>).",
)
parser.add_argument(
    "--lr",
    default=None,
    help="learning_rate",
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
    "--new_reward",
    default="ground_truth",
    help="New R"
)
parser.add_argument(
    "--switch_step",
    default=None,
    help="Step at which to switch epsilon params and reward"
)
parser.add_argument(
    "--save_step",
    default=None,
    help="Step at which to switch to save checkpoints"
)
parser.add_argument(
    "--new_eps_low",
    default=0.20,
    help="Epsilon low after switch"
)
parser.add_argument(
    "--new_eps_high",
    default=0.28,
    help="Epsilon high after switch"
)
parser.add_argument(
    "--entropy_thermostat",
    action="store_true",
    help="Train on GT and inject SR entropy correction only when target bounds are violated.",
)
parser.add_argument(
    "--thermostat_normal_reward",
    choices=available_rewards(),
    default=None,
    help="Normal task reward used outside SR corrections; thermostat SR regions always use random.",
)
parser.add_argument(
    "--thermostat_target",
    type=float,
    default=1.0,
    help="Target raw train/entropy value (default: 1.0).",
)
parser.add_argument(
    "--thermostat_deadband",
    type=float,
    default=0.4,
    help="Half-width of the raw-entropy target band (default: 0.4 => H in [0.6, 1.4]).",
)
parser.add_argument(
    "--thermostat_ema_alpha",
    type=float,
    default=0.2,
    help="EMA weight for new entropy observations; 1.0 disables smoothing (default: 0.2).",
)
parser.add_argument(
    "--thermostat_interval",
    type=int,
    default=16,
    help="Optimizer steps between control decisions (default: 16, one current rollout cycle).",
)
parser.add_argument(
    "--thermostat_up_eps_low",
    type=float,
    default=0.05,
    help="Clip-low epsilon used to push entropy upward (default: 0.05).",
)
parser.add_argument(
    "--thermostat_up_eps_high",
    default="inf",
    help="Clip-high epsilon used to push entropy upward (default: inf, disabled).",
)
parser.add_argument(
    "--thermostat_down_eps_low",
    type=float,
    default=1.0,
    help="Clip-low epsilon used to push entropy downward (default: 1.0, disabled).",
)
parser.add_argument(
    "--thermostat_down_eps_high",
    default="0.10",
    help="Clip-high epsilon used to push entropy downward (default: 0.10).",
)
args = parser.parse_args()


if __name__ == "__main__":
    args.run_name = f"{args.run_name}_low={args.eps_low}_high={args.eps_high}_s{args.seed}"  

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
            "entropy_thermostat": args.entropy_thermostat,
            "thermostat_target": args.thermostat_target,
            "thermostat_normal_reward": args.thermostat_normal_reward,
            "thermostat_deadband": args.thermostat_deadband,
            "thermostat_ema_alpha": args.thermostat_ema_alpha,
            "thermostat_interval": args.thermostat_interval,
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
        attn_implementation="sdpa" if HPC else "flash_attention_2", # args.attn, 
    )

    post_trainer.run(model, tokenizer, args)
