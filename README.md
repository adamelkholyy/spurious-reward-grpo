# spurious-reward-grpo

GRPO training and evaluation framework for running GRPO post-training with Spurious Rewards. Includes evaluation scripts and entropy distribution reconstruction.

## Install

```bash
pip install -r requirements.txt
pip install vllm wandb   
```

## Train

```bash
python trainer.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset gsm8k --reward random --eps_low 0.2 --eps_high 0.28 --run_name SR_grpo
```

Key flags:
- `--model`: HF model id or local directory
- `--dataset`: one of `gsm8k`, `deepscaler`, `wordle`, `countdown`, `dapo`, `aime2024`, `mbpp`
- `--reward`: choose  `random` for Spurious Rewards, else dataset name for the relevant ground-truth rewards
- `--eps_low` / `--eps_high`: GRPO clipping bounds (default `0.2` / `0.28`, DAPO-style), `inf` disables clip-high
 `--run_name`: wandb run name and saved output file
- `--max_steps`, `--lr`, `--seed`, `--lora`, `--output_dir`

Checkpoints/logs go to `outputs/<run_name>` and metrics log to Weights & Biases  



## Evaluate

```bash
python eval.py --dataset gsm8k --n 256
```

Evaluates the accuracy and pass@k for every model found in the output directory that matches the dataset (and hasn't already been evaluated)
```bash
python eval_output_dist.py --dataset aime2024
```
Constructs the entropy distributions of every model in the output dir matching the dataset (that hasn't already been evaluated). 

Both eval scripts can take a `--tag <substring>` argument to only evaluate models containing the substring.
 