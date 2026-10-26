# spurious-reward-grpo

GRPO training and evaluation framework for running GRPO post-training with Spurious Rewards. Includes evaluation scripts and entropy distribution reconstruction.

## Install

```bash
pip install -r requirements.txt
pip install vllm wandb
```

Both training and eval import a local, untracked `settings.py` (gitignored) defining:
- `HPC`: `True` on the CSD3 cluster, `False` on the workstation (selects attention backend, vLLM memory and the eval working directory)
- `GRPO_CONFIG`: base `trl.GRPOConfig` kwargs, overridden per model/dataset and by CLI flags in `GRPORunner.build_grpo_config`
- `INF_EPS_HIGH`: the `epsilon_high` value used for `--eps_high inf`

## Train

```bash
python trainer.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset gsm8k --reward random --eps_low 0.2 --eps_high 0.28 --run_name SR_grpo
```

Key flags:
- `--model`: HF model id or local directory
- `--dataset`: one of `gsm8k`, `deepscaler`, `wordle`, `countdown`, `dapo`, `aime2024`, `mbpp`
- `--reward`: choose  `random` for Spurious Rewards, else dataset name for the relevant ground-truth rewards (defaults to the dataset's `default_reward`)
- `--eps_low` / `--eps_high`: GRPO clipping bounds (default `0.2` / `0.28`, DAPO-style), `inf` disables clip-high
- `--run_name`: wandb run name and output dir (default `<dataset>_<model>_gt_<max_steps>_s<seed>`; note it does not encode the reward or clipping bounds)
- `--max_steps`, `--lr`, `--seed`, `--save_step`, `--output_dir`
- `--switch_step`: at this step switch to `--new_reward` / `--new_eps_low` / `--new_eps_high` (`ScheduledGRPOTrainer`); a checkpoint is saved at the switch

Checkpoints/logs go to `outputs/<run_name>` and metrics log to Weights & Biases



## Evaluate

```bash
python eval.py --dataset gsm8k --n 256
```

Evaluates the accuracy and pass@n for every model found in the output directory that matches the dataset (and hasn't already been evaluated). Pass `--eval-baselines` to also evaluate the base models, or `--models label=path ...` to pick models explicitly.
```bash
python eval_output_dist.py --dataset aime2024
```
Constructs the entropy distributions of every model in the output dir matching the dataset (that hasn't already been evaluated). 

`eval_output_dist.py` takes `--tag <substring>` to only measure runs whose folder name contains the substring; runs listed in `results/.disabled_models.json` (toggled with `disable_models.py`) are skipped.

`print_table.py` / `print_results_pct.py` reprint the results tables (absolute / relative deltas; the latter adds entropy deltas from `results_output_dist.json`).
 