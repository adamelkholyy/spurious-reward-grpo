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
- `--dataset`: one of `gsm8k`, `deepscaler`, `wordle`, `countdown`, `dapo`, `aime2024`, `mbpp`, `capitals`, `numbers`
- `--reward`: choose  `random` for Spurious Rewards, else dataset name for the relevant ground-truth rewards (defaults to the dataset's `default_reward`). For GSM8K, `gsm8k_strategy` (α = 0.5) and `gsm8k_strategy_a025` / `gsm8k_strategy_a075` reward each correct sample by α + (1 − α)/n, where n counts the correct samples in its group sharing its strategy, i.e. the set of verified intermediate values on the path to the answer (`strategy_reward.py`). `gsm8k_embedding` (α = 0.5) and `gsm8k_embedding_a025` use a soft count of semantically similar correct solutions instead (`embedding_reward.py`): they need `--embed_tau`, calibrated on base-model rollouts with `python calibrate_embedding.py --model <base>` (which also says whether to pass `--embed_keep_numbers`); `--embed_model` picks the frozen embedder (default `BAAI/bge-small-en-v1.5`)
- `--eps_low` / `--eps_high`: GRPO clipping bounds (default `0.2` / `0.28`, DAPO-style), `inf` disables clip-high
- `--run_name`: wandb run name and output dir (default `<dataset>_<model>_gt_<max_steps>_s<seed>`; note it does not encode the reward or clipping bounds)
- `--max_steps`, `--lr`, `--seed`, `--save_step`, `--output_dir`
- `--data_fraction F` / `--single_example`: train on a random fraction `F` of the training set, or on one random example (both seeded by `--seed`, mutually exclusive). A subset smaller than one generation batch is repeated to fill it, so it is still only those distinct problems
- `--switch_step`: at this step switch to `--new_reward` / `--new_eps_low` / `--new_eps_high` (`ScheduledGRPOTrainer`); a checkpoint is saved at the switch
- `--range LO-HI`: inclusive number range for `--dataset numbers` (default `1-100`); also accepted by both eval scripts, so e.g. train with `--range 1-100` and evaluate the held-out `--range 101-200`. eval.py writes every range to `results/results_numbers.json` (one row per model × range) and plots each row's number distribution to `results/plots/numbers/LO-HI/<label>.png`; `python merge_numbers_results.py` folds old per-range `results_numbers_LO-HI.json` files into it

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
 