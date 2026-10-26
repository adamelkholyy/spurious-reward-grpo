import argparse
import json
import os
import re
import sys
from typing import Any

# label -> HF id for the base models every run is compared against
BASE_MODELS = {
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "olmo": "allenai/OLMo-2-0425-1B-Instruct",
    "qwen-small": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
}

# eval dataset -> the training-run tag its runs are named with (DAPO runs are benchmarked on AIME 2024)
DATASET_ALIASES = {"aime2024": "dapo"}


def get_baseline_key(label: str):
    """Map a run label to its base-model label in BASE_MODELS (or None)."""
    lbl = label.lower()
    if "llama" in lbl:
        return "llama"
    if "olmo" in lbl:
        return "olmo"
    if "qwen" in lbl:
        if "small" in lbl:
            return "qwen-small"
        if "3b" in lbl:
            return "qwen3b"
        return "qwen"
    return None


def get_completion_text(completion: Any) -> str:

    if isinstance(completion, str):
        return completion

    if isinstance(completion, list) and completion:
        msg0 = completion[0]
        if isinstance(msg0, dict) and "content" in msg0:
            return str(msg0["content"])

    if isinstance(completion, dict) and "content" in completion:
        return str(completion["content"])

    return str(completion)


def resolve_output_dir(cli_args):
    out = cli_args.output_dir or os.path.join("outputs", cli_args.run_name)
    os.makedirs(out, exist_ok=True)
    return out


def resolve_run_name(cli_args):
    # case-sensitive substring checks catch local checkpoint paths; HF ids fall through to the lookup
    if "llama" in cli_args.model:
        model_name = "llama"
    elif "qwen" in cli_args.model:
        model_name = "qwen"
    elif "olmo" in cli_args.model:
        model_name = "olmo"
    else:
        model_lookup = {hf_id: label for label, hf_id in BASE_MODELS.items()}
        model_name = model_lookup.get(cli_args.model, "model")

    if cli_args.switch_step:
        info = f"{cli_args.switch_step}-{cli_args.max_steps}"
    else:
        info = f"gt_{cli_args.max_steps}"

    name = f"{cli_args.dataset}_{model_name}_{info}_s{cli_args.seed}"
    if cli_args.data_fraction < 1.0:
        name += f"_frac{cli_args.data_fraction:g}"
    if cli_args.range:
        from tasks.numbers import DEFAULT_RANGE

        if cli_args.range != DEFAULT_RANGE:
            name += f"_range{cli_args.range[0]}-{cli_args.range[1]}"
    return name


def answer_counts(task_metrics: dict):
    """Per-answer counts stored by a diversity task (capitals or numbers), else None."""
    return task_metrics.get("country_counts", task_metrics.get("number_counts"))


def parse_range(value: str) -> tuple[int, int]:
    """argparse type for --range: 'LO-HI' -> (LO, HI), inclusive."""
    m = re.fullmatch(r"(\d+)-(\d+)", value.strip())
    if not m or int(m[1]) > int(m[2]):
        raise argparse.ArgumentTypeError(f"expected LO-HI with LO <= HI, got '{value}'")
    return int(m[1]), int(m[2])


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def is_model_dir(d: str) -> bool:
    try:
        files = os.listdir(d)
    except OSError:
        return False
    return "config.json" in files and any(f.endswith((".safetensors", ".bin")) for f in files)


def checkpoint_step(name: str):
    """sort key: numeric step from a trailing integer (checkpoint-300 -> 300)."""
    m = re.search(r"(\d+)$", name)
    return (int(m.group(1)) if m else -1, name)


def check_for_alias(run_dir):
    """The run's original name from the TRL model card, which survives renaming the run dir."""
    path = os.path.join(run_dir, "README.md")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        m = re.search(r"(?m)^model_name:\s*(.+)$", f.read())
    return m.group(1) if m else None


def discover_models(outputs_dir: str, all_checkpoints, dataset_name, existing_labels):
    """(label, path) for each run under outputs_dir not in `existing_labels`: its latest checkpoint, or all of them."""
    if not os.path.isdir(outputs_dir):
        sys.exit(f"No such directory: {outputs_dir}")

    seen, pairs = {}, []
    for name in sorted(os.listdir(outputs_dir)):
        run_dir = os.path.join(outputs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        if DATASET_ALIASES.get(dataset_name, dataset_name) not in name:
            continue
        checkpoints = sorted(
            (c for c in os.listdir(run_dir) if is_model_dir(os.path.join(run_dir, c))),
            key=checkpoint_step,
        )
        if not checkpoints:
            print(f"{run_dir}: no model checkpoints found, skipping")
            continue
        if not all_checkpoints:
            checkpoints = checkpoints[-1:]

        # one label per run (deduped across runs); checkpoints add a step suffix.
        base = re.sub(r"-\d{8,}$", "", name)  # strip trailing timestamp
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0

        alias = check_for_alias(run_dir)
        for checkpoint in checkpoints:
            label = f"{base}-{checkpoint_step(checkpoint)[0]}" if all_checkpoints else base
            if label in existing_labels or alias in existing_labels:
                continue
            pairs.append((label, os.path.join(run_dir, checkpoint)))
    return pairs
