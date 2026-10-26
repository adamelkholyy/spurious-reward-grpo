import os
from typing import Any

# label -> HF id for the base models every run is compared against
BASE_MODELS = {
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "olmo": "allenai/OLMo-2-0425-1B-Instruct",
    "qwen-small": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
}


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
    return name
