import os
import re
from typing import Any, tuple


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
    if cli_args.output_dir:
        out = cli_args.output_dir
    else:
        run_name = cli_args.run_name or f"{getattr(cli_args, 'method', 'grpo')}_gsm8k"
        out = os.path.join("outputs", run_name)

    # if os.path.exists(out):
    #     out = f"{out}-{int(time.time())}"

    os.makedirs(out, exist_ok=True)
    return out


def save_model(trainer, label):
    is_lora = hasattr(trainer.model, "save_pretrained") and hasattr(
        trainer.model, "peft_config"
    )
    out_dir = os.path.join(
        trainer.args.output_dir, f"{'adapter' if is_lora else 'checkpoint'}-{label}"
    )
    trainer.model.save_pretrained(out_dir)
    trainer.processing_class.save_pretrained(out_dir)
    print(f"{'Adapter' if is_lora else 'Model'} saved to {out_dir}")


def strip_calculator_annotations(text: str) -> str:
    """Remove GSM8K-style calculator annotations (e.g. '<<48/2=24>>')."""
    return re.sub(r"<<[^>]*>>", "", text)


def split_prompt_answer(text: str) -> tuple[str, str]:

    split_pattern = re.compile(r"(\nAnswer:|\nCorrect:|\nSolution\n|\nEndings:\n)")
    matches = list(split_pattern.finditer(text))

    if matches:
        m = matches[-1]  # rightmost match
        split_start = m.start()
        split_end = m.end()

        prompt = text[:split_start]
        answer = text[split_end:]
        return prompt, answer

    # else default to last newline split
    idx = text.rfind("\n")
    if idx != -1:
        return text[: idx + 1], text[idx + 1 :]

    return text, ""
