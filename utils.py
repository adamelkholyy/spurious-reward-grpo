import os
import re
import time
from typing import Any, Tuple


def get_completion_text(completion: Any) -> str:
    """Return generated text for both plain-text and chat-style completions.

    In TRL GRPO:
      - If your dataset has `prompt: str`, completions are returned as plain strings.
      - If your dataset has `prompt: list[{'role','content'}]`, completions are returned as chat messages.
    """

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
        run_name = cli_args.run_name or f"{cli_args.method}_gsm8k"
        out = os.path.join("outputs", run_name)

    if os.path.exists(out):
        out = f"{out}-{int(time.time())}"

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
    """Remove GSM8K-style calculator annotations (e.g. '<<48/2=24>>').

    The raw GSM8K 'main' split embeds these inline within the reasoning
    (e.g. "...sold 48/2 = <<48/2=24>>24 clips..."). They aren't natural
    language and training on them as-is can push generations toward an
    unnatural format, which we've seen hurts flexible-match extraction at
    eval time. This strips them while leaving the surrounding text intact.
    """
    return re.sub(r"<<[^>]*>>", "", text)

def split_prompt_answer(text: str) -> Tuple[str, str]:

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
