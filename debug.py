import logging
from typing import Any

from settings import DEBUG_EVERY, DEBUG_N

logger = logging.getLogger(__name__)

_LAST_DEBUG_STEP: int | None = None


def maybe_debug_print_grpo(
    *,
    trainer_state: Any,
    prompts: list[Any],
    responses: list[str],
    answers: list[Any],
    extracted: list[Any] | None = None,
    scores: list[Any] | None = None,
    header: str = "GRPO DEBUG",
) -> None:
    """Periodically log a sample of GRPO rollouts.

    Emits every DEBUG_EVERY steps, capped at DEBUG_N samples per call.
    No-ops when DEBUG_EVERY <= 0 or the trainer state is unavailable.
    """
    global _LAST_DEBUG_STEP

    if trainer_state is None or not hasattr(trainer_state, "global_step"):
        return

    step = int(trainer_state.global_step)
    if DEBUG_EVERY <= 0 or step % DEBUG_EVERY != 0 or step == _LAST_DEBUG_STEP:
        return

    _LAST_DEBUG_STEP = step

    n = min(max(1, DEBUG_N), len(responses))
    sep = "=" * 100

    lines = [f"\n{sep}", f"[{header}] global_step={step} (printing {n} sample(s))"]
    for i in range(n):
        lines += [
            f"\n[PROMPT]\n{prompts[i]}",
            f"\n[COMPLETION]\n{responses[i]}",
            f"\n[GT_ANSWER] {answers[i]!r}",
        ]
        if extracted is not None:
            lines.append(f"[EXTRACTED_GUESS] {extracted[i]!r}")
        if scores is not None:
            lines.append(f"[REWARD] {scores[i]}")
    lines.append(f"{sep}\n")

    logger.debug("\n".join(lines))