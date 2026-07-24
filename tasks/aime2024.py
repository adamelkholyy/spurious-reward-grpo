"""AIME 2024 task — EVAL ONLY (the DAPO recipe's validation set).

Data: ``BytedTsinghua-SIA/AIME-2024``: the 30 AIME 2024 (I & II) problems in
the same verl schema as DAPO-Math-17k — one embedded-instruction user message
per row ('... The last line of your response should be of the form
Answer: $Answer ...'), gold integer at ``reward_model.ground_truth``, single
``train`` split. The hub intentionally repeats each problem 32x (960 rows,
for verl's Best-of-32 metric); the inherited prompt-content dedupe collapses
that back to the 30 unique problems.

Everything — loading/dedupe/flattening, the 'Answer:' extraction, grading,
and the split-name handling eval.py needs — is inherited from the DAPO task,
so DAPO-trained checkpoints are benchmarked with byte-identical prompt format
and grading. Held out: none of these problems are in (deduped) DAPO-Math-17k.

This task is eval-only; ``build_train`` refuses so a 30-problem benchmark can
never silently become training data. Note for eval.py: pass runs explicitly
via ``--models label=path`` — auto-discovery filters run labels by dataset
name and would skip runs labelled 'dapo'.
"""

from __future__ import annotations

from .dapo import DAPOMathTask
from .registry import register_task


@register_task
class AIME2024Task(DAPOMathTask):
    name = "aime2024"
    hf_path = "BytedTsinghua-SIA/AIME-2024"
    holdout_n = 0  # external benchmark: all 30 problems ARE the eval set

    # ---- eval-only ----
    def build_train(self, tokenizer):
        raise RuntimeError(
            "aime2024 is an eval-only task (the 30 AIME 2024 benchmark "
            "problems). Train with --dataset dapo and benchmark the "
            "checkpoints with --dataset aime2024."
        )
