"""Base class for a *dataset task*.

A "task file" bundles everything dataset-specific in ONE place so the rest of
the codebase (training + all eval scripts) stays dataset-agnostic:

  * how examples are turned into GRPO ``prompt`` / ``answer`` columns (training)
  * the prompt used at eval time (which may differ from the training prompt)
  * how the gold answer is extracted from a raw dataset row
  * how a model completion is graded against that gold
  * which reward it defaults to, and (optionally) which rewards are valid

To add a new dataset you subclass :class:`DatasetSpec`, fill in the class
attributes, implement the handful of methods below, and register it with
``@register_task`` (see ``tasks/registry.py``). No edits to ``GRPORunner`` /
``trainer`` / the eval scripts are needed — they all resolve the active task by
name via ``tasks.get_task(args.dataset)``.

Note on the two prompt "views": for some datasets the *training* prompt and the
*eval* prompt intentionally differ (e.g. GSM8K trains with a "#### <n>"
instruction but is benchmarked with a ``\\boxed{}`` prompt, matching the
Spurious-Rewards setup). ``format_train_example`` owns the former;
``build_eval_prompts`` owns the latter. Keeping both here means the two views
can never silently drift out of sync across files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


class DatasetSpec:
    # --- identity -----------------------------------------------------------
    #: Name used on the CLI (``--dataset <name>``) and as the registry key.
    name: str = ""

    #: HuggingFace dataset coordinates (informational; used by the default
    #: loaders in subclasses). ``hf_config`` is the builder config name.
    hf_path: str = ""
    hf_config: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "test"

    # --- rewards ------------------------------------------------------------
    #: Reward used when ``--reward`` is not passed on the CLI.
    default_reward: str = "ground_truth"
    #: Rewards that are meaningful for this task. ``None`` => any registered
    #: reward is allowed (useful for ablation sweeps like the random-reward
    #: study, where every reward is legitimately run against one dataset).
    allowed_rewards: Optional[Sequence[str]] = None

    # --- prompt text --------------------------------------------------------
    #: System prompt used at *eval* time (and by datasets that train with a
    #: system message, e.g. DeepScaleR). May be ``None``.
    system_prompt: Optional[str] = None
    #: Instruction appended to the question at *training* time. May be ``None``.
    train_instruction: Optional[str] = None

    # --- held-out eval for train-only hubs ---------------------------------
    #: If > 0, carve this many rows out of the training data to serve as the
    #: eval set (for datasets whose hub ships a train split only). Applied by
    #: the default ``build_train`` / ``_load_eval_split``; tasks with custom
    #: loaders apply :meth:`_split_holdout` themselves.
    holdout_n: int = 0
    #: Seed for the carve-out permutation. ``None`` => no shuffle: the
    #: holdout is literally the first ``holdout_n`` rows (only safe if the
    #: hub ordering is known to be random).
    holdout_seed: Optional[int] = 0

    def _split_holdout(self, ds) -> Tuple[Any, Any]:
        """``(train_ds, eval_ds)`` — deterministic disjoint carve-out.

        Selection is a seeded permutation (or a plain prefix when
        ``holdout_seed`` is ``None``); both halves are re-sorted by original
        index so row order stays stable and cache-friendly.
        """
        import random

        idx = list(range(len(ds)))
        if self.holdout_seed is not None:
            random.Random(self.holdout_seed).shuffle(idx)
        k = min(self.holdout_n, len(ds))
        held, rest = sorted(idx[:k]), sorted(idx[k:])
        return ds.select(rest), ds.select(held)

    def _holdout_active(self, split: Optional[str]) -> bool:
        """Holdout applies when evaluating the (train-only) training split."""
        return self.holdout_n > 0 and (split or self.eval_split) == self.train_split

    # =======================================================================
    # Training
    # =======================================================================
    def format_train_example(self, x: Dict[str, Any], tokenizer) -> Dict[str, str]:
        """Map one raw dataset row -> ``{"prompt": ..., "answer": ...}``.

        Pure and side-effect free so it is unit-testable with a fake tokenizer.
        Subclasses MUST implement this.
        """
        raise NotImplementedError

    def build_train(self, tokenizer):
        """Return the training ``datasets.Dataset`` with ``prompt``/``answer``.

        Default implementation loads ``hf_path``/``hf_config`` and maps
        :meth:`format_train_example` over it. Override for anything fancier.
        """
        from datasets import load_dataset

        ds = load_dataset(self.hf_path, self.hf_config, split=self.train_split)
        if self.holdout_n > 0:
            ds, _ = self._split_holdout(ds)  # never train on the eval rows
        return ds.map(
            lambda x: self.format_train_example(x, tokenizer),
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

    # =======================================================================
    # Eval — prompts
    # =======================================================================
    def _chatml_fallback(self, question: str) -> str:
        """Explicit Qwen ChatML, used when a tokenizer ships no chat template.

        Intentional: replicates the Spurious-Rewards setup of one prompt format
        across model families. Special tokens tokenize as plain text on
        non-Qwen vocabs — that is expected and recorded as ``chatml_fallback``.
        """
        sys_block = (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            if self.system_prompt
            else ""
        )
        return (
            f"{sys_block}"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _messages_for_eval(self, question: str) -> List[Dict[str, str]]:
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": question})
        return msgs

    def build_eval_prompts(
        self, questions: Sequence[str], tokenizer
    ) -> Tuple[List[str], str]:
        """Return ``(prompts, template_used)`` for a list of raw questions.

        ``template_used`` is ``"tokenizer_chat_template"`` or
        ``"chatml_fallback"``. This is the single prompt builder shared by all
        eval scripts; keeping it here guarantees they stay byte-identical.
        """
        if getattr(tokenizer, "chat_template", None):
            prompts = [
                tokenizer.apply_chat_template(
                    self._messages_for_eval(q),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for q in questions
            ]
            return prompts, "tokenizer_chat_template"
        return [self._chatml_fallback(q) for q in questions], "chatml_fallback"

    # =======================================================================
    # Eval — data loading
    # =======================================================================
    def _load_eval_split(self, split: Optional[str] = None):
        from datasets import load_dataset

        ds = load_dataset(
            self.hf_path, self.hf_config, split=split or self.eval_split
        )
        if self._holdout_active(split):
            _, ds = self._split_holdout(ds)
        return ds

    #: Column holding the natural-language problem/question.
    question_column: str = "question"
    #: Column holding the raw gold answer (passed to :meth:`extract_gold`).
    answer_column: str = "answer"

    def load_eval(
        self, split: Optional[str] = None, limit: Optional[int] = None
    ) -> Tuple[List[str], List[str]]:
        """Return ``(questions, golds)`` for an eval split.

        ``golds`` are already run through :meth:`extract_gold`.
        """
        ds = self._load_eval_split(split)
        if limit and limit > 0:
            ds = ds.select(range(min(limit, len(ds))))
        questions = [ex[self.question_column] for ex in ds]
        golds = [self.extract_gold(ex[self.answer_column]) for ex in ds]
        return questions, golds

    def sample_questions(
        self, split: str, num_prompts: int, seed: int
    ) -> List[str]:
        """Seeded sample of raw questions (order-stable across models/re-runs).

        Used by the entropy / output-distribution evals, which need the same
        prompt set for every checkpoint so differences aren't confounded by the
        prompt sample.
        """
        import random

        ds = self._load_eval_split(split)
        rng = random.Random(seed)
        idxs = rng.sample(range(len(ds)), k=min(num_prompts, len(ds)))
        return [ds[i][self.question_column] for i in idxs]

    # =======================================================================
    # Eval — grading
    # =======================================================================
    def extract_gold(self, answer_field: Any) -> str:
        """Normalise the raw gold from a dataset row into a comparable string."""
        return str(answer_field).strip()

    def extract_answer(self, completion: str) -> Optional[str]:
        """Extract the model's final answer from a completion (for display /
        cheap comparison). Subclasses override with their canonical form."""
        raise NotImplementedError

    def make_grader(self):
        """Return a picklable-friendly ``is_correct(completion, gold) -> bool``.

        Eval scripts call this once per process (incl. inside multiprocessing
        pool workers). Subclasses implement the dataset's grading policy.
        """
        raise NotImplementedError

    # =======================================================================
    # Misc
    # =======================================================================
    def resolve_reward(self, reward_name: Optional[str]) -> str:
        """Pick the reward: explicit CLI value, else this task's default."""
        return reward_name or self.default_reward

    def validate_reward(self, reward_name: str) -> None:
        """Warn (not error) if a reward isn't in ``allowed_rewards``.

        We warn rather than raise so ablation sweeps that deliberately pair a
        dataset with an unusual reward aren't blocked.
        """
        if self.allowed_rewards is not None and reward_name not in self.allowed_rewards:
            import warnings

            warnings.warn(
                f"Reward '{reward_name}' is not in {self.name}.allowed_rewards "
                f"({list(self.allowed_rewards)}). Proceeding anyway.",
                stacklevel=2,
            )

    def eval_dataset_id(self, split: Optional[str] = None) -> str:
        """Human-readable id recorded in eval JSON (e.g. 'openai/gsm8k:main:test').

        Includes the holdout config when active, so eval results and baseline
        caches keyed on this id can never silently mix the held-out eval with
        the old eval-on-train numbers.
        """
        cfg = f":{self.hf_config}" if self.hf_config else ""
        hold = (
            f":heldout{self.holdout_n}s{self.holdout_seed}"
            if self._holdout_active(split)
            else ""
        )
        return f"{self.hf_path}{cfg}:{split or self.eval_split}{hold}"
