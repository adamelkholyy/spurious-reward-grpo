# `tasks/` — modular datasets

Each dataset is one file under `tasks/`. That file is the **single source of
truth** for the dataset's prompts, gold extraction, grading, and reward config,
and it is consumed by both training (`trainer.py` → `GRPORunner`) and every eval
script (`eval_gsm8k.py`, `eval_entropy_gsm8k.py`, `eval_output_dist.py`).

Nothing outside `tasks/` hard-codes a dataset any more: everything resolves the
active dataset by name via `tasks.get_task(args.dataset)`, and `--dataset` /
`--reward` CLI choices are generated from the registries.

> The package is called `tasks` (not `datasets`) on purpose — a local
> `datasets` package would shadow the HuggingFace `datasets` library.

## Two prompt "views"

Some datasets train and evaluate with *different* prompts (GSM8K trains with a
`#### <n>` instruction but is benchmarked with a `\boxed{}` system prompt, per
the Spurious-Rewards setup). The spec keeps both so they can't drift apart:

- `format_train_example(x, tokenizer)` — the **training** view.
- `build_eval_prompts(questions, tokenizer)` — the **eval** view.

## Adding a dataset

1. Create `tasks/mydata.py`:

   ```python
   from .base import DatasetSpec
   from .registry import register_task

   @register_task
   class MyDataTask(DatasetSpec):
       name = "mydata"
       hf_path = "org/mydata"
       hf_config = None
       train_split = "train"
       eval_split = "test"

       default_reward = "ground_truth"   # used when --reward is omitted
       allowed_rewards = None            # None => any reward allowed

       system_prompt = "..."             # eval (and system-prompted training)
       train_instruction = None          # appended to the question at train time

       question_column = "question"
       answer_column = "answer"

       def format_train_example(self, x, tokenizer):
           msgs = [{"role": "user", "content": x[self.question_column]}]
           prompt = tokenizer.apply_chat_template(
               msgs, tokenize=False, add_generation_prompt=True)
           return {"prompt": prompt,
                   "answer": self.extract_gold(x[self.answer_column])}

       def extract_gold(self, answer_field):
           return str(answer_field).strip()

       def extract_answer(self, completion):        # for eval display/grading
           ...

       def make_grader(self):                       # is_correct(completion, gold)
           ...
   ```

   The base class already provides `build_train`, `load_eval`,
   `sample_questions`, `build_eval_prompts` (chat-template + ChatML fallback),
   and `eval_dataset_id`; override only what differs.

2. Register it by importing the module in `tasks/__init__.py`:

   ```python
   from . import gsm8k, deepscaler, mydata
   ```

That's it — `--dataset mydata` now works for training and all three eval
scripts, with no edits to `GRPORunner`, `trainer.py`, or the eval scripts.

## Rewards

Reward *functions* stay shared in `rewards.py` (e.g. the random/Bernoulli
rewards are dataset-agnostic). A task references them by name via
`default_reward` / `allowed_rewards`. If a dataset needs a bespoke reward,
add the function to `rewards.REWARD_REGISTRY` and point the task's
`default_reward` at it.
