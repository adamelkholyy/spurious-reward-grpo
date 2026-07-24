"""Dataset *tasks*: one file per dataset, each the single source of truth for
that dataset's prompts, gold/grading, and reward configuration.

Public API::

    from tasks import get_task, available_tasks
    task = get_task("gsm8k")
    ds   = task.build_train(tokenizer)          # training
    prompts, tmpl = task.build_eval_prompts(qs, tokenizer)  # eval

Add a dataset by creating ``tasks/<name>.py`` with a ``DatasetSpec`` subclass
decorated with ``@register_task``, then importing it below.
"""

from .base import DatasetSpec
from .registry import available_tasks, get_task, register_task

# Import concrete tasks so they register themselves on `import tasks`.
from . import gsm8k, deepscaler, wordle, countdown4, dapo, aime2024, mbpp  # noqa: E402,F401

__all__ = [
    "DatasetSpec",
    "register_task",
    "get_task",
    "available_tasks",
]
