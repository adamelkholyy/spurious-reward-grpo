"""Registry mapping ``--dataset`` names to :class:`DatasetSpec` instances.

Task modules register themselves at import time via ``@register_task``. The
package ``__init__`` imports every task module so the registry is fully
populated on ``import tasks``.
"""

from __future__ import annotations

from typing import Dict, List, Type

from .base import DatasetSpec

_REGISTRY: Dict[str, DatasetSpec] = {}


def register_task(cls: Type[DatasetSpec]) -> Type[DatasetSpec]:
    """Class decorator: instantiate ``cls`` and register it under its ``name``."""
    inst = cls()
    if not inst.name:
        raise ValueError(f"{cls.__name__} must set a non-empty `name`.")
    if inst.name in _REGISTRY:
        raise ValueError(f"Duplicate task name '{inst.name}' ({cls.__name__}).")
    _REGISTRY[inst.name] = inst
    return cls


def get_task(name: str) -> DatasetSpec:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Registered: {available_tasks()}"
        )
    return _REGISTRY[name]


def available_tasks() -> List[str]:
    return sorted(_REGISTRY)
