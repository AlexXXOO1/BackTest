"""
Selection strategy registry.

This module automatically discovers selection strategy modules under the
selection_strategies package.

A strategy module will be registered automatically if it defines:

    STRATEGY_NAME = "your_strategy_name"
    SELECT_FUNC = your_select_function

This avoids manually editing the registry every time a new strategy file is added.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Callable, Dict


SELECTION_STRATEGY_REGISTRY: Dict[str, Callable] = {}


def _discover_selection_strategies() -> Dict[str, Callable]:
    registry: Dict[str, Callable] = {}

    package_name = __package__
    if not package_name:
        raise RuntimeError("selection_strategies.registry must be imported as a package module.")

    package = importlib.import_module(package_name)

    for module_info in pkgutil.iter_modules(package.__path__):
        module_name = module_info.name

        if module_name.startswith("_"):
            continue

        if module_name in {"registry"}:
            continue

        module = importlib.import_module(f"{package_name}.{module_name}")

        strategy_name = getattr(module, "STRATEGY_NAME", None)
        select_func = getattr(module, "SELECT_FUNC", None)

        if strategy_name is None or select_func is None:
            continue

        if not callable(select_func):
            raise TypeError(f"{module_name}.SELECT_FUNC must be callable.")

        if strategy_name in registry:
            raise ValueError(f"Duplicate selection strategy name found: {strategy_name}")

        registry[strategy_name] = select_func

    return registry


SELECTION_STRATEGY_REGISTRY = _discover_selection_strategies()


def get_selection_strategy(name: str) -> Callable:
    if name not in SELECTION_STRATEGY_REGISTRY:
        available = ", ".join(sorted(SELECTION_STRATEGY_REGISTRY))
        raise ValueError(f"Unknown selection strategy: {name}. Available: {available}")

    return SELECTION_STRATEGY_REGISTRY[name]


def list_selection_strategies() -> list[str]:
    return sorted(SELECTION_STRATEGY_REGISTRY)