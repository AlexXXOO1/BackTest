from __future__ import annotations

"""
Selection strategy auto registry.

Rules:
1. Every strategy file under selection_strategies/ can be auto-discovered.
2. Strategy file name should usually match STRATEGY_NAME.
3. Every strategy module should provide:
   - STRATEGY_NAME: str
   - select(df, *args, **kwargs) or apply_strategy(df, *args, **kwargs) or run(df, *args, **kwargs)

Example:
    selection_strategies/thunder_bottom_j_strategy_v0.py

    STRATEGY_NAME = "thunder_bottom_j_strategy_v0"

    def select(df, ...):
        ...
        return df
"""

import importlib
import pkgutil
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any


SELECTION_STRATEGY_REGISTRY: dict[str, Callable[..., Any]] = {}


def _get_strategy_func(module: ModuleType) -> Callable[..., Any] | None:
    """
    Get the callable strategy entry from one strategy module.

    Priority:
        1. select
        2. apply_strategy
        3. run
    """
    for func_name in ("select", "apply_strategy", "run"):
        func = getattr(module, func_name, None)
        if callable(func):
            return func
    return None


def _register_module(module: ModuleType) -> None:
    """
    Register one strategy module if it has STRATEGY_NAME and a callable entry.
    """
    strategy_name = getattr(module, "STRATEGY_NAME", None)
    strategy_func = _get_strategy_func(module)

    if not isinstance(strategy_name, str):
        return

    if strategy_func is None:
        return

    SELECTION_STRATEGY_REGISTRY[strategy_name] = strategy_func


def _auto_discover_strategies() -> None:
    """
    Auto-discover all strategy modules in this package.

    Files ignored:
        - __init__.py
        - files starting with "_"
    """
    package_dir = Path(__file__).resolve().parent
    package_name = __name__

    for module_info in pkgutil.iter_modules([str(package_dir)]):
        module_name = module_info.name

        if module_name.startswith("_"):
            continue

        full_module_name = f"{package_name}.{module_name}"

        try:
            module = importlib.import_module(full_module_name)
        except Exception as exc:
            print(f"[WARN] Failed to import selection strategy module: {full_module_name}")
            print(f"[WARN] {type(exc).__name__}: {exc}")
            continue

        _register_module(module)


def get_selection_strategy(name: str) -> Callable[..., Any]:
    """
    Get a selection strategy by name.
    """
    if name not in SELECTION_STRATEGY_REGISTRY:
        available = ", ".join(sorted(SELECTION_STRATEGY_REGISTRY))
        raise KeyError(
            f"Selection strategy not found: {name}. "
            f"Available strategies: {available}"
        )

    return SELECTION_STRATEGY_REGISTRY[name]


def list_selection_strategies() -> list[str]:
    """
    Return all registered selection strategy names.
    """
    return sorted(SELECTION_STRATEGY_REGISTRY)


_auto_discover_strategies()