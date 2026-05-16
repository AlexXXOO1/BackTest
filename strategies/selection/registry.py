"""
Selection strategy registry.

Auto-discover all Python files under strategies/selection/.

A strategy module is registered automatically when it defines:
    STRATEGY_NAME = "your_strategy_name"

Required entry function:
    def select(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ...

Optional alias:
    SELECT_FUNC = select
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Callable, Dict


ENTRY_FUNC_CANDIDATES: tuple[str, ...] = (
    "select",
    "SELECT_FUNC",
)


def _resolve_select_func(module) -> Callable | None:
    """Resolve a strategy entry function without forcing a specific function name."""
    for attr_name in ENTRY_FUNC_CANDIDATES:
        func = getattr(module, attr_name, None)
        if callable(func):
            return func
    return None


def discover_selection_strategies(verbose: bool = False) -> Dict[str, Callable]:
    """Auto-discover valid selection strategy modules in this package."""
    registry: Dict[str, Callable] = {}

    package_name = __package__
    if not package_name:
        raise RuntimeError("strategies.selection.registry must be imported as a package module.")

    package = importlib.import_module(package_name)

    for module_info in pkgutil.iter_modules(package.__path__):
        module_name = module_info.name

        if module_name.startswith("_") or module_name == "registry":
            continue

        full_module_name = f"{package_name}.{module_name}"

        try:
            module = importlib.import_module(full_module_name)
        except Exception as e:
            if verbose:
                print(f"[WARN] Skip {full_module_name}: import failed: {type(e).__name__}: {e}")
            continue

        strategy_name = getattr(module, "STRATEGY_NAME", None)
        if not strategy_name:
            if verbose:
                print(f"[WARN] Skip {full_module_name}: missing STRATEGY_NAME")
            continue

        select_func = _resolve_select_func(module)
        if select_func is None:
            if verbose:
                print(
                    f"[WARN] Skip {full_module_name}: missing entry function. "
                    f"Supported names: {', '.join(ENTRY_FUNC_CANDIDATES)}"
                )
            continue

        strategy_name = str(strategy_name)
        if strategy_name in registry:
            raise ValueError(
                f"Duplicate selection strategy name found: {strategy_name}. "
                "Please make every STRATEGY_NAME unique."
            )

        registry[strategy_name] = select_func

    return registry


SELECTION_STRATEGY_REGISTRY: Dict[str, Callable] = discover_selection_strategies(verbose=False)


def get_selection_strategy(name: str) -> Callable:
    """Return one registered strategy function by strategy name."""
    if name not in SELECTION_STRATEGY_REGISTRY:
        available = ", ".join(sorted(SELECTION_STRATEGY_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown selection strategy: {name}. Available: {available}")
    return SELECTION_STRATEGY_REGISTRY[name]


def list_selection_strategies() -> list[str]:
    """List all registered selection strategy names."""
    return sorted(SELECTION_STRATEGY_REGISTRY)
