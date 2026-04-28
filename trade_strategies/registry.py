from __future__ import annotations

import importlib
import pkgutil
from typing import Callable, Optional

import pandas as pd

from .base import TradeRecord

TradeStrategyFunc = Callable[..., Optional[TradeRecord]]
CandidateSelectorFunc = Callable[..., pd.DataFrame]

_CANDIDATE_SELECTORS: dict[str, CandidateSelectorFunc] = {}


def _discover_trade_strategies() -> dict[str, TradeStrategyFunc]:
    """
    Automatically discover trade strategy modules in the trade_strategies package.

    A module is registered only when it defines both:
        STRATEGY_NAME = "your_strategy_name"
        EXECUTE_FUNC = your_execute_function

    Optional:
        select_candidates = your_candidate_selector

    Notes:
    - registry.py, base.py, __init__.py, and private modules are skipped.
    - If old archived strategy files remain in this folder and import broken legacy APIs,
      they may still break discovery. The cleanest structure is to keep only:
          __init__.py
          base.py
          registry.py
          renko_trade_strategy_v0.py
    """
    registry: dict[str, TradeStrategyFunc] = {}
    package_name = __package__ or "trade_strategies"
    package = importlib.import_module(package_name)

    skip_modules = {"registry", "base", "__init__"}

    for module_info in pkgutil.iter_modules(package.__path__):
        module_name = module_info.name

        if module_name.startswith("_") or module_name in skip_modules:
            continue

        module = importlib.import_module(f"{package_name}.{module_name}")

        strategy_name = getattr(module, "STRATEGY_NAME", None)
        strategy_func = getattr(module, "EXECUTE_FUNC", None)

        if strategy_name is None and strategy_func is None:
            continue

        if not strategy_name or not callable(strategy_func):
            raise RuntimeError(
                f"Trade strategy module {module_name} must define STRATEGY_NAME "
                f"and callable EXECUTE_FUNC."
            )

        strategy_name = str(strategy_name)

        if strategy_name in registry:
            raise ValueError(f"Duplicate trade strategy name found: {strategy_name}")

        registry[strategy_name] = strategy_func

        selector = getattr(module, "select_candidates", None)
        if callable(selector):
            _CANDIDATE_SELECTORS[strategy_name] = selector

    return registry


TRADE_STRATEGY_REGISTRY: dict[str, TradeStrategyFunc] = _discover_trade_strategies()


def get_trade_strategy(name: str) -> TradeStrategyFunc:
    """Return a registered trade strategy function by strategy name."""
    if name not in TRADE_STRATEGY_REGISTRY:
        available = ", ".join(sorted(TRADE_STRATEGY_REGISTRY))
        raise ValueError(f"Unsupported trade strategy: {name}. Available strategies: {available}")
    return TRADE_STRATEGY_REGISTRY[name]


def get_candidate_selector(name: str) -> CandidateSelectorFunc | None:
    """Return an optional daily candidate selector for a trade strategy."""
    return _CANDIDATE_SELECTORS.get(name)


def list_trade_strategies() -> list[str]:
    """Return all registered trade strategy names."""
    return sorted(TRADE_STRATEGY_REGISTRY)
