from __future__ import annotations

from pathlib import Path
import pandas as pd


def require_indicator_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    strategy_name: str = "unknown_strategy",
) -> None:
    """Fail fast when a strategy needs columns that are not available."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            "\n[Indicator Check Failed]\n"
            f"Strategy: {strategy_name}\n"
            "Missing indicator columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
            + "\n\nFix:\n"
            "1. Add the missing indicator under indicators/.\n"
            "2. Register it in indicators/__init__.py -> add_all_indicators().\n"
            "3. Rebuild indicator cache with scripts/build_indicators.py.\n"
        )


def require_indicator_cache_columns(
    indicator_cache_path: str | Path,
    required_columns: set[str],
    strategy_name: str = "unknown_strategy",
) -> None:
    """Check whether the parquet indicator cache contains all strategy columns."""
    path = Path(indicator_cache_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Indicator cache not found: {path}\n"
            "Please build indicator cache first."
        )
    df = pd.read_parquet(path)
    require_indicator_columns(df, required_columns, strategy_name)
