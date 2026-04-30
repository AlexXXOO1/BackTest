from __future__ import annotations

import pandas as pd


def add_v8_quality_indicators(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Backward-compatible wrapper for old imports.

    Strategy-quality booleans were moved out of indicators. Use
    selection_strategies.common_conditions.add_common_selection_conditions in
    strategy code to calculate small_rise_long_red_brick and score inputs.
    """
    from .quality import add_renko_quality_indicators

    return add_renko_quality_indicators(df, *args, **kwargs)


def add_weighted_score(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """
    Deprecated no-op wrapper.

    Weighted scores are strategy-layer outputs. This function is kept only to
    avoid breaking stale imports; it returns a copy of the input dataframe.
    """
    return df.copy()
