from __future__ import annotations

import pandas as pd


def add_renko_quality_indicators(
    df: pd.DataFrame,
    small_rise_max_pct: float = 3.0,
    long_red_lookback: int = 20,
    long_red_ratio: float = 1.3,
) -> pd.DataFrame:
    """
    Add raw reference metrics used by renko-quality strategies.

    Quant details:
    - red_height_reference is the rolling mean of positive current_red_height values.
    - The rolling reference uses previous bars only by applying shift(1).

    The threshold parameters are kept in the signature for backward
    compatibility. This function no longer creates small_rise_long_red_brick;
    strategies should apply small-rise and long-red-brick rules themselves.
    """
    out = df.copy()
    red_height_positive = out["current_red_height"].where(out["current_red_height"] > 0)
    out["red_height_reference"] = red_height_positive.shift(1).rolling(window=long_red_lookback, min_periods=3).mean()
    return out
