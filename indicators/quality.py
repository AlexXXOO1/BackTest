from __future__ import annotations

import pandas as pd


def add_renko_quality_indicators(
    df: pd.DataFrame,
    small_rise_max_pct: float = 3.0,
    long_red_lookback: int = 20,
    long_red_ratio: float = 1.3,
) -> pd.DataFrame:
    """
    Add reusable quality flags derived from the renko-like brick structure.

    Quant details:
    - red_height_reference is the rolling mean of positive current_red_height values.
    - The rolling reference uses the previous bars only by applying shift(1).
    - small_rise_long_red_brick is true when pct_change_close <= small_rise_max_pct,
      current_red_height > 0, and current_red_height >= red_height_reference * long_red_ratio.
    """
    df = df.copy()
    red_height_positive = df["current_red_height"].where(df["current_red_height"] > 0)
    df["red_height_reference"] = red_height_positive.shift(1).rolling(window=long_red_lookback, min_periods=3).mean()
    df["small_rise_long_red_brick"] = (
        (df["pct_change_close"] <= small_rise_max_pct)
        & (df["current_red_height"] > 0)
        & (df["current_red_height"] >= df["red_height_reference"] * long_red_ratio)
    ).fillna(False)
    return df
