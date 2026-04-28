from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_V8_WEIGHTS: dict[str, float] = {
    "two_day_above_trend_line": 1.2,
    "short_trend_above_trend_line": 1.3,
    "close_below_short_trend_cap": 0.8,
    "price_below_50": 0.6,
    "j_momentum_or_low": 1.0,
    "close_above_yellow_ma": 1.1,
    "surge_then_shrink_pullback": 1.5,
    "small_rise_long_red_brick": 2.5,
}


def add_v8_quality_indicators(
    df: pd.DataFrame,
    small_rise_max_pct: float = 3.0,
    long_red_lookback: int = 20,
    long_red_ratio: float = 1.3,
) -> pd.DataFrame:
    """
    Add the high-weight v8 quality indicator.

    Quant details:
    - red_height_reference is the rolling 20-bar average of positive current_red_height values, shifted by one bar.
    - small_rise_long_red_brick is true when pct_change_close <= 3%, current_red_height > 0, and current_red_height >= red_height_reference * 1.3.
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


def add_weighted_score(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """
    Add raw_score and score_pct for the v8 scoring layer.

    Quant details:
    - Each boolean indicator contributes its configured weight when true, otherwise 0.
    - Default total weight is 10.0.
    - score_pct = raw_score / total_weight * 100.
    """
    df = df.copy()
    weights = DEFAULT_V8_WEIGHTS if weights is None else weights
    df["raw_score"] = 0.0
    for col, weight in weights.items():
        weight = float(weight)
        df[f"{col}_weight"] = weight
        df[f"{col}_score"] = np.where(df[col].fillna(False), weight, 0.0)
        df["raw_score"] += df[f"{col}_score"]
    total_weight = float(sum(weights.values())) or 1.0
    df["score_pct"] = df["raw_score"] / total_weight * 100
    return df
