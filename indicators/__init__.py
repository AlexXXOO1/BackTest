from __future__ import annotations

import pandas as pd

from .brick import add_brick_indicators
from .momentum import add_kdj_indicators, add_macd_indicators
from .quality import add_renko_quality_indicators
from .candle_patterns import add_candle_pattern_indicators
from .trend import add_trend_indicators
from .volume import add_volume_indicators


def add_all_indicators(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    """
    Add reusable indicators inherited from the v1-v8 iteration chain.

    This function calculates facts only. Strategy hard filters, scoring weights,
    score thresholds, and selected flags should be applied in the strategy layer.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df = add_brick_indicators(df, n1=n1, n2=n2)
    df = add_trend_indicators(df)
    df = add_kdj_indicators(df)
    df = add_macd_indicators(df)
    df = add_volume_indicators(df)
    df = add_candle_pattern_indicators(
        df,
        hard_lookback=int(kwargs.get("v8_hard_lookback", 20)),
        high_pos_lookback=int(kwargs.get("v8_high_pos_lookback", 60)),
        high_pos_ratio=float(kwargs.get("v8_high_pos_ratio", 0.85)),
        accel_lookback=int(kwargs.get("v8_accel_lookback", 5)),
        accel_ret_pct=float(kwargs.get("v8_accel_ret_pct", 12.0)),
        huge_vol_ma_n=int(kwargs.get("v8_huge_vol_ma_n", 20)),
        huge_vol_ratio=float(kwargs.get("v8_huge_vol_ratio", 2.0)),
        big_bear_body_pct=float(kwargs.get("v8_big_bear_body_pct", 4.0)),
        limit_up_pct=float(kwargs.get("v8_limit_up_pct", 9.7)),
        shrink_limit_vol_ratio=float(kwargs.get("v8_shrink_limit_vol_ratio", 0.8)),
        hammer_lower_shadow_body_ratio=float(kwargs.get("v8_hammer_lower_shadow_body_ratio", 2.0)),
        hammer_lower_shadow_range_ratio=float(kwargs.get("v8_hammer_lower_shadow_range_ratio", 0.5)),
        hammer_upper_shadow_body_ratio=float(kwargs.get("v8_hammer_upper_shadow_body_ratio", 1.2)),
        hammer_max_body_range_ratio=float(kwargs.get("v8_hammer_max_body_range_ratio", 0.4)),
    )
    df = add_renko_quality_indicators(
        df,
        small_rise_max_pct=float(kwargs.get("renko_small_rise_max_pct", kwargs.get("v8_small_rise_max_pct", 3.0))),
        long_red_lookback=int(kwargs.get("renko_long_red_lookback", kwargs.get("v8_long_red_lookback", 20))),
        long_red_ratio=float(kwargs.get("renko_long_red_ratio", kwargs.get("v8_long_red_ratio", 1.3))),
    )
    return df


__all__ = [
    "add_all_indicators",
    "add_brick_indicators",
    "add_trend_indicators",
    "add_kdj_indicators",
    "add_macd_indicators",
    "add_volume_indicators",
    "add_candle_pattern_indicators",
    "add_renko_quality_indicators",
]
