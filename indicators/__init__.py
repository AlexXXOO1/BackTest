from __future__ import annotations

import pandas as pd

from .brick import add_brick_indicators
from .momentum import add_kdj_indicators, add_macd_indicators
from .quality import add_renko_quality_indicators
from .candle_patterns import add_candle_pattern_indicators
from .trend import add_trend_indicators
from .volume import add_volume_indicators

from .hard_brick_turn_strong import add_hard_brick_turn_strong
from .brick_reversal_strength import add_brick_reversal_strength_flags
from .price_rise_in_range import add_price_rise_in_range
from .close_to_short_trend import add_close_to_short_trend
from .strong_market_relaxed_flags import add_strong_market_relaxed_flags
from .tdx_renko_xg import add_tdx_renko_xg
from .two_day_above_trend_line import add_two_day_above_trend_line
from .short_trend_above_trend_line import add_short_trend_above_trend_line
from .close_below_short_trend_cap import add_close_below_short_trend_cap
from .price_below_50 import add_price_below_50
from .close_above_yellow_ma import add_close_above_yellow_ma
from .j_momentum_or_low import add_j_momentum_or_low
from .surge_then_shrink_pullback import add_surge_then_shrink_pullback
from .small_rise_long_red_brick import add_small_rise_long_red_brick
from .renko_v1_risk_flags import add_renko_v1_risk_flags
from .b2_confirm_v0 import add_b2_confirm_v0_indicators


def add_all_indicators(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    """
    Add reusable stock-level indicators.

    Rule:
    - Indicator layer calculates reusable facts only.
    - Strategy layer only combines these facts into selected / score / rank_key.
    - Market-regime fields are temporarily NOT included here. v5 still reads
      SH#999999.txt directly and merges market_regime separately.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Legacy base indicators.
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

    # Strategy-derived reusable indicators moved out of strategy layer.
    df = add_hard_brick_turn_strong(
        df,
        brick_reversal_ratio=float(kwargs.get("brick_reversal_ratio", kwargs.get("renko_brick_reversal_ratio", 0.70))),
    )
    df = add_tdx_renko_xg(
        df,
        reversal_ratio=float(kwargs.get("tdx_renko_reversal_ratio", kwargs.get("brick_reversal_ratio", 0.70))),
    )
    df = add_brick_reversal_strength_flags(df)
    df = add_price_rise_in_range(
        df,
        min_daily_return_pct=float(kwargs.get("min_daily_return_pct", 3.0)),
        max_daily_return_pct=float(kwargs.get("max_daily_return_pct", 7.0)),
    )
    df = add_close_to_short_trend(
        df,
        max_close_to_short_trend=float(kwargs.get("max_close_to_short_trend", 0.95)),
    )
    df = add_strong_market_relaxed_flags(
        df,
        strong_max_daily_return_pct=float(kwargs.get("strong_max_daily_return_pct", 9.0)),
        strong_max_close_to_short_trend=float(kwargs.get("strong_max_close_to_short_trend", 1.00)),
    )

    # Renko v1 reusable condition indicators.
    df = add_two_day_above_trend_line(df)
    df = add_short_trend_above_trend_line(df)
    df = add_close_below_short_trend_cap(df)
    df = add_price_below_50(df, max_price=float(kwargs.get("renko_price_below", 50.0)))
    df = add_close_above_yellow_ma(df)
    df = add_j_momentum_or_low(df, low_j_threshold=float(kwargs.get("low_j_threshold", 14.0)))
    df = add_surge_then_shrink_pullback(
        df,
        surge_return_pct=float(kwargs.get("surge_return_pct", 3.0)),
        surge_volume_ratio=float(kwargs.get("surge_volume_ratio", 1.5)),
        pullback_max_return_pct=float(kwargs.get("pullback_max_return_pct", 0.0)),
        shrink_volume_ratio=float(kwargs.get("shrink_volume_ratio", 1.0)),
    )
    df = add_small_rise_long_red_brick(
        df,
        small_rise_min_pct=float(kwargs.get("small_rise_min_pct", 0.0)),
        small_rise_max_pct=float(kwargs.get("small_rise_max_pct", kwargs.get("renko_small_rise_max_pct", 3.0))),
        long_red_ratio=float(kwargs.get("long_red_ratio", kwargs.get("renko_long_red_ratio", 1.3))),
    )
    df = add_renko_v1_risk_flags(
        df,
        high_pos_ratio=float(kwargs.get("v8_high_pos_ratio", 0.85)),
        accel_return_pct=float(kwargs.get("v8_accel_ret_pct", 12.0)),
        huge_volume_ratio=float(kwargs.get("v8_huge_vol_ratio", 2.0)),
        big_bear_body_pct=float(kwargs.get("v8_big_bear_body_pct", 4.0)),
        limit_up_pct=float(kwargs.get("v8_limit_up_pct", 9.7)),
        shrink_limit_vol_ratio=float(kwargs.get("v8_shrink_limit_vol_ratio", 0.8)),
        hammer_lower_shadow_body_ratio=float(kwargs.get("v8_hammer_lower_shadow_body_ratio", 2.0)),
        hammer_lower_shadow_range_ratio=float(kwargs.get("v8_hammer_lower_shadow_range_ratio", 0.5)),
        hammer_upper_shadow_body_ratio=float(kwargs.get("v8_hammer_upper_shadow_body_ratio", 1.2)),
        hammer_max_body_range_ratio=float(kwargs.get("v8_hammer_max_body_range_ratio", 0.4)),
        risk_lookback=int(kwargs.get("v8_hard_lookback", 20)),
    )

    # B2 confirmation v0 reusable condition indicators.
    df = add_b2_confirm_v0_indicators(
        df,
        b1_j_max=float(kwargs.get("b1_j_max", 14.0)),
        b2_lookback_days=int(kwargs.get("b2_lookback_days", 5)),
        b2_min_return_pct=float(kwargs.get("b2_min_return_pct", 4.0)),
        b2_j_max=float(kwargs.get("b2_j_max", 55.0)),
        b2_max_upper_shadow_ratio=float(kwargs.get("b2_max_upper_shadow_ratio", 0.25)),
        low_range_lookback=int(kwargs.get("low_range_lookback", 20)),
        position_in_range_max=float(kwargs.get("position_in_range_max", 0.25)),
        support_distance_max=float(kwargs.get("support_distance_max", 0.02)),
        previous_low_lookback=int(kwargs.get("previous_low_lookback", 20)),
        false_break_pct=float(kwargs.get("false_break_pct", 0.02)),
        low_volume_lookback=int(kwargs.get("low_volume_lookback", 20)),
        low_volume_quantile=float(kwargs.get("low_volume_quantile", 0.20)),
        volume_ma_lookback=int(kwargs.get("volume_ma_lookback", 5)),
        volume_ma_ratio_max=float(kwargs.get("volume_ma_ratio_max", 0.70)),
        range_width_max=float(kwargs.get("range_width_max", 0.25)),
        range_bottom_position_max=float(kwargs.get("range_bottom_position_max", 0.25)),
        range_bottom_false_break_pct=float(kwargs.get("range_bottom_false_break_pct", 0.02)),
        ma_support_distance_max=float(kwargs.get("ma_support_distance_max", 0.02)),
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
    "add_hard_brick_turn_strong",
    "add_brick_reversal_strength_flags",
    "add_price_rise_in_range",
    "add_close_to_short_trend",
    "add_strong_market_relaxed_flags",
    "add_tdx_renko_xg",
    "add_two_day_above_trend_line",
    "add_short_trend_above_trend_line",
    "add_close_below_short_trend_cap",
    "add_price_below_50",
    "add_close_above_yellow_ma",
    "add_j_momentum_or_low",
    "add_surge_then_shrink_pullback",
    "add_small_rise_long_red_brick",
    "add_renko_v1_risk_flags",
    "add_b2_confirm_v0_indicators",
]
