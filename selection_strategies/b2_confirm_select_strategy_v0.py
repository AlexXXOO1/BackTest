from __future__ import annotations

"""
B2 confirmation selection strategy v0.

Refactor rule:
- indicators/b2_confirm_v0.py calculates reusable B1/B2 condition columns.
- This strategy file only checks required indicator columns, combines final
  selected logic, and adds strategy-level attribution fields.

Workflow:
    T0 detects B2 confirmation signal
    T+1 open buy
    T+2 open or T+2 close sell
"""

import pandas as pd

from indicators import add_all_indicators
from indicators.required import require_indicator_columns


STRATEGY_NAME = "b2_confirm_select_strategy_v0"

# =============================================================================
# Adjustable parameters
# =============================================================================

B1_J_MAX = 14.0
B2_LOOKBACK_DAYS = 5
B2_MIN_RETURN_PCT = 4.0
B2_J_MAX = 55.0
B2_MAX_UPPER_SHADOW_RATIO = 0.25

LOW_RANGE_LOOKBACK = 20
POSITION_IN_RANGE_MAX = 0.25
SUPPORT_DISTANCE_MAX = 0.02
PREVIOUS_LOW_LOOKBACK = 20
FALSE_BREAK_PCT = 0.02
LOW_VOLUME_LOOKBACK = 20
LOW_VOLUME_QUANTILE = 0.20
VOLUME_MA_LOOKBACK = 5
VOLUME_MA_RATIO_MAX = 0.70
RANGE_WIDTH_MAX = 0.25
RANGE_BOTTOM_POSITION_MAX = 0.25
RANGE_BOTTOM_FALSE_BREAK_PCT = 0.02
MA_SUPPORT_DISTANCE_MAX = 0.02

REQUIRED_INDICATORS: set[str] = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "J",
    "daily_return_pct",
    "prev_volume",
    "volume_ratio_prev",
    "volume_ma5",
    "volume_ratio_ma5",
    "volume_q20_20",
    "volume_min_20",
    "range_high_20",
    "range_low_20",
    "range_width_20",
    "position_in_range_20",
    "previous_n_low",
    "distance_to_previous_n_low",
    "ma20",
    "distance_to_ma20",
    "bbi_for_b1",
    "distance_to_bbi",
    "yellow_for_b1",
    "distance_to_yellow",
    "upper_shadow_ratio",
    "b1_low_range_position",
    "b1_near_previous_n_low",
    "b1_in_range_bottom",
    "b1_near_ma_support",
    "b1_position_ok",
    "b1_j_ok",
    "b1_low_volume",
    "b1_extreme_low_volume",
    "b1_not_break_prev_low",
    "b1_valid",
    "b1_within_b2_lookback",
    "b1_days_ago_for_b2",
    "b2_return_ok",
    "b2_bullish_candle",
    "b2_volume_up",
    "b2_double_volume",
    "b2_sky_volume",
    "b2_j_ok",
    "b2_j_high_zone",
    "b2_upper_shadow_ok",
    "b2_tiny_upper_shadow",
    "b2_upper_shadow_warning",
    "b2_quality_score",
}

# Backward-compatible alias used by older tools.
REQUIRED_INDICATOR_COLUMNS = REQUIRED_INDICATORS


def _prepare_indicators(df: pd.DataFrame, n1: int, n2: int, **kwargs) -> pd.DataFrame:
    """
    Use indicator cache columns when present; otherwise calculate indicators.

    Important:
    - When selector reads daily_indicators.parquet, B1/B2 lookback fields are
      already precomputed on the full symbol history.
    - Therefore strategy must not recalculate B1/B2 fields after selector has
      date-filtered the cache, otherwise prior B1 signals outside the requested
      date range could be lost.
    """
    if REQUIRED_INDICATORS.issubset(set(df.columns)):
        out = df.copy()
        if "date" in out.columns:
            out = out.sort_values("date").reset_index(drop=True)
    else:
        kwargs = dict(kwargs)
        kwargs.setdefault("b1_j_max", B1_J_MAX)
        kwargs.setdefault("b2_lookback_days", B2_LOOKBACK_DAYS)
        kwargs.setdefault("b2_min_return_pct", B2_MIN_RETURN_PCT)
        kwargs.setdefault("b2_j_max", B2_J_MAX)
        kwargs.setdefault("b2_max_upper_shadow_ratio", B2_MAX_UPPER_SHADOW_RATIO)
        kwargs.setdefault("low_range_lookback", LOW_RANGE_LOOKBACK)
        kwargs.setdefault("position_in_range_max", POSITION_IN_RANGE_MAX)
        kwargs.setdefault("support_distance_max", SUPPORT_DISTANCE_MAX)
        kwargs.setdefault("previous_low_lookback", PREVIOUS_LOW_LOOKBACK)
        kwargs.setdefault("false_break_pct", FALSE_BREAK_PCT)
        kwargs.setdefault("low_volume_lookback", LOW_VOLUME_LOOKBACK)
        kwargs.setdefault("low_volume_quantile", LOW_VOLUME_QUANTILE)
        kwargs.setdefault("volume_ma_lookback", VOLUME_MA_LOOKBACK)
        kwargs.setdefault("volume_ma_ratio_max", VOLUME_MA_RATIO_MAX)
        kwargs.setdefault("range_width_max", RANGE_WIDTH_MAX)
        kwargs.setdefault("range_bottom_position_max", RANGE_BOTTOM_POSITION_MAX)
        kwargs.setdefault("range_bottom_false_break_pct", RANGE_BOTTOM_FALSE_BREAK_PCT)
        kwargs.setdefault("ma_support_distance_max", MA_SUPPORT_DISTANCE_MAX)
        out = add_all_indicators(df, n1=n1, n2=n2, **kwargs)

    require_indicator_columns(out, REQUIRED_INDICATORS, STRATEGY_NAME)
    return out


def add_final_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Add final B2 v0 strategy-level selected / score fields."""
    out = df.copy()

    out["selected"] = (
        out["b1_within_b2_lookback"].fillna(False).astype(bool)
        & out["b2_return_ok"].fillna(False).astype(bool)
        & out["b2_bullish_candle"].fillna(False).astype(bool)
        & out["b2_volume_up"].fillna(False).astype(bool)
        & out["b2_j_ok"].fillna(False).astype(bool)
        & out["b2_upper_shadow_ok"].fillna(False).astype(bool)
    )

    # Strategy-level attribution. Hard filters remain transparent for analysis.
    hard_cols = [
        "b1_within_b2_lookback",
        "b2_return_ok",
        "b2_bullish_candle",
        "b2_volume_up",
        "b2_j_ok",
        "b2_upper_shadow_ok",
    ]
    out["b2_hard_pass_count"] = out[hard_cols].fillna(False).astype(bool).sum(axis=1)
    out["score"] = out["b2_quality_score"].fillna(0).astype(float)
    out["score_pct"] = out["b2_hard_pass_count"] / float(len(hard_cols)) * 100.0
    out["score_rank_key"] = (
        out["b2_hard_pass_count"].fillna(0).astype(float) * 1000.0
        + out["b2_quality_score"].fillna(0).astype(float) * 100.0
        + out["daily_return_pct"].fillna(-999.0).clip(lower=-20.0, upper=20.0)
    )
    out["selection_strategy"] = STRATEGY_NAME
    return out


def select(
    df: pd.DataFrame,
    n1: int = 4,
    n2: int = 6,
    **kwargs,
) -> pd.DataFrame:
    out = _prepare_indicators(df, n1=n1, n2=n2, **kwargs)
    return add_final_selection(out)


def apply_strategy(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    return select(df=df, n1=n1, n2=n2, **kwargs)


def run(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    return select(df=df, n1=n1, n2=n2, **kwargs)


SELECT_FUNC = select
