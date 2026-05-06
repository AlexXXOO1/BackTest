from __future__ import annotations

"""
B2 confirmation selection strategy v0.

Purpose
-------
Find T0 B2 confirmation signals for the workflow:
    T0 detects B2 confirmation signal
    T+1 open buy
    T+2 open or T+2 close sell

This strategy is rebuilt from the B1/B2 confirmation document only.
It does NOT use the previous Renko / brick indicators.

Core document logic
-------------------
B2 confirmation = B1 appears first, then B2 confirmation candle appears.
B2 day is T0. Buy at T+1 open.

Current configurable defaults
-----------------------------
B1_J_MAX = 14
    B1 J condition. Currently J <= 14.
    Keep this as an interface for later tests, for example 0, -5, -10, 14.

B2_LOOKBACK_DAYS = 5
    B2 must appear within N trading days after B1.
    Currently 5 days. Keep this as an interface for later tests, for example 1, 3, 5.

Hard selected rule
------------------
selected =
    B1 appeared within previous B2_LOOKBACK_DAYS trading days
    AND T0 daily_return_pct > 4
    AND T0 close > open
    AND T0 volume > previous day volume
    AND T0 J < 55
    AND T0 upper_shadow_ratio <= 0.25

B1 rule
-------
B1 =
    low-position condition
    AND J <= B1_J_MAX
    AND low-volume condition
    AND not-effectively-break-previous-low condition

Only document-related indicators are used:
    KDJ J, daily return, volume, 5-day volume average,
    20-day high/low/range position, previous local low,
    upper shadow ratio, MA20/BBI/yellow-line-like support proximity.
"""

from typing import Iterable

import numpy as np
import pandas as pd


STRATEGY_NAME = "b2_confirm_select_strategy_v1"

# =============================================================================
# Adjustable parameters
# =============================================================================

# B1 J threshold. Current user requirement: J <= 14.
B1_J_MAX = 14.0

# B2 must appear within N trading days after B1.
# Current user requirement: within 5 days.
B2_LOOKBACK_DAYS = 5

# B2 confirmation candle parameters.
B2_MIN_RETURN_PCT = 3.0
B2_J_MAX = 55.0
B2_MAX_UPPER_SHADOW_RATIO = 0.25

# B1 low-position parameters.
LOW_RANGE_LOOKBACK = 20
POSITION_IN_RANGE_MAX = 0.25
SUPPORT_DISTANCE_MAX = 0.02

# Previous N low simplified definition:
# previous_n_low = prior 20-day rolling low.
PREVIOUS_LOW_LOOKBACK = 20
FALSE_BREAK_PCT = 0.02

# B1 low-volume parameters.
LOW_VOLUME_LOOKBACK = 20
LOW_VOLUME_QUANTILE = 0.20
VOLUME_MA_LOOKBACK = 5
VOLUME_MA_RATIO_MAX = 0.70

# Sideways range bottom parameters.
RANGE_WIDTH_MAX = 0.25
RANGE_BOTTOM_POSITION_MAX = 0.25
RANGE_BOTTOM_FALSE_BREAK_PCT = 0.02

# Support proximity for MA20 / BBI / yellow-line-like support.
MA_SUPPORT_DISTANCE_MAX = 0.02


# The strategy can compute most fields from OHLCV.
# J is required, but the column name may differ between projects.
REQUIRED_INDICATOR_COLUMNS: set[str] = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


J_COLUMN_CANDIDATES: tuple[str, ...] = (
    "j",
    "J",
    "kdj_j",
    "KDJ_J",
    "j_value",
    "J_VALUE",
)

YELLOW_LINE_CANDIDATES: tuple[str, ...] = (
    "yellow_line",
    "yellow_ma",
    "ma_yellow",
    "黄线",
)

BBI_COLUMN_CANDIDATES: tuple[str, ...] = (
    "bbi",
    "BBI",
)


def _find_first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first existing column from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safe division. Zero denominator returns NaN instead of inf."""
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _to_bool_series(s: pd.Series) -> pd.Series:
    """Convert a condition series to bool and fill missing values with False."""
    return s.fillna(False).astype(bool)


def add_b2_confirm_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add B1/B2 document-based indicators.

    Input must contain at least:
        date, open, high, low, close, volume, and one J column.

    Output adds:
        daily_return_pct
        prev_volume
        volume_ratio_prev
        volume_ma5
        volume_ratio_ma5
        range_high_20
        range_low_20
        range_width_20
        position_in_range_20
        previous_n_low
        upper_shadow_ratio
        b1_* flags
        b2_* flags
        b2_quality_score
        selected
    """
    out = df.copy()

    # Keep chronological order inside one symbol dataframe.
    if "date" in out.columns:
        out = out.sort_values("date").reset_index(drop=True)

    j_col = _find_first_existing_column(out, J_COLUMN_CANDIDATES)
    if j_col is None:
        raise KeyError(
            "Missing KDJ J column. Expected one of: "
            + ", ".join(J_COLUMN_CANDIDATES)
        )

    yellow_col = _find_first_existing_column(out, YELLOW_LINE_CANDIDATES)
    bbi_col = _find_first_existing_column(out, BBI_COLUMN_CANDIDATES)

    prev_close = out["close"].shift(1)
    out["daily_return_pct"] = (_safe_div(out["close"], prev_close) - 1.0) * 100.0

    out["prev_volume"] = out["volume"].shift(1)
    out["volume_ratio_prev"] = _safe_div(out["volume"], out["prev_volume"])

    out["volume_ma5"] = out["volume"].rolling(VOLUME_MA_LOOKBACK, min_periods=1).mean()
    out["volume_ratio_ma5"] = _safe_div(out["volume"], out["volume_ma5"])

    out["volume_q20_20"] = out["volume"].rolling(
        LOW_VOLUME_LOOKBACK, min_periods=LOW_VOLUME_LOOKBACK
    ).quantile(LOW_VOLUME_QUANTILE)
    out["volume_min_20"] = out["volume"].rolling(
        LOW_VOLUME_LOOKBACK, min_periods=LOW_VOLUME_LOOKBACK
    ).min()

    out["range_high_20"] = out["high"].rolling(
        LOW_RANGE_LOOKBACK, min_periods=LOW_RANGE_LOOKBACK
    ).max()
    out["range_low_20"] = out["low"].rolling(
        LOW_RANGE_LOOKBACK, min_periods=LOW_RANGE_LOOKBACK
    ).min()

    range_height = out["range_high_20"] - out["range_low_20"]
    out["range_width_20"] = _safe_div(range_height, out["range_low_20"])
    out["position_in_range_20"] = _safe_div(out["close"] - out["range_low_20"], range_height)

    # Simplified previous N low: prior 20-day rolling low, shifted to avoid using current day.
    out["previous_n_low"] = out["low"].shift(1).rolling(
        PREVIOUS_LOW_LOOKBACK, min_periods=5
    ).min()
    out["distance_to_previous_n_low"] = _safe_div(
        out["close"] - out["previous_n_low"], out["previous_n_low"]
    )

    # MA20 as a support proxy from the document.
    out["ma20"] = out["close"].rolling(20, min_periods=1).mean()
    out["distance_to_ma20"] = (out["close"] - out["ma20"]).abs() / out["ma20"].replace(0, np.nan)

    # BBI fallback: if project already has BBI, use it; otherwise compute common BBI proxy.
    if bbi_col is not None:
        out["bbi_for_b1"] = out[bbi_col]
    else:
        ma3 = out["close"].rolling(3, min_periods=1).mean()
        ma6 = out["close"].rolling(6, min_periods=1).mean()
        ma12 = out["close"].rolling(12, min_periods=1).mean()
        ma24 = out["close"].rolling(24, min_periods=1).mean()
        out["bbi_for_b1"] = (ma3 + ma6 + ma12 + ma24) / 4.0
    out["distance_to_bbi"] = (out["close"] - out["bbi_for_b1"]).abs() / out[
        "bbi_for_b1"
    ].replace(0, np.nan)

    # Yellow line fallback: use existing project column if available; otherwise fall back to MA20.
    if yellow_col is not None:
        out["yellow_for_b1"] = out[yellow_col]
    else:
        out["yellow_for_b1"] = out["ma20"]
    out["distance_to_yellow"] = (out["close"] - out["yellow_for_b1"]).abs() / out[
        "yellow_for_b1"
    ].replace(0, np.nan)

    daily_range = out["high"] - out["low"]
    out["upper_shadow_ratio"] = _safe_div(
        out["high"] - out[["open", "close"]].max(axis=1), daily_range
    )

    # =============================================================================
    # B1 conditions
    # =============================================================================
    out["b1_low_range_position"] = out["position_in_range_20"] <= POSITION_IN_RANGE_MAX

    out["b1_near_previous_n_low"] = (
        out["distance_to_previous_n_low"].abs() <= SUPPORT_DISTANCE_MAX
    )

    out["b1_in_range_bottom"] = (
        (out["range_width_20"] <= RANGE_WIDTH_MAX)
        & (out["position_in_range_20"] <= RANGE_BOTTOM_POSITION_MAX)
        & (out["close"] >= out["range_low_20"] * (1.0 - RANGE_BOTTOM_FALSE_BREAK_PCT))
    )

    out["b1_near_ma_support"] = (
        (out["distance_to_ma20"] <= MA_SUPPORT_DISTANCE_MAX)
        | (out["distance_to_bbi"] <= MA_SUPPORT_DISTANCE_MAX)
        | (out["distance_to_yellow"] <= MA_SUPPORT_DISTANCE_MAX)
    )

    out["b1_position_ok"] = (
        out["b1_low_range_position"]
        | out["b1_near_previous_n_low"]
        | out["b1_in_range_bottom"]
        | out["b1_near_ma_support"]
    )

    out["b1_j_ok"] = out[j_col] <= B1_J_MAX

    out["b1_low_volume"] = (
        (out["volume"] <= out["volume_q20_20"])
        | (out["volume_ratio_ma5"] <= VOLUME_MA_RATIO_MAX)
    )

    out["b1_extreme_low_volume"] = out["volume"] <= out["volume_min_20"]

    out["b1_not_break_prev_low"] = (
        (out["low"] >= out["previous_n_low"] * (1.0 - FALSE_BREAK_PCT))
        & (out["close"] >= out["previous_n_low"])
    )

    out["b1_valid"] = (
        out["b1_position_ok"]
        & out["b1_j_ok"]
        & out["b1_low_volume"]
        & out["b1_not_break_prev_low"]
    )

    # =============================================================================
    # B2 conditions on current day T0
    # =============================================================================
    b1_recent = pd.Series(False, index=out.index)
    b1_days_ago = pd.Series(np.nan, index=out.index, dtype="float64")
    for days_ago in range(1, B2_LOOKBACK_DAYS + 1):
        shifted = out["b1_valid"].shift(days_ago).fillna(False).astype(bool)
        b1_recent = b1_recent | shifted
        b1_days_ago = b1_days_ago.mask(shifted & b1_days_ago.isna(), float(days_ago))

    out["b1_within_b2_lookback"] = b1_recent
    out["b1_days_ago_for_b2"] = b1_days_ago

    out["b2_return_ok"] = out["daily_return_pct"] > B2_MIN_RETURN_PCT
    out["b2_bullish_candle"] = out["close"] > out["open"]
    out["b2_volume_up"] = out["volume"] > out["prev_volume"]
    out["b2_double_volume"] = out["volume"] > 1.90 * out["prev_volume"]
    out["b2_sky_volume"] = out["volume"] > 3.50 * out["prev_volume"]
    out["b2_j_ok"] = out[j_col] < B2_J_MAX
    out["b2_j_high_zone"] = (out[j_col] >= 45.0) & (out[j_col] < B2_J_MAX)
    out["b2_upper_shadow_ok"] = out["upper_shadow_ratio"] <= B2_MAX_UPPER_SHADOW_RATIO
    out["b2_tiny_upper_shadow"] = out["upper_shadow_ratio"] <= 0.10
    out["b2_upper_shadow_warning"] = (
        (out["upper_shadow_ratio"] > 0.20)
        & (out["upper_shadow_ratio"] <= B2_MAX_UPPER_SHADOW_RATIO)
    )

    out["selected"] = (
        out["b1_within_b2_lookback"]
        & out["b2_return_ok"]
        & out["b2_bullish_candle"]
        & out["b2_volume_up"]
        & out["b2_j_ok"]
        & out["b2_upper_shadow_ok"]
    )

    # Simple quality score for sorting / later attribution. Not used as a hard filter.
    out["b2_quality_score"] = (
        (out["b2_double_volume"].astype(int))
        + (out["b2_tiny_upper_shadow"].astype(int))
        + (out["b1_in_range_bottom"].shift(1).fillna(False).astype(int))
    )

    # Convert condition columns to stable bool dtype.
    bool_cols = [
        c
        for c in out.columns
        if c.startswith("b1_") or c.startswith("b2_") or c == "selected"
    ]
    for c in bool_cols:
        if c != "b1_days_ago_for_b2":
            out[c] = _to_bool_series(out[c])

    return out


def select(
    df: pd.DataFrame,
    n1: int | None = None,
    n2: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Project-compatible entry point.

    The selector framework usually passes one symbol dataframe into this function.
    Return the same dataframe with a boolean `selected` column and diagnostic columns.
    """
    missing = REQUIRED_INDICATOR_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for {STRATEGY_NAME}: {sorted(missing)}")

    return add_b2_confirm_indicators(df)


# Optional aliases for different selector loaders.
def apply_strategy(
    df: pd.DataFrame,
    n1: int | None = None,
    n2: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    return select(df, n1=n1, n2=n2, **kwargs)


def run(
    df: pd.DataFrame,
    n1: int | None = None,
    n2: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    return select(df, n1=n1, n2=n2, **kwargs)
