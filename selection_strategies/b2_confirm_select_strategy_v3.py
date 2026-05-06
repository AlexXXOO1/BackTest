from __future__ import annotations

"""
B2 confirmation selection strategy v0 with uptrend definition.

Base:
- This is a modified v0 version.
- It keeps the original B1/B2 confirmation logic style.
- It adds an uptrend condition based on:
    white_line = EMA(EMA(CLOSE, 10), 10)
    yellow_line = (MA(CLOSE, M1) + MA(CLOSE, M2) + MA(CLOSE, M3) + MA(CLOSE, M4)) / 4

Uptrend definition:
    close > yellow_line
    AND white_line > yellow_line

Default yellow-line params:
    M1 = 3
    M2 = 6
    M3 = 12
    M4 = 24

Strategy positioning:
- B1 = low position + low J + low volume + not effectively breaking previous low.
- B2 = bullish confirmation after B1 within configurable lookback days.
- New filter = B2 must also be in the uptrend state defined above.

Trade rule is not implemented here.
This file only builds the candidate pool.
"""

import numpy as np
import pandas as pd


STRATEGY_NAME = "b2_confirm_select_strategy_v4"


# =============================================================================
# Adjustable parameters
# =============================================================================

# B1 parameters
B1_J_MAX = 14.0
B1_LOOKBACK_N = 20
B1_POSITION_RANGE_MAX = 0.25
B1_SUPPORT_DISTANCE_MAX = 0.02
B1_VOLUME_QUANTILE = 0.20
B1_VOLUME_MA_N = 5
B1_VOLUME_MA_RATIO_MAX = 0.70
B1_PREV_LOW_LOOKBACK_MIN = 5
B1_PREV_LOW_LOOKBACK_MAX = 20
B1_FAKE_BREAK_MAX = 0.02
B1_RANGE_WIDTH_MAX = 0.25

# B2 parameters
B2_LOOKBACK_DAYS = 5

# v0 original B2 return rule:
# B2 daily return must be greater than this value.
B2_RETURN_MIN = 4.0

# v0 original B2 J rule:
# B2 J must be below this value.
# Unlike v1/v2, v0 does NOT require J > 14.
B2_J_MAX = 55.0

B2_UPPER_SHADOW_MAX = 0.25

# Uptrend parameters
# White line:
#   EMA(EMA(CLOSE, WHITE_EMA_N), WHITE_EMA_N)
WHITE_EMA_N = 10

# Yellow line:
#   (MA(CLOSE, M1) + MA(CLOSE, M2) + MA(CLOSE, M3) + MA(CLOSE, M4)) / 4
YELLOW_M1 = 14
YELLOW_M2 = 28
YELLOW_M3 = 57
YELLOW_M4 = 114

# If True, selected pool must satisfy:
#   close > yellow_line AND white_line > yellow_line
# If False, only output the tag b2_uptrend_ok, but do not filter by it.
REQUIRE_B2_UPTREND = True

# Optional tags / quality factors
B2_TINY_UPPER_SHADOW_MAX = 0.10
B2_DOUBLE_VOLUME_RATIO = 1.90
B2_SKY_VOLUME_RATIO = 3.50


REQUIRED_INDICATOR_COLUMNS: set[str] = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


# =============================================================================
# Column helpers
# =============================================================================

def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_col = {str(c).lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_col:
            return lower_to_col[cand.lower()]
    return None


def _get_j_series(df: pd.DataFrame) -> pd.Series:
    """
    Find KDJ J column from common names.

    Supported names include:
    - J
    - j
    - kdj_j
    - KDJ_J
    """
    col = _first_existing_col(df, ["kdj_j", "J", "j", "KDJ_J"])
    if col is None:
        raise ValueError(
            "Cannot find KDJ J column. Expected one of: kdj_j, J, j, KDJ_J."
        )
    return pd.to_numeric(df[col], errors="coerce")


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    b2 = b.replace(0, np.nan)
    return a / b2


def _rolling_quantile(s: pd.Series, window: int, q: float) -> pd.Series:
    return s.rolling(window=window, min_periods=window).quantile(q)


def _rolling_min_shifted(s: pd.Series, lookback: int) -> pd.Series:
    return s.shift(1).rolling(window=lookback, min_periods=lookback).min()


def _ema_tdx_like(s: pd.Series, n: int) -> pd.Series:
    """
    EMA approximation used by pandas.

    Formula:
        EMA today = alpha * close_today + (1 - alpha) * EMA_yesterday
        alpha = 2 / (n + 1)

    This is close to common EMA usage and matches the structure:
        EMA(EMA(CLOSE, 10), 10)
    """
    return pd.to_numeric(s, errors="coerce").ewm(span=n, adjust=False).mean()


# =============================================================================
# Indicator construction used by this strategy only
# =============================================================================

def _add_b2_strategy_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["prev_close"] = out["close"].shift(1)
    out["prev_volume"] = out["volume"].shift(1)
    out["daily_return_pct"] = (out["close"] / out["prev_close"] - 1.0) * 100.0

    # Keep a compatible alias with your current pool naming.
    out["pct_change_close"] = out["daily_return_pct"]

    out["j"] = _get_j_series(out)

    # 20-day range position.
    out["range_high_20"] = out["high"].rolling(B1_LOOKBACK_N, min_periods=B1_LOOKBACK_N).max()
    out["range_low_20"] = out["low"].rolling(B1_LOOKBACK_N, min_periods=B1_LOOKBACK_N).min()
    out["range_width_20"] = _safe_divide(
        out["range_high_20"] - out["range_low_20"],
        out["range_low_20"],
    )
    out["position_in_range_20"] = _safe_divide(
        out["close"] - out["range_low_20"],
        out["range_high_20"] - out["range_low_20"],
    )

    # Previous simplified N low: previous 20-day local low.
    out["previous_n_low"] = _rolling_min_shifted(out["low"], B1_PREV_LOW_LOOKBACK_MAX)
    out["distance_to_previous_n_low"] = _safe_divide(
        out["close"] - out["previous_n_low"],
        out["previous_n_low"],
    )

    # Moving averages / support alternatives.
    out["ma20"] = out["close"].rolling(20, min_periods=20).mean()
    out["ma5"] = out["close"].rolling(5, min_periods=5).mean()
    out["ma10"] = out["close"].rolling(10, min_periods=10).mean()
    out["bbi_for_b1"] = (out["ma5"] + out["ma10"] + out["ma20"]) / 3.0

    # If your indicator cache already has BBI, prefer existing columns.
    bbi_col = _first_existing_col(out, ["bbi_for_b1", "BBI", "bbi"])
    if bbi_col is not None:
        out["bbi_for_b1"] = pd.to_numeric(out[bbi_col], errors="coerce")

    # Keep old support yellow if present, but this is not the new trend yellow.
    old_yellow_col = _first_existing_col(out, ["yellow_for_b1", "yellow_ma", "yellow_line"])
    if old_yellow_col is not None:
        out["yellow_for_b1"] = pd.to_numeric(out[old_yellow_col], errors="coerce")
    else:
        out["yellow_for_b1"] = out["ma20"]

    out["distance_to_ma20"] = _safe_divide(out["close"] - out["ma20"], out["ma20"])
    out["distance_to_bbi"] = _safe_divide(out["close"] - out["bbi_for_b1"], out["bbi_for_b1"])
    out["distance_to_yellow"] = _safe_divide(out["close"] - out["yellow_for_b1"], out["yellow_for_b1"])

    # New uptrend white/yellow lines.
    out["white_line"] = _ema_tdx_like(_ema_tdx_like(out["close"], WHITE_EMA_N), WHITE_EMA_N)

    out["yellow_ma_m1"] = out["close"].rolling(YELLOW_M1, min_periods=YELLOW_M1).mean()
    out["yellow_ma_m2"] = out["close"].rolling(YELLOW_M2, min_periods=YELLOW_M2).mean()
    out["yellow_ma_m3"] = out["close"].rolling(YELLOW_M3, min_periods=YELLOW_M3).mean()
    out["yellow_ma_m4"] = out["close"].rolling(YELLOW_M4, min_periods=YELLOW_M4).mean()

    out["trend_yellow_line"] = (
        out["yellow_ma_m1"]
        + out["yellow_ma_m2"]
        + out["yellow_ma_m3"]
        + out["yellow_ma_m4"]
    ) / 4.0

    out["close_above_trend_yellow"] = out["close"] > out["trend_yellow_line"]
    out["white_above_trend_yellow"] = out["white_line"] > out["trend_yellow_line"]
    out["b2_uptrend_ok"] = (
        out["trend_yellow_line"].notna()
        & out["white_line"].notna()
        & out["close_above_trend_yellow"]
        & out["white_above_trend_yellow"]
    )

    # Volume.
    out["volume_ma5"] = out["volume"].rolling(B1_VOLUME_MA_N, min_periods=B1_VOLUME_MA_N).mean()
    out["volume_q20_20"] = _rolling_quantile(out["volume"], B1_LOOKBACK_N, B1_VOLUME_QUANTILE)
    out["volume_min_20"] = out["volume"].rolling(B1_LOOKBACK_N, min_periods=B1_LOOKBACK_N).min()
    out["volume_ratio_ma5"] = _safe_divide(out["volume"], out["volume_ma5"])

    # Upper shadow ratio.
    candle_range = out["high"] - out["low"]
    upper_shadow = out["high"] - out[["open", "close"]].max(axis=1)
    out["upper_shadow_ratio"] = _safe_divide(upper_shadow, candle_range).fillna(0.0)

    # Volume ratio for B2.
    out["volume_ratio_prev"] = _safe_divide(out["volume"], out["prev_volume"])
    out["b2_volume_ratio"] = out["volume_ratio_prev"]

    return out


# =============================================================================
# B1 / B2 logic
# =============================================================================

def _build_b1_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["b1_low_range_position"] = out["position_in_range_20"] <= B1_POSITION_RANGE_MAX

    out["b1_near_previous_n_low"] = (
        out["previous_n_low"].notna()
        & (out["close"] <= out["previous_n_low"] * (1.0 + B1_SUPPORT_DISTANCE_MAX))
        & (out["close"] >= out["previous_n_low"] * (1.0 - B1_FAKE_BREAK_MAX))
    )

    out["b1_in_range_bottom"] = (
        (out["range_width_20"] <= B1_RANGE_WIDTH_MAX)
        & (out["position_in_range_20"] <= B1_POSITION_RANGE_MAX)
        & (out["close"] >= out["range_low_20"] * (1.0 - B1_FAKE_BREAK_MAX))
    )

    out["b1_near_ma_support"] = (
        (
            out["bbi_for_b1"].notna()
            & (out["close"] >= out["bbi_for_b1"] * (1.0 - B1_SUPPORT_DISTANCE_MAX))
            & (out["close"] <= out["bbi_for_b1"] * (1.0 + B1_SUPPORT_DISTANCE_MAX))
        )
        | (
            out["yellow_for_b1"].notna()
            & (out["close"] >= out["yellow_for_b1"] * (1.0 - B1_SUPPORT_DISTANCE_MAX))
            & (out["close"] <= out["yellow_for_b1"] * (1.0 + B1_SUPPORT_DISTANCE_MAX))
        )
        | (
            out["ma20"].notna()
            & (out["close"] >= out["ma20"] * (1.0 - B1_SUPPORT_DISTANCE_MAX))
            & (out["close"] <= out["ma20"] * (1.0 + B1_SUPPORT_DISTANCE_MAX))
        )
    )

    out["b1_position_ok"] = (
        out["b1_low_range_position"]
        | out["b1_near_previous_n_low"]
        | out["b1_in_range_bottom"]
        | out["b1_near_ma_support"]
    )

    out["b1_j_ok"] = out["j"] <= B1_J_MAX

    out["b1_low_volume"] = (
        (out["volume"] <= out["volume_q20_20"])
        | (out["volume_ratio_ma5"] <= B1_VOLUME_MA_RATIO_MAX)
    )

    out["b1_not_break_prev_low"] = (
        out["previous_n_low"].notna()
        & (out["low"] >= out["previous_n_low"] * (1.0 - B1_FAKE_BREAK_MAX))
        & (out["close"] >= out["previous_n_low"])
    )

    out["b1_valid"] = (
        out["b1_position_ok"]
        & out["b1_j_ok"]
        & out["b1_low_volume"]
        & out["b1_not_break_prev_low"]
    )

    # Quality tags.
    out["b1_j_deep_negative"] = out["j"] <= -10
    out["b1_extreme_low_volume"] = out["volume"] <= out["volume_min_20"]

    return out


def _add_b1_lookback_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    b1_bool = out["b1_valid"].fillna(False).astype(bool)

    out["b1_days_ago_for_b2"] = np.nan
    out["b1_days_ago"] = np.nan
    out["b1_j_value"] = np.nan
    out["b1_volume_value"] = np.nan

    for days_ago in range(1, B2_LOOKBACK_DAYS + 1):
        shifted_b1 = b1_bool.shift(days_ago).fillna(False)
        need_fill = out["b1_days_ago_for_b2"].isna() & shifted_b1

        out.loc[need_fill, "b1_days_ago_for_b2"] = days_ago
        out.loc[need_fill, "b1_days_ago"] = days_ago
        out.loc[need_fill, "b1_j_value"] = out["j"].shift(days_ago).loc[need_fill]
        out.loc[need_fill, "b1_volume_value"] = out["volume"].shift(days_ago).loc[need_fill]

    out["b1_within_b2_lookback"] = out["b1_days_ago_for_b2"].notna()
    return out


def _build_b2_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["b2_after_b1"] = out["b1_within_b2_lookback"]

    # v0 return condition:
    # Only lower bound, no upper cap.
    out["b2_return_ok"] = out["daily_return_pct"] > B2_RETURN_MIN

    out["b2_bullish_candle"] = out["close"] > out["open"]
    out["b2_volume_up"] = out["volume"] > out["prev_volume"]

    # v0 J condition:
    # Only require J < 55, no J > 14 lower bound.
    out["b2_j_ok"] = out["j"] < B2_J_MAX

    out["b2_upper_shadow_ok"] = out["upper_shadow_ratio"] <= B2_UPPER_SHADOW_MAX

    out["b2_tiny_upper_shadow"] = out["upper_shadow_ratio"] <= B2_TINY_UPPER_SHADOW_MAX
    out["b2_double_volume"] = out["b2_volume_ratio"] >= B2_DOUBLE_VOLUME_RATIO
    out["b2_sky_volume"] = out["b2_volume_ratio"] >= B2_SKY_VOLUME_RATIO
    out["b2_j_high_zone"] = (out["j"] >= 45) & (out["j"] < B2_J_MAX)
    out["b2_upper_shadow_warning"] = (
        (out["upper_shadow_ratio"] > B2_TINY_UPPER_SHADOW_MAX)
        & (out["upper_shadow_ratio"] <= B2_UPPER_SHADOW_MAX)
    )

    out["b2_j_value"] = out["j"]

    base_selected = (
        out["b2_after_b1"]
        & out["b2_return_ok"]
        & out["b2_bullish_candle"]
        & out["b2_volume_up"]
        & out["b2_j_ok"]
        & out["b2_upper_shadow_ok"]
    )

    if REQUIRE_B2_UPTREND:
        out["selected"] = base_selected & out["b2_uptrend_ok"]
    else:
        out["selected"] = base_selected

    # Simple quality score for later sorting/debugging.
    quality_cols = [
        "b1_j_deep_negative",
        "b1_extreme_low_volume",
        "b1_in_range_bottom",
        "b2_tiny_upper_shadow",
        "b2_double_volume",
        "b2_uptrend_ok",
    ]

    out["b2_quality_score"] = 0
    for c in quality_cols:
        if c in out.columns:
            out["b2_quality_score"] += out[c].fillna(False).astype(int)

    out["score"] = out["b2_quality_score"]
    out["score_pct"] = out["b2_quality_score"] / max(len(quality_cols), 1) * 100.0

    return out


# =============================================================================
# Public strategy API
# =============================================================================

def select(
    df: pd.DataFrame,
    n1: int | None = None,
    n2: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Selection entry point compatible with the current selector framework.

    n1/n2 are accepted for compatibility only.
    This strategy does not use brick/renko parameters.
    """
    missing = [c for c in REQUIRED_INDICATOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{STRATEGY_NAME} missing required columns: {missing}")

    out = _add_b2_strategy_columns(df)
    out = _build_b1_flags(out)
    out = _add_b1_lookback_columns(out)
    out = _build_b2_flags(out)

    selected = out[out["selected"]].copy()

    keep_cols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "prev_close",
        "prev_volume",
        "daily_return_pct",
        "pct_change_close",
        "j",
        "J",

        "selected",
        "score",
        "score_pct",
        "b2_quality_score",

        # New trend definition.
        "white_line",
        "trend_yellow_line",
        "yellow_ma_m1",
        "yellow_ma_m2",
        "yellow_ma_m3",
        "yellow_ma_m4",
        "close_above_trend_yellow",
        "white_above_trend_yellow",
        "b2_uptrend_ok",

        # B1 features.
        "b1_days_ago_for_b2",
        "b1_days_ago",
        "b1_j_value",
        "b1_volume_value",
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

        # B2 features.
        "b2_after_b1",
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
        "b2_j_value",
        "b2_volume_ratio",
        "volume_ratio_prev",
        "upper_shadow_ratio",

        # Position/support features.
        "range_high_20",
        "range_low_20",
        "range_width_20",
        "position_in_range_20",
        "previous_n_low",
        "distance_to_previous_n_low",
        "volume_ratio_ma5",
        "volume_q20_20",
        "volume_min_20",
        "ma20",
        "bbi_for_b1",
        "yellow_for_b1",
        "distance_to_ma20",
        "distance_to_bbi",
        "distance_to_yellow",
    ]

    # If original J column is not literally named J, create J alias for downstream scripts.
    if "J" not in selected.columns:
        selected["J"] = selected["j"]

    keep_cols = [c for c in keep_cols if c in selected.columns]
    return selected[keep_cols].reset_index(drop=True)


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
