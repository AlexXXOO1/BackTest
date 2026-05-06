from __future__ import annotations

"""
B2 confirmation selection strategy v2.

Change from v1:
- Add B2_RETURN_MAX = 9.7
- B2 daily return condition becomes:
    B2_RETURN_MIN < daily_return_pct < B2_RETURN_MAX

Inherited from v1:
- B2 J condition:
    B2_J_MIN < J < B2_J_MAX

Purpose:
- Verify whether excluding near-limit-up B2 candles improves weak-year performance.
- Main target: reduce 2024 failure caused by B2 daily_return_pct >= 9.7 bucket.

Trade rule is not implemented here. This file only builds the candidate pool.
"""

import numpy as np
import pandas as pd

STRATEGY_NAME = "b2_confirm_select_strategy_v2"

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

# v2 core return range:
# v0/v1 only required daily_return_pct > 4.
# v2 excludes near-limit-up confirmation candles.
B2_RETURN_MIN = 4.0
B2_RETURN_MAX = 9.7

# v1 inherited J range:
# v0 only had J < 55.
# v1/v2 require 14 < J < 55.
B2_J_MIN = 14.0
B2_J_MAX = 55.0

B2_UPPER_SHADOW_MAX = 0.25

# Optional tags / quality factors
B2_TINY_UPPER_SHADOW_MAX = 0.10
B2_DOUBLE_VOLUME_RATIO = 1.90
B2_SKY_VOLUME_RATIO = 3.50

REQUIRED_INDICATOR_COLUMNS: set[str] = {"date", "open", "high", "low", "close", "volume"}


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_to_col = {str(c).lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_col:
            return lower_to_col[cand.lower()]
    return None


def _get_j_series(df: pd.DataFrame) -> pd.Series:
    col = _first_existing_col(df, ["kdj_j", "J", "j", "KDJ_J"])
    if col is None:
        raise ValueError("Cannot find KDJ J column. Expected one of: kdj_j, J, j, KDJ_J.")
    return pd.to_numeric(df[col], errors="coerce")


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _rolling_quantile(s: pd.Series, window: int, q: float) -> pd.Series:
    return s.rolling(window=window, min_periods=window).quantile(q)


def _rolling_min_shifted(s: pd.Series, lookback: int) -> pd.Series:
    return s.shift(1).rolling(window=lookback, min_periods=lookback).min()


def _add_b2_strategy_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["prev_close"] = out["close"].shift(1)
    out["prev_volume"] = out["volume"].shift(1)
    out["daily_return_pct"] = (out["close"] / out["prev_close"] - 1.0) * 100.0
    out["j"] = _get_j_series(out)

    out["range_high_20"] = out["high"].rolling(B1_LOOKBACK_N, min_periods=B1_LOOKBACK_N).max()
    out["range_low_20"] = out["low"].rolling(B1_LOOKBACK_N, min_periods=B1_LOOKBACK_N).min()
    out["range_width_20"] = _safe_divide(out["range_high_20"] - out["range_low_20"], out["range_low_20"])
    out["position_in_range_20"] = _safe_divide(
        out["close"] - out["range_low_20"],
        out["range_high_20"] - out["range_low_20"],
    )
    out["previous_n_low"] = _rolling_min_shifted(out["low"], B1_PREV_LOW_LOOKBACK_MAX)

    out["ma20"] = out["close"].rolling(20, min_periods=20).mean()
    out["ma5"] = out["close"].rolling(5, min_periods=5).mean()
    out["ma10"] = out["close"].rolling(10, min_periods=10).mean()
    out["bbi"] = (out["ma5"] + out["ma10"] + out["ma20"]) / 3.0

    bbi_col = _first_existing_col(out, ["BBI", "bbi"])
    if bbi_col is not None:
        out["bbi"] = pd.to_numeric(out[bbi_col], errors="coerce")

    yellow_col = _first_existing_col(out, ["yellow_line", "yellow_ma", "ma20", "MA20"])
    out["yellow_line"] = pd.to_numeric(out[yellow_col], errors="coerce") if yellow_col is not None else out["ma20"]

    out["volume_ma5"] = out["volume"].rolling(B1_VOLUME_MA_N, min_periods=B1_VOLUME_MA_N).mean()
    out["volume_20_q20"] = _rolling_quantile(out["volume"], B1_LOOKBACK_N, B1_VOLUME_QUANTILE)
    out["volume_20_min"] = out["volume"].rolling(B1_LOOKBACK_N, min_periods=B1_LOOKBACK_N).min()
    out["volume_to_ma5"] = _safe_divide(out["volume"], out["volume_ma5"])

    candle_range = out["high"] - out["low"]
    upper_shadow = out["high"] - out[["open", "close"]].max(axis=1)
    out["upper_shadow_ratio"] = _safe_divide(upper_shadow, candle_range).fillna(0.0)
    out["b2_volume_ratio"] = _safe_divide(out["volume"], out["prev_volume"])
    return out


def _build_b1_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    b1_low_range_position = out["position_in_range_20"] <= B1_POSITION_RANGE_MAX
    b1_near_previous_low = (
        out["previous_n_low"].notna()
        & (out["close"] <= out["previous_n_low"] * (1.0 + B1_SUPPORT_DISTANCE_MAX))
        & (out["close"] >= out["previous_n_low"] * (1.0 - B1_FAKE_BREAK_MAX))
    )
    b1_in_range_bottom = (
        (out["range_width_20"] <= B1_RANGE_WIDTH_MAX)
        & (out["position_in_range_20"] <= B1_POSITION_RANGE_MAX)
        & (out["close"] >= out["range_low_20"] * (1.0 - B1_FAKE_BREAK_MAX))
    )
    b1_near_ma_support = (
        (out["bbi"].notna() & (out["close"] >= out["bbi"] * (1.0 - B1_SUPPORT_DISTANCE_MAX)) & (out["close"] <= out["bbi"] * (1.0 + B1_SUPPORT_DISTANCE_MAX)))
        | (out["yellow_line"].notna() & (out["close"] >= out["yellow_line"] * (1.0 - B1_SUPPORT_DISTANCE_MAX)) & (out["close"] <= out["yellow_line"] * (1.0 + B1_SUPPORT_DISTANCE_MAX)))
        | (out["ma20"].notna() & (out["close"] >= out["ma20"] * (1.0 - B1_SUPPORT_DISTANCE_MAX)) & (out["close"] <= out["ma20"] * (1.0 + B1_SUPPORT_DISTANCE_MAX)))
    )

    out["b1_position_ok"] = b1_low_range_position | b1_near_previous_low | b1_in_range_bottom | b1_near_ma_support
    out["b1_j_ok"] = out["j"] <= B1_J_MAX
    out["b1_low_volume"] = (out["volume"] <= out["volume_20_q20"]) | (out["volume_to_ma5"] <= B1_VOLUME_MA_RATIO_MAX)
    out["b1_not_break_prev_low"] = (
        out["previous_n_low"].notna()
        & (out["low"] >= out["previous_n_low"] * (1.0 - B1_FAKE_BREAK_MAX))
        & (out["close"] >= out["previous_n_low"])
    )
    out["b1_valid"] = out["b1_position_ok"] & out["b1_j_ok"] & out["b1_low_volume"] & out["b1_not_break_prev_low"]

    out["b1_j_deep_negative"] = out["j"] <= -10
    out["b1_extreme_low_volume"] = out["volume"] <= out["volume_20_min"]
    out["b1_in_range_bottom"] = b1_in_range_bottom
    return out


def _add_b1_lookback_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    b1_bool = out["b1_valid"].fillna(False).astype(bool)
    out["b1_days_ago"] = np.nan
    out["b1_j_value"] = np.nan
    out["b1_volume_value"] = np.nan

    for days_ago in range(1, B2_LOOKBACK_DAYS + 1):
        shifted_b1 = b1_bool.shift(days_ago).fillna(False)
        need_fill = out["b1_days_ago"].isna() & shifted_b1
        out.loc[need_fill, "b1_days_ago"] = days_ago
        out.loc[need_fill, "b1_j_value"] = out["j"].shift(days_ago).loc[need_fill]
        out.loc[need_fill, "b1_volume_value"] = out["volume"].shift(days_ago).loc[need_fill]

    out["b1_within_lookback"] = out["b1_days_ago"].notna()
    return out


def _build_b2_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["b2_after_b1"] = out["b1_within_lookback"]

    # v2 change: require B2 return between 4% and 9.7%.
    out["b2_return_ok"] = (out["daily_return_pct"] > B2_RETURN_MIN) & (out["daily_return_pct"] < B2_RETURN_MAX)
    out["b2_bullish_candle"] = out["close"] > out["open"]
    out["b2_volume_up"] = out["volume"] > out["prev_volume"]
    out["b2_j_ok"] = (out["j"] > B2_J_MIN) & (out["j"] < B2_J_MAX)
    out["b2_upper_shadow_ok"] = out["upper_shadow_ratio"] <= B2_UPPER_SHADOW_MAX

    out["b2_tiny_upper_shadow"] = out["upper_shadow_ratio"] <= B2_TINY_UPPER_SHADOW_MAX
    out["b2_double_volume"] = out["b2_volume_ratio"] >= B2_DOUBLE_VOLUME_RATIO
    out["b2_sky_volume"] = out["b2_volume_ratio"] >= B2_SKY_VOLUME_RATIO
    out["b2_j_high_zone"] = (out["j"] >= 45) & (out["j"] < B2_J_MAX)
    out["b2_j_low_zone_removed_by_v1"] = out["j"] <= B2_J_MIN
    out["b2_near_limitup_removed_by_v2"] = out["daily_return_pct"] >= B2_RETURN_MAX
    out["b2_j_value"] = out["j"]

    out["selected"] = (
        out["b2_after_b1"]
        & out["b2_return_ok"]
        & out["b2_bullish_candle"]
        & out["b2_volume_up"]
        & out["b2_j_ok"]
        & out["b2_upper_shadow_ok"]
    )

    quality_cols = ["b1_j_deep_negative", "b1_extreme_low_volume", "b1_in_range_bottom", "b2_tiny_upper_shadow", "b2_double_volume"]
    out["quality_score"] = 0
    for c in quality_cols:
        if c in out.columns:
            out["quality_score"] += out[c].fillna(False).astype(int)
    out["score"] = out["quality_score"]
    out["score_pct"] = out["quality_score"] / max(len(quality_cols), 1) * 100.0
    return out


def select(df: pd.DataFrame, n1: int | None = None, n2: int | None = None, **kwargs) -> pd.DataFrame:
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
        "date", "open", "high", "low", "close", "volume",
        "prev_close", "prev_volume", "daily_return_pct", "j",
        "selected", "score", "score_pct", "quality_score",
        "b1_days_ago", "b1_j_value", "b1_volume_value",
        "b1_position_ok", "b1_j_ok", "b1_low_volume", "b1_not_break_prev_low",
        "b1_j_deep_negative", "b1_extreme_low_volume", "b1_in_range_bottom",
        "b2_after_b1", "b2_return_ok", "b2_bullish_candle", "b2_volume_up",
        "b2_j_ok", "b2_upper_shadow_ok", "b2_j_value", "b2_volume_ratio",
        "upper_shadow_ratio", "b2_tiny_upper_shadow", "b2_double_volume", "b2_sky_volume",
        "b2_j_high_zone", "b2_j_low_zone_removed_by_v1", "b2_near_limitup_removed_by_v2",
        "range_high_20", "range_low_20", "range_width_20", "position_in_range_20",
        "previous_n_low", "volume_to_ma5", "volume_20_q20", "volume_20_min",
        "ma20", "bbi", "yellow_line",
    ]
    keep_cols = [c for c in keep_cols if c in selected.columns]
    return selected[keep_cols].reset_index(drop=True)


def apply_strategy(df: pd.DataFrame, n1: int | None = None, n2: int | None = None, **kwargs) -> pd.DataFrame:
    return select(df, n1=n1, n2=n2, **kwargs)


def run(df: pd.DataFrame, n1: int | None = None, n2: int | None = None, **kwargs) -> pd.DataFrame:
    return select(df, n1=n1, n2=n2, **kwargs)
