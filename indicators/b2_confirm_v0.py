from __future__ import annotations

"""
Reusable B1/B2 confirmation indicators for b2_confirm_select_strategy_v0.

Layer rule:
- This file calculates reusable indicator / condition columns only.
- It does NOT create the final strategy-level `selected` column.
- The strategy file decides how to combine b1/b2 flags into selected / score.
"""

from typing import Iterable

import numpy as np
import pandas as pd


OUTPUT_COLUMNS: set[str] = {
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

J_COLUMN_CANDIDATES: tuple[str, ...] = (
    "j",
    "J",
    "kdj_j",
    "KDJ_J",
    "j_value",
    "J_VALUE",
)

YELLOW_LINE_CANDIDATES: tuple[str, ...] = (
    "z_slow_trend_line",
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
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _to_bool_series(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


def add_b2_confirm_v0_indicators(
    df: pd.DataFrame,
    *,
    b1_j_max: float = 14.0,
    b2_lookback_days: int = 5,
    b2_min_return_pct: float = 4.0,
    b2_j_max: float = 55.0,
    b2_max_upper_shadow_ratio: float = 0.25,
    low_range_lookback: int = 20,
    position_in_range_max: float = 0.25,
    support_distance_max: float = 0.02,
    previous_low_lookback: int = 20,
    false_break_pct: float = 0.02,
    low_volume_lookback: int = 20,
    low_volume_quantile: float = 0.20,
    volume_ma_lookback: int = 5,
    volume_ma_ratio_max: float = 0.70,
    range_width_max: float = 0.25,
    range_bottom_position_max: float = 0.25,
    range_bottom_false_break_pct: float = 0.02,
    ma_support_distance_max: float = 0.02,
) -> pd.DataFrame:
    """
    Add B1/B2 confirmation indicator columns.

    B1 = low-position condition AND J <= b1_j_max AND low-volume condition
         AND not-effectively-break-previous-low condition.

    B2 flags are calculated for current day T0. Final selected logic stays in
    selection_strategies/b2_confirm_select_strategy_v0.py.
    """
    out = df.copy()

    if "date" in out.columns:
        out = out.sort_values("date").reset_index(drop=True)

    j_col = _find_first_existing_column(out, J_COLUMN_CANDIDATES)
    if j_col is None:
        raise KeyError(
            "Missing KDJ J column. Expected one of: " + ", ".join(J_COLUMN_CANDIDATES)
        )

    yellow_col = _find_first_existing_column(out, YELLOW_LINE_CANDIDATES)
    bbi_col = _find_first_existing_column(out, BBI_COLUMN_CANDIDATES)

    open_ = pd.to_numeric(out["open"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce")
    j = pd.to_numeric(out[j_col], errors="coerce")

    # Keep daily_return_pct from base indicators if available; otherwise create it.
    if "daily_return_pct" not in out.columns:
        prev_close = close.shift(1)
        out["daily_return_pct"] = (_safe_div(close, prev_close) - 1.0) * 100.0

    out["prev_volume"] = volume.shift(1)
    out["volume_ratio_prev"] = _safe_div(volume, out["prev_volume"])

    out["volume_ma5"] = volume.rolling(volume_ma_lookback, min_periods=1).mean()
    out["volume_ratio_ma5"] = _safe_div(volume, out["volume_ma5"])

    out["volume_q20_20"] = volume.rolling(
        low_volume_lookback, min_periods=low_volume_lookback
    ).quantile(low_volume_quantile)
    out["volume_min_20"] = volume.rolling(
        low_volume_lookback, min_periods=low_volume_lookback
    ).min()

    out["range_high_20"] = high.rolling(
        low_range_lookback, min_periods=low_range_lookback
    ).max()
    out["range_low_20"] = low.rolling(
        low_range_lookback, min_periods=low_range_lookback
    ).min()

    range_height = out["range_high_20"] - out["range_low_20"]
    out["range_width_20"] = _safe_div(range_height, out["range_low_20"])
    out["position_in_range_20"] = _safe_div(close - out["range_low_20"], range_height)

    out["previous_n_low"] = low.shift(1).rolling(previous_low_lookback, min_periods=5).min()
    out["distance_to_previous_n_low"] = _safe_div(
        close - out["previous_n_low"], out["previous_n_low"]
    )

    if "ma20" not in out.columns:
        out["ma20"] = close.rolling(20, min_periods=1).mean()
    out["distance_to_ma20"] = (close - out["ma20"]).abs() / out["ma20"].replace(0, np.nan)

    if bbi_col is not None:
        out["bbi_for_b1"] = out[bbi_col]
    else:
        ma3 = close.rolling(3, min_periods=1).mean()
        ma6 = close.rolling(6, min_periods=1).mean()
        ma12 = close.rolling(12, min_periods=1).mean()
        ma24 = close.rolling(24, min_periods=1).mean()
        out["bbi_for_b1"] = (ma3 + ma6 + ma12 + ma24) / 4.0
    out["distance_to_bbi"] = (close - out["bbi_for_b1"]).abs() / out["bbi_for_b1"].replace(0, np.nan)

    if yellow_col is not None:
        out["yellow_for_b1"] = out[yellow_col]
    else:
        out["yellow_for_b1"] = out["ma20"]
    out["distance_to_yellow"] = (close - out["yellow_for_b1"]).abs() / out["yellow_for_b1"].replace(0, np.nan)

    daily_range = high - low
    out["upper_shadow_ratio"] = _safe_div(high - pd.concat([open_, close], axis=1).max(axis=1), daily_range)

    # B1 conditions.
    out["b1_low_range_position"] = out["position_in_range_20"] <= position_in_range_max
    out["b1_near_previous_n_low"] = out["distance_to_previous_n_low"].abs() <= support_distance_max
    out["b1_in_range_bottom"] = (
        (out["range_width_20"] <= range_width_max)
        & (out["position_in_range_20"] <= range_bottom_position_max)
        & (close >= out["range_low_20"] * (1.0 - range_bottom_false_break_pct))
    )
    out["b1_near_ma_support"] = (
        (out["distance_to_ma20"] <= ma_support_distance_max)
        | (out["distance_to_bbi"] <= ma_support_distance_max)
        | (out["distance_to_yellow"] <= ma_support_distance_max)
    )
    out["b1_position_ok"] = (
        out["b1_low_range_position"]
        | out["b1_near_previous_n_low"]
        | out["b1_in_range_bottom"]
        | out["b1_near_ma_support"]
    )
    out["b1_j_ok"] = j <= b1_j_max
    out["b1_low_volume"] = (
        (volume <= out["volume_q20_20"])
        | (out["volume_ratio_ma5"] <= volume_ma_ratio_max)
    )
    out["b1_extreme_low_volume"] = volume <= out["volume_min_20"]
    out["b1_not_break_prev_low"] = (
        (low >= out["previous_n_low"] * (1.0 - false_break_pct))
        & (close >= out["previous_n_low"])
    )
    out["b1_valid"] = (
        out["b1_position_ok"]
        & out["b1_j_ok"]
        & out["b1_low_volume"]
        & out["b1_not_break_prev_low"]
    )

    # B2 current-day flags.
    b1_recent = pd.Series(False, index=out.index)
    b1_days_ago = pd.Series(np.nan, index=out.index, dtype="float64")
    for days_ago in range(1, int(b2_lookback_days) + 1):
        shifted = out["b1_valid"].shift(days_ago).fillna(False).astype(bool)
        b1_recent = b1_recent | shifted
        b1_days_ago = b1_days_ago.mask(shifted & b1_days_ago.isna(), float(days_ago))

    out["b1_within_b2_lookback"] = b1_recent
    out["b1_days_ago_for_b2"] = b1_days_ago

    out["b2_return_ok"] = pd.to_numeric(out["daily_return_pct"], errors="coerce") > b2_min_return_pct
    out["b2_bullish_candle"] = close > open_
    out["b2_volume_up"] = volume > out["prev_volume"]
    out["b2_double_volume"] = volume > 1.90 * out["prev_volume"]
    out["b2_sky_volume"] = volume > 3.50 * out["prev_volume"]
    out["b2_j_ok"] = j < b2_j_max
    out["b2_j_high_zone"] = (j >= 45.0) & (j < b2_j_max)
    out["b2_upper_shadow_ok"] = out["upper_shadow_ratio"] <= b2_max_upper_shadow_ratio
    out["b2_tiny_upper_shadow"] = out["upper_shadow_ratio"] <= 0.10
    out["b2_upper_shadow_warning"] = (
        (out["upper_shadow_ratio"] > 0.20)
        & (out["upper_shadow_ratio"] <= b2_max_upper_shadow_ratio)
    )

    out["b2_quality_score"] = (
        out["b2_double_volume"].fillna(False).astype(int)
        + out["b2_tiny_upper_shadow"].fillna(False).astype(int)
        + out["b1_in_range_bottom"].shift(1).fillna(False).astype(int)
    )

    bool_cols = [c for c in out.columns if c.startswith("b1_") or c.startswith("b2_")]
    for c in bool_cols:
        if c not in {"b1_days_ago_for_b2", "b2_quality_score"}:
            out[c] = _to_bool_series(out[c])

    return out
