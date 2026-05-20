# -*- coding: utf-8 -*-
from __future__ import annotations

"""
B1 stage-low selection strategy v1.

Purpose:
- Select stage-low candidates only.
- Do not score candidates.
- Do not create final buy decisions.
- Keep enough factor columns for forward-return validation.

B1 v1 selection rule:
    1) kdj_j < j_threshold
    2) b1_structure_downtrend_20 == 0

Compared with v0, v1 adds the 20-day structural downtrend flag directly in
the strategy and uses it as a hard reject condition. Other B1 condition checks
are retained as factor columns in the pool parquet for later bucket analysis
and validation. T-1 J is retained as a continuous numeric factor, not as a
binary threshold flag. The former extreme-volume flag is not exported.
"""

import numpy as np
import pandas as pd


STRATEGY_NAME = "b1_stage_low_select_strategy_v1"

STRUCTURE_DOWNTREND_WINDOW = 20
STRUCTURE_HIGH_DOWN_THRESHOLD = 0.99
STRUCTURE_LOW_DOWN_THRESHOLD = 0.97
STRUCTURE_DOWNTREND_COL = "b1_structure_downtrend_20"


REQUIRED_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


FINAL_FACTOR_COLUMNS = [
    "kdj_k",
    "kdj_d",
    "kdj_j",
    "volume_ratio_ma5",
    "volume_ratio_ma10",
    "previous_20d_low",
    "close_to_previous_20d_low_pct",
    "low_to_previous_20d_low_pct",
    "ma20",
    "ma60",
    "close_to_ma20_pct",
    "close_to_ma60_pct",
    "range_60d_high",
    "range_60d_low",
    "range_60d_pct",
    "close_position_60d_pct",
    "b1_j_low",
    "b1_tminus1_j",
    "b1_structure_downtrend_20",
    "b1_not_break_previous_20d_low",
    "is_uptrend_pullback",
    "is_range_low",
    "b1_stage_low_valid",
]


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = _to_num(b).replace(0, np.nan)
    return _to_num(a) / denom


def _tdx_sma(series: pd.Series, period: int, weight: int = 1) -> pd.Series:
    """Approximate TDX SMA(X, N, M): Y=(M*X+(N-M)*Y')/N."""
    alpha = float(weight) / float(period)
    return _to_num(series).ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def _ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"{STRATEGY_NAME} missing required columns: {missing}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = _to_num(out[col])

    return out


def add_kdj_features(
    df: pd.DataFrame,
    kdj_n: int = 9,
    kdj_m1: int = 3,
    kdj_m2: int = 3,
) -> pd.DataFrame:
    """Use existing KDJ columns when present; otherwise calculate standard KDJ."""
    out = df.copy()

    has_kdj = all(c in out.columns for c in ["kdj_k", "kdj_d", "kdj_j"])
    if has_kdj:
        out["kdj_k"] = _to_num(out["kdj_k"])
        out["kdj_d"] = _to_num(out["kdj_d"])
        out["kdj_j"] = _to_num(out["kdj_j"])
        return out

    high_n = _to_num(out["high"]).rolling(int(kdj_n), min_periods=max(1, int(kdj_n) // 3)).max()
    low_n = _to_num(out["low"]).rolling(int(kdj_n), min_periods=max(1, int(kdj_n) // 3)).min()
    rsv = (_safe_div(_to_num(out["close"]) - low_n, high_n - low_n) * 100.0).clip(lower=0, upper=100)

    out["kdj_k"] = _tdx_sma(rsv, int(kdj_m1), 1)
    out["kdj_d"] = _tdx_sma(out["kdj_k"], int(kdj_m2), 1)
    out["kdj_j"] = 3.0 * out["kdj_k"] - 2.0 * out["kdj_d"]

    return out


def add_b1_features(
    df: pd.DataFrame,
    support_lookback: int = 20,
    range_lookback: int = 60,
) -> pd.DataFrame:
    out = df.copy()

    close = _to_num(out["close"])
    low = _to_num(out["low"])
    high = _to_num(out["high"])
    volume = _to_num(out["volume"])

    if "ma20" not in out.columns:
        out["ma20"] = close.rolling(20, min_periods=7).mean()
    else:
        out["ma20"] = _to_num(out["ma20"])

    if "ma60" not in out.columns:
        out["ma60"] = close.rolling(60, min_periods=20).mean()
    else:
        out["ma60"] = _to_num(out["ma60"])

    if "volume_ratio_ma5" not in out.columns:
        volume_ma5 = volume.rolling(5, min_periods=2).mean()
        out["volume_ratio_ma5"] = _safe_div(volume, volume_ma5)
    else:
        out["volume_ratio_ma5"] = _to_num(out["volume_ratio_ma5"])

    if "volume_ratio_ma10" not in out.columns:
        volume_ma10 = volume.rolling(10, min_periods=3).mean()
        out["volume_ratio_ma10"] = _safe_div(volume, volume_ma10)
    else:
        out["volume_ratio_ma10"] = _to_num(out["volume_ratio_ma10"])

    prev_low_window = int(support_lookback)
    range_window = int(range_lookback)

    out["previous_20d_low"] = low.shift(1).rolling(prev_low_window, min_periods=max(3, prev_low_window // 3)).min()
    out["close_to_previous_20d_low_pct"] = (_safe_div(close, out["previous_20d_low"]) - 1.0) * 100.0
    out["low_to_previous_20d_low_pct"] = (_safe_div(low, out["previous_20d_low"]) - 1.0) * 100.0

    out["close_to_ma20_pct"] = (_safe_div(close, out["ma20"]) - 1.0) * 100.0
    out["close_to_ma60_pct"] = (_safe_div(close, out["ma60"]) - 1.0) * 100.0

    out["range_60d_high"] = high.rolling(range_window, min_periods=max(10, range_window // 3)).max()
    out["range_60d_low"] = low.rolling(range_window, min_periods=max(10, range_window // 3)).min()
    out["range_60d_pct"] = (_safe_div(out["range_60d_high"], out["range_60d_low"]) - 1.0) * 100.0
    out["close_position_60d_pct"] = (
        _safe_div(close - out["range_60d_low"], out["range_60d_high"] - out["range_60d_low"]) * 100.0
    )

    recent_high = high.rolling(
        STRUCTURE_DOWNTREND_WINDOW,
        min_periods=STRUCTURE_DOWNTREND_WINDOW,
    ).max()
    previous_high = recent_high.shift(STRUCTURE_DOWNTREND_WINDOW)

    recent_low = low.rolling(
        STRUCTURE_DOWNTREND_WINDOW,
        min_periods=STRUCTURE_DOWNTREND_WINDOW,
    ).min()
    previous_low = recent_low.shift(STRUCTURE_DOWNTREND_WINDOW)

    high_clearly_lower = recent_high < previous_high * STRUCTURE_HIGH_DOWN_THRESHOLD
    low_clearly_lower = recent_low < previous_low * STRUCTURE_LOW_DOWN_THRESHOLD
    out[STRUCTURE_DOWNTREND_COL] = (
        high_clearly_lower & low_clearly_lower
    ).fillna(False).astype("int8")

    return out


def select(
    df: pd.DataFrame,
    j_threshold: float = 14.0,
    support_break_tolerance_pct: float = 2.0,
    uptrend_ma60_floor_pct: float = -3.0,
    uptrend_ma20_ceiling_pct: float = 3.0,
    range_width_60d_max_pct: float = 35.0,
    range_low_position_max_pct: float = 35.0,
    support_lookback: int = 20,
    range_lookback: int = 60,
    **kwargs,
) -> pd.DataFrame:
    """Return B1 v1 rows after low-J and non-downtrend hard filters."""
    out = _ensure_base_columns(df)
    out = add_kdj_features(out)
    out = add_b1_features(out, support_lookback=support_lookback, range_lookback=range_lookback)

    kdj_j = _to_num(out["kdj_j"])
    close = _to_num(out["close"])
    low = _to_num(out["low"])
    prev_low = _to_num(out["previous_20d_low"])
    structure_downtrend = pd.to_numeric(
        out[STRUCTURE_DOWNTREND_COL],
        errors="coerce",
    ).fillna(0).astype(int)

    out["b1_j_low"] = (kdj_j < float(j_threshold)).astype(int)

    # Numeric T-1 J factor. Keep it continuous so Analyze Pool Indicator can bucket it.
    out["b1_tminus1_j"] = kdj_j.shift(1)

    support_floor = prev_low * (1.0 - float(support_break_tolerance_pct) / 100.0)
    not_break_support = low.ge(support_floor) & close.ge(prev_low)
    out["b1_not_break_previous_20d_low"] = not_break_support.fillna(False).astype(int)

    uptrend_pullback = (
        (_to_num(out["ma20"]) > _to_num(out["ma60"]))
        & (_to_num(out["close_to_ma60_pct"]) >= float(uptrend_ma60_floor_pct))
        & (_to_num(out["close_to_ma20_pct"]) <= float(uptrend_ma20_ceiling_pct))
    )
    out["is_uptrend_pullback"] = uptrend_pullback.fillna(False).astype(int)

    range_low = (
        (_to_num(out["range_60d_pct"]) <= float(range_width_60d_max_pct))
        & (_to_num(out["close_position_60d_pct"]) <= float(range_low_position_max_pct))
    )
    out["is_range_low"] = range_low.fillna(False).astype(int)

    # Hard filter: keep low-J candidates only when the 20-day structural downtrend flag is off.
    hard_filter = out["b1_j_low"].eq(1) & structure_downtrend.eq(0)

    factor_valid = (
        out["b1_j_low"].eq(1)
        & structure_downtrend.eq(0)
        & out["b1_not_break_previous_20d_low"].eq(1)
        & (out["is_uptrend_pullback"].eq(1) | out["is_range_low"].eq(1))
    )
    out["b1_stage_low_valid"] = factor_valid.astype(int)

    selected = out.loc[hard_filter].copy()
    selected["selection_strategy"] = STRATEGY_NAME

    keep_cols = [
        c for c in [
            "symbol", "file", "date", "selection_strategy",
            "open", "high", "low", "close", "volume", "amount",
            "daily_return_pct", "intraday_return_pct", "amplitude_pct",
            "upper_shadow_pct", "lower_shadow_pct", "body_pct", "body_abs_pct",
            "is_red_k", "is_green_k", "is_flat_k",
            *FINAL_FACTOR_COLUMNS,
        ]
        if c in selected.columns
    ]

    extra_cols = [c for c in selected.columns if c not in keep_cols]
    return selected[[*keep_cols, *extra_cols]].copy()


SELECT_FUNC = select
