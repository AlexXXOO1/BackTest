# -*- coding: utf-8 -*-
"""
b2_confirm_select_strategy_v1

V1 keeps the B1 discovery + B2 confirmation entry logic from v0.
The old simple score_rank_key is replaced by a CSV-driven bucket scoring system.

Scoring design:
1. Only same-day or past-known factors are allowed.
2. fwd_* columns are never allowed as factors.
3. Duplicated, absolute-price, absolute-volume, weak, and middle-range-best factors are not used.
4. prefer_high_values: low buckets deduct points, high buckets add points.
5. prefer_low_values: low buckets add points, high buckets deduct points.
"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd


STRATEGY_NAME = "b2_confirm_select_strategy_v1"

KDJ_N = 9
KDJ_M1 = 3
KDJ_M2 = 3

B1_RECENT_DAYS = 5
SCORE_BUCKETS = 10

FactorDirection = Literal["high", "low"]

CSV_DERIVED_SCORING_RULES: tuple[tuple[str, FactorDirection], ...] = (
    ("daily_return_pct", "high"),
    ("body_pct", "high"),
    ("b2v0_dbbi", "high"),
    ("b2v0_dma20", "high"),
    ("amplitude_pct", "high"),
    ("lower_shadow_pct", "high"),
    ("b2v0_dzsl", "high"),
    ("b2v0_low_volume_count_20", "high"),
    ("b2v0_rw20", "high"),
    ("b2v0_kdj_j", "high"),
    ("b2v0_upper_shadow_ratio", "low"),
    ("volume_ratio_prev1", "low"),
    ("volume_ratio_ma5", "low"),
    ("volume_ratio_ma10", "low"),
    ("b2v0_kdj_d", "low"),
    ("b2v0_kdj_k", "low"),
)

FORBIDDEN_FACTOR_PREFIXES = ("fwd_",)


_schema_safe_re = re.compile(r"[^0-9a-zA-Z_]+")


def _num(s: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric values."""
    return pd.to_numeric(s, errors="coerce")


def _safe_div(a: pd.Series, b: pd.Series, default: float = np.nan) -> pd.Series:
    """Safe division. Zero denominator is replaced by default."""
    denom = _num(b).replace(0, np.nan)
    result = _num(a) / denom
    if not np.isnan(default):
        result = result.fillna(default)
    return result


def _tdx_sma(s: pd.Series, n: int, m: int = 1) -> pd.Series:
    """
    Calculate Tongdaxin-style SMA.

    TDX formula:
        SMA(X, N, M) = (M * X + (N - M) * REF(SMA, 1)) / N
    """
    return _num(s).ewm(alpha=m / n, adjust=False, min_periods=1).mean()


def _rolling_low_volume_count(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Count how many days in the latest window have volume >= current volume.

    This matches the TDX approximation:
        IF(V>=V,1,0) + IF(REF(V,1)>=V,1,0) + ... + IF(REF(V,19)>=V,1,0)
    """
    v = _num(volume)

    def _count(arr: np.ndarray) -> float:
        if len(arr) == 0 or np.isnan(arr[-1]):
            return np.nan
        return float(np.sum(arr >= arr[-1]))

    return v.rolling(window, min_periods=1).apply(_count, raw=True)


def _factor_col_name(prefix: str, factor: str) -> str:
    """Create a stable output column name for a factor."""
    safe = _schema_safe_re.sub("_", factor).strip("_").lower()
    return f"{prefix}_{safe}"


def _raw_bucket_score_from_bucket(bucket: pd.Series, bucket_count: int) -> pd.Series:
    """Map buckets into -5..-1 and +1..+5 when bucket_count is 10."""
    midpoint = bucket_count // 2
    score = pd.Series(0.0, index=bucket.index)
    low_mask = bucket <= midpoint
    high_mask = bucket > midpoint
    score.loc[low_mask] = bucket.loc[low_mask] - (midpoint + 1)
    score.loc[high_mask] = bucket.loc[high_mask] - midpoint
    return score


def _bucket_score(
    s: pd.Series,
    direction: FactorDirection,
    bucket_count: int = SCORE_BUCKETS,
) -> tuple[pd.Series, pd.Series]:
    """
    Convert one factor into bucket number and signed score.

    Missing values receive bucket 0 and score 0.
    Constant factors receive bucket 0 and score 0 to avoid fake signal.
    """
    x = _num(s)
    valid = x.notna() & np.isfinite(x)
    bucket = pd.Series(0, index=x.index, dtype="int64")
    score = pd.Series(0.0, index=x.index)

    if valid.sum() < 2 or x.loc[valid].nunique(dropna=True) < 2:
        return bucket, score

    pct_rank = x.loc[valid].rank(method="average", pct=True)
    raw_bucket = np.ceil(pct_rank * bucket_count).astype(int).clip(1, bucket_count)
    bucket.loc[valid] = raw_bucket.astype("int64")

    raw_score = _raw_bucket_score_from_bucket(bucket.loc[valid].astype(float), bucket_count)
    if direction == "low":
        raw_score = -raw_score
    score.loc[valid] = raw_score

    return bucket, score


def _ensure_scoring_factors(
    out: pd.DataFrame,
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    prev_close: pd.Series,
    prev_volume: pd.Series,
    daily_return_pct: pd.Series,
    volume_ratio_ma5: pd.Series,
) -> pd.DataFrame:
    """Create normalized same-day factor columns needed by the v1 score."""
    day_range = high - low
    min_open_close = pd.concat([open_, close], axis=1).min(axis=1)

    if "daily_return_pct" not in out.columns:
        out["daily_return_pct"] = daily_return_pct

    if "body_pct" not in out.columns:
        out["body_pct"] = _safe_div(close - open_, open_, default=np.nan) * 100.0

    if "amplitude_pct" not in out.columns:
        out["amplitude_pct"] = _safe_div(high - low, prev_close, default=np.nan) * 100.0

    if "lower_shadow_pct" not in out.columns:
        out["lower_shadow_pct"] = _safe_div(min_open_close - low, day_range, default=np.nan)

    if "volume_ratio_prev1" not in out.columns:
        out["volume_ratio_prev1"] = _safe_div(volume, prev_volume, default=np.nan)

    if "volume_ratio_ma5" not in out.columns:
        out["volume_ratio_ma5"] = volume_ratio_ma5

    if "volume_ratio_ma10" not in out.columns:
        volume_ma10 = volume.rolling(10, min_periods=1).mean()
        out["volume_ratio_ma10"] = _safe_div(volume, volume_ma10, default=np.nan)

    return out


def _apply_v1_bucket_score(out: pd.DataFrame) -> pd.DataFrame:
    """Apply CSV-derived bucket scoring rules and write detail columns."""
    total_score = pd.Series(0.0, index=out.index)
    used_factor_count = pd.Series(0, index=out.index, dtype="int64")
    used_factors: list[str] = []

    for factor, direction in CSV_DERIVED_SCORING_RULES:
        if factor.startswith(FORBIDDEN_FACTOR_PREFIXES):
            continue
        if factor not in out.columns:
            continue

        bucket, score = _bucket_score(out[factor], direction, SCORE_BUCKETS)
        bucket_col = _factor_col_name("b2v1_bucket", factor)
        score_col = _factor_col_name("b2v1_score", factor)

        out[bucket_col] = bucket
        out[score_col] = score

        valid_score = bucket > 0
        total_score = total_score + score.fillna(0.0)
        used_factor_count = used_factor_count + valid_score.astype("int64")
        used_factors.append(factor)

    out["b2v1_score_total"] = total_score
    out["b2v1_score_factor_count"] = used_factor_count
    out["b2v1_score_used_factors"] = ",".join(used_factors)
    return out


def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Apply B1 discovery, B2 confirmation, and v1 bucket scoring to one stock."""
    out = df.copy()

    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"{STRATEGY_NAME} missing required columns: {missing}")

    out = out.sort_values("date").reset_index(drop=True)

    open_ = _num(out["open"])
    high = _num(out["high"])
    low = _num(out["low"])
    close = _num(out["close"])
    volume = _num(out["volume"])

    prev_close = close.shift(1)
    prev_volume = volume.shift(1)

    daily_return_pct = (
        _num(out["daily_return_pct"])
        if "daily_return_pct" in out.columns
        else (_safe_div(close, prev_close) - 1.0) * 100.0
    )

    volume_ma5 = (
        _num(out["volume_ma5"])
        if "volume_ma5" in out.columns
        else volume.rolling(5, min_periods=1).mean()
    )
    volume_ratio_ma5 = _safe_div(volume, volume_ma5, default=999.0)

    ma20 = (
        _num(out["ma20"])
        if "ma20" in out.columns
        else close.rolling(20, min_periods=1).mean()
    )

    ma3 = close.rolling(3, min_periods=1).mean()
    ma6 = close.rolling(6, min_periods=1).mean()
    ma12 = close.rolling(12, min_periods=1).mean()
    ma24 = close.rolling(24, min_periods=1).mean()
    bbi = (ma3 + ma6 + ma12 + ma24) / 4.0
    zsl = close.ewm(span=21, adjust=False, min_periods=1).mean()

    ll9 = low.rolling(KDJ_N, min_periods=1).min()
    hh9 = high.rolling(KDJ_N, min_periods=1).max()
    r9 = hh9 - ll9
    rsv = pd.Series(
        np.where(r9 == 0, 0.0, (close - ll9) / r9 * 100.0),
        index=out.index,
    )
    k = _tdx_sma(rsv, KDJ_M1, 1)
    d = _tdx_sma(k, KDJ_M2, 1)
    j = 3.0 * k - 2.0 * d

    rh20 = high.rolling(20, min_periods=1).max()
    rl20 = low.rolling(20, min_periods=1).min()
    rhgt = rh20 - rl20
    rw20 = pd.Series(
        np.where(rl20 == 0, 999.0, rhgt / rl20),
        index=out.index,
    )
    pos20 = pd.Series(
        np.where(rhgt == 0, 999.0, (close - rl20) / rhgt),
        index=out.index,
    )

    pnlow = low.shift(1).rolling(20, min_periods=1).min()
    dplow = pd.Series(
        np.where(pnlow == 0, 999.0, (close - pnlow) / pnlow),
        index=out.index,
    )

    dma20 = pd.Series(
        np.where(ma20 == 0, 999.0, (close - ma20).abs() / ma20),
        index=out.index,
    )
    dbbi = pd.Series(
        np.where(bbi == 0, 999.0, (close - bbi).abs() / bbi),
        index=out.index,
    )
    dzsl = pd.Series(
        np.where(zsl == 0, 999.0, (close - zsl).abs() / zsl),
        index=out.index,
    )

    day_range = high - low
    max_open_close = pd.concat([open_, close], axis=1).max(axis=1)
    upper_shadow_ratio = pd.Series(
        np.where(day_range == 0, 999.0, (high - max_open_close) / day_range),
        index=out.index,
    )

    low_volume_count_20 = _rolling_low_volume_count(volume, 20)

    b1_low_position = pos20 <= 0.25
    b1_near_prev_low = dplow.abs() <= 0.02
    b1_bottom_zone = (rw20 <= 0.25) & (pos20 <= 0.25) & (close >= rl20 * 0.98)
    b1_ma_support = (dma20 <= 0.02) | (dbbi <= 0.02) | (dzsl <= 0.02)

    b1_position_ok = b1_low_position | b1_near_prev_low | b1_bottom_zone | b1_ma_support
    b1_j_ok = j <= 14
    b1_low_volume = (low_volume_count_20 >= 16) | (volume_ratio_ma5 <= 0.70)
    b1_not_break_prev_low = (low >= pnlow * 0.98) & (close >= pnlow)

    b1_discovery = b1_position_ok & b1_j_ok & b1_low_volume & b1_not_break_prev_low

    b1_recent_5d = pd.Series(False, index=out.index)
    for i in range(1, B1_RECENT_DAYS + 1):
        b1_recent_5d = b1_recent_5d | b1_discovery.shift(i, fill_value=False)

    b2_return_ok = daily_return_pct > 4.0
    b2_bull = close > open_
    b2_volume_ok = volume > prev_volume
    b2_j_ok = j < 55
    b2_upper_shadow_ok = upper_shadow_ratio <= 0.25

    b2_confirm = (
        b1_recent_5d
        & b2_return_ok
        & b2_bull
        & b2_volume_ok
        & b2_j_ok
        & b2_upper_shadow_ok
    )

    out["b2v0_kdj_k"] = k
    out["b2v0_kdj_d"] = d
    out["b2v0_kdj_j"] = j
    out["b2v0_rh20"] = rh20
    out["b2v0_rl20"] = rl20
    out["b2v0_rw20"] = rw20
    out["b2v0_pos20"] = pos20
    out["b2v0_pnlow"] = pnlow
    out["b2v0_dplow"] = dplow
    out["b2v0_bbi"] = bbi
    out["b2v0_zsl_ema21"] = zsl
    out["b2v0_dma20"] = dma20
    out["b2v0_dbbi"] = dbbi
    out["b2v0_dzsl"] = dzsl
    out["b2v0_upper_shadow_ratio"] = upper_shadow_ratio
    out["b2v0_low_volume_count_20"] = low_volume_count_20
    out["b2v0_volume_ratio_ma5"] = volume_ratio_ma5

    out = _ensure_scoring_factors(
        out=out,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        prev_close=prev_close,
        prev_volume=prev_volume,
        daily_return_pct=daily_return_pct,
        volume_ratio_ma5=volume_ratio_ma5,
    )
    out = _apply_v1_bucket_score(out)

    out["b1_discovery_v0"] = b1_discovery.fillna(False).astype(int)
    out["b1_recent_5d_v0"] = b1_recent_5d.fillna(False).astype(int)

    out["b2v0_cond_return_gt_4"] = b2_return_ok.fillna(False).astype(int)
    out["b2v0_cond_bull"] = b2_bull.fillna(False).astype(int)
    out["b2v0_cond_volume_gt_prev"] = b2_volume_ok.fillna(False).astype(int)
    out["b2v0_cond_j_lt_55"] = b2_j_ok.fillna(False).astype(int)
    out["b2v0_cond_upper_shadow_le_025"] = b2_upper_shadow_ok.fillna(False).astype(int)

    out["b2_confirm_v0"] = b2_confirm.fillna(False).astype(int)
    out["b2_confirm_v1"] = out["b2_confirm_v0"]
    out["selected"] = out["b2_confirm_v1"]
    out["selection_strategy"] = STRATEGY_NAME

    out["selected_score_base"] = out["b2v1_score_total"]
    out["score_rank_key"] = out["b2v1_score_total"]
    out["score_pct"] = out["score_rank_key"].rank(pct=True)

    return out


def select(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Entry point for scripts/build_pool.py."""
    return apply_strategy(df)


SELECT_FUNC = select
