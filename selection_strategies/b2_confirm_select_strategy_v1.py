# -*- coding: utf-8 -*-
"""
b2_confirm_select_strategy_v1

Single-file implementation of B1 discovery + B2 confirmation logic.

Compared with v0:
- Only the B1 discovery algorithm is changed.
- B1 adds MACD white line / DIF above 0 axis: DIF > 0.
- B1 relaxes KDJ J from J <= 14 to J < 22.
- B2 confirmation conditions remain aligned with v0.

Final pool output:
    selected = b2_confirm_v1
"""

from __future__ import annotations

import numpy as np
import pandas as pd


STRATEGY_NAME = "b2_confirm_select_strategy_v1"

KDJ_N = 9
KDJ_M1 = 3
KDJ_M2 = 3

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

B1_RECENT_DAYS = 5


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


def select(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Apply B1 v1 discovery and B2 confirmation to one stock's historical data."""
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

    ema12 = close.ewm(span=MACD_FAST, adjust=False, min_periods=1).mean()
    ema26 = close.ewm(span=MACD_SLOW, adjust=False, min_periods=1).mean()
    macd_dif = ema12 - ema26
    macd_dea = macd_dif.ewm(span=MACD_SIGNAL, adjust=False, min_periods=1).mean()
    macd_hist = 2.0 * (macd_dif - macd_dea)

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
    b1_j_ok = j < 0.0
    b1_macd_dif_above_zero = macd_dif > 0.0
    b1_low_volume = (low_volume_count_20 >= 16) | (volume_ratio_ma5 <= 0.70)
    b1_not_break_prev_low = (low >= pnlow * 0.98) & (close >= pnlow)

    b1_discovery = (
        b1_position_ok
        & b1_j_ok
        & b1_macd_dif_above_zero
        & b1_low_volume
        & b1_not_break_prev_low
    )

    b1_recent_5d = pd.Series(False, index=out.index)
    for i in range(1, B1_RECENT_DAYS + 1):
        b1_recent_5d = b1_recent_5d | b1_discovery.shift(i, fill_value=False)

    b2_return_ok = daily_return_pct > 4.0
    b2_bull = close > open_
    b2_volume_ok = volume > prev_volume
    b2_j_ok = j < 55.0
    b2_upper_shadow_ok = upper_shadow_ratio <= 0.25

    b2_confirm = (
        b1_recent_5d
        & b2_return_ok
        & b2_bull
        & b2_volume_ok
        & b2_j_ok
        & b2_upper_shadow_ok
    )

    out["b2v1_kdj_k"] = k
    out["b2v1_kdj_d"] = d
    out["b2v1_kdj_j"] = j
    out["b2v1_macd_dif"] = macd_dif
    out["b2v1_macd_dea"] = macd_dea
    out["b2v1_macd_hist"] = macd_hist
    out["b2v1_rh20"] = rh20
    out["b2v1_rl20"] = rl20
    out["b2v1_rw20"] = rw20
    out["b2v1_pos20"] = pos20
    out["b2v1_pnlow"] = pnlow
    out["b2v1_dplow"] = dplow
    out["b2v1_bbi"] = bbi
    out["b2v1_zsl_ema21"] = zsl
    out["b2v1_dma20"] = dma20
    out["b2v1_dbbi"] = dbbi
    out["b2v1_dzsl"] = dzsl
    out["b2v1_upper_shadow_ratio"] = upper_shadow_ratio
    out["b2v1_low_volume_count_20"] = low_volume_count_20
    out["b2v1_volume_ratio_ma5"] = volume_ratio_ma5

    out["b1_discovery_v1"] = b1_discovery.fillna(False).astype(int)
    out["b1_recent_5d_v1"] = b1_recent_5d.fillna(False).astype(int)
    out["b2v1_cond_b1_j_lt_22"] = b1_j_ok.fillna(False).astype(int)
    out["b2v1_cond_b1_macd_dif_gt_0"] = b1_macd_dif_above_zero.fillna(False).astype(int)

    out["b2v1_cond_return_gt_4"] = b2_return_ok.fillna(False).astype(int)
    out["b2v1_cond_bull"] = b2_bull.fillna(False).astype(int)
    out["b2v1_cond_volume_gt_prev"] = b2_volume_ok.fillna(False).astype(int)
    out["b2v1_cond_j_lt_55"] = b2_j_ok.fillna(False).astype(int)
    out["b2v1_cond_upper_shadow_le_025"] = b2_upper_shadow_ok.fillna(False).astype(int)

    out["b2_confirm_v1"] = b2_confirm.fillna(False).astype(int)
    out["selected"] = out["b2_confirm_v1"]
    out["selection_strategy"] = STRATEGY_NAME
    out["selected_score_base"] = out["selected"]

    out["score_rank_key"] = (
        daily_return_pct.fillna(0.0)
        - upper_shadow_ratio.fillna(999.0) * 10.0
        - j.fillna(999.0) * 0.01
        + (macd_dif > 0).astype(float)
    )
    out["score_pct"] = out["score_rank_key"].rank(pct=True)

    return out


apply_strategy = select
SELECT_FUNC = select
