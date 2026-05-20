# -*- coding: utf-8 -*-
from __future__ import annotations

"""
B2 confirm selection strategy based on the inline TDX B1+B2 formula.

This module is intentionally a selection strategy only:
- It receives one symbol dataframe from ops/daily_update/build_pool.py.
- It calculates the TDX-aligned B1 context and B2 confirmation factors.
- It returns selected B2 rows only.
- Forward fields are added centrally by ops/daily_update/build_pool.py.

Run:
    python .\\ops\\daily_update\\build_pool.py --strategy b2_confirm_tdx_b1_v0 --no-csv
"""

import numpy as np
import pandas as pd


STRATEGY_NAME = "b2_confirm_tdx_b1_v0"

REQUIRED_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

B2_FACTOR_COLUMNS = [
    "b1_source_strategy",
    "b1_date",
    "b1_lag_trade_days",
    "prev_close",
    "prev_volume",
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "volume_ma5",
    "volume_ma10",
    "volume_ratio_ma5",
    "volume_ratio_ma10",
    "volume_ratio_prev1",
    "kdj_k",
    "kdj_d",
    "kdj_j",
    "macd_dif",
    "macd_dea",
    "macd_hist",
    "b1_tdx_range_high_20",
    "b1_tdx_range_low_20",
    "b1_tdx_range_width_20",
    "b1_tdx_position_20",
    "b1_tdx_previous_20d_low",
    "b1_tdx_close_to_previous_20d_low",
    "b1_tdx_bbi",
    "b1_tdx_zsl_ema21",
    "b1_tdx_distance_to_ma20",
    "b1_tdx_distance_to_bbi",
    "b1_tdx_distance_to_zsl",
    "b1_tdx_low_volume_count_20",
    "b1_tdx_low_position",
    "b1_tdx_near_previous_low",
    "b1_tdx_bottom_zone",
    "b1_tdx_ma_support",
    "b1_tdx_position_ok",
    "b1_tdx_j_ok",
    "b1_tdx_low_volume",
    "b1_tdx_not_break_previous_low",
    "b1_tdx_signal",
    "b1_tdx_recent_1_to_5",
    "b2_tdx_return_gt_4",
    "b2_tdx_bullish_candle",
    "b2_tdx_volume_gt_prev",
    "b2_tdx_j_below_55",
    "b2_tdx_upper_shadow_ratio",
    "b2_tdx_upper_shadow_ok",
]

OUTPUT_FRONT_COLUMNS = [
    "symbol",
    "file",
    "date",
    "selection_strategy",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "daily_return_pct",
    "intraday_return_pct",
    "amplitude_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "body_pct",
    "body_abs_pct",
    "is_red_k",
    "is_green_k",
    "is_flat_k",
    *B2_FACTOR_COLUMNS,
]


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_div(a: pd.Series, b: pd.Series, fill_when_zero: float | None = None) -> pd.Series:
    denom = _to_num(b)
    out = _to_num(a) / denom.replace(0, np.nan)
    if fill_when_zero is not None:
        out = out.mask(denom.eq(0), fill_when_zero)
    return out


def _tdx_sma(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """Approximate TDX SMA(X, N, M): Y=(M*X+(N-M)*Y')/N."""
    alpha = float(m) / float(n)
    return _to_num(series).ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def _tdx_ema(series: pd.Series, n: int) -> pd.Series:
    return _to_num(series).ewm(span=int(n), adjust=False, min_periods=1).mean()


def _ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"{STRATEGY_NAME} missing required columns: {missing}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = _to_num(out[col])

    if "amount" not in out.columns:
        out["amount"] = np.nan
    else:
        out["amount"] = _to_num(out["amount"])

    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    out = out[out["close"].gt(0)]
    out = out.sort_values("date", kind="stable").drop_duplicates("date", keep="last").reset_index(drop=True)

    return out


def _rolling_low_volume_count_20(volume: pd.Series) -> pd.Series:
    v = _to_num(volume)
    return v.rolling(20, min_periods=20).apply(
        lambda x: float(np.sum(x >= x[-1])) if np.isfinite(x[-1]) else np.nan,
        raw=True,
    )


def add_tdx_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_base_columns(df)

    open_ = _to_num(out["open"])
    high = _to_num(out["high"])
    low = _to_num(out["low"])
    close = _to_num(out["close"])
    volume = _to_num(out["volume"])

    prev_close = close.shift(1)
    prev_volume = volume.shift(1)

    out["prev_close"] = prev_close
    out["prev_volume"] = prev_volume

    out["daily_return_pct"] = (_safe_div(close, prev_close) - 1.0) * 100.0
    out["intraday_return_pct"] = (_safe_div(close, open_) - 1.0) * 100.0
    out["amplitude_pct"] = _safe_div(high - low, prev_close) * 100.0
    out["body_pct"] = _safe_div(close - open_, prev_close) * 100.0
    out["body_abs_pct"] = out["body_pct"].abs()

    max_oc = pd.concat([open_, close], axis=1).max(axis=1)
    min_oc = pd.concat([open_, close], axis=1).min(axis=1)

    out["upper_shadow_pct"] = _safe_div(high - max_oc, prev_close) * 100.0
    out["lower_shadow_pct"] = _safe_div(min_oc - low, prev_close) * 100.0

    out["is_red_k"] = close.gt(open_).astype("int8")
    out["is_green_k"] = close.lt(open_).astype("int8")
    out["is_flat_k"] = close.eq(open_).astype("int8")

    volume_ma5 = volume.rolling(5, min_periods=5).mean()
    volume_ma10 = volume.rolling(10, min_periods=10).mean()
    out["volume_ma5"] = volume_ma5
    out["volume_ma10"] = volume_ma10
    out["volume_ratio_ma5"] = _safe_div(volume, volume_ma5)
    out["volume_ratio_ma10"] = _safe_div(volume, volume_ma10)
    out["volume_ratio_prev1"] = _safe_div(volume, prev_volume)

    out["ma5"] = close.rolling(5, min_periods=5).mean()
    out["ma10"] = close.rolling(10, min_periods=10).mean()
    out["ma20"] = close.rolling(20, min_periods=20).mean()
    out["ma60"] = close.rolling(60, min_periods=60).mean()

    ll9 = low.rolling(9, min_periods=9).min()
    hh9 = high.rolling(9, min_periods=9).max()
    r9 = hh9 - ll9
    rsv = _safe_div(close - ll9, r9, fill_when_zero=0.0) * 100.0

    out["kdj_k"] = _tdx_sma(rsv, 3, 1)
    out["kdj_d"] = _tdx_sma(out["kdj_k"], 3, 1)
    out["kdj_j"] = 3.0 * out["kdj_k"] - 2.0 * out["kdj_d"]

    ema12 = _tdx_ema(close, 12)
    ema26 = _tdx_ema(close, 26)
    out["macd_dif"] = ema12 - ema26
    out["macd_dea"] = _tdx_ema(out["macd_dif"], 9)
    out["macd_hist"] = (out["macd_dif"] - out["macd_dea"]) * 2.0

    rh20 = high.rolling(20, min_periods=20).max()
    rl20 = low.rolling(20, min_periods=20).min()
    rhgt = rh20 - rl20

    rw20 = _safe_div(rhgt, rl20, fill_when_zero=999.0)
    pos20 = _safe_div(close - rl20, rhgt, fill_when_zero=999.0)

    pnlow = low.shift(1).rolling(20, min_periods=20).min()
    dplow = _safe_div(close - pnlow, pnlow, fill_when_zero=999.0)

    bbi = (
        close.rolling(3, min_periods=3).mean()
        + close.rolling(6, min_periods=6).mean()
        + close.rolling(12, min_periods=12).mean()
        + close.rolling(24, min_periods=24).mean()
    ) / 4.0
    zsl = _tdx_ema(close, 21)

    dma20 = _safe_div((close - out["ma20"]).abs(), out["ma20"], fill_when_zero=999.0)
    dbbi = _safe_div((close - bbi).abs(), bbi, fill_when_zero=999.0)
    dzsl = _safe_div((close - zsl).abs(), zsl, fill_when_zero=999.0)

    day_range = high - low
    upsh = _safe_div(high - max_oc, day_range, fill_when_zero=999.0)

    lowvc = _rolling_low_volume_count_20(volume)

    j = _to_num(out["kdj_j"])
    vr5 = _to_num(out["volume_ratio_ma5"])

    b1_low_position = pos20.le(0.25)
    b1_near_previous_low = dplow.abs().le(0.02)
    b1_bottom_zone = rw20.le(0.25) & pos20.le(0.25) & close.ge(rl20 * 0.98)
    b1_ma_support = dma20.le(0.02) | dbbi.le(0.02) | dzsl.le(0.02)
    b1_position_ok = b1_low_position | b1_near_previous_low | b1_bottom_zone | b1_ma_support
    b1_j_ok = j.le(14.0)
    b1_low_volume = lowvc.ge(16.0) | vr5.le(0.70)
    b1_not_break_previous_low = low.ge(pnlow * 0.98) & close.ge(pnlow)
    b1_signal = b1_position_ok & b1_j_ok & b1_low_volume & b1_not_break_previous_low

    b1_recent = pd.Series(False, index=out.index)
    for lag in range(1, 6):
        b1_recent = b1_recent | b1_signal.shift(lag, fill_value=False)

    b2_return_gt_4 = _to_num(out["daily_return_pct"]).gt(4.0)
    b2_bullish_candle = close.gt(open_)
    b2_volume_gt_prev = volume.gt(prev_volume)
    b2_j_below_55 = j.lt(55.0)
    b2_upper_shadow_ok = upsh.le(0.25)

    xg = (
        b1_recent
        & b2_return_gt_4
        & b2_bullish_candle
        & b2_volume_gt_prev
        & b2_j_below_55
        & b2_upper_shadow_ok
    )

    out["b1_tdx_range_high_20"] = rh20
    out["b1_tdx_range_low_20"] = rl20
    out["b1_tdx_range_width_20"] = rw20
    out["b1_tdx_position_20"] = pos20
    out["b1_tdx_previous_20d_low"] = pnlow
    out["b1_tdx_close_to_previous_20d_low"] = dplow
    out["b1_tdx_bbi"] = bbi
    out["b1_tdx_zsl_ema21"] = zsl
    out["b1_tdx_distance_to_ma20"] = dma20
    out["b1_tdx_distance_to_bbi"] = dbbi
    out["b1_tdx_distance_to_zsl"] = dzsl
    out["b1_tdx_low_volume_count_20"] = lowvc

    out["b1_tdx_low_position"] = b1_low_position.fillna(False).astype("int8")
    out["b1_tdx_near_previous_low"] = b1_near_previous_low.fillna(False).astype("int8")
    out["b1_tdx_bottom_zone"] = b1_bottom_zone.fillna(False).astype("int8")
    out["b1_tdx_ma_support"] = b1_ma_support.fillna(False).astype("int8")
    out["b1_tdx_position_ok"] = b1_position_ok.fillna(False).astype("int8")
    out["b1_tdx_j_ok"] = b1_j_ok.fillna(False).astype("int8")
    out["b1_tdx_low_volume"] = b1_low_volume.fillna(False).astype("int8")
    out["b1_tdx_not_break_previous_low"] = b1_not_break_previous_low.fillna(False).astype("int8")
    out["b1_tdx_signal"] = b1_signal.fillna(False).astype("int8")
    out["b1_tdx_recent_1_to_5"] = b1_recent.fillna(False).astype("int8")

    out["b2_tdx_return_gt_4"] = b2_return_gt_4.fillna(False).astype("int8")
    out["b2_tdx_bullish_candle"] = b2_bullish_candle.fillna(False).astype("int8")
    out["b2_tdx_volume_gt_prev"] = b2_volume_gt_prev.fillna(False).astype("int8")
    out["b2_tdx_j_below_55"] = b2_j_below_55.fillna(False).astype("int8")
    out["b2_tdx_upper_shadow_ratio"] = upsh
    out["b2_tdx_upper_shadow_ok"] = b2_upper_shadow_ok.fillna(False).astype("int8")
    out["_selected"] = xg.fillna(False).astype("int8")

    b1_pos = np.where(out["b1_tdx_signal"].to_numpy(dtype=bool), np.arange(len(out)), np.nan)
    last_b1_pos = pd.Series(b1_pos, index=out.index).ffill().shift(1)

    out["b1_lag_trade_days"] = pd.Series(np.arange(len(out)), index=out.index) - last_b1_pos

    b1_date = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    valid_last = last_b1_pos.notna()
    if valid_last.any():
        b1_date.loc[valid_last] = out["date"].iloc[last_b1_pos.loc[valid_last].astype(int).to_numpy()].to_numpy()
    out["b1_date"] = b1_date

    return out


def select(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Return rows that satisfy the TDX B2 confirm formula."""
    full = add_tdx_features(df)
    selected = full.loc[full["_selected"].eq(1)].copy()

    if selected.empty:
        return pd.DataFrame(columns=[c for c in OUTPUT_FRONT_COLUMNS if c in full.columns])

    selected["selection_strategy"] = STRATEGY_NAME
    selected["b1_source_strategy"] = "tdx_formula_inline_from_daily_cache"
    selected = selected.drop(columns=["_selected"], errors="ignore")

    keep_cols = [c for c in OUTPUT_FRONT_COLUMNS if c in selected.columns]
    extra_cols = [c for c in selected.columns if c not in keep_cols]

    return selected[[*keep_cols, *extra_cols]].copy()


SELECT_FUNC = select
