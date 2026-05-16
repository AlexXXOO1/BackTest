# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Basic reusable indicator layer.

Principle:
- Only calculate stable, generic, reusable facts.
- Do NOT calculate strategy decisions, selected flags, scores, renko signals,
  market regime, distance-to-line conditions, or any v0/v4/v5-specific columns here.

Input columns expected:
    symbol, file, date, open, high, low, close, volume, amount

Output columns include:
    base OHLCVA fields
    K-line features
    MA
    volume MA / volume ratios
    MACD
    basic brick-chart / renko algorithm
"""

from typing import Iterable

import numpy as np
import pandas as pd

from .renko_basic import add_renko_basic_indicator
from .auto_loader import apply_auto_indicators


BASE_COLUMNS = [
    "symbol",
    "file",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

DEFAULT_MA_WINDOWS = (5, 10, 20, 60)
DEFAULT_VOLUME_WINDOWS = (5, 10)


def _safe_div(a, b):
    """Safe vectorized division. 0 denominator becomes NaN."""
    if not isinstance(a, pd.Series):
        a = pd.Series(a)
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=a.index)
    return a / b.replace(0, np.nan)


def _ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in BASE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out[out["close"] > 0]
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    return out[BASE_COLUMNS].copy()


def add_kline_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add generic K-line features."""
    out = df.copy()

    open_ = out["open"]
    high = out["high"]
    low = out["low"]
    close = out["close"]
    volume = out["volume"]
    prev_close = close.shift(1)
    prev_volume = volume.shift(1)

    out["prev_close"] = prev_close
    out["prev_volume"] = prev_volume

    out["daily_return_pct"] = (_safe_div(close, prev_close) - 1.0) * 100.0
    out["intraday_return_pct"] = (_safe_div(close, open_) - 1.0) * 100.0

    # Standard A-share amplitude definition:
    # amplitude = (high - low) / previous close * 100
    out["amplitude_pct"] = _safe_div(high - low, prev_close) * 100.0

    # Keep K-line percentage factors on the same previous-close denominator.
    out["body_pct"] = _safe_div(close - open_, prev_close) * 100.0
    out["body_abs_pct"] = out["body_pct"].abs()

    max_oc = pd.concat([open_, close], axis=1).max(axis=1)
    min_oc = pd.concat([open_, close], axis=1).min(axis=1)

    out["upper_shadow_pct"] = _safe_div(high - max_oc, prev_close) * 100.0
    out["lower_shadow_pct"] = _safe_div(min_oc - low, prev_close) * 100.0

    out["is_red_k"] = (close > open_).astype(int)
    out["is_green_k"] = (close < open_).astype(int)
    out["is_flat_k"] = (close == open_).astype(int)

    return out


def add_ma_indicators(
    df: pd.DataFrame,
    ma_windows: Iterable[int] = DEFAULT_MA_WINDOWS,
) -> pd.DataFrame:
    """
    Add close moving averages only.

    注意：
    close_to_maX 属于判断条件 / 分析条件，不再写入基础 indicator cache。
    如果策略或分析需要距离均线，直接在 strategy/analyze_tools 中临时计算：
        close_to_ma5 = (close / ma5 - 1.0) * 100.0
    """
    out = df.copy()
    close = out["close"]

    for window in ma_windows:
        w = int(window)
        if w <= 0:
            continue

        ma_col = f"ma{w}"
        out[ma_col] = close.rolling(w, min_periods=max(1, w // 3)).mean()

    return out


def add_volume_indicators(
    df: pd.DataFrame,
    volume_windows: Iterable[int] = DEFAULT_VOLUME_WINDOWS,
) -> pd.DataFrame:
    """Add generic volume moving averages and volume ratios."""
    out = df.copy()
    volume = out["volume"]

    out["volume_ratio_prev1"] = _safe_div(volume, volume.shift(1))

    for window in volume_windows:
        w = int(window)
        if w <= 0:
            continue

        ma_col = f"volume_ma{w}"
        ratio_col = f"volume_ratio_ma{w}"

        out[ma_col] = volume.rolling(w, min_periods=max(1, w // 3)).mean()
        out[ratio_col] = _safe_div(volume, out[ma_col])

    return out


def add_macd_indicators(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Add standard MACD columns.

    Naming:
    - macd_dif: EMA(close, fast) - EMA(close, slow)
    - macd_dea: EMA(macd_dif, signal)
    - macd_hist: DIF - DEA

    Note:
    Some software displays MACD bar as 2 * (DIF - DEA).
    Here macd_hist uses the cleaner one-times version. If you need the TDX-style
    doubled bar later, add macd_bar = 2 * macd_hist in strategy or analysis.
    """
    out = df.copy()
    close = out["close"]

    ema_fast = close.ewm(span=int(fast), adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=int(slow), adjust=False, min_periods=1).mean()

    out["macd_dif"] = ema_fast - ema_slow
    out["macd_dea"] = out["macd_dif"].ewm(span=int(signal), adjust=False, min_periods=1).mean()
    out["macd_hist"] = out["macd_dif"] - out["macd_dea"]

    return out


def add_all_indicators(
    df: pd.DataFrame,
    n1: int = 4,
    n2: int = 6,
    ma_windows: Iterable[int] = DEFAULT_MA_WINDOWS,
    volume_windows: Iterable[int] = DEFAULT_VOLUME_WINDOWS,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    **kwargs,
) -> pd.DataFrame:
    """
    Public entry point for IndicatorStore.

    n1/n2 are used by the basic brick-chart / renko formula.
    """
    out = _ensure_base_columns(df)
    out = add_kline_indicators(out)
    out = add_ma_indicators(out, ma_windows=ma_windows)
    out = add_volume_indicators(out, volume_windows=volume_windows)
    out = add_macd_indicators(out, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    out = add_renko_basic_indicator(out, n1=n1, n2=n2, prefix="renko")
    out = apply_auto_indicators(out, **kwargs)

    # Keep deterministic column order.
    ordered_cols = [
        "symbol", "file", "date", "open", "high", "low", "close", "volume", "amount",
        "daily_return_pct", "intraday_return_pct", "amplitude_pct",
        "upper_shadow_pct", "lower_shadow_pct", "body_pct", "body_abs_pct",
        "is_red_k", "is_green_k", "is_flat_k",
    ]

    for w in ma_windows:
        w = int(w)
        ordered_cols.append(f"ma{w}")

    ordered_cols.append("volume_ratio_prev1")

    for w in volume_windows:
        w = int(w)
        ordered_cols.extend([f"volume_ma{w}", f"volume_ratio_ma{w}"])

    ordered_cols.extend(["macd_dif", "macd_dea", "macd_hist"])
    ordered_cols.extend(["renko_value"])

    ordered_cols = [c for c in ordered_cols if c in out.columns]
    extra_cols = [c for c in out.columns if c not in ordered_cols]

    return out[ordered_cols + extra_cols].copy()
