# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Basic brick-chart / Renko-like indicator translated from the original TDX formula.

Only the algorithmic value is cached:
    - renko_value: original 砖型图 value

No color/state columns and no strategy signals are produced here. Do not add
color/state fields, selected, score, turn_red, turn_strong, AA, BB, CC,
relax_level, or any v0/v4-specific columns here.
"""

import numpy as np
import pandas as pd


def tdx_sma(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """
    TongDaXin SMA(X, N, M).

    Recursive formula:
        Y = (M * X + (N - M) * REF(Y, 1)) / N

    This is not pandas rolling mean.
    """
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    prev = np.nan

    for i, value in enumerate(values):
        if np.isnan(value):
            out[i] = prev
            continue

        if np.isnan(prev):
            prev = value
        else:
            prev = (m * value + (n - m) * prev) / n

        out[i] = prev

    return pd.Series(out, index=series.index)


def add_renko_basic_indicator(
    df: pd.DataFrame,
    n1: int = 4,
    n2: int = 6,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    prefix: str = "renko",
) -> pd.DataFrame:
    """
    Add the original brick-chart algorithm into the basic indicator cache.

    Original formula:
        VAR1A := (HHV(HIGH,N1)-CLOSE)/(HHV(HIGH,N1)-LLV(LOW,N1))*100-90
        VAR2A := SMA(VAR1A,N1,1)+100
        VAR3A := (CLOSE-LLV(LOW,N1))/(HHV(HIGH,N1)-LLV(LOW,N1))*100
        VAR4A := SMA(VAR3A,N2,1)
        VAR5A := SMA(VAR4A,N2,1)+100
        VAR6A := VAR5A-VAR2A
        砖型图 := IF(VAR6A>N1, VAR6A-N1, 0)

    Cached columns:
        {prefix}_value

    If a strategy needs red/green state, derive it from {prefix}_value:
        red   when REF(砖型图, 1) < 砖型图
        green when REF(砖型图, 1) > 砖型图
    """
    out = df.copy()

    n1 = int(n1)
    n2 = int(n2)
    if n1 <= 0:
        raise ValueError("n1 must be positive for renko basic indicator.")
    if n2 <= 0:
        raise ValueError("n2 must be positive for renko basic indicator.")

    high = pd.to_numeric(out[high_col], errors="coerce")
    low = pd.to_numeric(out[low_col], errors="coerce")
    close = pd.to_numeric(out[close_col], errors="coerce")

    hhv_high_n1 = high.rolling(window=n1, min_periods=n1).max()
    llv_low_n1 = low.rolling(window=n1, min_periods=n1).min()

    price_range = (hhv_high_n1 - llv_low_n1).replace(0, np.nan)

    var1a = (hhv_high_n1 - close) / price_range * 100.0 - 90.0
    var2a = tdx_sma(var1a, n=n1, m=1) + 100.0

    var3a = (close - llv_low_n1) / price_range * 100.0
    var4a = tdx_sma(var3a, n=n2, m=1)
    var5a = tdx_sma(var4a, n=n2, m=1) + 100.0

    var6a = var5a - var2a

    value = pd.Series(
        np.where(var6a > n1, var6a - n1, 0.0),
        index=out.index,
        dtype="float64",
    )

    out[f"{prefix}_value"] = value

    return out
