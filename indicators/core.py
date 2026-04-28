from __future__ import annotations

import numpy as np
import pandas as pd


def ref(series: pd.Series, n: int = 1) -> pd.Series:
    """Return the value from n bars ago, equivalent to TDX REF(X, n)."""
    return series.shift(n)


def hhv(series: pd.Series, n: int) -> pd.Series:
    """Return the highest value over the latest n bars, equivalent to TDX HHV(X, n)."""
    return series.rolling(window=n, min_periods=n).max()


def llv(series: pd.Series, n: int) -> pd.Series:
    """Return the lowest value over the latest n bars, equivalent to TDX LLV(X, n)."""
    return series.rolling(window=n, min_periods=n).min()


def ma(series: pd.Series, n: int) -> pd.Series:
    """Return a simple moving average over n bars."""
    return series.rolling(window=n, min_periods=n).mean()


def sma_tdx(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """
    Return the TDX-style SMA value.

    Quant detail:
    SMA(X, N, M) = (M * X + (N - M) * previous_SMA) / N.
    The first non-null X value is used as the initial SMA value.
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


def pct_change(current: pd.Series, base: pd.Series) -> pd.Series:
    """Return percentage change as (current / base - 1) * 100."""
    safe_base = pd.to_numeric(base, errors="coerce").replace(0, np.nan)
    return (pd.to_numeric(current, errors="coerce") / safe_base - 1) * 100
