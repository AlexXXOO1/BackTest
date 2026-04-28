from __future__ import annotations

import numpy as np
import pandas as pd

from .core import sma_tdx


def add_kdj_indicators(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3, low_j_threshold: float = 14.0) -> pd.DataFrame:
    """
    Add KDJ momentum indicators.

    Quant details:
    - RSV = (close - LLV(low, n)) / (HHV(high, n) - LLV(low, n)) * 100.
    - K = TDX SMA(RSV, m1, 1).
    - D = TDX SMA(K, m2, 1).
    - J = 3 * K - 2 * D.
    - j_three_day_rising is true when J[t-3] < J[t-2] < J[t-1].
    - j_two_day_rising is true when J[t] > J[t-1] > J[t-2].
    - j_below_14 is true when J < 14.
    - j_momentum_or_low is true when j_two_day_rising is true or J < 14.
    """
    df = df.copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    lowest_low = low.rolling(window=n, min_periods=n).min()
    highest_high = high.rolling(window=n, min_periods=n).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
    df["K"] = sma_tdx(rsv, m1, 1)
    df["D"] = sma_tdx(df["K"], m2, 1)
    df["J"] = 3 * df["K"] - 2 * df["D"]
    df["j_three_day_rising"] = (df["J"].shift(3) < df["J"].shift(2)) & (df["J"].shift(2) < df["J"].shift(1))
    df["j_two_day_rising"] = (df["J"] > df["J"].shift(1)) & (df["J"].shift(1) > df["J"].shift(2))
    df["j_below_14"] = df["J"] < low_j_threshold
    df["j_momentum_or_low"] = df["j_two_day_rising"] | df["j_below_14"]
    return df


def add_macd_indicators(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Add MACD indicators.

    Quant details:
    - diff = EMA(close, 12) - EMA(close, 26).
    - dea = EMA(diff, 9).
    - macd = 2 * (diff - dea).
    - diff_above_zero is true when diff > 0.
    """
    df = df.copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    df["diff"] = ema_fast - ema_slow
    df["dea"] = df["diff"].ewm(span=signal, adjust=False).mean()
    df["macd"] = 2 * (df["diff"] - df["dea"])
    df["diff_above_zero"] = df["diff"] > 0
    return df
