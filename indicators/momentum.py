from __future__ import annotations

import numpy as np
import pandas as pd

from .core import sma_tdx


def add_kdj_indicators(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3, low_j_threshold: float = 14.0) -> pd.DataFrame:
    """
    Add raw KDJ values only.

    Quant details:
    - RSV = (close - LLV(low, n)) / (HHV(high, n) - LLV(low, n)) * 100.
    - K = TDX SMA(RSV, m1, 1).
    - D = TDX SMA(K, m2, 1).
    - J = 3 * K - 2 * D.

    low_j_threshold is kept in the signature for backward compatibility, but
    this function no longer creates J-condition booleans. Strategy modules
    should decide whether J is low, rising, or inside a selected range.
    """
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    lowest_low = low.rolling(window=n, min_periods=n).min()
    highest_high = high.rolling(window=n, min_periods=n).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
    out["K"] = sma_tdx(rsv, m1, 1)
    out["D"] = sma_tdx(out["K"], m2, 1)
    out["J"] = 3 * out["K"] - 2 * out["D"]
    return out


def add_macd_indicators(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Add raw MACD values only.

    Quant details:
    - diff = EMA(close, fast) - EMA(close, slow).
    - dea = EMA(diff, signal).
    - macd = 2 * (diff - dea).
    """
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    out["diff"] = ema_fast - ema_slow
    out["dea"] = out["diff"].ewm(span=signal, adjust=False).mean()
    out["macd"] = 2 * (out["diff"] - out["dea"])
    return out
