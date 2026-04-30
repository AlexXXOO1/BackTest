from __future__ import annotations

import numpy as np
import pandas as pd

from .core import hhv, llv, ref, sma_tdx


def add_brick_indicators(df: pd.DataFrame, n1: int = 4, n2: int = 6) -> pd.DataFrame:
    """
    Add raw brick-chart metrics only.

    Quant details:
    - var1a = (HHV(high, n1) - close) / (HHV(high, n1) - LLV(low, n1)) * 100 - 90.
    - var2a = SMA(var1a, n1, 1) + 100.
    - var3a = (close - LLV(low, n1)) / (HHV(high, n1) - LLV(low, n1)) * 100.
    - var4a = SMA(var3a, n2, 1).
    - var5a = SMA(var4a, n2, 1) + 100.
    - var6a = var5a - var2a.
    - brick_value = max(var6a - n1, 0) when var6a > n1, otherwise 0.

    This indicator module intentionally does not create selection booleans such
    as hard_brick_turn_strong. Strategy modules should decide whether a brick
    sequence is turning strong, rising, falling, valid, or selected.
    """
    out = df.copy()
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")

    highest_high = hhv(high, n1)
    lowest_low = llv(low, n1)
    range_n = (highest_high - lowest_low).replace(0, np.nan)

    out["var1a"] = (highest_high - close) / range_n * 100 - 90
    out["var2a"] = sma_tdx(out["var1a"], n1, 1) + 100
    out["var3a"] = (close - lowest_low) / range_n * 100
    out["var4a"] = sma_tdx(out["var3a"], n2, 1)
    out["var5a"] = sma_tdx(out["var4a"], n2, 1) + 100
    out["var6a"] = out["var5a"] - out["var2a"]
    out["brick_value"] = np.where(out["var6a"] > n1, out["var6a"] - n1, 0.0)

    out["brick_prev_1"] = ref(out["brick_value"], 1)
    out["brick_prev_2"] = ref(out["brick_value"], 2)
    out["brick_open"] = out["brick_prev_1"]
    out["brick_close"] = out["brick_value"]
    out["brick_delta"] = out["brick_value"] - out["brick_prev_1"]
    out["current_red_height"] = out["brick_value"] - out["brick_prev_1"]
    out["previous_green_height"] = out["brick_prev_2"] - out["brick_prev_1"]
    return out
