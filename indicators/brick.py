from __future__ import annotations

import numpy as np
import pandas as pd

from .core import hhv, llv, ref, sma_tdx


def add_brick_indicators(df: pd.DataFrame, n1: int = 4, n2: int = 6) -> pd.DataFrame:
    """
    Add the brick momentum indicator and its structural flags.

    Quant details:
    - var1a = (HHV(high, n1) - close) / (HHV(high, n1) - LLV(low, n1)) * 100 - 90.
    - var2a = SMA(var1a, n1, 1) + 100.
    - var3a = (close - LLV(low, n1)) / (HHV(high, n1) - LLV(low, n1)) * 100.
    - var4a = SMA(var3a, n2, 1).
    - var5a = SMA(var4a, n2, 1) + 100.
    - var6a = var5a - var2a.
    - brick_value = max(var6a - n1, 0) when var6a > n1, otherwise 0.
    - green_to_red is true when the previous brick direction was not rising and the current brick direction is rising.
    - current_red_height = brick_value - previous brick_value.
    - previous_green_height = brick_value two bars ago - brick_value one bar ago.
    - brick_reversal_strength is true when the red brick height is greater than 70% of the previous green brick height.
    """
    df = df.copy()
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    highest_high = hhv(high, n1)
    lowest_low = llv(low, n1)
    range_n = (highest_high - lowest_low).replace(0, np.nan)

    df["var1a"] = (highest_high - close) / range_n * 100 - 90
    df["var2a"] = sma_tdx(df["var1a"], n1, 1) + 100
    df["var3a"] = (close - lowest_low) / range_n * 100
    df["var4a"] = sma_tdx(df["var3a"], n2, 1)
    df["var5a"] = sma_tdx(df["var4a"], n2, 1) + 100
    df["var6a"] = df["var5a"] - df["var2a"]
    df["brick_value"] = np.where(df["var6a"] > n1, df["var6a"] - n1, 0.0)

    prev2 = ref(df["brick_value"], 2)
    prev1 = ref(df["brick_value"], 1)
    rising_now = prev1 < df["brick_value"]

    df["green_brick"] = prev1 > df["brick_value"]
    df["red_brick"] = rising_now
    df["brick_open"] = prev1
    df["brick_close"] = df["brick_value"]
    df["brick_delta"] = df["brick_value"] - prev1
    df["green_to_red"] = (rising_now.shift(1) == 0) & rising_now
    df["valid_red_brick"] = df["brick_value"] > 0
    df["current_red_height"] = df["brick_value"] - prev1
    df["valid_green_brick"] = (prev2 > prev1) & ((prev2 - prev1) > 0)
    df["previous_green_height"] = prev2 - prev1
    df["green_height_70pct"] = df["previous_green_height"] * 0.7
    df["brick_reversal_strength"] = df["current_red_height"] > df["green_height_70pct"]
    df["hard_brick_turn_strong"] = (
    (prev2 > prev1)
    & (prev1 < df["brick_value"])
    & (
        df["brick_value"]
        > prev1 + (prev2 - prev1) * 0.7
    )
)
    return df
