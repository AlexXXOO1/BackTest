from __future__ import annotations

import pandas as pd


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume and pullback indicators.

    Quant details:
    - vol_ma5 = MA(volume, 5).
    - volume_confirm is true when volume is greater than the previous day's volume and greater than vol_ma5.
    - surge_then_shrink_pullback is true when the stock rose two bars ago with volume above 1.2 times vol_ma5, then pulled back yesterday with volume below 0.8 times the volume from two bars ago.
    """
    df = df.copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    df["vol_ma5"] = volume.rolling(window=5, min_periods=5).mean()
    df["volume_above_prev"] = volume > volume.shift(1)
    df["volume_above_ma5"] = volume > df["vol_ma5"]
    df["volume_confirm"] = df["volume_above_prev"] & df["volume_above_ma5"]
    df["surge_then_shrink_pullback"] = (
        (close.shift(2) > close.shift(3))
        & (volume.shift(2) > df["vol_ma5"].shift(2) * 1.2)
        & (close.shift(1) < close.shift(2))
        & (volume.shift(1) < volume.shift(2) * 0.8)
    )
    return df
