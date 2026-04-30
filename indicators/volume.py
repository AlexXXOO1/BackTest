from __future__ import annotations

import pandas as pd


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add raw volume metrics only.

    Quant details:
    - vol_ma5 = MA(volume, 5).
    - volume_prev_1 = previous bar volume.
    - volume_prev_2 = volume from two bars ago.

    This indicator module does not create volume-confirmation or pullback
    booleans. Strategy modules should decide whether a volume pattern is a
    valid selection condition.
    """
    out = df.copy()
    volume = pd.to_numeric(out["volume"], errors="coerce")
    out["vol_ma5"] = volume.rolling(window=5, min_periods=5).mean()
    out["volume_prev_1"] = volume.shift(1)
    out["volume_prev_2"] = volume.shift(2)
    return out
