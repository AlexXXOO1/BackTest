from __future__ import annotations

import pandas as pd

OUTPUT_COLUMNS = {"j_low", "j_rising_2d", "j_momentum_or_low"}


def add_j_momentum_or_low(df: pd.DataFrame, low_j_threshold: float = 14.0) -> pd.DataFrame:
    """Add J-low OR two-day-rising momentum flag used by renko v1."""
    out = df.copy()
    j = pd.to_numeric(out["J"], errors="coerce")
    out["j_low"] = (j < float(low_j_threshold)).fillna(False)
    out["j_rising_2d"] = ((j > j.shift(1)) & (j.shift(1) > j.shift(2))).fillna(False)
    out["j_momentum_or_low"] = (
        out["j_low"].astype(bool) | out["j_rising_2d"].astype(bool)
    ).fillna(False)
    return out
