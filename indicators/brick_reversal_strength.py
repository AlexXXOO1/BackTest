from __future__ import annotations

import pandas as pd

OUTPUT_COLUMNS = {
    "brick_reversal_strength_070", "brick_reversal_strength_090",
    "brick_reversal_strength_100", "brick_reversal_strength_120",
    "brick_reversal_strength_below_100", "brick_reversal_strength_below_090",
}


def add_brick_reversal_strength_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ratio = pd.to_numeric(out["brick_reversal_ratio"], errors="coerce")
    out["brick_reversal_strength_070"] = (ratio >= 0.70).fillna(False)
    out["brick_reversal_strength_090"] = (ratio >= 0.90).fillna(False)
    out["brick_reversal_strength_100"] = (ratio >= 1.00).fillna(False)
    out["brick_reversal_strength_120"] = (ratio >= 1.20).fillna(False)
    out["brick_reversal_strength_below_100"] = (ratio.notna() & (ratio < 1.00)).fillna(False)
    out["brick_reversal_strength_below_090"] = (ratio.notna() & (ratio < 0.90)).fillna(False)
    return out
