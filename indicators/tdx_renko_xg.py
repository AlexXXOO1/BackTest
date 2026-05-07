from __future__ import annotations

"""
TongDaXin original renko XG indicator.

This module preserves the earliest v0 formula exactly at the indicator layer:

条件1 := REF(砖型图,2) > REF(砖型图,1)
         AND REF(砖型图,1) < 砖型图
         AND 砖型图 > REF(砖型图,1) + (REF(砖型图,2)-REF(砖型图,1))*0.7;

XG: IF(条件1, 1, 0)

Notes:
- brick_value is the translated 砖型图.
- brick_prev_1 is REF(砖型图, 1).
- brick_prev_2 is REF(砖型图, 2).
- This is a reusable indicator fact, not a strategy decision.
"""

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = {
    "tdx_renko_condition1",
    "tdx_renko_xg",
    "tdx_renko_xg_int",
}


def add_tdx_renko_xg(
    df: pd.DataFrame,
    *,
    reversal_ratio: float = 0.70,
) -> pd.DataFrame:
    """Add the original TongDaXin renko 条件1 / XG columns."""
    out = df.copy()

    brick = pd.to_numeric(out["brick_value"], errors="coerce")
    brick_prev_1 = pd.to_numeric(out["brick_prev_1"], errors="coerce")
    brick_prev_2 = pd.to_numeric(out["brick_prev_2"], errors="coerce")

    # TDX:
    # REF(砖型图,2) > REF(砖型图,1)
    previous_bar_is_green = brick_prev_2 > brick_prev_1

    # TDX:
    # REF(砖型图,1) < 砖型图
    current_bar_is_red = brick_prev_1 < brick

    # TDX:
    # 砖型图 > REF(砖型图,1) + (REF(砖型图,2)-REF(砖型图,1))*0.7
    reversal_is_strong = brick > (
        brick_prev_1 + (brick_prev_2 - brick_prev_1) * float(reversal_ratio)
    )

    out["tdx_renko_condition1"] = (
        previous_bar_is_green & current_bar_is_red & reversal_is_strong
    ).fillna(False)

    out["tdx_renko_xg"] = out["tdx_renko_condition1"].astype(bool)
    out["tdx_renko_xg_int"] = np.where(out["tdx_renko_xg"], 1, 0).astype(int)

    return out
