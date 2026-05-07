from __future__ import annotations

import pandas as pd

OUTPUT_COLUMNS = {"small_price_rise", "long_red_brick", "small_rise_long_red_brick"}


def add_small_rise_long_red_brick(
    df: pd.DataFrame,
    small_rise_min_pct: float = 0.0,
    small_rise_max_pct: float = 3.0,
    long_red_ratio: float = 1.3,
) -> pd.DataFrame:
    """Add low daily rise + long current red brick flag used by renko v1."""
    out = df.copy()
    daily_return_pct = pd.to_numeric(out["daily_return_pct"], errors="coerce")
    current_red_height = pd.to_numeric(out["current_red_height"], errors="coerce")
    red_height_reference = pd.to_numeric(out["red_height_reference"], errors="coerce")

    out["small_price_rise"] = (
        (daily_return_pct > float(small_rise_min_pct))
        & (daily_return_pct <= float(small_rise_max_pct))
    ).fillna(False)
    out["long_red_brick"] = (
        (current_red_height > 0)
        & red_height_reference.notna()
        & (current_red_height >= red_height_reference * float(long_red_ratio))
    ).fillna(False)
    out["small_rise_long_red_brick"] = (
        out["small_price_rise"].astype(bool) & out["long_red_brick"].astype(bool)
    ).fillna(False)
    return out
