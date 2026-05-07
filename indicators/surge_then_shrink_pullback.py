from __future__ import annotations

import pandas as pd

OUTPUT_COLUMNS = {"surge_then_shrink_pullback"}


def add_surge_then_shrink_pullback(
    df: pd.DataFrame,
    surge_return_pct: float = 3.0,
    surge_volume_ratio: float = 1.5,
    pullback_max_return_pct: float = 0.0,
    shrink_volume_ratio: float = 1.0,
) -> pd.DataFrame:
    """Add previous-day surge plus current shrink-volume pullback flag."""
    out = df.copy()
    daily_return_pct = pd.to_numeric(out["daily_return_pct"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce")
    volume_prev_1 = pd.to_numeric(out["volume_prev_1"], errors="coerce")
    volume_prev_2 = pd.to_numeric(out["volume_prev_2"], errors="coerce")

    prev_day_surge = (
        (daily_return_pct.shift(1) >= float(surge_return_pct))
        & (volume_prev_1 >= volume_prev_2 * float(surge_volume_ratio))
    ).fillna(False)
    current_pullback = (daily_return_pct <= float(pullback_max_return_pct)).fillna(False)
    current_shrink_volume = (volume <= volume_prev_1 * float(shrink_volume_ratio)).fillna(False)

    out["surge_then_shrink_pullback"] = (
        prev_day_surge.astype(bool)
        & current_pullback.astype(bool)
        & current_shrink_volume.astype(bool)
    ).fillna(False)
    return out
