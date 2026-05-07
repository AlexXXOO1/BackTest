from __future__ import annotations

import pandas as pd

OUTPUT_COLUMNS = {"close_below_short_trend_cap", "close_above_short_trend_cap"}


def add_close_below_short_trend_cap(df: pd.DataFrame) -> pd.DataFrame:
    """Add close <= short_trend_cap and its inverse flag."""
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    short_trend_cap = pd.to_numeric(out["short_trend_cap"], errors="coerce")
    out["close_below_short_trend_cap"] = (close <= short_trend_cap).fillna(False)
    out["close_above_short_trend_cap"] = (close > short_trend_cap).fillna(False)
    return out
