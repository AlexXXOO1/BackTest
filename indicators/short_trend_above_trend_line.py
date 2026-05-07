from __future__ import annotations

import pandas as pd

OUTPUT_COLUMNS = {"short_trend_above_trend_line"}


def add_short_trend_above_trend_line(df: pd.DataFrame) -> pd.DataFrame:
    """Add whether short_trend is above trend_line."""
    out = df.copy()
    short_trend = pd.to_numeric(out["short_trend"], errors="coerce")
    trend_line = pd.to_numeric(out["trend_line"], errors="coerce")
    out["short_trend_above_trend_line"] = (short_trend > trend_line).fillna(False)
    return out
