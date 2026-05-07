from __future__ import annotations

import pandas as pd

OUTPUT_COLUMNS = {
    "close_above_trend_line",
    "close_prev_1_above_trend_line_prev_1",
    "two_day_above_trend_line",
}


def add_two_day_above_trend_line(df: pd.DataFrame) -> pd.DataFrame:
    """Add reusable close-above-trend-line flags."""
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    trend_line = pd.to_numeric(out["trend_line"], errors="coerce")

    out["close_above_trend_line"] = (close > trend_line).fillna(False)
    out["close_prev_1_above_trend_line_prev_1"] = (
        close.shift(1) > trend_line.shift(1)
    ).fillna(False)
    out["two_day_above_trend_line"] = (
        out["close_above_trend_line"].astype(bool)
        & out["close_prev_1_above_trend_line_prev_1"].astype(bool)
    ).fillna(False)
    return out
