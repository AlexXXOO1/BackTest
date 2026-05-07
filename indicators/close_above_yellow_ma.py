from __future__ import annotations

import pandas as pd

OUTPUT_COLUMNS = {"above_yellow_ma", "close_above_yellow_ma"}


def add_close_above_yellow_ma(df: pd.DataFrame) -> pd.DataFrame:
    """Add close > yellow_ma flag and backward-compatible alias."""
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    yellow_ma = pd.to_numeric(out["yellow_ma"], errors="coerce")
    out["close_above_yellow_ma"] = (close > yellow_ma).fillna(False)
    out["above_yellow_ma"] = out["close_above_yellow_ma"]
    return out
