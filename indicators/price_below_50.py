from __future__ import annotations

import pandas as pd

OUTPUT_COLUMNS = {"price_below_50"}


def add_price_below_50(df: pd.DataFrame, max_price: float = 50.0) -> pd.DataFrame:
    """Add low-absolute-price flag used by legacy renko v1 scoring."""
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    out["price_below_50"] = (close < float(max_price)).fillna(False)
    return out
