from __future__ import annotations

import pandas as pd

OUTPUT_COLUMNS = {
    "price_rise_above_min", "price_rise_below_max", "price_rise_in_range",
    "daily_return_3_to_4", "daily_return_4_to_5", "daily_return_5_to_6",
    "daily_return_5_to_7", "daily_return_55_to_7", "daily_return_6_to_7",
    "daily_return_6_to_7_exact", "daily_return_7_to_10_5",
}


def add_price_rise_in_range(
    df: pd.DataFrame,
    min_daily_return_pct: float = 3.0,
    max_daily_return_pct: float = 7.0,
) -> pd.DataFrame:
    out = df.copy()
    ret = pd.to_numeric(out["daily_return_pct"], errors="coerce")
    out["price_rise_above_min"] = (ret > min_daily_return_pct).fillna(False)
    out["price_rise_below_max"] = (ret <= max_daily_return_pct).fillna(False)
    out["price_rise_in_range"] = (out["price_rise_above_min"].astype(bool) & out["price_rise_below_max"].astype(bool)).fillna(False)
    out["daily_return_3_to_4"] = ((ret > 3.0) & (ret <= 4.0)).fillna(False)
    out["daily_return_4_to_5"] = ((ret > 4.0) & (ret <= 5.0)).fillna(False)
    out["daily_return_5_to_6"] = ((ret > 5.0) & (ret <= 6.0)).fillna(False)
    out["daily_return_5_to_7"] = ((ret > 5.0) & (ret <= 7.0)).fillna(False)
    out["daily_return_55_to_7"] = ((ret > 5.5) & (ret <= 7.0)).fillna(False)
    out["daily_return_6_to_7"] = ((ret > 6.0) & (ret <= 7.0)).fillna(False)
    out["daily_return_6_to_7_exact"] = out["daily_return_6_to_7"]
    out["daily_return_7_to_10_5"] = ((ret > 7.0) & (ret <= 10.5)).fillna(False)
    return out
