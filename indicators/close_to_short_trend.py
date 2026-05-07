from __future__ import annotations

import numpy as np
import pandas as pd

OUTPUT_COLUMNS = {
    "close_to_short_trend", "close_to_short_trend_below_limit",
    "close_to_short_trend_below_084", "close_to_short_trend_below_086",
    "close_to_short_trend_below_088", "close_to_short_trend_below_090",
    "close_to_short_trend_below_095", "close_to_short_trend_below_100",
    "close_to_short_trend_below_105", "below_short_trend", "not_below_short_trend",
    "close_below_short_trend_cap_calc", "price_rise_range_and_close_to_short_trend_below_limit",
}


def add_close_to_short_trend(df: pd.DataFrame, max_close_to_short_trend: float = 0.95) -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    short_trend = pd.to_numeric(out["short_trend"], errors="coerce")
    valid = short_trend > 0
    out["close_to_short_trend"] = close / short_trend.replace(0, np.nan)
    out["close_to_short_trend_below_limit"] = (valid & out["close_to_short_trend"].notna() & (out["close_to_short_trend"] < max_close_to_short_trend)).fillna(False)
    thresholds = {
        "084": 0.84,
        "086": 0.86,
        "088": 0.88,
        "090": 0.90,
        "095": 0.95,
        "100": 1.00,
        "105": 1.05,
    }
    for suffix, v in thresholds.items():
        out[f"close_to_short_trend_below_{suffix}"] = (
            valid
            & out["close_to_short_trend"].notna()
            & (out["close_to_short_trend"] < v)
        ).fillna(False)
    out["below_short_trend"] = (valid & out["close_to_short_trend"].notna() & (out["close_to_short_trend"] < 1.0)).fillna(False)
    out["not_below_short_trend"] = (valid & out["close_to_short_trend"].notna() & (out["close_to_short_trend"] >= 1.0)).fillna(False)
    out["close_below_short_trend_cap_calc"] = out["below_short_trend"]
    if "price_rise_in_range" in out.columns:
        out["price_rise_range_and_close_to_short_trend_below_limit"] = (
            out["price_rise_in_range"].fillna(False).astype(bool)
            & out["close_to_short_trend_below_limit"].fillna(False).astype(bool)
        ).fillna(False)
    return out
