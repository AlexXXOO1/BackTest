from __future__ import annotations

import pandas as pd

OUTPUT_COLUMNS = {
    "strong_price_rise_below_max", "strong_price_rise_in_range",
    "strong_close_to_short_trend_below_limit",
    "strong_price_rise_range_and_close_to_short_trend_below_limit",
}


def add_strong_market_relaxed_flags(
    df: pd.DataFrame,
    strong_max_daily_return_pct: float = 9.0,
    strong_max_close_to_short_trend: float = 1.00,
) -> pd.DataFrame:
    """Stock-level relaxed flags. Market-regime itself is still merged by v5 from sh#999999."""
    out = df.copy()
    ret = pd.to_numeric(out["daily_return_pct"], errors="coerce")
    short_trend = pd.to_numeric(out["short_trend"], errors="coerce")
    ratio = pd.to_numeric(out["close_to_short_trend"], errors="coerce")
    valid = short_trend > 0

    out["strong_price_rise_below_max"] = (ret <= strong_max_daily_return_pct).fillna(False)
    out["strong_price_rise_in_range"] = (
        out["price_rise_above_min"].fillna(False).astype(bool)
        & out["strong_price_rise_below_max"].fillna(False).astype(bool)
    ).fillna(False)
    out["strong_close_to_short_trend_below_limit"] = (valid & ratio.notna() & (ratio < strong_max_close_to_short_trend)).fillna(False)
    out["strong_price_rise_range_and_close_to_short_trend_below_limit"] = (
        out["strong_price_rise_in_range"].fillna(False).astype(bool)
        & out["strong_close_to_short_trend_below_limit"].fillna(False).astype(bool)
    ).fillna(False)
    return out
