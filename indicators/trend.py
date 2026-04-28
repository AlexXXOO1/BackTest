from __future__ import annotations

import pandas as pd

from .core import ma


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trend indicators used by the v1-v8 strategy chain.

    Quant details:
    - short_trend = EMA(EMA(close, 10), 10).
    - trend_line = average of MA(close, 14), MA(close, 28), MA(close, 57), and MA(close, 114).
    - yellow_ma = MA(close, 20).
    - two_day_above_trend_line is true when close is above trend_line today and yesterday.
    - short_trend_above_trend_line is true when short_trend is greater than trend_line.
    - close_below_short_trend_cap is true when close is below short_trend * 1.02.
    - price_below_50 is true when close is below 50.
    - close_above_yellow_ma is true when close is above yellow_ma.
    """
    df = df.copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    df["short_trend"] = close.ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    df["trend_line"] = (ma(close, 14) + ma(close, 28) + ma(close, 57) + ma(close, 114)) / 4
    df["yellow_ma"] = ma(close, 20)
    df["short_trend_cap"] = df["short_trend"] * 1.02
    df["above_trend_line_today"] = close > df["trend_line"]
    df["above_trend_line_prev"] = close.shift(1) > df["trend_line"].shift(1)
    df["two_day_above_trend_line"] = df["above_trend_line_today"] & df["above_trend_line_prev"]
    df["short_trend_above_trend_line"] = df["short_trend"] > df["trend_line"]
    df["close_below_short_trend_cap"] = close < df["short_trend_cap"]
    df["price_below_50"] = close < 50
    df["price_zone_ok"] = df["two_day_above_trend_line"] & df["close_below_short_trend_cap"]
    df["trend_condition_ok"] = df["short_trend_above_trend_line"]
    df["price_condition_ok"] = df["price_below_50"]
    df["close_above_yellow_ma"] = close > df["yellow_ma"]
    return df
