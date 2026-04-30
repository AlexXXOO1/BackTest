from __future__ import annotations

import pandas as pd

from .core import ma


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add raw trend metrics only.

    Quant details:
    - short_trend = EMA(EMA(close, 10), 10).
    - trend_line = average of MA(close, 14), MA(close, 28), MA(close, 57), and MA(close, 114).
    - yellow_ma = MA(close, 20).
    - short_trend_cap = short_trend * 1.02.

    This indicator module does not decide whether price is above a trend line,
    below a cap, or inside a price zone. Those booleans belong to the strategy
    layer.
    """
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    out["short_trend"] = close.ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    out["trend_line"] = (ma(close, 14) + ma(close, 28) + ma(close, 57) + ma(close, 114)) / 4
    out["yellow_ma"] = ma(close, 20)
    out["short_trend_cap"] = out["short_trend"] * 1.02
    return out
