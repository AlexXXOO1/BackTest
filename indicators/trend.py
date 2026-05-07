from __future__ import annotations

import pandas as pd

from .core import ma


Z_FAST_TREND_SPAN = 8
Z_SLOW_TREND_SPAN = 21


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add reusable trend-line metrics only.

    Existing legacy trend fields are preserved:
    - short_trend = EMA(EMA(close, 10), 10).
    - trend_line = average of MA(close, 14), MA(close, 28), MA(close, 57), and MA(close, 114).
    - yellow_ma = MA(close, 20).
    - short_trend_cap = short_trend * 1.02.

    New project-standard yellow/white-line style fields:
    - z_fast_trend_line = EMA(close, 8).
    - z_slow_trend_line = EMA(close, 21).

    Notes:
    - z_fast_trend_line / z_slow_trend_line are independent indicator fields.
      They are not direct aliases of the legacy short_trend / trend_line fields.
    - Keep strategy decisions such as price above/below a line in the strategy layer.
    """
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")

    # Legacy lines: keep unchanged for backward compatibility.
    out["short_trend"] = close.ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    out["trend_line"] = (ma(close, 14) + ma(close, 28) + ma(close, 57) + ma(close, 114)) / 4
    out["yellow_ma"] = ma(close, 20)
    out["short_trend_cap"] = out["short_trend"] * 1.02

    # New standard naming for the custom yellow/white-line concept.
    # These are computed independently and are intentionally not simple aliases
    # of short_trend / trend_line.
    out["z_fast_trend_line"] = close.ewm(span=Z_FAST_TREND_SPAN, adjust=False).mean()
    out["z_slow_trend_line"] = close.ewm(span=Z_SLOW_TREND_SPAN, adjust=False).mean()

    return out
