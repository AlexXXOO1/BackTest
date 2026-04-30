from __future__ import annotations

import numpy as np
import pandas as pd

from .core import pct_change


def add_candle_pattern_indicators(
    df: pd.DataFrame,
    hard_lookback: int = 20,
    high_pos_lookback: int = 60,
    high_pos_ratio: float = 0.85,
    accel_lookback: int = 5,
    accel_ret_pct: float = 12.0,
    huge_vol_ma_n: int = 20,
    huge_vol_ratio: float = 2.0,
    big_bear_body_pct: float = 4.0,
    limit_up_pct: float = 9.7,
    shrink_limit_vol_ratio: float = 0.8,
    hammer_lower_shadow_body_ratio: float = 2.0,
    hammer_lower_shadow_range_ratio: float = 0.5,
    hammer_upper_shadow_body_ratio: float = 1.2,
    hammer_max_body_range_ratio: float = 0.4,
) -> pd.DataFrame:
    """
    Add raw candle, return, position, and volume-reference metrics only.

    Quant details:
    - pct_change_close = close / previous_close - 1, expressed as a percentage.
    - high_position_line = rolling high_pos_lookback-day high * high_pos_ratio.
    - accel_return_pct = close / close.shift(accel_lookback) - 1, expressed as a percentage.
    - huge_volume_ma = MA(volume, huge_vol_ma_n).
    - bear_body_pct = (open - close) / previous_close * 100.
    - candle_range = high - low.
    - body_abs = abs(close - open).
    - lower_shadow = min(open, close) - low.
    - upper_shadow = high - max(open, close).
    - shrink_volume_ma5 = MA(volume, 5), used by strategies that define shrink-volume rules.

    Parameters with threshold names are retained for backward compatibility and
    to compute reference lines. This function no longer creates risk or pattern
    booleans such as limit_up, long_lower_shadow_hammer, or prior_20d risks.
    """
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    open_ = pd.to_numeric(out["open"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce")
    prev_close = close.shift(1)

    out["pct_change_close"] = pct_change(close, prev_close)
    out["daily_return_pct"] = out["pct_change_close"]
    out["high_position_line"] = high.rolling(window=high_pos_lookback, min_periods=20).max() * high_pos_ratio
    out["accel_return_pct"] = pct_change(close, close.shift(accel_lookback))
    out["huge_volume_ma"] = volume.rolling(window=huge_vol_ma_n, min_periods=5).mean()
    out["shrink_volume_ma5"] = volume.rolling(window=5, min_periods=5).mean()

    out["candle_range"] = high - low
    out["body_abs"] = (close - open_).abs()
    out["bear_body_pct"] = (open_ - close) / prev_close.replace(0, np.nan) * 100
    out["lower_shadow"] = np.minimum(open_, close) - low
    out["upper_shadow"] = high - np.maximum(open_, close)
    return out
