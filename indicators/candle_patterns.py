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
    Add reusable candle-pattern and price-action indicators.

    Quant details:
    - pct_change_close = close / previous_close - 1, expressed as a percentage.
    - high_position_line = rolling 60-day high * 0.85.
    - high_position is true when close >= high_position_line.
    - accelerated_move is true when the latest 5-bar return is >= 12%.
    - huge_volume is true when volume >= MA(volume, 20) * 2.0.
    - big_bear_body is true when close < open and the bearish body is >= 4% of previous close.
    - accelerated_huge_volume_bear is true when high_position, accelerated_move, huge_volume, and big_bear_body are all true.
    - prior_20d_accelerated_huge_volume_bear is true if accelerated_huge_volume_bear appeared in the prior 20 bars.
    - limit_up is true when pct_change_close >= 9.7%.
    - shrink_volume is true when volume is below 80% of either previous volume or MA(volume, 5).
    - shrink_limit_up is true when limit_up and shrink_volume are both true.
    - prior_20d_shrink_limit_up is true if shrink_limit_up appeared in the prior 20 bars.
    - long_lower_shadow_hammer is true when lower shadow >= body * 2.0, lower shadow >= 50% of full range, upper shadow <= body * 1.2, and body <= 40% of full range.
    - limit_up_red_brick is true when limit_up and red_brick are both true.
    """
    df = df.copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    prev_close = close.shift(1)

    df["pct_change_close"] = pct_change(close, prev_close)
    candle_range = (high - low).replace(0, np.nan)
    body_abs = (close - open_).abs()
    bear_body_pct = (open_ - close) / prev_close.replace(0, np.nan) * 100

    df["high_position_line"] = high.rolling(window=high_pos_lookback, min_periods=20).max() * high_pos_ratio
    df["high_position"] = close >= df["high_position_line"]
    df["accel_return_pct"] = pct_change(close, close.shift(accel_lookback))
    df["accelerated_move"] = df["accel_return_pct"] >= accel_ret_pct
    df["huge_volume_ma"] = volume.rolling(window=huge_vol_ma_n, min_periods=5).mean()
    df["huge_volume"] = volume >= df["huge_volume_ma"] * huge_vol_ratio
    df["big_bear_body"] = (close < open_) & (bear_body_pct >= big_bear_body_pct)
    df["accelerated_huge_volume_bear"] = df["high_position"] & df["accelerated_move"] & df["huge_volume"] & df["big_bear_body"]
    df["prior_20d_accelerated_huge_volume_bear"] = (
        df["accelerated_huge_volume_bear"].shift(1).rolling(window=hard_lookback, min_periods=1).max().fillna(False).astype(bool)
    )

    vol_ma5 = volume.rolling(window=5, min_periods=5).mean()
    df["limit_up"] = df["pct_change_close"] >= limit_up_pct
    df["shrink_volume"] = (volume < volume.shift(1) * shrink_limit_vol_ratio) | (volume < vol_ma5 * shrink_limit_vol_ratio)
    df["shrink_limit_up"] = df["limit_up"] & df["shrink_volume"]
    df["prior_20d_shrink_limit_up"] = (
        df["shrink_limit_up"].shift(1).rolling(window=hard_lookback, min_periods=1).max().fillna(False).astype(bool)
    )

    lower_shadow = np.minimum(open_, close) - low
    upper_shadow = high - np.maximum(open_, close)
    safe_body = body_abs.replace(0, np.nan)
    df["long_lower_shadow_hammer"] = (
        (candle_range > 0)
        & (lower_shadow >= safe_body * hammer_lower_shadow_body_ratio)
        & ((lower_shadow / candle_range) >= hammer_lower_shadow_range_ratio)
        & (upper_shadow <= safe_body * hammer_upper_shadow_body_ratio)
        & ((body_abs / candle_range) <= hammer_max_body_range_ratio)
    ).fillna(False)
    df["limit_up_red_brick"] = df["limit_up"] & df.get("red_brick", False)
    return df
