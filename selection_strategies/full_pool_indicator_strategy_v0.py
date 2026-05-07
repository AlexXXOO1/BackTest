from __future__ import annotations

"""
Full pool indicator snapshot strategy v0.

Purpose:
- Select every valid stock trading row.
- Write current numeric and boolean indicator columns into pool.
- This strategy is NOT for direct trading.
- It is only used for reverse analysis:

    T0 indicators -> T1 open buy -> T2 open sell return.

Registry rule:
- STRATEGY_NAME must be unique.
- SELECT_FUNC should point to the strategy entry function.

Important:
- This strategy does not use scoring.
- It selects all valid trading rows.
- Boolean factors are explicitly calculated here so that single-factor analysis works.
- select_strategy accepts n1/n2/**kwargs because selector.py may pass them.
"""

import numpy as np
import pandas as pd

from indicators import add_all_indicators


STRATEGY_NAME = "full_pool_indicator_strategy_v0"


REQUIRED_INDICATOR_COLUMNS: set[str] = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


# =============================================================================
# Boolean factor parameters
# =============================================================================

BRICK_REVERSAL_RATIO = 0.70
LOW_DAILY_RETURN_MAX_PCT = 3.0
LONG_RED_BRICK_MIN_HEIGHT = 8.0

SURGE_LOOKBACK = 5
SURGE_RET_MIN_PCT = 5.0
SURGE_VOL_RATIO = 1.8
PULLBACK_MAX_RET_PCT = 1.5
SHRINK_VOL_RATIO = 0.85

TREND_BUFFER_PCT = 0.0

LIMIT_UP_PCT = 9.7
SHRINK_LIMIT_VOL_RATIO = 0.8

HIGH_POS_LOOKBACK = 60
HIGH_POS_RATIO = 0.85
ACCEL_LOOKBACK = 5
ACCEL_RET_PCT = 12.0
HUGE_VOL_MA_N = 20
HUGE_VOL_RATIO = 2.0
BIG_BEAR_BODY_PCT = 4.0

HAMMER_LOWER_SHADOW_BODY_RATIO = 2.0
HAMMER_LOWER_SHADOW_RANGE_RATIO = 0.5
HAMMER_UPPER_SHADOW_BODY_RATIO = 1.2
HAMMER_MAX_BODY_RANGE_RATIO = 0.4


def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _ref(s: pd.Series, n: int = 1) -> pd.Series:
    return s.shift(n)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _rolling_max(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).max()


def _rolling_min(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).min()


def _rolling_mean(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).mean()


def _get_numeric_series(
    df: pd.DataFrame,
    col: str,
    default: float = np.nan,
) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def add_boolean_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicitly calculate boolean factors used by reverse analysis.

    This function only uses columns that exist after add_all_indicators().
    Missing required columns will result in conservative False values.
    """
    out = df.copy()

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            out[col] = np.nan

    open_ = pd.to_numeric(out["open"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce")

    prev_close = _ref(close, 1)

    if "daily_return_pct" not in out.columns:
        out["daily_return_pct"] = (close / prev_close - 1.0) * 100.0

    daily_return_pct = pd.to_numeric(out["daily_return_pct"], errors="coerce")

    # -------------------------------------------------------------------------
    # Brick factors
    # -------------------------------------------------------------------------
    brick_value = _get_numeric_series(out, "brick_value")
    brick_prev_1 = (
        _get_numeric_series(out, "brick_prev_1")
        if "brick_prev_1" in out.columns
        else _ref(brick_value, 1)
    )
    brick_prev_2 = (
        _get_numeric_series(out, "brick_prev_2")
        if "brick_prev_2" in out.columns
        else _ref(brick_value, 2)
    )

    current_red_height = (
        _get_numeric_series(out, "current_red_height")
        if "current_red_height" in out.columns
        else brick_value
    )

    previous_green_height = (
        _get_numeric_series(out, "previous_green_height")
        if "previous_green_height" in out.columns
        else brick_prev_1
    )

    out["brick_value"] = brick_value
    out["brick_prev_1"] = brick_prev_1
    out["brick_prev_2"] = brick_prev_2
    out["current_red_height"] = current_red_height
    out["previous_green_height"] = previous_green_height

    out["hard_brick_turn_strong"] = (
        (brick_value > 0)
        & (
            (brick_prev_1 <= 0)
            | (brick_value >= brick_prev_1 * BRICK_REVERSAL_RATIO)
            | (brick_value > brick_prev_1)
        )
    ).fillna(False)

    out["small_rise_long_red_brick"] = (
        (daily_return_pct <= LOW_DAILY_RETURN_MAX_PCT)
        & (daily_return_pct >= -10.0)
        & (current_red_height >= LONG_RED_BRICK_MIN_HEIGHT)
    ).fillna(False)

    # -------------------------------------------------------------------------
    # Surge then shrink pullback
    # -------------------------------------------------------------------------
    vol_ma20 = _rolling_mean(volume, 20)
    prior_surge = (
        (daily_return_pct >= SURGE_RET_MIN_PCT)
        & (volume >= vol_ma20 * SURGE_VOL_RATIO)
    )

    prior_surge_recent = (
        prior_surge.shift(1)
        .rolling(SURGE_LOOKBACK, min_periods=1)
        .max()
        .fillna(False)
        .astype(bool)
    )

    out["surge_then_shrink_pullback"] = (
        prior_surge_recent
        & (daily_return_pct <= PULLBACK_MAX_RET_PCT)
        & (volume <= _ref(volume, 1) * SHRINK_VOL_RATIO)
    ).fillna(False)

    # -------------------------------------------------------------------------
    # Trend factors
    # -------------------------------------------------------------------------
    trend_line = _get_numeric_series(out, "trend_line")
    short_trend = _get_numeric_series(out, "short_trend")
    yellow_ma = _get_numeric_series(out, "yellow_ma")

    out["two_day_above_trend_line"] = (
        (close > trend_line * (1.0 + TREND_BUFFER_PCT / 100.0))
        & (_ref(close, 1) > _ref(trend_line, 1) * (1.0 + TREND_BUFFER_PCT / 100.0))
    ).fillna(False)

    out["short_trend_above_trend_line"] = (
        short_trend > trend_line
    ).fillna(False)

    out["close_above_yellow_ma"] = (
        close > yellow_ma
    ).fillna(False)

    out["close_below_short_trend_cap"] = (
        close < short_trend
    ).fillna(False)

    out["close_above_short_trend_cap"] = (
        close >= short_trend
    ).fillna(False)

    # -------------------------------------------------------------------------
    # KDJ factors
    # -------------------------------------------------------------------------
    j = _get_numeric_series(out, "j")

    out["j"] = j
    out["j_below_0"] = (j < 0).fillna(False)
    out["j_30_50"] = ((j >= 30) & (j < 50)).fillna(False)

    j_rising_2days = (j > _ref(j, 1)) & (_ref(j, 1) > _ref(j, 2))
    out["j_momentum_or_low"] = (
        (j < 14)
        | j_rising_2days
    ).fillna(False)

    # -------------------------------------------------------------------------
    # Risk factors
    # -------------------------------------------------------------------------
    limit_up = daily_return_pct >= LIMIT_UP_PCT
    shrink_limit_up = limit_up & (volume <= vol_ma20 * SHRINK_LIMIT_VOL_RATIO)

    out["prior_20d_shrink_limit_up"] = (
        shrink_limit_up.shift(1)
        .rolling(20, min_periods=1)
        .max()
        .fillna(False)
        .astype(bool)
    )

    highest_60 = _rolling_max(high, HIGH_POS_LOOKBACK)
    lowest_60 = _rolling_min(low, HIGH_POS_LOOKBACK)

    high_pos = _safe_div(close - lowest_60, highest_60 - lowest_60) >= HIGH_POS_RATIO

    accel_ret = (close / _ref(close, ACCEL_LOOKBACK) - 1.0) * 100.0
    accelerated = accel_ret >= ACCEL_RET_PCT

    vol_ma_n = _rolling_mean(volume, HUGE_VOL_MA_N)
    huge_volume = volume >= vol_ma_n * HUGE_VOL_RATIO

    bear_body_pct = (open_ / close - 1.0) * 100.0
    big_bear = (close < open_) & (bear_body_pct >= BIG_BEAR_BODY_PCT)

    out["accelerated_huge_volume_bear"] = (
        high_pos
        & accelerated
        & huge_volume
        & big_bear
    ).fillna(False)

    body = (close - open_).abs()
    full_range = high - low
    upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low

    out["long_lower_shadow_hammer"] = (
        (full_range > 0)
        & (body > 0)
        & (lower_shadow >= body * HAMMER_LOWER_SHADOW_BODY_RATIO)
        & (lower_shadow >= full_range * HAMMER_LOWER_SHADOW_RANGE_RATIO)
        & (upper_shadow <= body * HAMMER_UPPER_SHADOW_BODY_RATIO)
        & (body <= full_range * HAMMER_MAX_BODY_RANGE_RATIO)
    ).fillna(False)

    bool_cols = [
        "hard_brick_turn_strong",
        "small_rise_long_red_brick",
        "surge_then_shrink_pullback",
        "two_day_above_trend_line",
        "short_trend_above_trend_line",
        "close_above_yellow_ma",
        "close_below_short_trend_cap",
        "close_above_short_trend_cap",
        "prior_20d_shrink_limit_up",
        "accelerated_huge_volume_bear",
        "long_lower_shadow_hammer",
        "j_below_0",
        "j_30_50",
        "j_momentum_or_low",
    ]

    for col in bool_cols:
        out[col] = out[col].fillna(False).astype(bool)

    return out


def select_strategy(
    df: pd.DataFrame,
    n1: int = 4,
    n2: int = 6,
    **kwargs,
) -> pd.DataFrame:
    """
    Full-pool selector.

    The selector engine may pass n1/n2 or other keyword args.
    This function accepts them to avoid:
        select_strategy() got an unexpected keyword argument 'n1'
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Add numeric indicators using your current indicator structure.
    # Some versions of add_all_indicators accept n1/n2, some do not.
    try:
        out = add_all_indicators(out, n1=n1, n2=n2)
    except TypeError:
        out = add_all_indicators(out)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "daily_return_pct",
        "brick_value",
        "brick_prev_1",
        "brick_prev_2",
        "current_red_height",
        "previous_green_height",
        "j",
        "k",
        "d",
        "yellow_ma",
        "short_trend",
        "trend_line",
        "score",
        "score_pct",
    ]

    out = _ensure_numeric(out, numeric_cols)

    # Explicitly calculate boolean factors.
    out = add_boolean_factors(out)

    valid = pd.Series(True, index=out.index)

    if "date" in out.columns:
        valid &= out["date"].notna()

    for col in ["open", "high", "low", "close"]:
        if col in out.columns:
            valid &= out[col].notna()
            valid &= out[col] > 0
        else:
            valid &= False

    if "volume" in out.columns:
        valid &= out["volume"].notna()
        valid &= out["volume"] > 0
    else:
        valid &= False

    out["selected"] = valid.astype(bool)

    # This is not a scoring strategy.
    out["score"] = 0.0
    out["score_pct"] = 100.0
    out["selection_strategy"] = STRATEGY_NAME

    out = out.loc[out["selected"]].copy()

    return out


def select(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    """Standard selection strategy entry point."""
    return select_strategy(df=df, **kwargs)


def apply_strategy(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    return select(df=df, n1=n1, n2=n2, **kwargs)


def run(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    return select(df=df, n1=n1, n2=n2, **kwargs)


SELECT_FUNC = select