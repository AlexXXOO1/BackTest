from __future__ import annotations

"""
Renko chart selection strategy v3.

Purpose:
Test whether close / short_trend < 0.95 improves the 3-7 daily return pool.

Known daily_return_pct findings:
- daily_return_pct <= 3: ineffective / negative.
- daily_return_pct > 3: effective.
- 3 < daily_return_pct <= 7: best overall balance, selected as base range.
- 3 < daily_return_pct <= 5: effective, but weaker average return than 3-7.
- 5 < daily_return_pct <= 7: highest average return, but sample size is smaller.

Known red brick length findings:
- current_red_height >= 8: not suitable as a main hard filter.
- current_red_height >= 6: not suitable as a main hard filter.
- current_red_height >= previous_green_height * 1.0: negative as a hard filter.

Known short trend findings:
- close >= short_trend was negative as a hard filter.
- close < short_trend was strongly positive.
- Bucket analysis showed that close / short_trend < 0.95 performed better
  than merely requiring close < short_trend.

Final selected rule:
selected =
    hard_brick_turn_strong
    AND 3 < daily_return_pct <= 7
    AND close / short_trend < 0.95

Meaning:
The stock must satisfy the base 3-7 price rise pool,
and the close price must still be meaningfully below the short trend line.

Important:
This strategy recalculates the short-trend ratio directly from:
    close_to_short_trend = close / short_trend

It does NOT rely on the old close_below_short_trend_cap column,
because that column may be None / NaN in historical pool files.
"""

import pandas as pd

from indicators import add_all_indicators


STRATEGY_NAME = "renko_chart_select_strategy_v3"

REQUIRED_INDICATOR_COLUMNS: set[str] = {
    "date", "open", "high", "low", "close", "volume",
    "brick_value", "brick_prev_1", "brick_prev_2",
    "current_red_height", "previous_green_height", "daily_return_pct",
    "short_trend",
}

# =============================================================================
# Strategy condition block
# Edit this block when creating a new strategy version.
# =============================================================================

# Basic brick reversal strength used inside hard_brick_turn_strong.
# This is the baseline reversal definition:
# current_red_height >= previous_green_height * 0.70
BRICK_REVERSAL_RATIO = 0.70

# T0 daily return lower bound.
# 3.0 means daily_return_pct must be > 3.0
DEFAULT_MIN_DAILY_RETURN_PCT = 3.0

# T0 daily return upper bound.
# 7.0 means daily_return_pct must be <= 7.0
DEFAULT_MAX_DAILY_RETURN_PCT = 7.0

# Short trend ratio upper bound.
# 0.95 means close / short_trend must be < 0.95
DEFAULT_MAX_CLOSE_TO_SHORT_TREND = 0.95


def add_strategy_conditions(
    df: pd.DataFrame,
    *,
    min_daily_return_pct: float = DEFAULT_MIN_DAILY_RETURN_PCT,
    max_daily_return_pct: float = DEFAULT_MAX_DAILY_RETURN_PCT,
    max_close_to_short_trend: float = DEFAULT_MAX_CLOSE_TO_SHORT_TREND,
) -> pd.DataFrame:
    """Add every boolean condition used by this strategy."""
    out = df.copy()

    brick_value = pd.to_numeric(out["brick_value"], errors="coerce")
    brick_prev_1 = pd.to_numeric(out["brick_prev_1"], errors="coerce")
    brick_prev_2 = pd.to_numeric(out["brick_prev_2"], errors="coerce")
    current_red_height = pd.to_numeric(out["current_red_height"], errors="coerce")
    previous_green_height = pd.to_numeric(out["previous_green_height"], errors="coerce")
    daily_return_pct = pd.to_numeric(out["daily_return_pct"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    short_trend = pd.to_numeric(out["short_trend"], errors="coerce")

    # -------------------------------------------------------------------------
    # Basic brick reversal condition
    # -------------------------------------------------------------------------
    out["red_brick"] = brick_value > brick_prev_1
    out["green_brick"] = brick_value < brick_prev_1

    out["green_to_red"] = (
        (~out["red_brick"].shift(1).fillna(False).astype(bool))
        & out["red_brick"]
    )

    out["valid_red_brick"] = brick_value > 0
    out["valid_previous_green_brick"] = brick_prev_2 > brick_prev_1
    out["valid_green_brick"] = out["valid_previous_green_brick"]

    out["brick_reversal_strength"] = (
        current_red_height >= previous_green_height * BRICK_REVERSAL_RATIO
    ).fillna(False)

    out["hard_brick_turn_strong"] = (
        out["green_to_red"]
        & out["valid_red_brick"]
        & out["valid_previous_green_brick"]
        & out["brick_reversal_strength"]
    ).fillna(False)

    # -------------------------------------------------------------------------
    # T0 daily return range condition
    # Base range:
    # 3 < daily_return_pct <= 7
    # -------------------------------------------------------------------------
    out["price_rise_above_min"] = (
        daily_return_pct > min_daily_return_pct
    ).fillna(False)

    out["price_rise_below_max"] = (
        daily_return_pct <= max_daily_return_pct
    ).fillna(False)

    out["price_rise_in_range"] = (
        out["price_rise_above_min"].astype(bool)
        & out["price_rise_below_max"].astype(bool)
    ).fillna(False)

    # -------------------------------------------------------------------------
    # Short trend ratio condition
    #
    # Do not use old close_below_short_trend_cap directly.
    # Your existing v2 pool showed:
    # close_below_short_trend_cap = None for all rows.
    #
    # Here we recalculate:
    #
    # close_to_short_trend = close / short_trend
    #
    # Final test condition:
    # close_to_short_trend < 0.95
    #
    # This tests whether stocks that are still meaningfully below short_trend
    # have better forward returns.
    # -------------------------------------------------------------------------
    out["close_to_short_trend"] = close / short_trend

    out["close_to_short_trend_below_limit"] = (
        (short_trend > 0)
        & out["close_to_short_trend"].notna()
        & (out["close_to_short_trend"] < max_close_to_short_trend)
    ).fillna(False)

    # Diagnostic columns.
    out["below_short_trend"] = (
        (short_trend > 0)
        & out["close_to_short_trend"].notna()
        & (out["close_to_short_trend"] < 1.0)
    ).fillna(False)

    out["not_below_short_trend"] = (
        (short_trend > 0)
        & out["close_to_short_trend"].notna()
        & (out["close_to_short_trend"] >= 1.0)
    ).fillna(False)

    out["close_below_short_trend_cap_calc"] = out["below_short_trend"]

    out["price_rise_range_and_close_to_short_trend_below_limit"] = (
        out["price_rise_in_range"].astype(bool)
        & out["close_to_short_trend_below_limit"].astype(bool)
    ).fillna(False)

    return out


def add_final_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Final selected rule for v3."""
    out = df.copy()

    out["selected_score_base"] = (
        out["hard_brick_turn_strong"].fillna(False).astype(bool)
        & out["price_rise_range_and_close_to_short_trend_below_limit"].fillna(False).astype(bool)
    ).astype(int)

    out["selected"] = out["selected_score_base"]

    return out


# =============================================================================
# Strategy execution wrapper
# Usually no need to edit below this line when creating a similar strategy.
# =============================================================================

def _prepare_indicators(df: pd.DataFrame, n1: int, n2: int, **kwargs) -> pd.DataFrame:
    if REQUIRED_INDICATOR_COLUMNS.issubset(set(df.columns)):
        return df.copy().sort_values("date").reset_index(drop=True)

    return add_all_indicators(df, n1=n1, n2=n2, **kwargs)


def select_renko_chart(
    df: pd.DataFrame,
    n1: int = 4,
    n2: int = 6,
    min_daily_return_pct: float = DEFAULT_MIN_DAILY_RETURN_PCT,
    max_daily_return_pct: float = DEFAULT_MAX_DAILY_RETURN_PCT,
    max_close_to_short_trend: float = DEFAULT_MAX_CLOSE_TO_SHORT_TREND,
    **kwargs,
) -> pd.DataFrame:
    out = _prepare_indicators(df, n1=n1, n2=n2, **kwargs)

    out = add_strategy_conditions(
        out,
        min_daily_return_pct=min_daily_return_pct,
        max_daily_return_pct=max_daily_return_pct,
        max_close_to_short_trend=max_close_to_short_trend,
    )

    out = add_final_selection(out)
    out["selection_strategy"] = STRATEGY_NAME

    return out


SELECT_FUNC = select_renko_chart