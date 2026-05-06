from __future__ import annotations

"""
Renko chart selection strategy v4.

Purpose:
Refined weighted score version based on previous v3/v4 data analysis.

Hard selected rule:
selected =
    hard_brick_turn_strong
    AND 3 < daily_return_pct <= 7
    AND close / short_trend < 0.95

Score design:
- close / short_trend < 0.90 is the strongest positive factor.
- Deeper close / short_trend levels receive extra fine-grained bonus.
- daily_return_pct in 5~7 receives moderate bonus.
- current_red_height >= previous_green_height * 1.00 is negative, so it is penalized.
- current_red_height >= previous_green_height * 1.20 receives additional penalty.
- score_rank_key is added to reduce same-score ties when choosing one stock per day.

Important:
STRATEGY_NAME remains unchanged:
    renko_chart_select_strategy_v4
"""

import pandas as pd

from indicators import add_all_indicators


STRATEGY_NAME = "renko_chart_select_strategy_v4"

REQUIRED_INDICATOR_COLUMNS: set[str] = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "brick_value",
    "brick_prev_1",
    "brick_prev_2",
    "current_red_height",
    "previous_green_height",
    "daily_return_pct",
    "short_trend",
}


# =============================================================================
# Strategy condition parameters
# =============================================================================

BRICK_REVERSAL_RATIO = 0.70

DEFAULT_MIN_DAILY_RETURN_PCT = 3.0
DEFAULT_MAX_DAILY_RETURN_PCT = 7.0
DEFAULT_MAX_CLOSE_TO_SHORT_TREND = 0.95


# =============================================================================
# Weighted score config
# =============================================================================
# Positive max score:
# 30 + 20 + 12 + 8 + 10 + 6 + 4 + 4 + 3 = 97
#
# Negative penalty:
# -22 and -10
#
# score_pct = score / POSITIVE_MAX_SCORE * 100
# score_pct may be negative if penalties are large.
# selected does NOT depend on score_pct.
# score is used for ranking only.

SCORE_WEIGHTS = {
    # Strongest factor: price is meaningfully below short_trend.
    "close_to_short_trend_below_090": 30.0,
    "close_to_short_trend_below_088": 20.0,
    "close_to_short_trend_below_086": 12.0,
    "close_to_short_trend_below_084": 8.0,

    # Daily return bonus.
    "daily_return_5_to_7": 10.0,
    "daily_return_55_to_7": 6.0,
    "daily_return_6_to_7": 4.0,

    # Prefer red brick not too long.
    "brick_reversal_strength_below_100": 4.0,
    "brick_reversal_strength_below_090": 3.0,

    # Penalty factors.
    "penalty_brick_reversal_strength_100": -22.0,
    "penalty_brick_reversal_strength_120": -10.0,
}

POSITIVE_MAX_SCORE = sum(v for v in SCORE_WEIGHTS.values() if v > 0)


def add_strategy_conditions(
    df: pd.DataFrame,
    *,
    min_daily_return_pct: float = DEFAULT_MIN_DAILY_RETURN_PCT,
    max_daily_return_pct: float = DEFAULT_MAX_DAILY_RETURN_PCT,
    max_close_to_short_trend: float = DEFAULT_MAX_CLOSE_TO_SHORT_TREND,
) -> pd.DataFrame:
    """Add all boolean conditions and diagnostic columns used by this strategy."""
    out = df.copy()

    brick_value = pd.to_numeric(out["brick_value"], errors="coerce")
    brick_prev_1 = pd.to_numeric(out["brick_prev_1"], errors="coerce")
    brick_prev_2 = pd.to_numeric(out["brick_prev_2"], errors="coerce")
    current_red_height = pd.to_numeric(out["current_red_height"], errors="coerce")
    previous_green_height = pd.to_numeric(out["previous_green_height"], errors="coerce")
    daily_return_pct = pd.to_numeric(out["daily_return_pct"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    short_trend = pd.to_numeric(out["short_trend"], errors="coerce")

    # =========================================================================
    # Brick reversal condition
    # =========================================================================
    out["red_brick"] = brick_value > brick_prev_1
    out["green_brick"] = brick_value < brick_prev_1

    out["green_to_red"] = (
        (~out["red_brick"].shift(1).fillna(False).astype(bool))
        & out["red_brick"]
    )

    out["valid_red_brick"] = brick_value > 0
    out["valid_previous_green_brick"] = brick_prev_2 > brick_prev_1
    out["valid_green_brick"] = out["valid_previous_green_brick"]

    out["brick_reversal_ratio"] = current_red_height / previous_green_height

    out["brick_reversal_strength"] = (
        current_red_height >= previous_green_height * BRICK_REVERSAL_RATIO
    ).fillna(False)

    out["hard_brick_turn_strong"] = (
        out["green_to_red"].fillna(False).astype(bool)
        & out["valid_red_brick"].fillna(False).astype(bool)
        & out["valid_previous_green_brick"].fillna(False).astype(bool)
        & out["brick_reversal_strength"].fillna(False).astype(bool)
    ).fillna(False)

    # =========================================================================
    # Daily return range
    # =========================================================================
    out["price_rise_above_min"] = (
        daily_return_pct > min_daily_return_pct
    ).fillna(False)

    out["price_rise_below_max"] = (
        daily_return_pct <= max_daily_return_pct
    ).fillna(False)

    out["price_rise_in_range"] = (
        out["price_rise_above_min"].fillna(False).astype(bool)
        & out["price_rise_below_max"].fillna(False).astype(bool)
    ).fillna(False)

    # Score-only daily return flags.
    out["daily_return_5_to_7"] = (
        (daily_return_pct > 5.0)
        & (daily_return_pct <= 7.0)
    ).fillna(False)

    out["daily_return_55_to_7"] = (
        (daily_return_pct > 5.5)
        & (daily_return_pct <= 7.0)
    ).fillna(False)

    out["daily_return_6_to_7"] = (
        (daily_return_pct > 6.0)
        & (daily_return_pct <= 7.0)
    ).fillna(False)

    # Diagnostic buckets.
    out["daily_return_3_to_4"] = (
        (daily_return_pct > 3.0)
        & (daily_return_pct <= 4.0)
    ).fillna(False)

    out["daily_return_4_to_5"] = (
        (daily_return_pct > 4.0)
        & (daily_return_pct <= 5.0)
    ).fillna(False)

    out["daily_return_5_to_6"] = (
        (daily_return_pct > 5.0)
        & (daily_return_pct <= 6.0)
    ).fillna(False)

    out["daily_return_6_to_7_exact"] = (
        (daily_return_pct > 6.0)
        & (daily_return_pct <= 7.0)
    ).fillna(False)

    # =========================================================================
    # close / short_trend
    # =========================================================================
    out["close_to_short_trend"] = close / short_trend

    out["close_to_short_trend_below_limit"] = (
        (short_trend > 0)
        & out["close_to_short_trend"].notna()
        & (out["close_to_short_trend"] < max_close_to_short_trend)
    ).fillna(False)

    out["close_to_short_trend_below_095"] = (
        (short_trend > 0)
        & out["close_to_short_trend"].notna()
        & (out["close_to_short_trend"] < 0.95)
    ).fillna(False)

    out["close_to_short_trend_below_090"] = (
        (short_trend > 0)
        & out["close_to_short_trend"].notna()
        & (out["close_to_short_trend"] < 0.90)
    ).fillna(False)

    out["close_to_short_trend_below_088"] = (
        (short_trend > 0)
        & out["close_to_short_trend"].notna()
        & (out["close_to_short_trend"] < 0.88)
    ).fillna(False)

    out["close_to_short_trend_below_086"] = (
        (short_trend > 0)
        & out["close_to_short_trend"].notna()
        & (out["close_to_short_trend"] < 0.86)
    ).fillna(False)

    out["close_to_short_trend_below_084"] = (
        (short_trend > 0)
        & out["close_to_short_trend"].notna()
        & (out["close_to_short_trend"] < 0.84)
    ).fillna(False)

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
        out["price_rise_in_range"].fillna(False).astype(bool)
        & out["close_to_short_trend_below_limit"].fillna(False).astype(bool)
    ).fillna(False)

    # =========================================================================
    # Brick reversal ratio flags
    # =========================================================================
    out["brick_reversal_strength_070"] = (
        out["brick_reversal_ratio"] >= 0.70
    ).fillna(False)

    out["brick_reversal_strength_090"] = (
        out["brick_reversal_ratio"] >= 0.90
    ).fillna(False)

    # Negative factors.
    out["brick_reversal_strength_100"] = (
        out["brick_reversal_ratio"] >= 1.00
    ).fillna(False)

    out["brick_reversal_strength_120"] = (
        out["brick_reversal_ratio"] >= 1.20
    ).fillna(False)

    # Positive fine flags.
    out["brick_reversal_strength_below_100"] = (
        out["brick_reversal_ratio"].notna()
        & (out["brick_reversal_ratio"] < 1.00)
    ).fillna(False)

    out["brick_reversal_strength_below_090"] = (
        out["brick_reversal_ratio"].notna()
        & (out["brick_reversal_ratio"] < 0.90)
    ).fillna(False)

    return out


def add_weighted_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add refined weighted score columns."""
    out = df.copy()

    # =========================================================================
    # Positive scores: close / short_trend
    # =========================================================================
    out["score_close_to_short_trend_below_090"] = (
        out["close_to_short_trend_below_090"].fillna(False).astype(bool).astype(float)
        * SCORE_WEIGHTS["close_to_short_trend_below_090"]
    )

    out["score_close_to_short_trend_below_088"] = (
        out["close_to_short_trend_below_088"].fillna(False).astype(bool).astype(float)
        * SCORE_WEIGHTS["close_to_short_trend_below_088"]
    )

    out["score_close_to_short_trend_below_086"] = (
        out["close_to_short_trend_below_086"].fillna(False).astype(bool).astype(float)
        * SCORE_WEIGHTS["close_to_short_trend_below_086"]
    )

    out["score_close_to_short_trend_below_084"] = (
        out["close_to_short_trend_below_084"].fillna(False).astype(bool).astype(float)
        * SCORE_WEIGHTS["close_to_short_trend_below_084"]
    )

    # =========================================================================
    # Positive scores: daily return
    # =========================================================================
    out["score_daily_return_5_to_7"] = (
        out["daily_return_5_to_7"].fillna(False).astype(bool).astype(float)
        * SCORE_WEIGHTS["daily_return_5_to_7"]
    )

    out["score_daily_return_55_to_7"] = (
        out["daily_return_55_to_7"].fillna(False).astype(bool).astype(float)
        * SCORE_WEIGHTS["daily_return_55_to_7"]
    )

    out["score_daily_return_6_to_7"] = (
        out["daily_return_6_to_7"].fillna(False).astype(bool).astype(float)
        * SCORE_WEIGHTS["daily_return_6_to_7"]
    )

    # =========================================================================
    # Positive scores: brick not too long
    # =========================================================================
    out["score_brick_reversal_strength_below_100"] = (
        out["brick_reversal_strength_below_100"].fillna(False).astype(bool).astype(float)
        * SCORE_WEIGHTS["brick_reversal_strength_below_100"]
    )

    out["score_brick_reversal_strength_below_090"] = (
        out["brick_reversal_strength_below_090"].fillna(False).astype(bool).astype(float)
        * SCORE_WEIGHTS["brick_reversal_strength_below_090"]
    )

    # =========================================================================
    # Penalty scores
    # =========================================================================
    out["penalty_brick_reversal_strength_100"] = (
        out["brick_reversal_strength_100"].fillna(False).astype(bool).astype(float)
        * SCORE_WEIGHTS["penalty_brick_reversal_strength_100"]
    )

    out["penalty_brick_reversal_strength_120"] = (
        out["brick_reversal_strength_120"].fillna(False).astype(bool).astype(float)
        * SCORE_WEIGHTS["penalty_brick_reversal_strength_120"]
    )

    score_cols = [
        "score_close_to_short_trend_below_090",
        "score_close_to_short_trend_below_088",
        "score_close_to_short_trend_below_086",
        "score_close_to_short_trend_below_084",
        "score_daily_return_5_to_7",
        "score_daily_return_55_to_7",
        "score_daily_return_6_to_7",
        "score_brick_reversal_strength_below_100",
        "score_brick_reversal_strength_below_090",
        "penalty_brick_reversal_strength_100",
        "penalty_brick_reversal_strength_120",
    ]

    out["score"] = out[score_cols].sum(axis=1)

    if POSITIVE_MAX_SCORE > 0:
        out["score_pct"] = out["score"] / POSITIVE_MAX_SCORE * 100.0
    else:
        out["score_pct"] = 0.0

    # =========================================================================
    # Fine-grained rank key to reduce same-score ties
    # =========================================================================
    close_to_short_trend = pd.to_numeric(out["close_to_short_trend"], errors="coerce")
    daily_return_pct = pd.to_numeric(out["daily_return_pct"], errors="coerce")
    brick_reversal_ratio = pd.to_numeric(out["brick_reversal_ratio"], errors="coerce")

    out["rank_close_to_short_trend"] = close_to_short_trend.fillna(999.0)
    out["rank_daily_return_pct"] = daily_return_pct.fillna(-999.0)
    out["rank_brick_reversal_ratio"] = brick_reversal_ratio.fillna(999.0)

    # Higher is better.
    # score_pct is primary.
    # lower close_to_short_trend is better.
    # higher daily_return_pct within 3~7 is better.
    # lower brick_reversal_ratio is better because >=1.0 was negative.
    out["score_rank_key"] = (
        out["score_pct"].fillna(0.0) * 100000.0
        + (1.0 - out["rank_close_to_short_trend"].clip(lower=0.0, upper=2.0)) * 10000.0
        + out["rank_daily_return_pct"].clip(lower=-20.0, upper=20.0) * 100.0
        + (1.0 - out["rank_brick_reversal_ratio"].clip(lower=0.0, upper=3.0)) * 1000.0
    )

    return out


def add_final_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Final selected rule for v4."""
    out = df.copy()

    out["selected_score_base"] = (
        out["hard_brick_turn_strong"].fillna(False).astype(bool)
        & out["price_rise_range_and_close_to_short_trend_below_limit"].fillna(False).astype(bool)
    ).astype(int)

    # Keep selected unchanged.
    # Score is for ranking, not for shrinking the pool.
    out["selected"] = out["selected_score_base"]

    return out


# =============================================================================
# Strategy execution wrapper
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

    out = add_weighted_score(out)
    out = add_final_selection(out)

    out["selection_strategy"] = STRATEGY_NAME

    return out


SELECT_FUNC = select_renko_chart