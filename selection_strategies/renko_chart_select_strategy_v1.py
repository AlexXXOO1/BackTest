from __future__ import annotations

"""
Renko chart selection strategy v1.

Copy-edit workflow:
1. Copy this file.
2. Change STRATEGY_NAME.
3. Edit only the "Strategy condition block", weights, and final selected rule.

Indicator modules only calculate raw values. All selection conditions used by
this strategy are declared near the top of this file.

Hard selection rules:
1. hard_brick_turn_strong is true.
2. j_momentum_or_low is true.
3. small_rise_long_red_brick is true.

Scoring is kept for attribution only; score_pct does not filter candidates.
"""

import numpy as np
import pandas as pd

from indicators import add_all_indicators

STRATEGY_NAME = "renko_chart_select_strategy_v1"

REQUIRED_INDICATOR_COLUMNS: set[str] = {
    "date", "open", "high", "low", "close", "volume",
    "brick_value", "brick_prev_1", "brick_prev_2",
    "current_red_height", "previous_green_height",
    "trend_line", "short_trend", "short_trend_cap", "yellow_ma",
    "J", "vol_ma5", "pct_change_close", "daily_return_pct", "red_height_reference",
    "high_position_line", "accel_return_pct", "huge_volume_ma",
    "candle_range", "body_abs", "lower_shadow", "upper_shadow",
    "bear_body_pct", "shrink_volume_ma5",
}

# =============================================================================
# Strategy condition block
# Edit this block when creating a new strategy version.
# =============================================================================
BRICK_REVERSAL_RATIO = 0.70
LOW_J_THRESHOLD = 14.0
SMALL_RISE_MAX_PCT = 3.0
LONG_RED_RATIO = 1.3
ACCEL_RET_PCT = 12.0
HUGE_VOL_RATIO = 2.0
BIG_BEAR_BODY_PCT = 4.0
LIMIT_UP_PCT = 9.7
SHRINK_LIMIT_VOL_RATIO = 0.8
HARD_LOOKBACK = 20
HAMMER_LOWER_SHADOW_BODY_RATIO = 2.0
HAMMER_LOWER_SHADOW_RANGE_RATIO = 0.5
HAMMER_UPPER_SHADOW_BODY_RATIO = 1.2
HAMMER_MAX_BODY_RANGE_RATIO = 0.4

DEFAULT_RENKO_CHART_SELECT_WEIGHTS: dict[str, float] = {
    "two_day_above_trend_line": 0.3,
    "short_trend_above_trend_line": 0.3,
    "close_below_short_trend_cap": 1.5,
    "price_below_50": 0.8,
    "j_momentum_or_low": 2.5,
    "above_yellow_ma": 0.3,
    "surge_then_shrink_pullback": -2.0,
    "small_rise_long_red_brick": 3.0,
    "prior_20d_accelerated_huge_volume_bear": -1.5,
    "prior_20d_shrink_limit_up": -1.2,
    "long_lower_shadow_hammer": -1.0,
    "limit_up_red_brick": -1.3,
}

RENKO_CHART_RISK_RULE_COLUMNS: tuple[str, ...] = (
    "prior_20d_accelerated_huge_volume_bear",
    "prior_20d_shrink_limit_up",
    "long_lower_shadow_hammer",
    "limit_up_red_brick",
)


def find_j_column(df: pd.DataFrame) -> str:
    """Return the KDJ J column name from common project naming variants."""
    candidates = ("J", "j", "kdj_j", "KDJ_J", "kdj_J", "j_value", "J_VALUE", "KDJJ")
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError("Cannot find KDJ J column. Expected one of: " + ", ".join(candidates))


def add_strategy_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Add every boolean condition used by this strategy."""
    out = df.copy()

    close = pd.to_numeric(out["close"], errors="coerce")
    open_ = pd.to_numeric(out["open"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce")
    brick_value = pd.to_numeric(out["brick_value"], errors="coerce")
    brick_prev_1 = pd.to_numeric(out["brick_prev_1"], errors="coerce")
    brick_prev_2 = pd.to_numeric(out["brick_prev_2"], errors="coerce")
    current_red_height = pd.to_numeric(out["current_red_height"], errors="coerce")
    previous_green_height = pd.to_numeric(out["previous_green_height"], errors="coerce")

    # Brick turn-strong condition.
    out["red_brick"] = brick_value > brick_prev_1
    out["green_brick"] = brick_value < brick_prev_1
    out["green_to_red"] = (~out["red_brick"].shift(1).fillna(False).astype(bool)) & out["red_brick"]
    out["valid_red_brick"] = brick_value > 0
    out["valid_previous_green_brick"] = brick_prev_2 > brick_prev_1
    out["valid_green_brick"] = out["valid_previous_green_brick"]
    out["brick_reversal_strength"] = current_red_height >= previous_green_height * BRICK_REVERSAL_RATIO
    out["hard_brick_turn_strong"] = (
        out["green_to_red"]
        & out["valid_red_brick"]
        & out["valid_previous_green_brick"]
        & out["brick_reversal_strength"]
    ).fillna(False)

    # Trend and price-zone conditions.
    trend_line = pd.to_numeric(out["trend_line"], errors="coerce")
    out["above_trend_line_today"] = close > trend_line
    out["above_trend_line_prev"] = close.shift(1) > trend_line.shift(1)
    out["two_day_above_trend_line"] = out["above_trend_line_today"] & out["above_trend_line_prev"]
    out["short_trend_above_trend_line"] = pd.to_numeric(out["short_trend"], errors="coerce") > trend_line
    out["close_below_short_trend_cap"] = close <= pd.to_numeric(out["short_trend_cap"], errors="coerce")
    out["price_below_50"] = close < 50
    out["above_yellow_ma"] = close > pd.to_numeric(out["yellow_ma"], errors="coerce")

    # J condition.
    j_col = find_j_column(out)
    j_value = pd.to_numeric(out[j_col], errors="coerce")
    out["j_three_day_rising"] = (j_value.shift(3) < j_value.shift(2)) & (j_value.shift(2) < j_value.shift(1))
    out["j_two_day_rising"] = (j_value > j_value.shift(1)) & (j_value.shift(1) > j_value.shift(2))
    out["j_below_14"] = j_value < LOW_J_THRESHOLD
    out["j_momentum_or_low"] = out["j_two_day_rising"] | out["j_below_14"]

    # Volume pullback condition.
    out["surge_then_shrink_pullback"] = (
        (close.shift(2) > close.shift(3))
        & (volume.shift(2) > pd.to_numeric(out["vol_ma5"], errors="coerce").shift(2) * 1.2)
        & (close.shift(1) < close.shift(2))
        & (volume.shift(1) < volume.shift(2) * 0.8)
    ).fillna(False)

    # Small-rise and long-red-brick condition.
    out["small_rise_long_red_brick"] = (
        (pd.to_numeric(out["pct_change_close"], errors="coerce") <= SMALL_RISE_MAX_PCT)
        & (current_red_height > 0)
        & (current_red_height >= pd.to_numeric(out["red_height_reference"], errors="coerce") * LONG_RED_RATIO)
    ).fillna(False)

    # Risk attribution conditions.
    candle_range = pd.to_numeric(out["candle_range"], errors="coerce").replace(0, np.nan)
    body_abs = pd.to_numeric(out["body_abs"], errors="coerce")
    lower_shadow = pd.to_numeric(out["lower_shadow"], errors="coerce")
    upper_shadow = pd.to_numeric(out["upper_shadow"], errors="coerce")
    safe_body = body_abs.replace(0, np.nan)

    out["high_position"] = close >= pd.to_numeric(out["high_position_line"], errors="coerce")
    out["accelerated_move"] = pd.to_numeric(out["accel_return_pct"], errors="coerce") >= ACCEL_RET_PCT
    out["huge_volume"] = volume >= pd.to_numeric(out["huge_volume_ma"], errors="coerce") * HUGE_VOL_RATIO
    out["big_bear_body"] = (close < open_) & (pd.to_numeric(out["bear_body_pct"], errors="coerce") >= BIG_BEAR_BODY_PCT)
    out["accelerated_huge_volume_bear"] = out["high_position"] & out["accelerated_move"] & out["huge_volume"] & out["big_bear_body"]
    out["prior_20d_accelerated_huge_volume_bear"] = (
        out["accelerated_huge_volume_bear"].shift(1).rolling(window=HARD_LOOKBACK, min_periods=1).max().fillna(False).astype(bool)
    )

    out["limit_up"] = pd.to_numeric(out["pct_change_close"], errors="coerce") >= LIMIT_UP_PCT
    out["shrink_volume"] = (
        (volume < volume.shift(1) * SHRINK_LIMIT_VOL_RATIO)
        | (volume < pd.to_numeric(out["shrink_volume_ma5"], errors="coerce") * SHRINK_LIMIT_VOL_RATIO)
    )
    out["shrink_limit_up"] = out["limit_up"] & out["shrink_volume"]
    out["prior_20d_shrink_limit_up"] = (
        out["shrink_limit_up"].shift(1).rolling(window=HARD_LOOKBACK, min_periods=1).max().fillna(False).astype(bool)
    )
    out["long_lower_shadow_hammer"] = (
        (candle_range > 0)
        & (lower_shadow >= safe_body * HAMMER_LOWER_SHADOW_BODY_RATIO)
        & ((lower_shadow / candle_range) >= HAMMER_LOWER_SHADOW_RANGE_RATIO)
        & (upper_shadow <= safe_body * HAMMER_UPPER_SHADOW_BODY_RATIO)
        & ((body_abs / candle_range) <= HAMMER_MAX_BODY_RANGE_RATIO)
    ).fillna(False)
    out["limit_up_red_brick"] = out["limit_up"] & out["red_brick"]
    return out


def add_strategy_score(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Add signed strategy score columns for attribution."""
    out = df.copy()
    weights = DEFAULT_RENKO_CHART_SELECT_WEIGHTS if weights is None else weights
    out["raw_score"] = 0.0
    for col, weight in weights.items():
        weight = float(weight)
        if col not in out.columns:
            out[col] = False
        out[f"{col}_weight"] = weight
        out[f"{col}_score"] = np.where(out[col].fillna(False).astype(bool), weight, 0.0)
        out["raw_score"] += out[f"{col}_score"]
    positive_weight_total = float(sum(weight for weight in weights.values() if weight > 0)) or 1.0
    absolute_weight_total = float(sum(abs(weight) for weight in weights.values())) or 1.0
    out["positive_weight_total"] = positive_weight_total
    out["absolute_weight_total"] = absolute_weight_total
    out["score_pct"] = out["raw_score"] / positive_weight_total * 100
    out["score_abs_pct"] = out["raw_score"] / absolute_weight_total * 100
    return out


def add_strategy_risk_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Add risk attribution tags without rejecting candidates."""
    out = df.copy()
    for col in RENKO_CHART_RISK_RULE_COLUMNS:
        if col not in out.columns:
            out[col] = False
    risk_bool_df = out[list(RENKO_CHART_RISK_RULE_COLUMNS)].fillna(False).astype(bool)
    out["risk_tag_any"] = risk_bool_df.any(axis=1)
    out["risk_tag_count"] = risk_bool_df.sum(axis=1)
    out["risk_tags"] = risk_bool_df.apply(
        lambda row: "|".join([col for col, value in row.items() if bool(value)]),
        axis=1,
    )
    out["risk_filter_pass"] = True
    return out


def add_final_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Final selected rule for v1."""
    out = df.copy()
    out["condition6_hard_pass"] = out["j_momentum_or_low"].fillna(False).astype(bool)
    out["condition9_hard_pass"] = out["small_rise_long_red_brick"].fillna(False).astype(bool)
    out["selected_score_base"] = (
        out["hard_brick_turn_strong"].fillna(False).astype(bool)
        & out["condition6_hard_pass"]
        & out["condition9_hard_pass"]
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


def select_renko_chart(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    weights = kwargs.get("weights", DEFAULT_RENKO_CHART_SELECT_WEIGHTS)
    out = _prepare_indicators(df, n1=n1, n2=n2, **kwargs)
    out = add_strategy_conditions(out)
    out = add_strategy_score(out, weights=weights)
    out = add_strategy_risk_tags(out)
    out = add_final_selection(out)
    out["selection_strategy"] = STRATEGY_NAME
    return out


SELECT_FUNC = select_renko_chart
