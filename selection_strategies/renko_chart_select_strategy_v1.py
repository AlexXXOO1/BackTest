from __future__ import annotations

"""
Renko chart selection strategy v1.

This version is adapted to the current indicator structure.

Current project rule:
- indicators/* calculate raw reusable values only.
- strategy files create strategy-specific boolean conditions, scores, and selected flags.

v1 hard selection rules:
1. hard_brick_turn_strong == True
2. j_momentum_or_low == True
3. small_rise_long_red_brick == True

v1 scoring rules:
- score_pct is calculated for attribution and later bucket analysis.
- score_pct is NOT used as a hard selection filter.
- former hard-risk rules are NOT direct rejection filters in v1.
- former hard-risk rules are kept as negative score factors and risk tags.

Important compatibility notes:
- This file does not rely on old indicator boolean columns.
- It recalculates all strategy booleans from raw columns such as:
  brick_value, current_red_height, trend_line, short_trend, yellow_ma, J,
  volume, daily_return_pct, high_position_line, accel_return_pct,
  huge_volume_ma, bear_body_pct, lower_shadow, upper_shadow, body_abs.
"""

import numpy as np
import pandas as pd

from indicators import add_all_indicators


STRATEGY_NAME = "renko_chart_select_strategy_v1"


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
    "red_height_reference",
    "short_trend",
    "trend_line",
    "yellow_ma",
    "short_trend_cap",
    "J",
    "vol_ma5",
    "volume_prev_1",
    "volume_prev_2",
    "pct_change_close",
    "high_position_line",
    "accel_return_pct",
    "huge_volume_ma",
    "shrink_volume_ma5",
    "candle_range",
    "body_abs",
    "bear_body_pct",
    "lower_shadow",
    "upper_shadow",
}


# =============================================================================
# Strategy condition parameters
# =============================================================================

# Brick reversal strength used by hard_brick_turn_strong.
BRICK_REVERSAL_RATIO = 0.70

# Condition 6: J momentum or low value.
DEFAULT_LOW_J_THRESHOLD = 14.0

# Condition 9: low daily price rise + relatively long red brick.
DEFAULT_SMALL_RISE_MIN_PCT = 0.0
DEFAULT_SMALL_RISE_MAX_PCT = 3.0
DEFAULT_LONG_RED_RATIO = 1.3
DEFAULT_MIN_RED_HEIGHT_REFERENCE_PERIODS = 3

# Condition 8: surge then shrink-volume pullback.
DEFAULT_SURGE_RETURN_PCT = 3.0
DEFAULT_SURGE_VOLUME_RATIO = 1.5
DEFAULT_PULLBACK_MAX_RETURN_PCT = 0.0
DEFAULT_SHRINK_VOLUME_RATIO = 1.0

# Former risk tag parameters.
DEFAULT_HIGH_POS_RATIO = 0.85
DEFAULT_ACCEL_RETURN_PCT = 12.0
DEFAULT_HUGE_VOLUME_RATIO = 2.0
DEFAULT_BIG_BEAR_BODY_PCT = 4.0
DEFAULT_LIMIT_UP_PCT = 9.7
DEFAULT_SHRINK_LIMIT_VOL_RATIO = 0.8
DEFAULT_HAMMER_LOWER_SHADOW_BODY_RATIO = 2.0
DEFAULT_HAMMER_LOWER_SHADOW_RANGE_RATIO = 0.5
DEFAULT_HAMMER_UPPER_SHADOW_BODY_RATIO = 1.2
DEFAULT_HAMMER_MAX_BODY_RANGE_RATIO = 0.4
DEFAULT_RISK_LOOKBACK = 20


DEFAULT_RENKO_CHART_SELECT_WEIGHTS: dict[str, float] = {
    # Condition 2: two consecutive days above trend line.
    "two_day_above_trend_line": 0.3,

    # Condition 3: short trend above medium/long trend line.
    "short_trend_above_trend_line": 0.3,

    # Condition 4: close not higher than short trend cap.
    "close_below_short_trend_cap": 1.5,

    # Condition 5: low absolute price.
    "price_below_50": 0.8,

    # Condition 6: KDJ J momentum or low value.
    # It is also a v1 hard selection condition.
    "j_momentum_or_low": 2.5,

    # Condition 7: close above yellow moving average.
    "above_yellow_ma": 0.3,

    # Condition 8: volume surge up followed by shrink-volume pullback.
    "surge_then_shrink_pullback": -2.0,

    # Condition 9: low daily rise + long red brick.
    # It is also a v1 hard selection condition.
    "small_rise_long_red_brick": 3.0,

    # Former hard-risk rules. v1 keeps them as negative attribution factors.
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


def _to_num(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric Series. Missing columns become all-NaN."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def add_strategy_conditions(
    df: pd.DataFrame,
    *,
    low_j_threshold: float = DEFAULT_LOW_J_THRESHOLD,
    small_rise_min_pct: float = DEFAULT_SMALL_RISE_MIN_PCT,
    small_rise_max_pct: float = DEFAULT_SMALL_RISE_MAX_PCT,
    long_red_ratio: float = DEFAULT_LONG_RED_RATIO,
    surge_return_pct: float = DEFAULT_SURGE_RETURN_PCT,
    surge_volume_ratio: float = DEFAULT_SURGE_VOLUME_RATIO,
    pullback_max_return_pct: float = DEFAULT_PULLBACK_MAX_RETURN_PCT,
    shrink_volume_ratio: float = DEFAULT_SHRINK_VOLUME_RATIO,
    high_pos_ratio: float = DEFAULT_HIGH_POS_RATIO,
    accel_return_pct: float = DEFAULT_ACCEL_RETURN_PCT,
    huge_volume_ratio: float = DEFAULT_HUGE_VOLUME_RATIO,
    big_bear_body_pct: float = DEFAULT_BIG_BEAR_BODY_PCT,
    limit_up_pct: float = DEFAULT_LIMIT_UP_PCT,
    shrink_limit_vol_ratio: float = DEFAULT_SHRINK_LIMIT_VOL_RATIO,
    hammer_lower_shadow_body_ratio: float = DEFAULT_HAMMER_LOWER_SHADOW_BODY_RATIO,
    hammer_lower_shadow_range_ratio: float = DEFAULT_HAMMER_LOWER_SHADOW_RANGE_RATIO,
    hammer_upper_shadow_body_ratio: float = DEFAULT_HAMMER_UPPER_SHADOW_BODY_RATIO,
    hammer_max_body_range_ratio: float = DEFAULT_HAMMER_MAX_BODY_RANGE_RATIO,
    risk_lookback: int = DEFAULT_RISK_LOOKBACK,
) -> pd.DataFrame:
    """Add every strategy-specific boolean condition used by v1."""
    out = df.copy()

    open_ = _to_num(out, "open")
    high = _to_num(out, "high")
    low = _to_num(out, "low")
    close = _to_num(out, "close")
    volume = _to_num(out, "volume")

    brick_value = _to_num(out, "brick_value")
    brick_prev_1 = _to_num(out, "brick_prev_1")
    brick_prev_2 = _to_num(out, "brick_prev_2")
    current_red_height = _to_num(out, "current_red_height")
    previous_green_height = _to_num(out, "previous_green_height")

    daily_return_pct = _to_num(out, "daily_return_pct")
    red_height_reference = _to_num(out, "red_height_reference")
    short_trend = _to_num(out, "short_trend")
    trend_line = _to_num(out, "trend_line")
    yellow_ma = _to_num(out, "yellow_ma")
    short_trend_cap = _to_num(out, "short_trend_cap")
    j_value = _to_num(out, "J")
    volume_prev_1 = _to_num(out, "volume_prev_1")
    volume_prev_2 = _to_num(out, "volume_prev_2")

    pct_change_close = _to_num(out, "pct_change_close")
    high_position_line = _to_num(out, "high_position_line")
    accel_return = _to_num(out, "accel_return_pct")
    huge_volume_ma = _to_num(out, "huge_volume_ma")
    shrink_volume_ma5 = _to_num(out, "shrink_volume_ma5")
    candle_range = _to_num(out, "candle_range")
    body_abs = _to_num(out, "body_abs")
    bear_body_pct = _to_num(out, "bear_body_pct")
    lower_shadow = _to_num(out, "lower_shadow")
    upper_shadow = _to_num(out, "upper_shadow")

    # -------------------------------------------------------------------------
    # Core brick reversal condition.
    # -------------------------------------------------------------------------
    out["red_brick"] = brick_value > brick_prev_1
    out["green_brick"] = brick_value < brick_prev_1
    red_brick_bool = out["red_brick"].fillna(False).astype(bool)
    out["green_to_red"] = (
        (~red_brick_bool.shift(1, fill_value=False))
        & red_brick_bool
    )
    out["valid_red_brick"] = brick_value > 0
    out["valid_previous_green_brick"] = brick_prev_2 > brick_prev_1
    out["valid_green_brick"] = out["valid_previous_green_brick"]
    out["brick_reversal_strength"] = (
        current_red_height >= previous_green_height * BRICK_REVERSAL_RATIO
    ).fillna(False)
    out["hard_brick_turn_strong"] = (
        out["green_to_red"].astype(bool)
        & out["valid_red_brick"].fillna(False).astype(bool)
        & out["valid_previous_green_brick"].fillna(False).astype(bool)
        & out["brick_reversal_strength"].fillna(False).astype(bool)
    ).fillna(False)

    # -------------------------------------------------------------------------
    # Trend and price-position conditions.
    # -------------------------------------------------------------------------
    out["close_above_trend_line"] = (close > trend_line).fillna(False)
    out["close_prev_1_above_trend_line_prev_1"] = (
        close.shift(1) > trend_line.shift(1)
    ).fillna(False)
    out["two_day_above_trend_line"] = (
        out["close_above_trend_line"].astype(bool)
        & out["close_prev_1_above_trend_line_prev_1"].astype(bool)
    ).fillna(False)

    out["short_trend_above_trend_line"] = (short_trend > trend_line).fillna(False)
    out["close_below_short_trend_cap"] = (close <= short_trend_cap).fillna(False)
    out["close_to_short_trend"] = close / short_trend.replace(0, np.nan)
    out["price_below_50"] = (close < 50).fillna(False)
    out["above_yellow_ma"] = (close > yellow_ma).fillna(False)
    out["close_above_yellow_ma"] = out["above_yellow_ma"]

    # -------------------------------------------------------------------------
    # KDJ J condition.
    # j_momentum_or_low = J is low OR J has risen for two consecutive bars.
    # -------------------------------------------------------------------------
    out["j_low"] = (j_value < low_j_threshold).fillna(False)
    out["j_rising_2d"] = (
        (j_value > j_value.shift(1))
        & (j_value.shift(1) > j_value.shift(2))
    ).fillna(False)
    out["j_momentum_or_low"] = (
        out["j_low"].astype(bool)
        | out["j_rising_2d"].astype(bool)
    ).fillna(False)

    # -------------------------------------------------------------------------
    # Volume pattern condition.
    # surge_then_shrink_pullback = previous bar surged with volume, current bar
    # is a shrink-volume pullback.
    # -------------------------------------------------------------------------
    prev_day_return = daily_return_pct.shift(1)
    prev_day_surge = (
        (prev_day_return >= surge_return_pct)
        & (volume_prev_1 >= volume_prev_2 * surge_volume_ratio)
    ).fillna(False)
    current_pullback = (daily_return_pct <= pullback_max_return_pct).fillna(False)
    current_shrink_volume = (volume <= volume_prev_1 * shrink_volume_ratio).fillna(False)
    out["surge_then_shrink_pullback"] = (
        prev_day_surge.astype(bool)
        & current_pullback.astype(bool)
        & current_shrink_volume.astype(bool)
    ).fillna(False)

    # -------------------------------------------------------------------------
    # Low daily rise + long red brick condition.
    # This is condition 9 in v1.
    # -------------------------------------------------------------------------
    out["small_price_rise"] = (
        (daily_return_pct > small_rise_min_pct)
        & (daily_return_pct <= small_rise_max_pct)
    ).fillna(False)
    out["long_red_brick"] = (
        (current_red_height > 0)
        & red_height_reference.notna()
        & (current_red_height >= red_height_reference * long_red_ratio)
    ).fillna(False)
    out["small_rise_long_red_brick"] = (
        out["small_price_rise"].astype(bool)
        & out["long_red_brick"].astype(bool)
    ).fillna(False)

    # -------------------------------------------------------------------------
    # Former hard-risk conditions, now retained as tags and negative scores.
    # -------------------------------------------------------------------------
    high_position = (close >= high_position_line).fillna(False)
    accelerated = (accel_return >= accel_return_pct).fillna(False)
    huge_volume = (volume >= huge_volume_ma * huge_volume_ratio).fillna(False)
    big_bear_body = (bear_body_pct >= big_bear_body_pct).fillna(False)

    out["accelerated_huge_volume_bear"] = (
        high_position.astype(bool)
        & accelerated.astype(bool)
        & huge_volume.astype(bool)
        & big_bear_body.astype(bool)
    ).fillna(False)

    out["prior_20d_accelerated_huge_volume_bear"] = (
        out["accelerated_huge_volume_bear"]
        .fillna(False)
        .astype(bool)
        .astype(int)
        .shift(1, fill_value=0)
        .rolling(window=risk_lookback, min_periods=1)
        .max()
        .fillna(0)
        .astype(bool)
    )

    out["limit_up"] = (pct_change_close >= limit_up_pct).fillna(False)
    out["shrink_limit_up"] = (
        out["limit_up"].astype(bool)
        & (volume <= shrink_volume_ma5 * shrink_limit_vol_ratio).fillna(False)
    ).fillna(False)
    out["prior_20d_shrink_limit_up"] = (
        out["shrink_limit_up"]
        .fillna(False)
        .astype(bool)
        .astype(int)
        .shift(1, fill_value=0)
        .rolling(window=risk_lookback, min_periods=1)
        .max()
        .fillna(0)
        .astype(bool)
    )

    body_safe = body_abs.replace(0, np.nan)
    range_safe = candle_range.replace(0, np.nan)
    out["long_lower_shadow_hammer"] = (
        (lower_shadow >= body_safe * hammer_lower_shadow_body_ratio)
        & (lower_shadow >= range_safe * hammer_lower_shadow_range_ratio)
        & (upper_shadow <= body_safe * hammer_upper_shadow_body_ratio)
        & (body_abs <= range_safe * hammer_max_body_range_ratio)
    ).fillna(False)

    out["limit_up_red_brick"] = (
        out["limit_up"].astype(bool)
        & out["red_brick"].fillna(False).astype(bool)
    ).fillna(False)

    return out


def add_strategy_score(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Add signed score columns used by renko_chart_select_strategy_v1.

    Scoring details:
    - Each boolean condition contributes its configured signed weight when true.
    - Positive weights increase raw_score.
    - Negative weights reduce raw_score.
    - score_pct = raw_score / positive_weight_total * 100.
    - score_abs_pct = raw_score / absolute_weight_total * 100.
    - score_pct is exported only; it is not a hard filter in v1.
    """
    out = df.copy()
    weights = DEFAULT_RENKO_CHART_SELECT_WEIGHTS if weights is None else weights

    out["raw_score"] = 0.0

    for col, weight in weights.items():
        weight = float(weight)
        if col not in out.columns:
            out[col] = False

        condition = out[col].fillna(False).astype(bool)
        out[f"{col}_weight"] = weight
        out[f"{col}_score"] = np.where(condition, weight, 0.0)
        out["raw_score"] += out[f"{col}_score"]

    positive_weight_total = float(sum(weight for weight in weights.values() if weight > 0)) or 1.0
    absolute_weight_total = float(sum(abs(weight) for weight in weights.values())) or 1.0

    out["positive_weight_total"] = positive_weight_total
    out["absolute_weight_total"] = absolute_weight_total
    out["score_pct"] = out["raw_score"] / positive_weight_total * 100
    out["score_abs_pct"] = out["raw_score"] / absolute_weight_total * 100

    return out


def add_strategy_risk_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add risk tags used for performance attribution.

    In v1 these tags are not direct rejection filters.
    """
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

    # Backward-compatible column. It is no longer a rejection filter in v1.
    out["risk_filter_pass"] = True

    return out


def add_final_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Final v1 selected rule."""
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
# =============================================================================

def _prepare_indicators(df: pd.DataFrame, n1: int, n2: int, **kwargs) -> pd.DataFrame:
    if REQUIRED_INDICATOR_COLUMNS.issubset(set(df.columns)):
        return df.copy().sort_values("date").reset_index(drop=True)
    return add_all_indicators(df, n1=n1, n2=n2, **kwargs)


def select_renko_chart(
    df: pd.DataFrame,
    n1: int = 4,
    n2: int = 6,
    **kwargs,
) -> pd.DataFrame:
    """Build the v1 renko chart selection result from raw reusable indicators."""
    weights = kwargs.get("weights", DEFAULT_RENKO_CHART_SELECT_WEIGHTS)

    out = _prepare_indicators(df, n1=n1, n2=n2, **kwargs)

    out = add_strategy_conditions(
        out,
        low_j_threshold=float(kwargs.get("low_j_threshold", DEFAULT_LOW_J_THRESHOLD)),
        small_rise_min_pct=float(kwargs.get("small_rise_min_pct", DEFAULT_SMALL_RISE_MIN_PCT)),
        small_rise_max_pct=float(kwargs.get("small_rise_max_pct", kwargs.get("renko_small_rise_max_pct", DEFAULT_SMALL_RISE_MAX_PCT))),
        long_red_ratio=float(kwargs.get("long_red_ratio", kwargs.get("renko_long_red_ratio", DEFAULT_LONG_RED_RATIO))),
        surge_return_pct=float(kwargs.get("surge_return_pct", DEFAULT_SURGE_RETURN_PCT)),
        surge_volume_ratio=float(kwargs.get("surge_volume_ratio", DEFAULT_SURGE_VOLUME_RATIO)),
        pullback_max_return_pct=float(kwargs.get("pullback_max_return_pct", DEFAULT_PULLBACK_MAX_RETURN_PCT)),
        shrink_volume_ratio=float(kwargs.get("shrink_volume_ratio", DEFAULT_SHRINK_VOLUME_RATIO)),
        high_pos_ratio=float(kwargs.get("v8_high_pos_ratio", DEFAULT_HIGH_POS_RATIO)),
        accel_return_pct=float(kwargs.get("v8_accel_ret_pct", DEFAULT_ACCEL_RETURN_PCT)),
        huge_volume_ratio=float(kwargs.get("v8_huge_vol_ratio", DEFAULT_HUGE_VOLUME_RATIO)),
        big_bear_body_pct=float(kwargs.get("v8_big_bear_body_pct", DEFAULT_BIG_BEAR_BODY_PCT)),
        limit_up_pct=float(kwargs.get("v8_limit_up_pct", DEFAULT_LIMIT_UP_PCT)),
        shrink_limit_vol_ratio=float(kwargs.get("v8_shrink_limit_vol_ratio", DEFAULT_SHRINK_LIMIT_VOL_RATIO)),
        hammer_lower_shadow_body_ratio=float(kwargs.get("v8_hammer_lower_shadow_body_ratio", DEFAULT_HAMMER_LOWER_SHADOW_BODY_RATIO)),
        hammer_lower_shadow_range_ratio=float(kwargs.get("v8_hammer_lower_shadow_range_ratio", DEFAULT_HAMMER_LOWER_SHADOW_RANGE_RATIO)),
        hammer_upper_shadow_body_ratio=float(kwargs.get("v8_hammer_upper_shadow_body_ratio", DEFAULT_HAMMER_UPPER_SHADOW_BODY_RATIO)),
        hammer_max_body_range_ratio=float(kwargs.get("v8_hammer_max_body_range_ratio", DEFAULT_HAMMER_MAX_BODY_RANGE_RATIO)),
        risk_lookback=int(kwargs.get("v8_hard_lookback", DEFAULT_RISK_LOOKBACK)),
    )

    out = add_strategy_score(out, weights=weights)
    out = add_strategy_risk_tags(out)
    out = add_final_selection(out)
    out["selection_strategy"] = STRATEGY_NAME

    return out


SELECT_FUNC = select_renko_chart
