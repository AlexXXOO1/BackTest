from __future__ import annotations

"""
Renko chart selection strategy v1.

Refactor rule:
- Strategy layer does not define indicator columns.
- All reusable conditions used by v1 are generated in indicators/.
- This file only checks required indicators, adds score/risk-tag attribution,
  and combines final selected conditions.

v1 hard selection rules:
1. hard_brick_turn_strong == True
2. j_momentum_or_low == True
3. small_rise_long_red_brick == True

v1 scoring rules:
- score_pct is exported for attribution and bucket analysis.
- score_pct is NOT used as a hard selection filter.
- legacy risk rules are NOT direct rejection filters in v1; they remain
  negative score factors and risk tags.
"""

import numpy as np
import pandas as pd

from indicators import add_all_indicators
from indicators.required import require_indicator_columns


STRATEGY_NAME = "renko_chart_select_strategy_v1"

REQUIRED_INDICATORS: set[str] = {
    "date", "open", "high", "low", "close", "volume",
    "brick_value", "brick_prev_1", "brick_prev_2",
    "current_red_height", "previous_green_height", "red_height_reference",
    "hard_brick_turn_strong", "brick_reversal_ratio", "brick_reversal_strength",

    "daily_return_pct", "pct_change_close",
    "short_trend", "trend_line", "yellow_ma", "short_trend_cap",
    "close_above_trend_line", "close_prev_1_above_trend_line_prev_1",
    "two_day_above_trend_line", "short_trend_above_trend_line",
    "close_below_short_trend_cap", "price_below_50",
    "above_yellow_ma", "close_above_yellow_ma",

    "J", "j_low", "j_rising_2d", "j_momentum_or_low",

    "vol_ma5", "volume_prev_1", "volume_prev_2",
    "surge_then_shrink_pullback",

    "small_price_rise", "long_red_brick", "small_rise_long_red_brick",

    "high_position_line", "accel_return_pct", "huge_volume_ma", "shrink_volume_ma5",
    "candle_range", "body_abs", "bear_body_pct", "lower_shadow", "upper_shadow",
    "accelerated_huge_volume_bear", "prior_20d_accelerated_huge_volume_bear",
    "limit_up", "shrink_limit_up", "prior_20d_shrink_limit_up",
    "long_lower_shadow_hammer", "limit_up_red_brick",
}

# Backward-compatible alias used by older tools.
REQUIRED_INDICATOR_COLUMNS = REQUIRED_INDICATORS

DEFAULT_LOW_J_THRESHOLD = 14.0
DEFAULT_SMALL_RISE_MIN_PCT = 0.0
DEFAULT_SMALL_RISE_MAX_PCT = 3.0
DEFAULT_LONG_RED_RATIO = 1.3

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


def _prepare_indicators(df: pd.DataFrame, n1: int, n2: int, **kwargs) -> pd.DataFrame:
    if REQUIRED_INDICATORS.issubset(set(df.columns)):
        out = df.copy().sort_values("date").reset_index(drop=True)
    else:
        out = add_all_indicators(df, n1=n1, n2=n2, **kwargs)
    require_indicator_columns(out, REQUIRED_INDICATORS, STRATEGY_NAME)
    return out


def add_strategy_score(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add signed score columns used by v1."""
    out = df.copy()
    weights = DEFAULT_RENKO_CHART_SELECT_WEIGHTS if weights is None else weights

    out["raw_score"] = 0.0

    for col, weight in weights.items():
        weight = float(weight)
        condition = out[col].fillna(False).astype(bool)
        out[f"{col}_weight"] = weight
        out[f"{col}_score"] = np.where(condition, weight, 0.0)
        out["raw_score"] += out[f"{col}_score"]

    positive_weight_total = float(sum(weight for weight in weights.values() if weight > 0)) or 1.0
    absolute_weight_total = float(sum(abs(weight) for weight in weights.values())) or 1.0

    out["positive_weight_total"] = positive_weight_total
    out["absolute_weight_total"] = absolute_weight_total
    out["score_pct"] = out["raw_score"] / positive_weight_total * 100.0
    out["score_abs_pct"] = out["raw_score"] / absolute_weight_total * 100.0
    out["score"] = out["raw_score"]

    # Stable ranking key for tools that sort selected rows.
    brick_reversal_ratio = pd.to_numeric(out["brick_reversal_ratio"], errors="coerce")
    daily_return_pct = pd.to_numeric(out["daily_return_pct"], errors="coerce")
    out["score_rank_key"] = (
        out["score_pct"].fillna(0.0) * 10000.0
        + (1.0 - brick_reversal_ratio.fillna(999.0).clip(lower=0.0, upper=3.0)) * 1000.0
        + daily_return_pct.fillna(-999.0).clip(lower=-20.0, upper=20.0) * 10.0
    )
    return out


def add_strategy_risk_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Add v1 risk tags for attribution; these are not rejection filters."""
    out = df.copy()
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


def select_renko_chart(
    df: pd.DataFrame,
    n1: int = 4,
    n2: int = 6,
    low_j_threshold: float = DEFAULT_LOW_J_THRESHOLD,
    small_rise_min_pct: float = DEFAULT_SMALL_RISE_MIN_PCT,
    small_rise_max_pct: float = DEFAULT_SMALL_RISE_MAX_PCT,
    long_red_ratio: float = DEFAULT_LONG_RED_RATIO,
    **kwargs,
) -> pd.DataFrame:
    """Build the v1 renko chart selection result from indicator cache columns."""
    kwargs = dict(kwargs)
    kwargs.setdefault("low_j_threshold", low_j_threshold)
    kwargs.setdefault("small_rise_min_pct", small_rise_min_pct)
    kwargs.setdefault("small_rise_max_pct", small_rise_max_pct)
    kwargs.setdefault("long_red_ratio", long_red_ratio)

    weights = kwargs.get("weights", DEFAULT_RENKO_CHART_SELECT_WEIGHTS)
    out = _prepare_indicators(df, n1=n1, n2=n2, **kwargs)
    out = add_strategy_score(out, weights=weights)
    out = add_strategy_risk_tags(out)
    out = add_final_selection(out)
    out["selection_strategy"] = STRATEGY_NAME
    return out


def select(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    return select_renko_chart(df=df, n1=n1, n2=n2, **kwargs)


def apply_strategy(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    return select(df=df, n1=n1, n2=n2, **kwargs)


def run(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    return select(df=df, n1=n1, n2=n2, **kwargs)


SELECT_FUNC = select
