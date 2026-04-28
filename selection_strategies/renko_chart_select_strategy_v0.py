from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import add_all_indicators

STRATEGY_NAME = "renko_chart_select_strategy_v0"

DEFAULT_RENKO_CHART_SELECT_WEIGHTS: dict[str, float] = {
    "two_day_above_trend_line": 1.2,
    "short_trend_above_trend_line": 1.3,
    "close_below_short_trend_cap": 0.8,
    "price_below_50": 0.6,
    "j_momentum_or_low": 1.0,
    "close_above_yellow_ma": 1.1,
    "surge_then_shrink_pullback": 1.5,
    "small_rise_long_red_brick": 2.5,
}


RENKO_CHART_RISK_RULE_COLUMNS: tuple[str, ...] = (
    "prior_20d_accelerated_huge_volume_bear",
    "prior_20d_shrink_limit_up",
    "long_lower_shadow_hammer",
    "limit_up_red_brick",
)


def add_strategy_score(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """
    Add the score columns used by this selection strategy.

    Quant details:
    - Each boolean condition contributes its configured weight when true.
    - raw_score is the sum of all active weighted conditions.
    - score_pct = raw_score / total_weight * 100.
    - This function belongs to the strategy layer because weights are strategy decisions,
      not reusable market indicators.
    """
    out = df.copy()
    weights = DEFAULT_RENKO_CHART_SELECT_WEIGHTS if weights is None else weights
    out["raw_score"] = 0.0
    for col, weight in weights.items():
        weight = float(weight)
        out[f"{col}_weight"] = weight
        out[f"{col}_score"] = np.where(out[col].fillna(False), weight, 0.0)
        out["raw_score"] += out[f"{col}_score"]
    total_weight = float(sum(weights.values())) or 1.0
    out["score_pct"] = out["raw_score"] / total_weight * 100
    return out


def add_strategy_risk_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the hard risk filter used by this selection strategy.

    Quant details:
    - The source columns are reusable candle-pattern facts from the indicators layer.
    - risk_filter_pass is false when any configured hard-risk condition is true.
    - The decision to reject these patterns belongs to the strategy layer because another
      strategy may score or trade the same patterns differently.
    """
    out = df.copy()
    hard_risk = pd.Series(False, index=out.index)
    for col in RENKO_CHART_RISK_RULE_COLUMNS:
        hard_risk = hard_risk | out[col].fillna(False).astype(bool)
    out["risk_filter_pass"] = ~hard_risk
    return out


def select_renko_chart(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    """
    Build the renko chart selection result from reusable indicators.

    Strategy rules:
    - hard_brick_turn_strong must be true.
    - risk_filter_pass must be true.
    - score_pct must be greater than or equal to score_threshold_pct.
    - selected is stored as 1 or 0.
    """
    score_threshold_pct = float(kwargs.get("score_threshold_pct", 60.0))
    weights = kwargs.get("weights", DEFAULT_RENKO_CHART_SELECT_WEIGHTS)

    required_indicator_columns = {
        "hard_brick_turn_strong",
        "two_day_above_trend_line",
        "short_trend_above_trend_line",
        "close_below_short_trend_cap",
        "price_below_50",
        "j_momentum_or_low",
        "close_above_yellow_ma",
        "surge_then_shrink_pullback",
        "small_rise_long_red_brick",
        *RENKO_CHART_RISK_RULE_COLUMNS,
    }
    if required_indicator_columns.issubset(set(df.columns)):
        out = df.copy().sort_values("date").reset_index(drop=True)
    else:
        out = add_all_indicators(df, n1=n1, n2=n2, **kwargs)
    out = add_strategy_score(out, weights=weights)
    out = add_strategy_risk_filter(out)

    out["selected_score_base"] = (out["hard_brick_turn_strong"] & (out["score_pct"] >= score_threshold_pct)).astype(int)
    out["selected"] = (
        out["hard_brick_turn_strong"]
        & out["risk_filter_pass"]
        & (out["score_pct"] >= score_threshold_pct)
    ).astype(int)
    out["selection_strategy"] = STRATEGY_NAME
    return out


SELECT_FUNC = select_renko_chart
