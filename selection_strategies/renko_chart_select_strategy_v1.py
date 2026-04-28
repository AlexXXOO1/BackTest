from __future__ import annotations

"""
Renko chart selection strategy v1.

Changes from renko_chart_select_strategy_v0:

1. Former hard-risk filters are no longer direct rejection rules.
   - prior_20d_accelerated_huge_volume_bear
   - prior_20d_shrink_limit_up
   - long_lower_shadow_hammer
   - limit_up_red_brick
   These four fields are now kept as weighted score factors and analysis tags.

2. The score range filter is removed.
   - v0 previously tested or supported score_pct ranges such as 80-90.
   - v1 keeps all candidates as long as the core hard conditions are satisfied.
   - score_pct is still calculated and exported for later performance attribution.

3. Condition 6 is promoted to a hard condition.
   - Field: j_momentum_or_low
   - v1 requires j_momentum_or_low == True.

4. Condition 9 is promoted to a hard condition.
   - Field: small_rise_long_red_brick
   - v1 requires small_rise_long_red_brick == True.

5. Signed scoring is used.
   Positive factors add score. Negative factors reduce score.
   Exact default weights:
   - two_day_above_trend_line: 0.3
   - short_trend_above_trend_line: 0.3
   - close_below_short_trend_cap: 1.5
   - price_below_50: 0.8
   - j_momentum_or_low: 2.5
   - close_above_yellow_ma: 0.3
   - surge_then_shrink_pullback: -2.0
   - small_rise_long_red_brick: 3.0
   - prior_20d_accelerated_huge_volume_bear: -1.5
   - prior_20d_shrink_limit_up: -1.2
   - long_lower_shadow_hammer: -1.0
   - limit_up_red_brick: -1.3

6. The T+1 opening gap filter is not used in this strategy layer.
   - No future-data filter is applied during T+0 selection.
   - If needed, T+1 opening gap checks should be handled in the trade execution layer.
"""

import numpy as np
import pandas as pd

from indicators import add_all_indicators

STRATEGY_NAME = "renko_chart_select_strategy_v1"


DEFAULT_RENKO_CHART_SELECT_WEIGHTS: dict[str, float] = {
    # Condition 2: relatively negative in the latest attribution result.
    # Keep it only as a weak reference factor.
    "two_day_above_trend_line": 0.3,

    # Condition 3: not strong enough in the latest attribution result.
    # Keep it only as a weak reference factor.
    "short_trend_above_trend_line": 0.3,

    # Condition 4: positive attribution result.
    "close_below_short_trend_cap": 1.5,

    # Condition 5: mildly positive attribution result.
    "price_below_50": 0.8,

    # Condition 6: clearly positive attribution result.
    # It is also required as a hard selection condition in v1.
    "j_momentum_or_low": 2.5,

    # Condition 7: not strong enough in the latest attribution result.
    # Keep it only as a weak reference factor.
    "close_above_yellow_ma": 0.3,

    # Condition 8: clearly negative attribution result.
    # Use a negative weight instead of deleting the column, so it remains measurable.
    "surge_then_shrink_pullback": -2.0,

    # Condition 9: positive attribution result.
    # It is also required as a hard selection condition in v1.
    "small_rise_long_red_brick": 3.0,

    # Former hard-risk rules.
    # In v1 they are no longer direct filters.
    # They are weighted negatively by default and exported as tags for attribution.
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


def add_strategy_score(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Add signed score columns used by renko_chart_select_strategy_v1.

    v1 scoring details:
    - Each boolean condition contributes its configured signed weight when true.
    - Positive weights increase raw_score.
    - Negative weights reduce raw_score.
    - raw_score is the sum of all active signed condition scores.
    - positive_weight_total is the sum of positive weights only.
    - score_pct = raw_score / positive_weight_total * 100.
    - score_abs_pct = raw_score / absolute_weight_total * 100.
    - score_pct is not used as a hard selection filter in v1.
    - score_pct remains in the output for later bucket analysis.

    Exact default weights:
    - two_day_above_trend_line: 0.3
    - short_trend_above_trend_line: 0.3
    - close_below_short_trend_cap: 1.5
    - price_below_50: 0.8
    - j_momentum_or_low: 2.5
    - close_above_yellow_ma: 0.3
    - surge_then_shrink_pullback: -2.0
    - small_rise_long_red_brick: 3.0
    - prior_20d_accelerated_huge_volume_bear: -1.5
    - prior_20d_shrink_limit_up: -1.2
    - long_lower_shadow_hammer: -1.0
    - limit_up_red_brick: -1.3
    """
    out = df.copy()
    weights = DEFAULT_RENKO_CHART_SELECT_WEIGHTS if weights is None else weights

    out["raw_score"] = 0.0

    for col, weight in weights.items():
        weight = float(weight)

        if col not in out.columns:
            out[col] = False

        out[f"{col}_weight"] = weight
        out[f"{col}_score"] = np.where(
            out[col].fillna(False).astype(bool),
            weight,
            0.0,
        )
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

    v1 change from v0:
    - The four former hard-risk rules are no longer used to reject candidates.
    - They are retained as boolean columns, weighted score factors, and tag outputs.
    - risk_tag_any is true when at least one former hard-risk condition is true.
    - risk_tag_count counts how many former hard-risk conditions are true.
    - risk_tags stores triggered condition names as a pipe-separated string.
    - risk_filter_pass is kept as True only for backward compatibility.
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

    # Backward-compatible column.
    # It is no longer used as a hard rejection filter in v1.
    out["risk_filter_pass"] = True

    return out


def select_renko_chart(
    df: pd.DataFrame,
    n1: int = 4,
    n2: int = 6,
    **kwargs,
) -> pd.DataFrame:
    """
    Build the v1 renko chart selection result from reusable indicators.

    v1 hard selection rules:
    - hard_brick_turn_strong must be true.
    - Condition 6, j_momentum_or_low, must be true.
    - Condition 9, small_rise_long_red_brick, must be true.

    v1 non-filter rules:
    - score_pct is calculated but does not filter candidates.
    - score_pct > 90 is not rejected.
    - score_pct 80-90 is not specially required.
    - Former hard-risk rules are tagged and scored, but not directly filtered.
    - T+1 opening gap is not checked here to avoid future-data leakage.

    Output fields:
    - selected_score_base: candidate passes the three v1 hard rules.
    - selected: same as selected_score_base in v1.
    - score_pct: signed raw_score divided by positive_weight_total.
    - score_abs_pct: signed raw_score divided by absolute_weight_total.
    - risk_tag_any, risk_tag_count, risk_tags: former hard-risk attribution fields.
    """
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
    out = add_strategy_risk_tags(out)

    out["condition6_hard_pass"] = out["j_momentum_or_low"].fillna(False).astype(bool)
    out["condition9_hard_pass"] = out["small_rise_long_red_brick"].fillna(False).astype(bool)

    out["selected_score_base"] = (
        out["hard_brick_turn_strong"].fillna(False).astype(bool)
        & out["condition6_hard_pass"]
        & out["condition9_hard_pass"]
    ).astype(int)

    out["selected"] = (
        out["hard_brick_turn_strong"].fillna(False).astype(bool)
        & out["condition6_hard_pass"]
        & out["condition9_hard_pass"]
    ).astype(int)

    out["selection_strategy"] = STRATEGY_NAME

    return out


SELECT_FUNC = select_renko_chart
