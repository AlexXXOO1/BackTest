from __future__ import annotations

"""
Renko chart selection strategy v0.

This strategy keeps only one hard selection condition:

1. hard_brick_turn_strong must be true.

Kept for compatibility:
- The output column selected.
- The output column selected_score_base.
- The output column selection_strategy.
"""

import pandas as pd

from indicators import add_all_indicators

STRATEGY_NAME = "renko_chart_select_strategy_v0"


def select_renko_chart(
    df: pd.DataFrame,
    n1: int = 4,
    n2: int = 6,
    **kwargs,
) -> pd.DataFrame:
    """
    Build the v2_renko_only selection result.

    Final selected rule:
    selected = hard_brick_turn_strong

    This strategy intentionally does not use:
    - J value filters.
    - Weighted scores.
    - score_pct.
    - raw_score.
    - small_rise_long_red_brick as a hard rule.
    - Risk-rule negative weights.
    - T+1 opening gap filtering.
    """
    required_indicator_columns = {
        "hard_brick_turn_strong",
    }

    if required_indicator_columns.issubset(set(df.columns)):
        out = df.copy().sort_values("date").reset_index(drop=True)
    else:
        out = add_all_indicators(df, n1=n1, n2=n2, **kwargs)

    out["selected_score_base"] = (
        out["hard_brick_turn_strong"].fillna(False).astype(bool)
    ).astype(int)

    out["selected"] = out["selected_score_base"]

    out["selection_strategy"] = STRATEGY_NAME

    return out


SELECT_FUNC = select_renko_chart
