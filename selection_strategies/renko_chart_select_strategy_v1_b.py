from __future__ import annotations

"""
Renko chart selection strategy v1_b.

This strategy uses hard selection conditions only.

Final rule:
1. T0 hard_brick_turn_strong must be true.
2. J confirmation must be true:
   - T0 J is below 14, or
   - T-1 J is greater than T-2 J.

Definitions:
- T0 is the day when hard_brick_turn_strong is true.
- T-1 is the previous trading day.
- T-2 is two trading days before T0.

No weighted score is used.
No score_pct filter is used.
No T+1 opening gap filter is used.
"""

import pandas as pd

from indicators import add_all_indicators

STRATEGY_NAME = "renko_chart_select_strategy_v1_b"


def _find_j_column(df: pd.DataFrame) -> str:
    """
    Find the J-value column from common column names.

    Supported names:
    - j
    - J
    - kdj_j
    - KDJ_J
    - j_value
    - J_VALUE
    """
    candidates = [
        "j",
        "J",
        "kdj_j",
        "KDJ_J",
        "j_value",
        "J_VALUE",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        "J value column not found. Expected one of: "
        "j, J, kdj_j, KDJ_J, j_value, J_VALUE. "
        "Please check the output column name from indicators.py."
    )


def select_renko_chart(
    df: pd.DataFrame,
    n1: int = 4,
    n2: int = 6,
    **kwargs,
) -> pd.DataFrame:
    """
    Build the selection result.

    Final selected rule:
    selected = hard_brick_turn_strong
               AND
               (
                   T0 J < 14
                   OR
                   T-1 J > T-2 J
               )

    Quant details:
    - hard_brick_turn_strong:
      Comes from indicators.add_all_indicators().
      It represents the renko chart turning strong on T0.

    - j_t0_below_14:
      True when:
          J_T0 < 14

    - j_tminus1_rise_vs_tminus2:
      True when:
          J_T-1 > J_T-2

    - j_low_or_tminus_rise:
      True when:
          j_t0_below_14 OR j_tminus1_rise_vs_tminus2

    This strategy intentionally does not use:
    - Weighted scores.
    - score_pct.
    - raw_score.
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

    j_col = _find_j_column(out)

    out["j_t0_below_14"] = (out[j_col] < 14).fillna(False)

    out["j_tminus1_rise_vs_tminus2"] = (
        out[j_col].shift(1) > out[j_col].shift(2)
    ).fillna(False)

    out["j_low_or_tminus_rise"] = (
        out["j_t0_below_14"].astype(bool)
        | out["j_tminus1_rise_vs_tminus2"].astype(bool)
    )

    out["selected_score_base"] = (
        out["hard_brick_turn_strong"].fillna(False).astype(bool)
        & out["j_low_or_tminus_rise"].astype(bool)
    ).astype(int)

    out["selected"] = out["selected_score_base"]

    out["selection_strategy"] = STRATEGY_NAME

    return out


SELECT_FUNC = select_renko_chart