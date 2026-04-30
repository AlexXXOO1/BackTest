from __future__ import annotations

"""
Renko chart selection strategy v1.

This strategy is based on renko_chart_select_strategy_v0 and adds
one J-value filter.

Hard selection rules:
1. hard_brick_turn_strong must be true.
2. J condition must be true:
   - j < 0, or
   - 30 <= j <= 50

Final selected rule:
selected = hard_brick_turn_strong AND (j < 0 OR 30 <= j <= 50)

"""

import pandas as pd

from indicators import add_all_indicators

STRATEGY_NAME = "renko_chart_select_strategy_v1_a"


J_COLUMN_CANDIDATES: tuple[str, ...] = (
    "j",
    "J",
    "kdj_j",
    "KDJ_J",
    "kdj_J",
    "KDJJ",
)


def _find_j_column(df: pd.DataFrame) -> str:
    """
    Return the KDJ J column name from the dataframe.

    Different project versions may use different J column names, so this helper
    keeps the strategy compatible as long as one common J column exists.
    """
    for col in J_COLUMN_CANDIDATES:
        if col in df.columns:
            return col

    raise KeyError(
        "Cannot find KDJ J column. Expected one of: "
        + ", ".join(J_COLUMN_CANDIDATES)
    )


def add_j_range_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the J range condition used by this strategy.

    J condition:
    - j < 0, or
    - 30 <= j <= 50

    Output columns:
    - j_lt_0: True when J is below 0.
    - j_30_to_50: True when J is between 30 and 50, inclusive.
    - j_condition_pass: True when either j_lt_0 or j_30_to_50 is true.
    - j_momentum_or_low: backward-compatible alias for older scripts.
    - j_condition_rule: documents the active J rule.
    - j_condition_source_col: records which J column was used.
    """
    out = df.copy()
    j_col = _find_j_column(out)
    j_value = pd.to_numeric(out[j_col], errors="coerce")

    out["j_lt_0"] = j_value < 0
    out["j_30_to_50"] = (j_value >= 30) & (j_value <= 50)
    out["j_condition_pass"] = out["j_lt_0"] | out["j_30_to_50"]

    # Backward-compatible alias.
    out["j_momentum_or_low"] = out["j_condition_pass"]

    out["j_condition_rule"] = "j_lt_0_or_j_30_to_50"
    out["j_condition_source_col"] = j_col

    return out


def select_renko_chart(
    df: pd.DataFrame,
    n1: int = 4,
    n2: int = 6,
    **kwargs,
) -> pd.DataFrame:
    """
    Build the v2_renko_j_range selection result.

    Final selected rule:
    selected = hard_brick_turn_strong AND j_condition_pass

    This strategy intentionally keeps only:
    - Renko chart turning strong.
    - J below 0 or J between 30 and 50.

    """
    required_indicator_columns = {
        "hard_brick_turn_strong",
    }

    if required_indicator_columns.issubset(set(df.columns)):
        out = df.copy().sort_values("date").reset_index(drop=True)
    else:
        out = add_all_indicators(df, n1=n1, n2=n2, **kwargs)

    out = add_j_range_condition(out)

    out["condition6_hard_pass"] = out["j_condition_pass"].fillna(False).astype(bool)

    out["selected_score_base"] = (
        out["hard_brick_turn_strong"].fillna(False).astype(bool)
        & out["condition6_hard_pass"]
    ).astype(int)

    out["selected"] = out["selected_score_base"]

    out["selection_strategy"] = STRATEGY_NAME

    return out


SELECT_FUNC = select_renko_chart
