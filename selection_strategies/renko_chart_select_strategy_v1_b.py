from __future__ import annotations

"""
Renko chart selection strategy v1_b.

Copy-edit workflow:
1. Copy this file.
2. Change STRATEGY_NAME.
3. Edit only the "Strategy condition block" and final selected rule.

Final selected rule:
selected = hard_brick_turn_strong AND (T0 J < 14 OR T-1 J > T-2 J)
"""

import pandas as pd

from indicators import add_all_indicators

STRATEGY_NAME = "renko_chart_select_strategy_v1_b"

REQUIRED_INDICATOR_COLUMNS: set[str] = {
    "date", "open", "high", "low", "close", "volume",
    "brick_value", "brick_prev_1", "brick_prev_2",
    "current_red_height", "previous_green_height", "J",
}

# =============================================================================
# Strategy condition block
# Edit this block when creating a new strategy version.
# =============================================================================
BRICK_REVERSAL_RATIO = 0.70
LOW_J_THRESHOLD = 14.0


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
    brick_value = pd.to_numeric(out["brick_value"], errors="coerce")
    brick_prev_1 = pd.to_numeric(out["brick_prev_1"], errors="coerce")
    brick_prev_2 = pd.to_numeric(out["brick_prev_2"], errors="coerce")
    current_red_height = pd.to_numeric(out["current_red_height"], errors="coerce")
    previous_green_height = pd.to_numeric(out["previous_green_height"], errors="coerce")

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

    j_col = find_j_column(out)
    j_value = pd.to_numeric(out[j_col], errors="coerce")
    out["j_t0_below_14"] = (j_value < LOW_J_THRESHOLD).fillna(False)
    out["j_tminus1_rise_vs_tminus2"] = (j_value.shift(1) > j_value.shift(2)).fillna(False)
    out["j_low_or_tminus_rise"] = out["j_t0_below_14"].astype(bool) | out["j_tminus1_rise_vs_tminus2"].astype(bool)
    return out


def add_final_selection(df: pd.DataFrame) -> pd.DataFrame:
    """Final selected rule for v1_b."""
    out = df.copy()
    out["selected_score_base"] = (
        out["hard_brick_turn_strong"].fillna(False).astype(bool)
        & out["j_low_or_tminus_rise"].astype(bool)
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
    out = _prepare_indicators(df, n1=n1, n2=n2, **kwargs)
    out = add_strategy_conditions(out)
    out = add_final_selection(out)
    out["selection_strategy"] = STRATEGY_NAME
    return out


SELECT_FUNC = select_renko_chart
