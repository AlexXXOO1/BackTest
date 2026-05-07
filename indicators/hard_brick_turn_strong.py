from __future__ import annotations

import pandas as pd

OUTPUT_COLUMNS = {
    "red_brick", "green_brick", "green_to_red",
    "valid_red_brick", "valid_previous_green_brick", "valid_green_brick",
    "brick_reversal_ratio", "brick_reversal_strength", "hard_brick_turn_strong",
}


def add_hard_brick_turn_strong(df: pd.DataFrame, brick_reversal_ratio: float = 0.70) -> pd.DataFrame:
    """Add reusable brick turn-strong facts used by renko strategies."""
    out = df.copy()
    brick_value = pd.to_numeric(out["brick_value"], errors="coerce")
    brick_prev_1 = pd.to_numeric(out["brick_prev_1"], errors="coerce")
    brick_prev_2 = pd.to_numeric(out["brick_prev_2"], errors="coerce")
    current_red_height = pd.to_numeric(out["current_red_height"], errors="coerce")
    previous_green_height = pd.to_numeric(out["previous_green_height"], errors="coerce")

    out["red_brick"] = brick_value > brick_prev_1
    out["green_brick"] = brick_value < brick_prev_1
    out["green_to_red"] = ((~out["red_brick"].shift(1).fillna(False).astype(bool)) & out["red_brick"]).fillna(False)
    out["valid_red_brick"] = (brick_value > 0).fillna(False)
    out["valid_previous_green_brick"] = (brick_prev_2 > brick_prev_1).fillna(False)
    out["valid_green_brick"] = out["valid_previous_green_brick"]
    out["brick_reversal_ratio"] = current_red_height / previous_green_height.replace(0, pd.NA)
    out["brick_reversal_strength"] = (current_red_height >= previous_green_height * brick_reversal_ratio).fillna(False)
    out["hard_brick_turn_strong"] = (
        out["green_to_red"].astype(bool)
        & out["valid_red_brick"].astype(bool)
        & out["valid_previous_green_brick"].astype(bool)
        & out["brick_reversal_strength"].astype(bool)
    ).fillna(False)
    return out
