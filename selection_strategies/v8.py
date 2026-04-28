from __future__ import annotations

import pandas as pd

from indicators import add_all_indicators

STRATEGY_NAME = "v8"


def select_v8(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    """
    Build the v8 selection result from reusable indicators.

    Strategy rules:
    - hard_brick_turn_strong must be true.
    - v8_hard_filter_pass must be true.
    - score_pct must be greater than or equal to 60 by default.
    - selected is stored as 1 or 0.
    """
    score_threshold_pct = float(kwargs.get("score_threshold_pct", 60.0))
    out = add_all_indicators(df, n1=n1, n2=n2, **kwargs)
    out["selected_v7_base"] = (out["hard_brick_turn_strong"] & (out["score_pct"] >= score_threshold_pct)).astype(int)
    out["selected"] = (
        out["hard_brick_turn_strong"]
        & out["v8_hard_filter_pass"]
        & (out["score_pct"] >= score_threshold_pct)
    ).astype(int)
    out["selection_strategy"] = STRATEGY_NAME
    return out


SELECT_FUNC = select_v8
