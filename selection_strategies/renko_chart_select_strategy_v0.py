from __future__ import annotations

"""
Renko chart selection strategy v0.

Purpose:
- Preserve the earliest TongDaXin XG logic as the baseline strategy.
- Strategy layer only consumes indicator columns and creates selected / score / rank_key.
- The original TongDaXin formula is implemented in indicators/tdx_renko_xg.py.

Original TongDaXin selection rule:

条件1 := REF(砖型图,2) > REF(砖型图,1)
         AND REF(砖型图,1) < 砖型图
         AND 砖型图 > REF(砖型图,1) + (REF(砖型图,2)-REF(砖型图,1))*0.7;

XG: IF(条件1, 1, 0);
"""

import pandas as pd

from indicators import add_all_indicators
from indicators.required import require_indicator_columns


STRATEGY_NAME = "renko_chart_select_strategy_v0"


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
    "brick_open",
    "brick_close",
    "brick_delta",
    "current_red_height",
    "previous_green_height",
    "tdx_renko_condition1",
    "tdx_renko_xg",
    "tdx_renko_xg_int",
}


EXPORT_COLUMNS: list[str] = [
    "date",
    "code",
    "name",
    "stock_name",
    "selection_strategy",
    "selected",
    "selected_score_base",
    "score",
    "score_pct",
    "score_rank_key",
    "tdx_renko_condition1",
    "tdx_renko_xg",
    "tdx_renko_xg_int",
    "brick_value",
    "brick_prev_1",
    "brick_prev_2",
    "brick_open",
    "brick_close",
    "brick_delta",
    "current_red_height",
    "previous_green_height",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def _prepare_indicators(df: pd.DataFrame, n1: int, n2: int, **kwargs) -> pd.DataFrame:
    """
    Support both usage modes:
    1. selector.py passes indicator cache: require all needed columns.
    2. Direct testing passes raw OHLCV data: calculate indicators first.
    """
    if REQUIRED_INDICATOR_COLUMNS.issubset(set(df.columns)):
        out = df.copy()
    else:
        raw_required = {"date", "open", "high", "low", "close", "volume"}
        if raw_required.issubset(set(df.columns)):
            out = add_all_indicators(df, n1=n1, n2=n2, **kwargs)
        else:
            out = df.copy()

    require_indicator_columns(
        df=out,
        required_columns=REQUIRED_INDICATOR_COLUMNS,
        strategy_name=STRATEGY_NAME,
    )
    return out.sort_values("date").reset_index(drop=True)


def select(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    """Select stocks using the original TongDaXin renko XG condition."""
    out = _prepare_indicators(df, n1=n1, n2=n2, **kwargs).copy()

    selected_bool = out["tdx_renko_xg"].fillna(False).astype(bool)

    out["selected"] = selected_bool.astype(int)
    out["selected_score_base"] = out["selected"]

    # v0 is a pure baseline signal. Scores are intentionally simple.
    out["score"] = selected_bool.astype(float)
    out["score_pct"] = selected_bool.astype(float) * 100.0

    # Higher current red height / reversal strength ranks first within same date.
    out["score_rank_key"] = (
        pd.to_numeric(out["current_red_height"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["brick_delta"], errors="coerce").fillna(0.0) * 0.01
    )

    out["selection_strategy"] = STRATEGY_NAME

    existing_export_cols = [c for c in EXPORT_COLUMNS if c in out.columns]
    other_cols = [c for c in out.columns if c not in existing_export_cols]
    return out[existing_export_cols + other_cols]


def apply_strategy(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    return select(df=df, n1=n1, n2=n2, **kwargs)


def run(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    return select(df=df, n1=n1, n2=n2, **kwargs)
