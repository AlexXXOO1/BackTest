from __future__ import annotations

"""
Renko chart selection strategy v4.

Refactor rule:
- Strategy layer does not define indicator columns.
- All reusable facts are generated in indicators/ and checked by REQUIRED_INDICATORS.
- Strategy only combines indicators into selected / score / rank_key.
"""

import pandas as pd

from indicators import add_all_indicators
from indicators.required import require_indicator_columns


STRATEGY_NAME = "renko_chart_select_strategy_v4"

REQUIRED_INDICATORS: set[str] = {
    "date", "open", "high", "low", "close", "volume",
    "brick_value", "brick_prev_1", "brick_prev_2",
    "current_red_height", "previous_green_height",
    "daily_return_pct", "short_trend",
    "hard_brick_turn_strong",
    "price_rise_range_and_close_to_short_trend_below_limit",
    "close_to_short_trend",
    "close_to_short_trend_below_084",
    "close_to_short_trend_below_086",
    "close_to_short_trend_below_088",
    "close_to_short_trend_below_090",
    "daily_return_5_to_7",
    "daily_return_55_to_7",
    "daily_return_6_to_7",
    "brick_reversal_ratio",
    "brick_reversal_strength_100",
    "brick_reversal_strength_120",
    "brick_reversal_strength_below_100",
    "brick_reversal_strength_below_090",
}

# Backward-compatible alias used by older tools.
REQUIRED_INDICATOR_COLUMNS = REQUIRED_INDICATORS

DEFAULT_MIN_DAILY_RETURN_PCT = 3.0
DEFAULT_MAX_DAILY_RETURN_PCT = 7.0
DEFAULT_MAX_CLOSE_TO_SHORT_TREND = 0.95

SCORE_WEIGHTS = {
    "close_to_short_trend_below_090": 30.0,
    "close_to_short_trend_below_088": 20.0,
    "close_to_short_trend_below_086": 12.0,
    "close_to_short_trend_below_084": 8.0,
    "daily_return_5_to_7": 10.0,
    "daily_return_55_to_7": 6.0,
    "daily_return_6_to_7": 4.0,
    "brick_reversal_strength_below_100": 4.0,
    "brick_reversal_strength_below_090": 3.0,
    "brick_reversal_strength_100": -22.0,
    "brick_reversal_strength_120": -10.0,
}

POSITIVE_MAX_SCORE = sum(v for v in SCORE_WEIGHTS.values() if v > 0)


def _prepare_indicators(df: pd.DataFrame, n1: int, n2: int, **kwargs) -> pd.DataFrame:
    if REQUIRED_INDICATORS.issubset(set(df.columns)):
        out = df.copy().sort_values("date").reset_index(drop=True)
    else:
        out = add_all_indicators(df, n1=n1, n2=n2, **kwargs)
    require_indicator_columns(out, REQUIRED_INDICATORS, STRATEGY_NAME)
    return out


def add_weighted_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    score_cols: list[str] = []
    for col, weight in SCORE_WEIGHTS.items():
        score_col = f"score_{col}" if weight > 0 else f"penalty_{col}"
        out[score_col] = out[col].fillna(False).astype(bool).astype(float) * weight
        score_cols.append(score_col)

    out["score"] = out[score_cols].sum(axis=1)
    out["score_pct"] = out["score"] / POSITIVE_MAX_SCORE * 100.0 if POSITIVE_MAX_SCORE > 0 else 0.0

    close_to_short_trend = pd.to_numeric(out["close_to_short_trend"], errors="coerce")
    daily_return_pct = pd.to_numeric(out["daily_return_pct"], errors="coerce")
    brick_reversal_ratio = pd.to_numeric(out["brick_reversal_ratio"], errors="coerce")

    out["rank_close_to_short_trend"] = close_to_short_trend.fillna(999.0)
    out["rank_daily_return_pct"] = daily_return_pct.fillna(-999.0)
    out["rank_brick_reversal_ratio"] = brick_reversal_ratio.fillna(999.0)

    out["score_rank_key"] = (
        out["score_pct"].fillna(0.0) * 100000.0
        + (1.0 - out["rank_close_to_short_trend"].clip(lower=0.0, upper=2.0)) * 10000.0
        + out["rank_daily_return_pct"].clip(lower=-20.0, upper=20.0) * 100.0
        + (1.0 - out["rank_brick_reversal_ratio"].clip(lower=0.0, upper=3.0)) * 1000.0
    )
    return out


def add_final_selection(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["selected_score_base"] = (
        out["hard_brick_turn_strong"].fillna(False).astype(bool)
        & out["price_rise_range_and_close_to_short_trend_below_limit"].fillna(False).astype(bool)
    ).astype(int)
    out["selected"] = out["selected_score_base"]
    return out


def select_renko_chart(
    df: pd.DataFrame,
    n1: int = 4,
    n2: int = 6,
    min_daily_return_pct: float = DEFAULT_MIN_DAILY_RETURN_PCT,
    max_daily_return_pct: float = DEFAULT_MAX_DAILY_RETURN_PCT,
    max_close_to_short_trend: float = DEFAULT_MAX_CLOSE_TO_SHORT_TREND,
    **kwargs,
) -> pd.DataFrame:
    kwargs = dict(kwargs)
    kwargs.setdefault("min_daily_return_pct", min_daily_return_pct)
    kwargs.setdefault("max_daily_return_pct", max_daily_return_pct)
    kwargs.setdefault("max_close_to_short_trend", max_close_to_short_trend)

    out = _prepare_indicators(df, n1=n1, n2=n2, **kwargs)
    out = add_weighted_score(out)
    out = add_final_selection(out)
    out["selection_strategy"] = STRATEGY_NAME
    return out


def select(
    df: pd.DataFrame,
    n1: int = 4,
    n2: int = 6,
    min_daily_return_pct: float = DEFAULT_MIN_DAILY_RETURN_PCT,
    max_daily_return_pct: float = DEFAULT_MAX_DAILY_RETURN_PCT,
    max_close_to_short_trend: float = DEFAULT_MAX_CLOSE_TO_SHORT_TREND,
    **kwargs,
) -> pd.DataFrame:
    return select_renko_chart(
        df=df,
        n1=n1,
        n2=n2,
        min_daily_return_pct=min_daily_return_pct,
        max_daily_return_pct=max_daily_return_pct,
        max_close_to_short_trend=max_close_to_short_trend,
        **kwargs,
    )


def apply_strategy(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    return select(df=df, n1=n1, n2=n2, **kwargs)


def run(df: pd.DataFrame, n1: int = 4, n2: int = 6, **kwargs) -> pd.DataFrame:
    return select(df=df, n1=n1, n2=n2, **kwargs)


SELECT_FUNC = select
