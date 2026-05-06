from __future__ import annotations

"""
Oversold rebound selection strategy v0.

Purpose:
    Build a candidate pool for the T1 open buy -> T2 close sell trading model.

Core idea:
    This strategy is NOT a trend continuation model.
    It is an oversold-rebound model.

Main hypothesis:
    Low KDJ J values indicate short-term oversold status.
    Based on previous data validation, low J zones are positive for:
        T1 open buy -> T2 close sell

Strategy logic:
    selected =
        J < 20
        AND T0 daily return > -5%
        AND close / short_trend > 0.92

Meaning:
    1. J < 20:
        The stock is in a short-term oversold state.

    2. daily_return_pct > -5:
        Avoid stocks that are still in strong downside collapse on T0.

    3. close / short_trend > 0.92:
        Avoid stocks that have fallen too far below the short trend line.
        This is used to reduce weak-trend continuation risk.

Notes:
    - This strategy only creates the T0 candidate pool.
    - T1 open gap filtering should be handled by the trade strategy.
    - Recommended trade rule:
        Buy at T1 open only if T1 open gap is between -2% and +2%.
        Sell at T2 close.
"""

import numpy as np
import pandas as pd

from indicators import add_all_indicators


STRATEGY_NAME = "oversold_rebound_select_strategy_v0"


# =============================================================================
# Strategy parameters
# =============================================================================

# 一级筛选：J 低位
J_MAX = 20.0

# 避免 T0 当日极端杀跌
MIN_DAILY_RETURN_PCT = -5.0

# 避免价格严重跌破短趋势
MIN_CLOSE_TO_SHORT_TREND_RATIO = 0.92

# 是否过滤 ST
EXCLUDE_ST = True

# 是否过滤价格异常高的票
MAX_CLOSE_PRICE = 50.0

# 是否过滤成交量为 0 / 无效行情
REQUIRE_POSITIVE_VOLUME = True


# =============================================================================
# Required columns
# =============================================================================

REQUIRED_BASE_COLUMNS: set[str] = {
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


# 下面这些列名会做兼容识别，不强制要求完全同名
J_COLUMN_CANDIDATES = [
    "j",
    "J",
    "kdj_j",
    "KDJ_J",
    "j_value",
    "kdj_j_value",
]

DAILY_RETURN_COLUMN_CANDIDATES = [
    "daily_return_pct",
    "pct_change",
    "close_pct_change",
    "ret_pct",
]

SHORT_TREND_COLUMN_CANDIDATES = [
    "short_trend",
    "short_trend_line",
    "yellow_ma",
    "ma_short",
]


# =============================================================================
# Helper functions
# =============================================================================

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Find a column from candidate names, case-insensitive.
    """

    exact_map = {str(c): c for c in df.columns}
    lower_map = {str(c).lower(): c for c in df.columns}

    for c in candidates:
        if c in exact_map:
            return exact_map[c]

        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None


def _to_bool_series(value, index: pd.Index) -> pd.Series:
    """
    Convert scalar/list/Series to bool Series.
    """

    if isinstance(value, pd.Series):
        return value.fillna(False).astype(bool)

    return pd.Series(bool(value), index=index)


def _get_j_series(df: pd.DataFrame) -> pd.Series:
    col = _find_col(df, J_COLUMN_CANDIDATES)

    if col is None:
        raise ValueError(
            "Cannot find KDJ J column. "
            f"Candidates={J_COLUMN_CANDIDATES}. "
            f"Current columns={list(df.columns)}"
        )

    return pd.to_numeric(df[col], errors="coerce")


def _get_daily_return_pct(df: pd.DataFrame) -> pd.Series:
    """
    Get T0 daily return pct.

    Prefer existing daily_return_pct column.
    If not found, calculate from previous close by symbol:
        close / previous_close - 1
    """

    col = _find_col(df, DAILY_RETURN_COLUMN_CANDIDATES)

    if col is not None:
        return pd.to_numeric(df[col], errors="coerce")

    if "close" not in df.columns:
        raise ValueError("Cannot calculate daily_return_pct because close column is missing.")

    out = df.copy()

    if "symbol" in out.columns and "date" in out.columns:
        out = out.sort_values(["symbol", "date"]).copy()
        prev_close = out.groupby("symbol", sort=False)["close"].shift(1)
        daily_return_pct = (pd.to_numeric(out["close"], errors="coerce") / prev_close - 1.0) * 100.0

        # restore original order
        daily_return_pct = daily_return_pct.reindex(out.index)
        result = pd.Series(index=df.index, dtype="float64")
        result.loc[out.index] = daily_return_pct
        return result

    prev_close = pd.to_numeric(df["close"], errors="coerce").shift(1)
    return (pd.to_numeric(df["close"], errors="coerce") / prev_close - 1.0) * 100.0


def _get_short_trend(df: pd.DataFrame) -> pd.Series:
    col = _find_col(df, SHORT_TREND_COLUMN_CANDIDATES)

    if col is None:
        raise ValueError(
            "Cannot find short trend column. "
            f"Candidates={SHORT_TREND_COLUMN_CANDIDATES}. "
            f"Current columns={list(df.columns)}"
        )

    return pd.to_numeric(df[col], errors="coerce")


def _get_stock_name_series(df: pd.DataFrame) -> pd.Series:
    for col in ["name", "stock_name", "股票名称", "证券简称"]:
        if col in df.columns:
            return df[col].astype(str)

    return pd.Series("", index=df.index)


def _is_st_stock(df: pd.DataFrame) -> pd.Series:
    """
    Detect ST stocks by name if stock name is available.
    If name is unavailable, return False.
    """

    name = _get_stock_name_series(df)

    return (
        name.str.contains("ST", case=False, na=False)
        | name.str.contains("退", case=False, na=False)
        | name.str.contains("\\*ST", case=False, na=False)
    )


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


# =============================================================================
# Main strategy function
# =============================================================================

def select(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Select oversold rebound candidates.

    Parameters
    ----------
    df:
        Daily market data for one stock or a multi-stock dataframe.
        It should contain at least:
            date, symbol, open, high, low, close, volume

        If indicators are not already available, this function will call:
            add_all_indicators(df)

    Returns
    -------
    pd.DataFrame
        Selected rows with strategy columns attached.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    missing_base_cols = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing_base_cols:
        raise ValueError(
            f"{STRATEGY_NAME}: missing required base columns: {missing_base_cols}. "
            f"Current columns={list(df.columns)}"
        )

    work = df.copy()

    # Ensure date sorting before calculating indicators / previous close.
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"])

    if "symbol" in work.columns:
        work = work.sort_values(["symbol", "date"]).reset_index(drop=True)
    else:
        work = work.sort_values("date").reset_index(drop=True)

    # Add indicators.
    # If your pipeline already added indicators, add_all_indicators should keep existing data compatible.
    work = add_all_indicators(work)

    close = pd.to_numeric(work["close"], errors="coerce")
    volume = pd.to_numeric(work["volume"], errors="coerce")

    j = _get_j_series(work)
    daily_return_pct = _get_daily_return_pct(work)
    short_trend = _get_short_trend(work)

    close_to_short_trend_ratio = _safe_ratio(close, short_trend)

    # -------------------------------------------------------------------------
    # Level 1: oversold condition
    # -------------------------------------------------------------------------
    condition_j_low = j < J_MAX

    # -------------------------------------------------------------------------
    # Level 2: avoid weak continuation / collapse
    # -------------------------------------------------------------------------
    condition_not_big_selloff = daily_return_pct > MIN_DAILY_RETURN_PCT
    condition_not_far_below_short_trend = close_to_short_trend_ratio > MIN_CLOSE_TO_SHORT_TREND_RATIO

    # -------------------------------------------------------------------------
    # Basic market data filters
    # -------------------------------------------------------------------------
    condition_valid_price = close.notna() & (close > 0)

    if MAX_CLOSE_PRICE is not None:
        condition_price_cap = close <= MAX_CLOSE_PRICE
    else:
        condition_price_cap = pd.Series(True, index=work.index)

    if REQUIRE_POSITIVE_VOLUME:
        condition_positive_volume = volume.notna() & (volume > 0)
    else:
        condition_positive_volume = pd.Series(True, index=work.index)

    if EXCLUDE_ST:
        condition_not_st = ~_is_st_stock(work)
    else:
        condition_not_st = pd.Series(True, index=work.index)

    # -------------------------------------------------------------------------
    # Final selected rule
    # -------------------------------------------------------------------------
    selected = (
        condition_j_low
        & condition_not_big_selloff
        & condition_not_far_below_short_trend
        & condition_valid_price
        & condition_price_cap
        & condition_positive_volume
        & condition_not_st
    )

    # -------------------------------------------------------------------------
    # Attach strategy fields for later analysis
    # -------------------------------------------------------------------------
    work["strategy_name"] = STRATEGY_NAME
    work["selected"] = selected.fillna(False).astype(bool)

    work["j"] = j
    work["daily_return_pct"] = daily_return_pct
    work["short_trend"] = short_trend
    work["close_to_short_trend_ratio"] = close_to_short_trend_ratio

    work["oversold_j_low"] = condition_j_low.fillna(False).astype(bool)
    work["oversold_not_big_selloff"] = condition_not_big_selloff.fillna(False).astype(bool)
    work["oversold_not_far_below_short_trend"] = condition_not_far_below_short_trend.fillna(False).astype(bool)
    work["oversold_valid_price"] = condition_valid_price.fillna(False).astype(bool)
    work["oversold_price_cap"] = condition_price_cap.fillna(False).astype(bool)
    work["oversold_positive_volume"] = condition_positive_volume.fillna(False).astype(bool)
    work["oversold_not_st"] = condition_not_st.fillna(False).astype(bool)

    work["oversold_model_level_1"] = condition_j_low.fillna(False).astype(bool)
    work["oversold_model_quality_filter"] = (
        condition_not_big_selloff
        & condition_not_far_below_short_trend
        & condition_valid_price
        & condition_price_cap
        & condition_positive_volume
        & condition_not_st
    ).fillna(False).astype(bool)

    selected_df = work[work["selected"]].copy()

    return selected_df


# =============================================================================
# Optional aliases for compatibility with different selector loaders
# =============================================================================

def run(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return select(df, **kwargs)


def apply_strategy(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return select(df, **kwargs)


def get_strategy_name() -> str:
    return STRATEGY_NAME