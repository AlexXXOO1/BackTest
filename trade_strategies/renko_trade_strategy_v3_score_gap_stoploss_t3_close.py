from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from .base import TradeRecord


STRATEGY_NAME = "renko_trade_strategy_v3_score_gap_stoploss_t3_close"

MIN_T1_OPEN_GAP_PCT = -2.0
MAX_T1_OPEN_GAP_PCT = 2.0
STOP_LOSS_RATIO = 0.97


def _is_true_value(value) -> bool:
    """Return True for common boolean-like values from pandas, parquet, or CSV rows."""
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) == 1.0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "t"}
    return False


def _pct_change(current: float, base: float) -> float:
    """Calculate percentage change from base to current."""
    if base <= 0 or pd.isna(base) or pd.isna(current):
        return np.nan
    return (float(current) / float(base) - 1.0) * 100.0


def _safe_numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    """Convert a pandas Series to numeric values without raising on invalid inputs."""
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _score_bucket_allowed(score: float) -> bool:
    """
    Exclude open score intervals:
    - 80 < score_pct < 85
    - 90 < score_pct < 95

    Boundary values are allowed:
    - 80, 85, 90, 95 are all allowed.
    """
    if pd.isna(score):
        return False

    score = float(score)

    if 80.0 < score < 85.0:
        return False

    if 90.0 < score < 95.0:
        return False

    return True


def select_candidates(signal_df: pd.DataFrame, signal_date: pd.Timestamp, config) -> pd.DataFrame:
    """
    Return ranked candidates for one signal date.

    Rules:
    1. Require selected=True/1 when the column exists.
    2. Exclude score_pct in open intervals 80-85 and 90-95.
    3. Rank by score_pct descending.
    4. Return multiple rows so the engine can try the next candidate when the
       previous candidate fails the T+1 open gap filter.
    """
    if signal_df.empty:
        return signal_df

    df = signal_df.copy()
    before_count = len(df)

    if "selected" in df.columns:
        df["selected_bool"] = df["selected"].apply(_is_true_value)
        df = df[df["selected_bool"]].copy()
    else:
        df["selected_bool"] = True

    if df.empty:
        print(
            f"{pd.Timestamp(signal_date).strftime('%Y-%m-%d')} {STRATEGY_NAME} "
            f"no candidate after filter: selected=True required. Original signals: {before_count}."
        )
        return pd.DataFrame(columns=signal_df.columns)

    if "score_pct" in df.columns:
        df["score_pct"] = _safe_numeric(df["score_pct"], default=np.nan)
    else:
        df["score_pct"] = np.nan

    df["score_bucket_allowed"] = df["score_pct"].apply(_score_bucket_allowed)
    df = df[df["score_bucket_allowed"]].copy()

    if df.empty:
        print(
            f"{pd.Timestamp(signal_date).strftime('%Y-%m-%d')} {STRATEGY_NAME} "
            f"no candidate after score bucket filter. "
            f"Excluded intervals: 80<score_pct<85 and 90<score_pct<95. "
            f"Original signals: {before_count}."
        )
        return pd.DataFrame(columns=signal_df.columns)

    for bool_col in [
        "close_above_short_trend_cap",
        "small_rise_long_red_brick",
        "j_v3_range",
        "risk_filter_pass",
    ]:
        if bool_col in df.columns:
            df[f"{bool_col}_bool"] = df[bool_col].apply(_is_true_value)
        else:
            df[f"{bool_col}_bool"] = False

    if "pct_change_close" in df.columns:
        df["pct_change_close"] = pd.to_numeric(df["pct_change_close"], errors="coerce")
    else:
        df["pct_change_close"] = np.nan

    if "current_red_height" in df.columns:
        df["current_red_height"] = pd.to_numeric(df["current_red_height"], errors="coerce")
    else:
        df["current_red_height"] = np.nan

    if "symbol" not in df.columns:
        df["symbol"] = df.get("file", "")

    df = df.sort_values(
        by=[
            "score_pct",
            "close_above_short_trend_cap_bool",
            "small_rise_long_red_brick_bool",
            "j_v3_range_bool",
            "risk_filter_pass_bool",
            "pct_change_close",
            "current_red_height",
            "symbol",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            True,
        ],
        na_position="last",
    ).reset_index(drop=True)

    top = df.iloc[0]

    print(
        f"{pd.Timestamp(signal_date).strftime('%Y-%m-%d')} {STRATEGY_NAME} ranked candidates: "
        f"{len(df)} / {before_count}"
        f" | top={top.get('symbol', top.get('file', ''))}"
        f" | top_score_pct={top.get('score_pct', np.nan)}"
        f" | T+1 open gap will be checked during execution."
    )

    return df


def execute_trade_renko_trade_strategy_v3_score_gap_stoploss_t3_close(
    df: pd.DataFrame,
    signal_date: pd.Timestamp,
    capital_alloc: float,
    lot_size: int = 100,
    commission_rate: float = 0.0003,
    stamp_tax_rate: float = 0.001,
    slippage_rate: float = 0.0,
    code: str = "",
    file_name: str = "",
    **kwargs,
) -> Optional[TradeRecord]:
    """
    Execute a v3 score-gap strategy with T0-close-based stop loss.

    Trading logic:
    - T0 is the signal date.
    - Exclude score_pct in open intervals 80-85 and 90-95.
    - Buy only when -2% <= T+1 open gap <= 2%.
    - Buy at T+1 open.
    - stop_price = T0 close * 0.98.
    - If T+1 close < stop_price, sell at T+2 open.
    - Else if T+2 close < stop_price, sell at T+3 open.
    - Else sell at T+3 close.
    """
    if df.empty:
        return None

    df = df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    signal_ts = pd.Timestamp(signal_date).normalize()
    idx_list = df.index[df["date"].dt.normalize() == signal_ts].tolist()
    if not idx_list:
        return None

    signal_idx = int(idx_list[0])
    buy_idx = signal_idx + 1
    t2_idx = signal_idx + 2
    t3_idx = signal_idx + 3

    if buy_idx >= len(df) or t2_idx >= len(df):
        return None

    signal_row = df.loc[signal_idx]

    if "selected" in signal_row.index and not _is_true_value(signal_row.get("selected")):
        return None

    score_pct = pd.to_numeric(
        pd.Series([signal_row.get("score_pct", np.nan)]),
        errors="coerce",
    ).iloc[0]

    if not _score_bucket_allowed(score_pct):
        return None

    t0_close = float(df.loc[signal_idx, "close"])
    t1_open_raw = float(df.loc[buy_idx, "open"])
    t1_close = float(df.loc[buy_idx, "close"])

    if t0_close <= 0 or t1_open_raw <= 0:
        return None

    t1_open_gap_pct = _pct_change(t1_open_raw, t0_close)

    if pd.isna(t1_open_gap_pct):
        return None

    if not (MIN_T1_OPEN_GAP_PCT <= t1_open_gap_pct <= MAX_T1_OPEN_GAP_PCT):
        print(
            f"{pd.Timestamp(signal_date).strftime('%Y-%m-%d')} {STRATEGY_NAME} skip: "
            f"{code or file_name}"
            f" | score_pct={score_pct}"
            f" | T+1 open gap={t1_open_gap_pct:.4f}%"
            f" | required {MIN_T1_OPEN_GAP_PCT}% to {MAX_T1_OPEN_GAP_PCT}%."
        )
        return None

    stop_price = t0_close * STOP_LOSS_RATIO

    exit_rule = ""
    sell_idx = None
    sell_price_raw = np.nan

    if t1_close < stop_price:
        sell_idx = t2_idx
        sell_price_raw = float(df.loc[t2_idx, "open"])
        exit_rule = "stop_loss_t1_close_below_t0_close_98pct_sell_t2_open"
    else:
        if t3_idx >= len(df):
            return None

        t2_close = float(df.loc[t2_idx, "close"])

        if t2_close < stop_price:
            sell_idx = t3_idx
            sell_price_raw = float(df.loc[t3_idx, "open"])
            exit_rule = "stop_loss_t2_close_below_t0_close_98pct_sell_t3_open"
        else:
            sell_idx = t3_idx
            sell_price_raw = float(df.loc[t3_idx, "close"])
            exit_rule = "force_exit_t3_close"

    buy_price = t1_open_raw * (1.0 + float(slippage_rate))
    sell_price = sell_price_raw * (1.0 - float(slippage_rate))

    if buy_price <= 0 or sell_price <= 0 or capital_alloc <= 0:
        return None

    shares = math.floor(float(capital_alloc) / buy_price / int(lot_size)) * int(lot_size)
    if shares <= 0:
        return None

    buy_amount = shares * buy_price
    buy_cost = buy_amount * float(commission_rate)

    sell_amount = shares * sell_price
    sell_cost = sell_amount * (float(commission_rate) + float(stamp_tax_rate))

    gross_pnl = sell_amount - buy_amount
    net_pnl = sell_amount - sell_cost - buy_amount - buy_cost
    ret_pct = net_pnl / (buy_amount + buy_cost) * 100.0 if (buy_amount + buy_cost) > 0 else np.nan

    t1_close_ret_pct = _pct_change(t1_close, buy_price)

    t2_close_ret_pct = np.nan
    if t2_idx < len(df):
        t2_close_ret_pct = _pct_change(float(df.loc[t2_idx, "close"]), buy_price)

    t3_close_ret_pct = np.nan
    if t3_idx < len(df):
        t3_close_ret_pct = _pct_change(float(df.loc[t3_idx, "close"]), buy_price)

    print(
        f"{pd.Timestamp(signal_date).strftime('%Y-%m-%d')} {STRATEGY_NAME} executed: "
        f"{code or file_name}"
        f" | score_pct={score_pct}"
        f" | T+1 open gap={t1_open_gap_pct:.4f}%"
        f" | stop_price={stop_price:.4f}"
        f" | exit_rule={exit_rule}"
        f" | buy={buy_price:.4f}"
        f" | sell={sell_price:.4f}"
        f" | net_pnl={net_pnl:.2f}"
        f" | ret_pct={ret_pct:.4f}%"
    )

    return TradeRecord(
        code=code,
        file=file_name,
        signal_date=str(pd.Timestamp(df.loc[signal_idx, "date"]).date()),
        buy_date=str(pd.Timestamp(df.loc[buy_idx, "date"]).date()),
        buy_price=round(float(buy_price), 4),
        shares=int(shares),
        buy_amount=round(float(buy_amount), 2),
        buy_cost=round(float(buy_cost), 2),
        exit_rule=(
            f"{exit_rule}; T0 signal; score bucket filter; "
            f"-2%<=T+1 open gap<=2%; stop_price=T0 close*0.98."
        ),
        sell_date=str(pd.Timestamp(df.loc[sell_idx, "date"]).date()),
        sell_price=round(float(sell_price), 4),
        sell_amount=round(float(sell_amount), 2),
        sell_cost=round(float(sell_cost), 2),
        gross_pnl=round(float(gross_pnl), 2),
        net_pnl=round(float(net_pnl), 2),
        ret_pct=round(float(ret_pct), 4) if not pd.isna(ret_pct) else np.nan,
        hold_days=int(sell_idx - buy_idx + 1),
        t2_close_ret_pct=round(float(t2_close_ret_pct), 4) if not pd.isna(t2_close_ret_pct) else np.nan,
        trade_strategy=STRATEGY_NAME,
    )


EXECUTE_FUNC = execute_trade_renko_trade_strategy_v3_score_gap_stoploss_t3_close
