from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from .base import TradeRecord


STRATEGY_NAME = "renko_trade_strategy_v0"


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


def select_candidates(signal_df: pd.DataFrame, signal_date: pd.Timestamp, config) -> pd.DataFrame:
    """
    Select at most one candidate from the daily pool.

    Rules:
    1. Only rows with selected=True/1 are eligible.
    2. Only one stock is selected per signal date to avoid repeated full-capital orders.
    3. Ranking prefers higher score_pct, valid condition9/condition6, smaller T0 rise,
       stronger red-brick height, and then stable symbol order.
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
        df["score_pct"] = _safe_numeric(df["score_pct"], default=-np.inf)
    else:
        df["score_pct"] = -np.inf

    for bool_col in ["small_rise_long_red_brick", "j_momentum_or_low", "risk_filter_pass"]:
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
            "small_rise_long_red_brick_bool",
            "j_momentum_or_low_bool",
            "risk_filter_pass_bool",
            "pct_change_close",
            "current_red_height",
            "symbol",
        ],
        ascending=[False, False, False, False, True, False, True],
        na_position="last",
    ).reset_index(drop=True)

    selected_df = df.iloc[[0]].reset_index(drop=True)
    row = selected_df.iloc[0]
    print(
        f"{pd.Timestamp(signal_date).strftime('%Y-%m-%d')} {STRATEGY_NAME} auto selected: "
        f"{row.get('symbol', row.get('file', ''))}"
        f" | score_pct={row.get('score_pct', np.nan)}"
        f" | selected={row.get('selected', True)}"
        f" | condition9={row.get('small_rise_long_red_brick', np.nan)}"
        f" | condition6={row.get('j_momentum_or_low', np.nan)}"
    )
    return selected_df


def execute_trade_renko_trade_strategy_v0(
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
    Execute a fixed short-term Renko pool trade.

    Trading logic:
    - T0 is the signal date from the pool.
    - Buy at T+1 open using available capital.
    - Sell at T+3 close.
    - The backtest engine controls position occupancy; while capital is occupied,
      no new position should be opened.
    - This strategy itself only accepts rows where selected=True/1 when the field exists.
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
    sell_idx = signal_idx + 3

    if buy_idx >= len(df) or sell_idx >= len(df):
        return None

    signal_row = df.loc[signal_idx]
    if "selected" in signal_row.index and not _is_true_value(signal_row.get("selected")):
        return None

    buy_price = float(df.loc[buy_idx, "open"]) * (1.0 + float(slippage_rate))
    sell_price = float(df.loc[sell_idx, "close"]) * (1.0 - float(slippage_rate))

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

    t2_close_ret_pct = np.nan
    t2_idx = signal_idx + 2
    if t2_idx < len(df):
        t2_close_ret_pct = _pct_change(float(df.loc[t2_idx, "close"]), buy_price)

    return TradeRecord(
        code=code,
        file=file_name,
        signal_date=str(pd.Timestamp(df.loc[signal_idx, "date"]).date()),
        buy_date=str(pd.Timestamp(df.loc[buy_idx, "date"]).date()),
        buy_price=round(float(buy_price), 4),
        shares=int(shares),
        buy_amount=round(float(buy_amount), 2),
        buy_cost=round(float(buy_cost), 2),
        exit_rule="T0 signal, T+1 open buy, T+3 close sell; skip new entries while capital is occupied.",
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


EXECUTE_FUNC = execute_trade_renko_trade_strategy_v0
