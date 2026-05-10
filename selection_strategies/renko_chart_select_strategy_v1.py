# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd


STRATEGY_NAME = "renko_chart_select_strategy_v1"

RENKO_RATIO_THRESHOLD = 0.75
DAILY_RETURN_MIN = 3.0
CLOSE_TO_MA5_MIN = 0.0
CLOSE_TO_MA5_MAX = 1.0


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    v0 优化版条件：

    1. 前一砖是绿砖：
       REF(renko_value, 2) > REF(renko_value, 1)

    2. 今天是红砖：
       REF(renko_value, 1) < renko_value

    3. 红砖长度 > 绿砖长度 * 0.75：
       renko_value > REF(renko_value,1) + (REF(renko_value,2)-REF(renko_value,1))*0.75

    4. T0 当日涨幅 >= 3：
       daily_return_pct >= 3

    5. T0 收盘价刚站上 MA5：
       0 <= close_to_ma5 <= 1

    注意：
    close_to_ma5 不再从 indicator cache 读取，而是在策略内临时计算。
    """
    out = df.copy()

    required = [
        "date",
        "close",
        "ma5",
        "daily_return_pct",
        "renko_value",
    ]

    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"{STRATEGY_NAME} missing required columns: {missing}")

    out = out.sort_values("date").reset_index(drop=True)

    out["close"] = _to_numeric(out["close"])
    out["ma5"] = _to_numeric(out["ma5"])
    out["daily_return_pct"] = _to_numeric(out["daily_return_pct"])
    out["renko_value"] = _to_numeric(out["renko_value"])

    out["renko_ref1"] = out["renko_value"].shift(1)
    out["renko_ref2"] = out["renko_value"].shift(2)

    out["green_len"] = out["renko_ref2"] - out["renko_ref1"]
    out["red_len"] = out["renko_value"] - out["renko_ref1"]

    out["close_to_ma5_tmp"] = (out["close"] / out["ma5"].replace(0, np.nan) - 1.0) * 100.0

    cond_prev_green = out["renko_ref2"] > out["renko_ref1"]
    cond_today_red = out["renko_ref1"] < out["renko_value"]

    cond_red_len_ok = (
        out["renko_value"]
        > out["renko_ref1"] + out["green_len"] * RENKO_RATIO_THRESHOLD
    )

    cond_daily_return = out["daily_return_pct"] >= DAILY_RETURN_MIN

    cond_close_to_ma5 = (
        (out["close_to_ma5_tmp"] >= CLOSE_TO_MA5_MIN)
        & (out["close_to_ma5_tmp"] <= CLOSE_TO_MA5_MAX)
    )

    selected = (
        cond_prev_green
        & cond_today_red
        & cond_red_len_ok
        & cond_daily_return
        & cond_close_to_ma5
    )

    out["selected"] = selected.fillna(False).astype(int)
    out["selection_strategy"] = STRATEGY_NAME

    out["selected_score_base"] = 0
    out.loc[out["selected"] == 1, "selected_score_base"] = 1

    # 调试字段保留在 pool 里，方便后续分析
    out["v0_cond_prev_green"] = cond_prev_green.fillna(False).astype(int)
    out["v0_cond_today_red"] = cond_today_red.fillna(False).astype(int)
    out["v0_cond_red_len_ok"] = cond_red_len_ok.fillna(False).astype(int)
    out["v0_cond_daily_return_ge_3"] = cond_daily_return.fillna(False).astype(int)
    out["v0_cond_close_to_ma5_0_1"] = cond_close_to_ma5.fillna(False).astype(int)

    out["v0_close_to_ma5_tmp"] = out["close_to_ma5_tmp"]
    out["v0_green_len"] = out["green_len"]
    out["v0_red_len"] = out["red_len"]

    return out


def select(df: pd.DataFrame) -> pd.DataFrame:
    return apply_strategy(df)


def run(df: pd.DataFrame) -> pd.DataFrame:
    return apply_strategy(df)


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    return apply_strategy(df)