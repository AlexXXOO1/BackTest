# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

STRATEGY_NAME = "renko_chart_select_strategy_v0"


def add_v0_features(
    df: pd.DataFrame,
    min_red_green_ratio: float = 0.75,
) -> pd.DataFrame:
    out = df.copy()

    required = ["date", "symbol", "renko_value"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    renko = pd.to_numeric(out["renko_value"], errors="coerce")
    ref1 = renko.shift(1)
    ref2 = renko.shift(2)

    out["prev_green_brick_len"] = ref2 - ref1
    out["red_brick_len"] = renko - ref1
    out["red_vs_prev_green_ratio"] = (
        out["red_brick_len"] / out["prev_green_brick_len"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    out["is_prev_green"] = (out["prev_green_brick_len"] > 0).fillna(False).astype(bool)
    out["is_today_red"] = (out["red_brick_len"] > 0).fillna(False).astype(bool)
    out["is_red_len_enough"] = (
        out["red_brick_len"] > out["prev_green_brick_len"] * float(min_red_green_ratio)
    ).fillna(False).astype(bool)

    out["renko_v0_turn_strong"] = (
        out["is_prev_green"] & out["is_today_red"] & out["is_red_len_enough"]
    ).fillna(False).astype(bool)

    return out


def select(
    df: pd.DataFrame,
    min_red_green_ratio: float = 0.5,
    **kwargs,
) -> pd.DataFrame:
    out = add_v0_features(df, min_red_green_ratio=min_red_green_ratio)

    mask = out["renko_v0_turn_strong"].fillna(False).astype(bool)
    out = out.loc[mask].copy()
    out["selection_strategy"] = STRATEGY_NAME

    return out


SELECT_FUNC = select
