# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd


STRATEGY_NAME = "renko_chart_select_strategy_v0"


def add_v0_features(df: pd.DataFrame, min_red_green_ratio: float = 0.75) -> pd.DataFrame:
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

    out["renko_ref1"] = ref1
    out["renko_ref2"] = ref2
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


def _add_score_pct(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    selected_score = pd.to_numeric(out.loc[out["selected"] == 1, "score_rank_key"], errors="coerce")
    if selected_score.empty:
        out["score_pct"] = 0.0
        return out
    min_s = selected_score.min()
    max_s = selected_score.max()
    denom = max(float(max_s - min_s), 1e-9)
    out["score_pct"] = ((out["score_rank_key"] - min_s) / denom * 100.0).clip(0.0, 100.0)
    return out


def select(df: pd.DataFrame, min_red_green_ratio: float = 0.5, **kwargs) -> pd.DataFrame:
    out = add_v0_features(df, min_red_green_ratio=min_red_green_ratio)
    out["selected"] = 0
    out["selected_score_base"] = 0

    mask = out["renko_v0_turn_strong"].fillna(False).astype(bool)
    out.loc[mask, "selected"] = 1
    out.loc[mask, "selected_score_base"] = 1

    out["score_rank_key"] = pd.to_numeric(out["red_vs_prev_green_ratio"], errors="coerce")
    out["score_rank_key"] = out["score_rank_key"].replace([np.inf, -np.inf], np.nan).fillna(-9999.0)
    out = _add_score_pct(out)
    out["selection_strategy"] = STRATEGY_NAME
    return out


SELECT_FUNC = select
