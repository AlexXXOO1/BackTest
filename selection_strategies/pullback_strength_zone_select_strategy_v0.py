# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd


STRATEGY_NAME = "pullback_strength_zone_select_strategy_v0"


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _add_score_pct(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()

    selected_score = pd.to_numeric(
        out.loc[out["selected"] == 1, "score_rank_key"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if selected_score.empty:
        out["score_pct"] = 0.0
        return out

    min_s = float(selected_score.min())
    max_s = float(selected_score.max())
    denom = max(max_s - min_s, 1e-9)

    out["score_pct"] = (
        (pd.to_numeric(out["score_rank_key"], errors="coerce") - min_s)
        / denom
        * 100.0
    ).clip(0.0, 100.0)

    out["score_pct"] = (
        out["score_pct"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return out


def add_pullback_strength_zone_features(
    df: pd.DataFrame,
    short_window: int = 3,
    long_window: int = 21,
) -> pd.DataFrame:
    out = df.copy()

    required = ["date", "symbol", "close", "low"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n1 = int(short_window)
    n2 = int(long_window)

    if n1 <= 0 or n2 <= 0:
        raise ValueError("short_window and long_window must be positive integers.")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")

    out = (
        out.dropna(subset=["date", "symbol"])
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    g = out.groupby("symbol", group_keys=False)

    short_llv_col = f"short_llv_low_{n1}"
    short_hhv_col = f"short_hhv_close_{n1}"
    long_llv_col = f"long_llv_low_{n2}"
    long_hhv_col = f"long_hhv_close_{n2}"

    short_pos_col = f"short_pos_{n1}"
    long_pos_col = f"long_pos_{n2}"

    out[short_llv_col] = (
        g["low"]
        .rolling(n1, min_periods=n1)
        .min()
        .reset_index(level=0, drop=True)
    )

    out[short_hhv_col] = (
        g["close"]
        .rolling(n1, min_periods=n1)
        .max()
        .reset_index(level=0, drop=True)
    )

    out[long_llv_col] = (
        g["low"]
        .rolling(n2, min_periods=n2)
        .min()
        .reset_index(level=0, drop=True)
    )

    out[long_hhv_col] = (
        g["close"]
        .rolling(n2, min_periods=n2)
        .max()
        .reset_index(level=0, drop=True)
    )

    close = out["close"]

    out[short_pos_col] = (
        _safe_div(close - out[short_llv_col], out[short_hhv_col] - out[short_llv_col])
        * 100.0
    )

    out[long_pos_col] = (
        _safe_div(close - out[long_llv_col], out[long_hhv_col] - out[long_llv_col])
        * 100.0
    )

    out[short_pos_col] = out[short_pos_col].replace([np.inf, -np.inf], np.nan)
    out[long_pos_col] = out[long_pos_col].replace([np.inf, -np.inf], np.nan)

    out[f"{short_pos_col}_le_20"] = (
        pd.to_numeric(out[short_pos_col], errors="coerce") <= 20.0
    ).fillna(False).astype(bool)

    out[f"{long_pos_col}_ge_60"] = (
        pd.to_numeric(out[long_pos_col], errors="coerce") >= 60.0
    ).fillna(False).astype(bool)

    out[f"{long_pos_col}_ge_80"] = (
        pd.to_numeric(out[long_pos_col], errors="coerce") >= 80.0
    ).fillna(False).astype(bool)

    signal_60 = out[f"{short_pos_col}_le_20"] & out[f"{long_pos_col}_ge_60"]
    signal_80 = out[f"{short_pos_col}_le_20"] & out[f"{long_pos_col}_ge_80"]

    out["pullback_strength_zone_v0"] = signal_60.fillna(False).astype(bool)

    out["tdx_signal_short_le_20_long_ge_60"] = np.where(signal_60, 10, 0)
    out["tdx_signal_short_le_20_long_ge_80"] = np.where(signal_80, 10, 0)

    if n1 == 3 and short_pos_col != "short_pos_3":
        out["short_pos_3"] = out[short_pos_col]

    if n2 == 21 and long_pos_col != "long_pos_21":
        out["long_pos_21"] = out[long_pos_col]

    return out


def select(
    df: pd.DataFrame,
    short_window: int = 3,
    long_window: int = 21,
    short_threshold: float = 20.0,
    long_threshold: float = 60.0,
    **kwargs,
) -> pd.DataFrame:
    out = add_pullback_strength_zone_features(
        df,
        short_window=short_window,
        long_window=long_window,
    )

    short_col = f"short_pos_{int(short_window)}"
    long_col = f"long_pos_{int(long_window)}"

    short_pos = pd.to_numeric(out[short_col], errors="coerce")
    long_pos = pd.to_numeric(out[long_col], errors="coerce")

    mask = (
        (short_pos <= float(short_threshold))
        & (long_pos >= float(long_threshold))
    ).fillna(False)

    out["selected"] = 0
    out.loc[mask, "selected"] = 1

    out["selected_score_base"] = out["selected"]

    out["score_rank_key"] = (
        long_pos - short_pos
    ).replace([np.inf, -np.inf], np.nan).fillna(-9999.0)

    out = _add_score_pct(out)

    out["selection_strategy"] = STRATEGY_NAME

    return out


SELECT_FUNC = select