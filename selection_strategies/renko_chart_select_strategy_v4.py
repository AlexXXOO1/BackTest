# -*- coding: utf-8 -*-
"""
renko_chart_select_strategy_v4

Strategy purpose
----------------
This strategy selects stocks that show a Renko-style green-to-red reversal,
with a controlled daily return filter and a z_short_trend_line low-position filter.

Core selection logic
--------------------
A row is selected when all conditions below are true:

1. Renko reversal condition:
   - The previous Renko segment is falling.
   - The current Renko segment turns upward.
   - The current red/upward segment length is at least 70% of the previous
     green/downward segment length.

2. Daily return condition:
   - daily_return_pct > 3.0
   - daily_return_pct <= 7.0

3. Z short trend line position condition:
   - close_to_z_short_trend_line <= -5.0
   - close_to_z_short_trend_line is calculated inside this strategy as:
     (close / z_short_trend_line - 1) * 100

Output columns added by this strategy
-------------------------------------
selected:
    1 if the row is selected by this strategy, otherwise 0.

selection_strategy:
    Strategy name. The value is always "renko_chart_select_strategy_v4".

selected_score_base:
    Basic selected score. It equals selected.

v4_brk:
    Calculated Renko value based on the original formula.

v4_brk1:
    Previous trading day's v4_brk.

v4_brk2:
    v4_brk from two trading days ago.

v4_crh:
    Current red/upward Renko segment length.
    Formula: v4_brk - v4_brk1.

v4_pgh:
    Previous green/downward Renko segment length.
    Formula: v4_brk2 - v4_brk1.

v4_drp:
    Daily return percentage used by this strategy.
    It is copied from daily_return_pct.

v4_close_to_z_short_trend_line:
    Closing price distance from z_short_trend_line in percentage.
    Formula: (close / z_short_trend_line - 1) * 100.

v4_close_to_ma5:
    Closing price distance from MA5 in percentage.
    This is only a review/hint field and does not affect selected.

v4_cond_g2r:
    1 if the Renko direction changes from non-red to red/upward.

v4_cond_vrb:
    1 if v4_brk > 0.

v4_cond_vgb:
    1 if the previous Renko segment was green/downward.

v4_cond_brs:
    1 if the current red/upward Renko segment is at least 70% of the previous
    green/downward segment.

v4_cond_hbts:
    1 if the full Renko reversal condition is satisfied.

v4_cond_prir:
    1 if daily_return_pct is greater than 3.0 and less than or equal to 7.0.

v4_cond_z_short_below_minus_5:
    1 if close_to_z_short_trend_line <= -5.0.

v4_up_hint_score:
    Opportunity hint score. Higher means more positive hint conditions.

v4_risk_hint_score:
    Risk hint score. Higher means more risk hint conditions.

v4_net_hint_score:
    v4_up_hint_score - v4_risk_hint_score.

v4_hint_label:
    up_potential / neutral / risk.
    This is only a manual review label and does not affect selected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


STRATEGY_NAME = "renko_chart_select_strategy_v4"

N1 = 4
N2 = 6

RENKO_RATIO_THRESHOLD = 0.70
DAILY_RETURN_MIN = 3.0
DAILY_RETURN_MAX = 7.0
CLOSE_TO_Z_SHORT_MAX = -5.0


def _num(s: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric values."""
    return pd.to_numeric(s, errors="coerce")


def _tdx_sma(s: pd.Series, n: int, m: int = 1) -> pd.Series:
    """
    Calculate the Tongdaxin-style SMA.

    Tongdaxin formula:
        Y = (M * X + (N - M) * REF(Y, 1)) / N

    With M=1, this is equivalent to an EMA with alpha=1/N.
    """
    return _num(s).ewm(alpha=m / n, adjust=False, min_periods=1).mean()


def _rank_pct(s: pd.Series) -> pd.Series:
    """Historical percentile rank within the same stock."""
    return _num(s).rank(pct=True)


def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply renko_chart_select_strategy_v4 to one stock's historical data.

    The input DataFrame is expected to contain one symbol's daily data.
    """
    out = df.copy()

    required = [
        "date",
        "high",
        "low",
        "close",
        "daily_return_pct",
        "z_short_trend_line",
    ]

    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"{STRATEGY_NAME} missing required columns: {missing}")

    out = out.sort_values("date").reset_index(drop=True)

    high = _num(out["high"])
    low = _num(out["low"])
    close = _num(out["close"])
    daily_return_pct = _num(out["daily_return_pct"])
    z_short_trend_line = _num(out["z_short_trend_line"])

    hh = high.rolling(N1, min_periods=1).max()
    ll = low.rolling(N1, min_periods=1).min()
    hlr = hh - ll

    v1a = np.where(
        hlr == 0,
        0,
        (hh - close) / hlr * 100 - 90,
    )
    v1a = pd.Series(v1a, index=out.index)

    v2a = _tdx_sma(v1a, N1, 1) + 100

    v3a = np.where(
        hlr == 0,
        0,
        (close - ll) / hlr * 100,
    )
    v3a = pd.Series(v3a, index=out.index)

    v4a = _tdx_sma(v3a, N2, 1)
    v5a = _tdx_sma(v4a, N2, 1) + 100
    v6a = v5a - v2a

    brk = pd.Series(
        np.where(v6a > N1, v6a - N1, 0),
        index=out.index,
    )

    brk1 = brk.shift(1)
    brk2 = brk.shift(2)

    red = brk > brk1
    g2r = (~red.shift(1).fillna(False)) & red

    crh = brk - brk1
    pgh = brk2 - brk1

    vrb = brk > 0
    vgb = brk2 > brk1
    brs = crh >= pgh * RENKO_RATIO_THRESHOLD
    hbts = g2r & vrb & vgb & brs

    prir = (
        (daily_return_pct > DAILY_RETURN_MIN)
        & (daily_return_pct <= DAILY_RETURN_MAX)
    )

    close_to_z_short = (
        close / z_short_trend_line.replace(0, np.nan) - 1.0
    ) * 100.0

    z_short_ok = close_to_z_short <= CLOSE_TO_Z_SHORT_MAX

    selected = hbts & prir & z_short_ok

    out["selected"] = selected.fillna(False).astype(int)
    out["selection_strategy"] = STRATEGY_NAME
    out["selected_score_base"] = out["selected"]

    out["v4_brk"] = brk
    out["v4_brk1"] = brk1
    out["v4_brk2"] = brk2
    out["v4_crh"] = crh
    out["v4_pgh"] = pgh
    out["v4_drp"] = daily_return_pct
    out["v4_close_to_z_short_trend_line"] = close_to_z_short

    if "ma5" in out.columns:
        ma5 = _num(out["ma5"])
        out["v4_close_to_ma5"] = (close / ma5.replace(0, np.nan) - 1.0) * 100.0
    else:
        ma5 = pd.Series(np.nan, index=out.index)
        out["v4_close_to_ma5"] = np.nan

    out["v4_cond_g2r"] = g2r.fillna(False).astype(int)
    out["v4_cond_vrb"] = vrb.fillna(False).astype(int)
    out["v4_cond_vgb"] = vgb.fillna(False).astype(int)
    out["v4_cond_brs"] = brs.fillna(False).astype(int)
    out["v4_cond_hbts"] = hbts.fillna(False).astype(int)
    out["v4_cond_prir"] = prir.fillna(False).astype(int)
    out["v4_cond_z_short_below_minus_5"] = z_short_ok.fillna(False).astype(int)

    # =========================
    # Up / risk hint fields
    # These fields are only for manual review.
    # They do not affect selected.
    # =========================

    brk_rank = _rank_pct(brk)
    z_short_rank = _rank_pct(z_short_trend_line)

    out["v4_hint_brk_rank_pct"] = brk_rank
    out["v4_hint_brk_low"] = (brk_rank <= 0.20).fillna(False).astype(int)
    out["v4_hint_brk_high"] = (brk_rank >= 0.80).fillna(False).astype(int)

    out["v4_hint_drp_strong"] = (daily_return_pct >= 4.5).fillna(False).astype(int)
    out["v4_hint_drp_near_limit"] = (daily_return_pct >= 6.5).fillna(False).astype(int)

    out["v4_hint_z_short_rank_pct"] = z_short_rank
    out["v4_hint_z_short_low"] = (z_short_rank <= 0.20).fillna(False).astype(int)
    out["v4_hint_z_short_high"] = (z_short_rank >= 0.80).fillna(False).astype(int)

    if "ma5" in out.columns:
        ma5_rank = _rank_pct(ma5)
        out["v4_hint_ma5_rank_pct"] = ma5_rank
        out["v4_hint_ma5_low"] = (ma5_rank <= 0.20).fillna(False).astype(int)
        out["v4_hint_ma5_high"] = (ma5_rank >= 0.80).fillna(False).astype(int)
    else:
        out["v4_hint_ma5_rank_pct"] = np.nan
        out["v4_hint_ma5_low"] = 0
        out["v4_hint_ma5_high"] = 0

    if "volume_ratio_prev1" in out.columns:
        volume_ratio_prev1 = _num(out["volume_ratio_prev1"])
        volume_rank = volume_ratio_prev1.rank(pct=True)

        out["v4_hint_volume_ratio_prev1_rank_pct"] = volume_rank
        out["v4_hint_volume_extreme"] = (
            volume_rank >= 0.80
        ).fillna(False).astype(int)
    else:
        out["v4_hint_volume_ratio_prev1_rank_pct"] = np.nan
        out["v4_hint_volume_extreme"] = 0

    up_cols = [
        "v4_hint_brk_low",
        "v4_hint_drp_strong",
        "v4_hint_z_short_low",
        "v4_hint_ma5_low",
    ]

    risk_cols = [
        "v4_hint_brk_high",
        "v4_hint_drp_near_limit",
        "v4_hint_z_short_high",
        "v4_hint_ma5_high",
        "v4_hint_volume_extreme",
    ]

    out["v4_up_hint_score"] = out[up_cols].sum(axis=1)
    out["v4_risk_hint_score"] = out[risk_cols].sum(axis=1)
    out["v4_net_hint_score"] = (
        out["v4_up_hint_score"] - out["v4_risk_hint_score"]
    )

    out["v4_hint_label"] = "neutral"
    out.loc[out["v4_net_hint_score"] >= 2, "v4_hint_label"] = "up_potential"
    out.loc[out["v4_net_hint_score"] <= -2, "v4_hint_label"] = "risk"

    return out


def select(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility entry point for the pool builder."""
    return apply_strategy(df)


def run(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility entry point for the pool builder."""
    return apply_strategy(df)


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility entry point for the pool builder."""
    return apply_strategy(df)