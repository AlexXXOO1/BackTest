from __future__ import annotations

import numpy as np
import pandas as pd

OUTPUT_COLUMNS = {
    "accelerated_huge_volume_bear",
    "prior_20d_accelerated_huge_volume_bear",
    "limit_up",
    "shrink_limit_up",
    "prior_20d_shrink_limit_up",
    "long_lower_shadow_hammer",
    "limit_up_red_brick",
}


def add_renko_v1_risk_flags(
    df: pd.DataFrame,
    high_pos_ratio: float = 0.85,  # kept for signature compatibility; high_position_line already includes it.
    accel_return_pct: float = 12.0,
    huge_volume_ratio: float = 2.0,
    big_bear_body_pct: float = 4.0,
    limit_up_pct: float = 9.7,
    shrink_limit_vol_ratio: float = 0.8,
    hammer_lower_shadow_body_ratio: float = 2.0,
    hammer_lower_shadow_range_ratio: float = 0.5,
    hammer_upper_shadow_body_ratio: float = 1.2,
    hammer_max_body_range_ratio: float = 0.4,
    risk_lookback: int = 20,
) -> pd.DataFrame:
    """Add legacy renko v1 risk flags for scoring/risk-tag attribution."""
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce")
    pct_change_close = pd.to_numeric(out["pct_change_close"], errors="coerce")
    high_position_line = pd.to_numeric(out["high_position_line"], errors="coerce")
    accel_return = pd.to_numeric(out["accel_return_pct"], errors="coerce")
    huge_volume_ma = pd.to_numeric(out["huge_volume_ma"], errors="coerce")
    shrink_volume_ma5 = pd.to_numeric(out["shrink_volume_ma5"], errors="coerce")
    candle_range = pd.to_numeric(out["candle_range"], errors="coerce")
    body_abs = pd.to_numeric(out["body_abs"], errors="coerce")
    bear_body_pct = pd.to_numeric(out["bear_body_pct"], errors="coerce")
    lower_shadow = pd.to_numeric(out["lower_shadow"], errors="coerce")
    upper_shadow = pd.to_numeric(out["upper_shadow"], errors="coerce")

    high_position = (close >= high_position_line).fillna(False)
    accelerated = (accel_return >= float(accel_return_pct)).fillna(False)
    huge_volume = (volume >= huge_volume_ma * float(huge_volume_ratio)).fillna(False)
    big_bear_body = (bear_body_pct >= float(big_bear_body_pct)).fillna(False)

    out["accelerated_huge_volume_bear"] = (
        high_position.astype(bool)
        & accelerated.astype(bool)
        & huge_volume.astype(bool)
        & big_bear_body.astype(bool)
    ).fillna(False)

    out["prior_20d_accelerated_huge_volume_bear"] = (
        out["accelerated_huge_volume_bear"]
        .fillna(False)
        .astype(bool)
        .astype(int)
        .shift(1, fill_value=0)
        .rolling(window=int(risk_lookback), min_periods=1)
        .max()
        .fillna(0)
        .astype(bool)
    )

    out["limit_up"] = (pct_change_close >= float(limit_up_pct)).fillna(False)
    out["shrink_limit_up"] = (
        out["limit_up"].astype(bool)
        & (volume <= shrink_volume_ma5 * float(shrink_limit_vol_ratio)).fillna(False)
    ).fillna(False)
    out["prior_20d_shrink_limit_up"] = (
        out["shrink_limit_up"]
        .fillna(False)
        .astype(bool)
        .astype(int)
        .shift(1, fill_value=0)
        .rolling(window=int(risk_lookback), min_periods=1)
        .max()
        .fillna(0)
        .astype(bool)
    )

    body_safe = body_abs.replace(0, np.nan)
    range_safe = candle_range.replace(0, np.nan)
    out["long_lower_shadow_hammer"] = (
        (lower_shadow >= body_safe * float(hammer_lower_shadow_body_ratio))
        & (lower_shadow >= range_safe * float(hammer_lower_shadow_range_ratio))
        & (upper_shadow <= body_safe * float(hammer_upper_shadow_body_ratio))
        & (body_abs <= range_safe * float(hammer_max_body_range_ratio))
    ).fillna(False)

    if "red_brick" not in out.columns:
        brick_value = pd.to_numeric(out["brick_value"], errors="coerce")
        brick_prev_1 = pd.to_numeric(out["brick_prev_1"], errors="coerce")
        out["red_brick"] = (brick_value > brick_prev_1).fillna(False)

    out["limit_up_red_brick"] = (
        out["limit_up"].astype(bool) & out["red_brick"].fillna(False).astype(bool)
    ).fillna(False)
    return out
