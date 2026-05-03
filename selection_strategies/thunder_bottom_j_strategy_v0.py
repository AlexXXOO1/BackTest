from __future__ import annotations

"""
Thunder bottom J selection strategy v0.

Chinese name:
    绝对底部后回调 J 低点异动策略 v0

Core logic:
    This strategy does NOT use any brick / renko logic.

    The selected date T0 should follow this structure:

        Prior absolute bottom
            -> valid rebound after absolute bottom
            -> pullback relative bottom after rebound
            -> KDJ J confirms low position
            -> optional abnormal move / big bull volume / scary key position diagnostics

Important:
    - No future function.
    - No brick_value.
    - No brick_turn_strong.
    - No renko indicator.
    - n1 and n2 are accepted only for compatibility with the current engine.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


STRATEGY_NAME = "thunder_bottom_j_strategy_v0"


REQUIRED_INDICATOR_COLUMNS: set[str] = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


# =============================================================================
# Adjustable parameters
# =============================================================================

@dataclass(frozen=True)
class ThunderBottomJParams:
    # -------------------------------------------------------------------------
    # Diagnostic selection layer
    # -------------------------------------------------------------------------
    # Available values:
    #   "layer_4_j_confirm"
    #   "layer_5_sudden"
    #   "layer_6_big_bull_volume"
    #   "layer_7_scary"
    #
    # First test should use layer_4_j_confirm to avoid empty pool.
    selection_layer: str = "layer_7_scary"

    # -------------------------------------------------------------------------
    # Absolute bottom: prior event
    # -------------------------------------------------------------------------
    absolute_bottom_lookback: int = 120
    absolute_bottom_position_max: float = 0.30

    # The prior absolute-bottom event must be within this many trading days.
    absolute_bottom_valid_days: int = 120

    # -------------------------------------------------------------------------
    # Rebound after absolute bottom
    # -------------------------------------------------------------------------
    # After absolute bottom, price must rebound at least this much.
    min_rebound_from_absolute_bottom: float = 0.05

    # T0 must pull back from the post-bottom rebound high by at least this much.
    min_pullback_from_rebound_high: float = 0.025

    # T0 should not break too far below the prior absolute-bottom low.
    # Example: 0.90 means close >= absolute_bottom_low * 0.90.
    min_close_vs_absolute_bottom_low: float = 0.90

    # -------------------------------------------------------------------------
    # Relative bottom on T0
    # -------------------------------------------------------------------------
    relative_low_lookback: int = 20
    near_recent_low_max: float = 0.12

    # -------------------------------------------------------------------------
    # KDJ J low confirmation
    # -------------------------------------------------------------------------
    kdj_n: int = 9
    kdj_m1: int = 3
    kdj_m2: int = 3

    # For first test, use 35 instead of 20, otherwise the pool can be too narrow.
    j_low_max: float = 35.0

    # If True, require J to be near recent low.
    # First test recommends False.
    use_j_recent_low_filter: bool = False
    j_recent_low_lookback: int = 5
    j_recent_low_buffer: float = 5.0

    # If True, require J to turn upward on T0.
    # First test recommends False.
    require_j_turn_up: bool = False

    # If True, allow recent J low in T0/T-1/T-2.
    # This avoids the conflict where T0 big bullish candle lifts J sharply.
    use_recent_j_low_window: bool = True
    recent_j_low_window: int = 3

    # -------------------------------------------------------------------------
    # Sudden abnormal movement
    # -------------------------------------------------------------------------
    quiet_lookback: int = 10
    quiet_abs_ret_mean_max: float = 0.040

    sudden_return_min: float = 0.020

    range_shock_lookback: int = 10
    range_shock_ratio_min: float = 1.20

    body_shock_lookback: int = 10
    body_shock_ratio_min: float = 1.20

    # -------------------------------------------------------------------------
    # Big bullish candle and huge volume
    # -------------------------------------------------------------------------
    big_body_pct_min: float = 0.015
    big_daily_return_min: float = 0.020
    close_position_min: float = 0.55

    volume_ma_short_n: int = 5
    volume_ma_long_n: int = 20
    volume_ratio_5_min: float = 1.20
    volume_ratio_20_min: float = 1.30

    # -------------------------------------------------------------------------
    # Scary key position
    # -------------------------------------------------------------------------
    break_high_lookback: int = 20

    platform_lookback: int = 15
    platform_range_max: float = 0.25

    prev_big_bear_body_min: float = 0.030
    reversal_open_above_prev_close_max: float = 1.02

    ma_short_n: int = 20
    ma_long_n: int = 60

    gap_open_min: float = 0.005
    gap_hold_low_min: float = 0.995

    # -------------------------------------------------------------------------
    # Base filters
    # -------------------------------------------------------------------------
    min_volume: float = 1.0
    min_close: float = 1.0
    max_close: float = 9999.0


DEFAULT_PARAMS = ThunderBottomJParams()


# =============================================================================
# Helper functions
# =============================================================================

def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _rolling_high(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).max()


def _rolling_low(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).min()


def _rolling_mean(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _as_bool(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool)


def _tdx_sma(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """
    TongDaXin-style SMA:
        SMA(X, N, M) = (M * X + (N - M) * REF(SMA, 1)) / N

    No future data is used.
    """
    x = series.astype(float)
    result = pd.Series(index=x.index, dtype="float64")

    prev = np.nan

    for i, value in enumerate(x):
        if pd.isna(value):
            result.iloc[i] = np.nan
            continue

        if pd.isna(prev):
            prev = value
        else:
            prev = (m * value + (n - m) * prev) / n

        result.iloc[i] = prev

    return result


# =============================================================================
# KDJ calculation
# =============================================================================

def add_kdj_indicators(
    df: pd.DataFrame,
    params: ThunderBottomJParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """
    Add KDJ indicators.

    Formula:
        RSV = (close - LLV(low, N)) / (HHV(high, N) - LLV(low, N)) * 100
        K = SMA(RSV, M1, 1)
        D = SMA(K, M2, 1)
        J = 3K - 2D

    No future data is used.
    """
    out = df.copy()

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    low_n = _rolling_low(low, params.kdj_n)
    high_n = _rolling_high(high, params.kdj_n)

    out["kdj_rsv"] = _safe_div(close - low_n, high_n - low_n) * 100.0
    out["kdj_k"] = _tdx_sma(out["kdj_rsv"], params.kdj_m1, 1)
    out["kdj_d"] = _tdx_sma(out["kdj_k"], params.kdj_m2, 1)
    out["kdj_j"] = 3.0 * out["kdj_k"] - 2.0 * out["kdj_d"]

    return out


# =============================================================================
# Prior absolute bottom and post-bottom pullback logic
# =============================================================================

def add_prior_absolute_bottom_pullback_logic(
    df: pd.DataFrame,
    params: ThunderBottomJParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """
    Calculate the main bottom structure:

        prior absolute bottom
        -> rebound
        -> T0 pullback relative bottom confirmed by J

    For each T0 row i:
        - only rows <= i are visible.
        - prior absolute-bottom event must be from rows < i.
        - post-bottom rebound high is calculated from abs_bottom_index + 1 to i - 1.
        - T0 itself is NOT used to calculate prior rebound high.
    """
    out = df.copy()

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    low_abs = _rolling_low(low, params.absolute_bottom_lookback)
    high_abs = _rolling_high(high, params.absolute_bottom_lookback)

    out["price_position_120"] = _safe_div(close - low_abs, high_abs - low_abs)

    out["absolute_bottom_event"] = (
        out["price_position_120"] <= params.absolute_bottom_position_max
    )

    n = len(out)

    prior_abs_bottom_seen = [False] * n
    prior_abs_bottom_index = [np.nan] * n
    prior_abs_bottom_low = [np.nan] * n
    post_abs_rebound_high = [np.nan] * n
    rebound_from_abs_bottom = [np.nan] * n
    pullback_from_rebound_high = [np.nan] * n

    absolute_event_bool = out["absolute_bottom_event"].fillna(False).to_numpy()

    high_arr = high.to_numpy(dtype="float64")
    low_arr = low.to_numpy(dtype="float64")
    close_arr = close.to_numpy(dtype="float64")

    for i in range(n):
        start = max(0, i - params.absolute_bottom_valid_days)

        # Search only before T0, so end is i, not i + 1.
        candidate_indices = np.where(absolute_event_bool[start:i])[0]

        if len(candidate_indices) == 0:
            continue

        last_abs_idx = start + int(candidate_indices[-1])

        prior_abs_bottom_seen[i] = True
        prior_abs_bottom_index[i] = last_abs_idx
        prior_abs_bottom_low[i] = low_arr[last_abs_idx]

        # Rebound high after absolute bottom and before T0.
        rebound_start = last_abs_idx + 1
        rebound_end = i

        if rebound_start >= rebound_end:
            continue

        rebound_high = np.nanmax(high_arr[rebound_start:rebound_end])

        if pd.isna(rebound_high) or pd.isna(prior_abs_bottom_low[i]):
            continue

        post_abs_rebound_high[i] = rebound_high
        rebound_from_abs_bottom[i] = rebound_high / prior_abs_bottom_low[i] - 1.0
        pullback_from_rebound_high[i] = close_arr[i] / rebound_high - 1.0

    out["prior_abs_bottom_seen"] = prior_abs_bottom_seen
    out["prior_abs_bottom_index"] = prior_abs_bottom_index
    out["prior_abs_bottom_low"] = prior_abs_bottom_low
    out["post_abs_rebound_high"] = post_abs_rebound_high
    out["rebound_from_abs_bottom"] = rebound_from_abs_bottom
    out["pullback_from_rebound_high"] = pullback_from_rebound_high

    recent_low = _rolling_low(low, params.relative_low_lookback)

    out["near_recent_low"] = _safe_div(close, recent_low) - 1.0

    out["has_valid_post_abs_rebound"] = (
        out["prior_abs_bottom_seen"]
        & (out["rebound_from_abs_bottom"] >= params.min_rebound_from_absolute_bottom)
    )

    out["pullback_after_abs_bottom"] = (
        out["has_valid_post_abs_rebound"]
        & (out["pullback_from_rebound_high"] <= -params.min_pullback_from_rebound_high)
        & (close >= out["prior_abs_bottom_low"] * params.min_close_vs_absolute_bottom_low)
    )

    out["relative_bottom_after_abs_bottom"] = (
        out["pullback_after_abs_bottom"]
        & (out["near_recent_low"] <= params.near_recent_low_max)
    )

    return out


# =============================================================================
# Main indicator calculation
# =============================================================================

def add_thunder_bottom_j_indicators(
    df: pd.DataFrame,
    params: ThunderBottomJParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    out = df.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    out = out.sort_values("date").reset_index(drop=True)

    close = out["close"].astype(float)
    open_ = out["open"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)

    prev_close = close.shift(1)
    prev_open = open_.shift(1)
    prev_high = high.shift(1)

    # -------------------------------------------------------------------------
    # KDJ J
    # -------------------------------------------------------------------------
    out = add_kdj_indicators(out, params=params)

    out["j_recent_low"] = _rolling_low(out["kdj_j"], params.j_recent_low_lookback)

    out["j_low_position"] = (
        out["kdj_j"] <= params.j_low_max
    )

    if params.use_recent_j_low_window:
        out["j_low_recent_window"] = (
            _rolling_low(out["kdj_j"], params.recent_j_low_window) <= params.j_low_max
        )
    else:
        out["j_low_recent_window"] = out["j_low_position"]

    if params.use_j_recent_low_filter:
        out["j_near_recent_low"] = (
            out["kdj_j"] <= out["j_recent_low"] + params.j_recent_low_buffer
        )
    else:
        out["j_near_recent_low"] = True

    if params.require_j_turn_up:
        out["j_turn_up"] = (
            out["kdj_j"] > out["kdj_j"].shift(1)
        )
    else:
        out["j_turn_up"] = True

    out["j_confirm_relative_low"] = (
        out["j_low_recent_window"]
        & out["j_near_recent_low"]
        & out["j_turn_up"]
    )

    # -------------------------------------------------------------------------
    # Bottom structure:
    # prior absolute bottom -> rebound -> pullback relative bottom
    # -------------------------------------------------------------------------
    out = add_prior_absolute_bottom_pullback_logic(out, params=params)

    out["bottom_position_ok"] = (
        out["relative_bottom_after_abs_bottom"]
        & out["j_confirm_relative_low"]
    )

    # -------------------------------------------------------------------------
    # Common candle metrics
    # -------------------------------------------------------------------------
    out["daily_return_pct"] = _safe_div(close, prev_close) - 1.0
    out["today_range_pct"] = _safe_div(high - low, prev_close)
    out["body_pct_abs"] = _safe_div((close - open_).abs(), prev_close)
    out["body_pct_bull"] = _safe_div(close - open_, prev_close)
    out["close_position_in_bar"] = _safe_div(close - low, high - low)
    out["gap_open_pct"] = _safe_div(open_, prev_close) - 1.0

    out["ma20"] = _rolling_mean(close, params.ma_short_n)
    out["ma60"] = _rolling_mean(close, params.ma_long_n)

    out["volume_ma5"] = _rolling_mean(volume, params.volume_ma_short_n)
    out["volume_ma20"] = _rolling_mean(volume, params.volume_ma_long_n)

    out["volume_ratio_5"] = _safe_div(volume, out["volume_ma5"])
    out["volume_ratio_20"] = _safe_div(volume, out["volume_ma20"])

    # -------------------------------------------------------------------------
    # Sudden abnormal movement
    # -------------------------------------------------------------------------
    out["ret_abs_mean_10"] = _rolling_mean(
        out["daily_return_pct"].abs(),
        params.quiet_lookback,
    )

    out["quiet_before"] = (
        out["ret_abs_mean_10"] <= params.quiet_abs_ret_mean_max
    )

    out["sudden_return"] = (
        out["daily_return_pct"] >= params.sudden_return_min
    )

    out["avg_range_10"] = _rolling_mean(
        out["today_range_pct"],
        params.range_shock_lookback,
    )

    out["range_shock_ratio"] = _safe_div(
        out["today_range_pct"],
        out["avg_range_10"],
    )

    out["sudden_range"] = (
        out["range_shock_ratio"] >= params.range_shock_ratio_min
    )

    out["avg_body_10"] = _rolling_mean(
        out["body_pct_abs"],
        params.body_shock_lookback,
    )

    out["body_shock_ratio"] = _safe_div(
        out["body_pct_abs"],
        out["avg_body_10"],
    )

    out["sudden_body"] = (
        out["body_shock_ratio"] >= params.body_shock_ratio_min
    )

    out["sudden_thunder_move"] = (
        out["quiet_before"]
        & out["sudden_return"]
        & out["sudden_range"]
        & out["sudden_body"]
    )

    # -------------------------------------------------------------------------
    # Big bullish candle and huge volume
    # -------------------------------------------------------------------------
    out["big_bull_body"] = (
        (close > open_)
        & (out["body_pct_bull"] >= params.big_body_pct_min)
        & (out["daily_return_pct"] >= params.big_daily_return_min)
        & (out["close_position_in_bar"] >= params.close_position_min)
    )

    out["huge_volume"] = (
        (out["volume_ratio_5"] >= params.volume_ratio_5_min)
        & (out["volume_ratio_20"] >= params.volume_ratio_20_min)
    )

    out["big_bull_volume_bar"] = (
        out["big_bull_body"]
        & out["huge_volume"]
    )

    # -------------------------------------------------------------------------
    # Scary key position
    # -------------------------------------------------------------------------
    prior_high_20 = _rolling_high(prev_high, params.break_high_lookback)

    out["scary_break_20d_high"] = (
        (close > prior_high_20)
        & (prev_close <= prior_high_20)
    )

    platform_high = _rolling_high(high, params.platform_lookback)
    platform_low = _rolling_low(low, params.platform_lookback)
    prior_platform_high = _rolling_high(prev_high, params.platform_lookback)

    out["platform_range_15"] = _safe_div(
        platform_high - platform_low,
        platform_low,
    )

    out["small_platform"] = (
        out["platform_range_15"] <= params.platform_range_max
    )

    out["break_platform"] = (
        close >= prior_platform_high
    )

    out["scary_platform_break"] = (
        out["small_platform"]
        & out["break_platform"]
    )

    prev_big_bear = (
        (prev_close < prev_open)
        & (_safe_div(prev_open - prev_close, close.shift(2)) >= params.prev_big_bear_body_min)
    )

    today_engulf = (
        (close > prev_open)
        & (open_ <= prev_close * params.reversal_open_above_prev_close_max)
    )

    out["scary_bearish_engulf_reversal"] = (
        prev_big_bear
        & today_engulf
    )

    out["scary_ma60_reclaim"] = (
        (prev_close < out["ma60"].shift(1))
        & (close > out["ma60"])
        & (close > out["ma20"])
    )

    out["scary_gap_hold"] = (
        (out["gap_open_pct"] >= params.gap_open_min)
        & (close > open_)
        & (low >= prev_close * params.gap_hold_low_min)
    )

    out["scary_key_position"] = (
        out["scary_break_20d_high"]
        | out["scary_platform_break"]
        | out["scary_bearish_engulf_reversal"]
        | out["scary_ma60_reclaim"]
        | out["scary_gap_hold"]
    )

    # -------------------------------------------------------------------------
    # Base filters
    # -------------------------------------------------------------------------
    out["base_filter_ok"] = (
        (volume >= params.min_volume)
        & (close >= params.min_close)
        & (close <= params.max_close)
    )

    # -------------------------------------------------------------------------
    # Reserved scoring interface
    # -------------------------------------------------------------------------
    out = add_reserved_scores(out, params=params)

    bool_cols = [
        "absolute_bottom_event",
        "prior_abs_bottom_seen",
        "has_valid_post_abs_rebound",
        "pullback_after_abs_bottom",
        "relative_bottom_after_abs_bottom",

        "j_low_position",
        "j_low_recent_window",
        "j_near_recent_low",
        "j_turn_up",
        "j_confirm_relative_low",

        "bottom_position_ok",

        "quiet_before",
        "sudden_return",
        "sudden_range",
        "sudden_body",
        "sudden_thunder_move",

        "big_bull_body",
        "huge_volume",
        "big_bull_volume_bar",

        "scary_break_20d_high",
        "small_platform",
        "break_platform",
        "scary_platform_break",
        "scary_bearish_engulf_reversal",
        "scary_ma60_reclaim",
        "scary_gap_hold",
        "scary_key_position",

        "base_filter_ok",
    ]

    for col in bool_cols:
        if col in out.columns:
            out[col] = _as_bool(out[col])

    return out


# =============================================================================
# Reserved scoring interface
# =============================================================================

def add_reserved_scores(
    df: pd.DataFrame,
    params: ThunderBottomJParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """
    Reserved scoring interface.

    This v0 does not use score as selection filter.
    These columns are only placeholders for future ranking / analysis.
    """
    out = df.copy()

    out["bottom_score"] = 0.0
    out["sudden_score"] = 0.0
    out["bull_volume_score"] = 0.0
    out["scary_score"] = 0.0
    out["thunder_score"] = 0.0

    return out


# =============================================================================
# Final / diagnostic selection rule
# =============================================================================

def apply_selection_rule(
    df: pd.DataFrame,
    params: ThunderBottomJParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """
    Diagnostic layered selection rule.

    Layers:
        layer_1_prior_abs_bottom:
            Prior absolute bottom appeared.

        layer_2_valid_rebound:
            After absolute bottom, price had a valid rebound.

        layer_3_relative_bottom:
            T0 is a pullback relative bottom after that rebound.

        layer_4_j_confirm:
            Layer 3 + J confirms low position.

        layer_5_sudden:
            Layer 4 + sudden abnormal move.

        layer_6_big_bull_volume:
            Layer 5 + big bullish candle with huge volume.

        layer_7_scary:
            Layer 6 + scary key position.

    params.selection_layer controls which layer becomes selected.
    First test recommends:
        selection_layer = "layer_4_j_confirm"

    No brick / renko condition is used.
    """
    out = df.copy()

    out["layer_1_prior_abs_bottom"] = out["prior_abs_bottom_seen"]

    out["layer_2_valid_rebound"] = (
        out["layer_1_prior_abs_bottom"]
        & out["has_valid_post_abs_rebound"]
    )

    out["layer_3_relative_bottom"] = (
        out["layer_2_valid_rebound"]
        & out["relative_bottom_after_abs_bottom"]
    )

    out["layer_4_j_confirm"] = (
        out["layer_3_relative_bottom"]
        & out["j_confirm_relative_low"]
    )

    out["layer_5_sudden"] = (
        out["layer_4_j_confirm"]
        & out["sudden_thunder_move"]
    )

    out["layer_6_big_bull_volume"] = (
        out["layer_5_sudden"]
        & out["big_bull_volume_bar"]
    )

    out["layer_7_scary"] = (
        out["layer_6_big_bull_volume"]
        & out["scary_key_position"]
    )

    valid_layers = {
        "layer_4_j_confirm",
        "layer_5_sudden",
        "layer_6_big_bull_volume",
        "layer_7_scary",
    }

    selected_layer = params.selection_layer
    if selected_layer not in valid_layers:
        raise ValueError(
            f"Invalid selection_layer: {selected_layer}. "
            f"Valid layers: {sorted(valid_layers)}"
        )

    out["selected"] = (
        out[selected_layer]
        & out["base_filter_ok"]
    )

    bool_cols = [
        "layer_1_prior_abs_bottom",
        "layer_2_valid_rebound",
        "layer_3_relative_bottom",
        "layer_4_j_confirm",
        "layer_5_sudden",
        "layer_6_big_bull_volume",
        "layer_7_scary",
        "selected",
    ]

    for col in bool_cols:
        out[col] = _as_bool(out[col])

    return out


def select(
    df: pd.DataFrame,
    params: ThunderBottomJParams = DEFAULT_PARAMS,
    n1: int | None = None,
    n2: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Standard strategy entry.

    n1 and n2 are accepted only for compatibility with the existing selector
    engine. This strategy does NOT use n1, n2, or any brick / renko logic.
    """
    out = df.copy()
    out = add_thunder_bottom_j_indicators(out, params=params)
    out = apply_selection_rule(out, params=params)
    return out


run = select
apply_strategy = select


# =============================================================================
# Debug summary
# =============================================================================

def debug_summary(df: pd.DataFrame) -> dict[str, int]:
    cols = [
        "absolute_bottom_event",
        "prior_abs_bottom_seen",
        "has_valid_post_abs_rebound",
        "pullback_after_abs_bottom",
        "relative_bottom_after_abs_bottom",

        "j_low_position",
        "j_low_recent_window",
        "j_near_recent_low",
        "j_turn_up",
        "j_confirm_relative_low",

        "bottom_position_ok",

        "sudden_thunder_move",
        "quiet_before",
        "sudden_return",
        "sudden_range",
        "sudden_body",

        "big_bull_volume_bar",
        "big_bull_body",
        "huge_volume",

        "scary_key_position",
        "scary_break_20d_high",
        "scary_platform_break",
        "scary_bearish_engulf_reversal",
        "scary_ma60_reclaim",
        "scary_gap_hold",

        "layer_1_prior_abs_bottom",
        "layer_2_valid_rebound",
        "layer_3_relative_bottom",
        "layer_4_j_confirm",
        "layer_5_sudden",
        "layer_6_big_bull_volume",
        "layer_7_scary",

        "base_filter_ok",
        "selected",
    ]

    result: dict[str, int] = {}

    for col in cols:
        if col in df.columns:
            result[col] = int(df[col].fillna(False).sum())

    return result
