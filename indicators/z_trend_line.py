# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Iterable

import pandas as pd


DEFAULT_LONG_TREND_WINDOWS = (5, 10, 20, 60)


def _ensure_close(df: pd.DataFrame) -> pd.Series:
    if "close" not in df.columns:
        raise ValueError("missing required column: close")
    return pd.to_numeric(df["close"], errors="coerce")


def calc_z_short_trend_line(
    df: pd.DataFrame,
    span: int = 10,
    adjust: bool = False,
) -> pd.Series:
    close = _ensure_close(df)
    ema1 = close.ewm(span=span, adjust=adjust, min_periods=1).mean()
    ema2 = ema1.ewm(span=span, adjust=adjust, min_periods=1).mean()
    return ema2


def calc_z_long_trend_line(
    df: pd.DataFrame,
    windows: Iterable[int] = DEFAULT_LONG_TREND_WINDOWS,
) -> pd.Series:
    close = _ensure_close(df)
    windows = tuple(int(x) for x in windows)

    if len(windows) != 4:
        raise ValueError(f"long trend windows must contain exactly 4 values, got: {windows}")

    ma_list = [
        close.rolling(window=w, min_periods=1).mean()
        for w in windows
    ]
    return sum(ma_list) / len(ma_list)


def add_z_trend_lines(
    df: pd.DataFrame,
    z_short_span: int = 10,
    z_long_windows: Iterable[int] = DEFAULT_LONG_TREND_WINDOWS,
) -> pd.DataFrame:
    out = df.copy()
    out["z_short_trend_line"] = calc_z_short_trend_line(
        out,
        span=z_short_span,
        adjust=False,
    )
    out["z_long_trend_line"] = calc_z_long_trend_line(
        out,
        windows=z_long_windows,
    )
    return out


def add_indicators(
    df: pd.DataFrame,
    z_short_span: int = 10,
    z_long_windows: Iterable[int] = DEFAULT_LONG_TREND_WINDOWS,
    **kwargs,
) -> pd.DataFrame:
    """
    Auto-loader entry point.

    build_indicators.py does not need to import this file manually.
    The generic auto loader will also create:
        close_to_z_short_trend_line
        close_to_z_long_trend_line
    """
    return add_z_trend_lines(
        df,
        z_short_span=z_short_span,
        z_long_windows=z_long_windows,
    )
