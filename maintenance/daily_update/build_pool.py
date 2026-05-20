# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generic pool builder.

Purpose:
- Read indicator cache.
- Dynamically load any selection strategy from strategies/selection/.
- Run strategy per symbol.
- Keep strategy-selected rows. A strategy may either return only selected rows or return a transient selected column.
- Add executable forward fields through T+4 using T+1 open as buy price.
- Normalize and validate the final pool contract before saving.
- Save pool parquet/csv.

Strategy file requirement:
    strategies/selection/xxx.py

Must provide either:
    def select(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ...
or:
    SELECT_FUNC = select

Strategy output requirement:
    A strategy can return only selected rows, or return a transient selected column where 1 means selected.

Final saved pool does not keep selected / selected_score_base / score_rank_key / score_pct.
"""

import argparse
import importlib.util
import json
import shutil
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.path_manager import DATA_ROOT, INDICATOR_CACHE_PATH, POOLS_DIR, MARKET_CACHE_DIR
from core.indicator_store import _read_table, _standardize_market_df
from data_engine.indicators import add_all_indicators

from core.pool_schema import (
    POOL_SCHEMA_VERSION,
    drop_removed_score_columns,
    validate_pool_schema,
)

DEFAULT_DATA_ROOT = DATA_ROOT
DEFAULT_INDICATOR_PATH = INDICATOR_CACHE_PATH
DEFAULT_POOL_DIR = POOLS_DIR

CORE_COLUMNS = [
    "symbol",
    "file",
    "date",
    "selection_strategy",
]

TRANSIENT_SELECTION_COLUMN = "selected"
REMOVED_SCORE_COLUMNS = [
    "selected",
    "selected_score_base",
    "score",
    "score_rank_key",
    "score_pct",
]

PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "prev_close",
    "prev_volume",
    "daily_return_pct",
    "intraday_return_pct",
    "amplitude_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "body_pct",
    "body_abs_pct",
    "upper_shadow_ratio",
    "is_red_k",
    "is_green_k",
    "is_flat_k",
]

COMMON_INDICATOR_COLUMNS = [
    "ma5",
    "ma10",
    "ma20",
    "ma50",
    "ma60",
    "volume_ma5",
    "volume_ratio_ma5",
    "volume_ma10",
    "volume_ratio_ma10",
    "volume_ratio_prev1",
    "macd_dif",
    "macd_dea",
    "macd_hist",
    "kdj_k",
    "kdj_d",
    "kdj_j",
    "j",
]

B2_EXTENSION_COLUMNS = [
    "range_high_20",
    "range_low_20",
    "range_width_20",
    "position_in_range_20",
    "previous_n_low",
    "distance_to_previous_n_low",
    "volume_to_ma5",
    "volume_q20_20",
    "volume_min_20",
    "volume_20_q20",
    "volume_20_min",
    "double_volume_bar",
    "prior_20d_double_volume_count",
    "prior_20d_has_double_volume",
    "bbi",
    "bbi_for_b1",
    "yellow_line",
    "yellow_for_b1",
    "distance_to_ma20",
    "distance_to_bbi",
    "distance_to_yellow",
    "b1_j_ok",
    "b1_low_range_position",
    "b1_near_previous_n_low",
    "b1_in_range_bottom",
    "b1_stage_low_position",
    "b1_position_ok",
    "b1_low_volume",
    "b1_extreme_low_volume",
    "b1_low_or_extreme_volume",
    "b1_ma20_gt_ma50",
    "b1_prior_20d_has_double_volume",
    "b1_not_break_prev_low",
    "b1_j_deep_negative",
    "b1_valid",
    "b1_within_b2_lookback",
    "b1_within_lookback",
    "b1_days_ago_for_b2",
    "b1_days_ago",
    "b1_j_value",
    "b1_volume_value",
    "b1_close_value",
    "b1_position_in_range_20_value",
    "b1_volume_ratio_ma5_value",
    "b2_after_b1",
    "b2_after_b1_within_3d",
    "b2_return_ok",
    "b2_volume_up",
    "b2_j_ok",
    "b2_upper_shadow_ok",
    "b2_no_or_tiny_upper_shadow",
    "b2_tiny_upper_shadow",
    "b2_upper_shadow_warning",
    "b2_long_upper_shadow_reject",
    "b2_bullish_candle",
    "b2_strong_volume",
    "b2_double_volume",
    "b2_sky_volume",
    "b2_j_high_zone",
    "b2_j_low_zone_removed_by_v1",
    "b2_j_value",
    "b2_volume_ratio",
    "b2_quality_score",
    "quality_score",
]

RENKO_EXTENSION_COLUMNS = [
    "renko_value",
    "brick_value",
    "brick_prev_1",
    "brick_prev_2",
    "brick_open",
    "brick_close",
    "brick_delta",
    "green_to_red",
    "red_brick",
    "green_brick",
    "valid_red_brick",
    "valid_previous_green_brick",
    "current_red_height",
    "previous_green_height",
    "brick_reversal_strength",
    "brick_reversal_ratio",
    "hard_brick_turn_strong",
    "t0_close_to_z_short_trend_line_pct",
    "t0_close_to_z_long_trend_line_pct",
    "z_short_trend_above_z_long_trend_line",
    "short_trend",
    "trend_line",
    "yellow_ma",
    "short_trend_cap",
    "close_to_short_trend",
    "close_to_trend_line",
    "close_to_yellow_ma",
    "close_above_yellow_ma",
    "above_yellow_ma",
    "two_day_above_trend_line",
    "short_trend_above_trend_line",
    "close_below_short_trend_cap",
    "price_below_50",
    "price_zone_ok",
    "trend_condition_ok",
    "price_condition_ok",
    "price_rise_range_and_close_to_short_trend_below_limit",
    "j_momentum_or_low",
    "j_below_14",
    "j_two_day_rising",
    "j_three_day_rising",
    "j_lt_0",
    "j_30_to_50",
    "j_condition_pass",
    "j_condition_rule",
    "j_condition_source_col",
    "surge_then_shrink_pullback",
    "small_rise_long_red_brick",
    "condition6_hard_pass",
    "condition8_hard_pass",
    "condition9_hard_pass",
    "risk_tag_any",
    "risk_tag_count",
    "risk_tags",
    "risk_filter_pass",
    "rank_close_to_short_trend",
    "rank_daily_return_pct",
    "rank_brick_reversal_ratio",
    "v4_brk",
    "v4_crh",
    "v4_pgh",
    "v4_close_to_ma5",
    "v4_cond_g2r",
    "v4_cond_vrb",
    "v4_cond_vgb",
    "v4_cond_brs",
    "v4_cond_hbts",
    "v4_cond_prir",
    "v4_cond_ma5_0_1",
    "v4_hint_brk_rank_pct",
    "v4_hint_brk_low",
    "v4_hint_brk_high",
    "v4_hint_drp_strong",
    "v4_hint_drp_near_limit",
    "v4_hint_ma5_rank_pct",
    "v4_hint_ma5_low",
    "v4_hint_ma5_high",
    "v4_hint_volume_ratio_prev1_rank_pct",
    "v4_hint_volume_extreme",
    "v4_up_hint_score",
    "v4_risk_hint_score",
    "v4_net_hint_score",
    "v4_hint_label",
]

RAW_ABSOLUTE_POOL_DROP_COLUMNS = {
    "z_short_trend_line",
    "z_long_trend_line",
}

FINAL_POOL_DROP_COLUMNS = {
    "renko_ref1",
    "renko_ref2",
}

FORWARD_HORIZON_MAX = 20

FORWARD_COLUMNS = [
    *[
        col
        for horizon in range(1, FORWARD_HORIZON_MAX + 1)
        for col in (
            f"t{horizon}_date",
            f"t{horizon}_open",
            f"t{horizon}_close",
        )
    ],
    *[f"fwd_return_pct_T{horizon}" for horizon in range(1, FORWARD_HORIZON_MAX + 1)],
    *[f"fwd_up_T{horizon}" for horizon in range(1, FORWARD_HORIZON_MAX + 1)],
    "forward_data_status",
]

REHYDRATE_COLUMNS = [
    *CORE_COLUMNS,
    *PRICE_COLUMNS,
    *COMMON_INDICATOR_COLUMNS,
    *B2_EXTENSION_COLUMNS,
    *RENKO_EXTENSION_COLUMNS,
]

DROP_COLUMN_PATTERNS = [
    re.compile(r"^b2v\d+_bucket_"),
    re.compile(r"^b2v\d+_score_(?!rank_key$|pct$)"),
    re.compile(r"^b2v\d+_score_total$"),
    re.compile(r"^b2v\d+_score_factor_count$"),
    re.compile(r"^b2v\d+_score_used_factors$"),
]

STRIP_PREFIX_PATTERNS = [
    re.compile(r"^b2v\d+_(.+)$"),
]

OLD_FORWARD_PATTERNS = [
    re.compile(r"^fwd_close_T\d+$"),
    re.compile(r"^fwd_return_pct_T\d+$"),
    re.compile(r"^fwd_up_T\d+$"),
    re.compile(r"^t\d+_date$"),
    re.compile(r"^t\d+_open$"),
    re.compile(r"^t\d+_close$"),
    re.compile(r"^forward_data_status$"),
]


def parse_extra_args(items: Optional[List[str]]) -> Dict[str, Any]:
    """Parse repeated --param values into strategy kwargs."""
    if not items:
        return {}

    result: Dict[str, Any] = {}

    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --param format: {item}. Expected key=value.")

        left, value = item.split("=", 1)

        if ":" in left:
            key, typ = left.split(":", 1)
            typ = typ.lower().strip()
        else:
            key, typ = left.strip(), "str"

        key = key.strip()
        value = value.strip()

        if not key:
            raise ValueError(f"Invalid --param key: {item}")

        if typ == "int":
            result[key] = int(value)
        elif typ == "float":
            result[key] = float(value)
        elif typ == "bool":
            result[key] = value.lower() in {"1", "true", "yes", "y", "on"}
        elif typ == "str":
            result[key] = value
        else:
            raise ValueError(f"Unsupported --param type: {typ}. Use int/float/bool/str.")

    return result


def load_strategy_func(strategy_name: str):
    """Load a strategy by file/module name or by STRATEGY_NAME registry value."""
    strategy_path = PROJECT_ROOT / "strategies" / "selection" / f"{strategy_name}.py"

    if strategy_path.exists():
        module_name = f"strategies.selection.{strategy_name}"

        spec = importlib.util.spec_from_file_location(module_name, strategy_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load strategy spec from: {strategy_path}")

        module = importlib.util.module_from_spec(spec)

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        spec.loader.exec_module(module)

        if hasattr(module, "SELECT_FUNC"):
            func = getattr(module, "SELECT_FUNC")
        elif hasattr(module, "select"):
            func = getattr(module, "select")
        elif hasattr(module, "apply_strategy"):
            func = getattr(module, "apply_strategy")
        else:
            raise AttributeError(
                f"Strategy {strategy_name} must define SELECT_FUNC, select(), or apply_strategy()."
            )

        if not callable(func):
            raise TypeError(f"Strategy function for {strategy_name} is not callable.")

        return func

    try:
        from strategies.selection.registry import get_selection_strategy, list_selection_strategies

        return get_selection_strategy(strategy_name)
    except Exception as exc:
        try:
            from strategies.selection.registry import list_selection_strategies
            available = ", ".join(list_selection_strategies())
        except Exception:
            available = "<registry unavailable>"

        raise FileNotFoundError(
            f"Strategy not found by module file or registry name: {strategy_name}\n"
            f"Tried file: {strategy_path}\n"
            f"Available registry strategy names: {available}"
        ) from exc


def normalize_symbol(x) -> str:
    if pd.isna(x):
        return ""

    s = str(x).strip().upper()
    digits = "".join(ch for ch in s if ch.isdigit())

    if len(digits) >= 6:
        return digits[-6:]

    if len(digits) > 0:
        return digits.zfill(6)

    return ""



def _enforce_canonical_kline_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force canonical OHLC-based factor definitions during pool build.

    This protects build_pool from stale indicator_cache files that were generated
    with older formula definitions.
    """
    required = {"symbol", "date", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return df

    out = df.copy(deep=False)

    open_ = pd.to_numeric(out["open"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")

    prev_close = close.groupby(out["symbol"], sort=False).shift(1)
    prev_close_safe = prev_close.replace(0, pd.NA)

    out["prev_close"] = prev_close
    out["daily_return_pct"] = (close / prev_close_safe - 1.0) * 100.0
    out["intraday_return_pct"] = (close / open_.replace(0, pd.NA) - 1.0) * 100.0

    # Canonical A-share amplitude:
    # amplitude_pct = (high - low) / previous close * 100
    out["amplitude_pct"] = (high - low) / prev_close_safe * 100.0

    out["body_pct"] = (close - open_) / prev_close_safe * 100.0
    out["body_abs_pct"] = out["body_pct"].abs()

    max_oc = pd.concat([open_, close], axis=1).max(axis=1)
    min_oc = pd.concat([open_, close], axis=1).min(axis=1)

    out["upper_shadow_pct"] = (high - max_oc) / prev_close_safe * 100.0
    out["lower_shadow_pct"] = (min_oc - low) / prev_close_safe * 100.0

    if "volume" in out.columns:
        volume = pd.to_numeric(out["volume"], errors="coerce")
        prev_volume = volume.groupby(out["symbol"], sort=False).shift(1)
        out["prev_volume"] = prev_volume
        out["volume_ratio_prev1"] = volume / prev_volume.replace(0, pd.NA)

    return out


def load_indicator_cache(indicator_path: Path) -> pd.DataFrame:
    if not indicator_path.exists():
        raise FileNotFoundError(
            f"Indicator cache not found: {indicator_path}\n"
            f"Please run ops/daily_update/build_indicators.py first."
        )

    # Do not read the whole daily_indicators.parquet.
    # The full table is large and can trigger pyarrow/pandas MemoryError.
    required_cols = [
        "symbol", "file", "date",
        "open", "high", "low", "close", "volume", "amount",
        "daily_return_pct", "intraday_return_pct", "amplitude_pct",
        "upper_shadow_pct", "lower_shadow_pct", "body_pct", "body_abs_pct",
        "is_red_k", "is_green_k", "is_flat_k",
        "ma5", "ma10", "ma20", "ma60",
        "volume_ma5", "volume_ma10",
        "volume_ratio_ma5", "volume_ratio_ma10", "volume_ratio_prev1",
        "kdj_k", "kdj_d", "kdj_j",
        "macd_dif", "macd_dea", "macd_hist",
        "renko_value",
    ]

    try:
        import pyarrow.parquet as pq

        schema_cols = list(pq.read_schema(indicator_path).names)
        schema_set = set(schema_cols)
        usecols = [c for c in required_cols if c in schema_set]

        print(
            f"[INFO] Loading indicator cache with column pruning: "
            f"{len(usecols)} / {len(schema_cols)} columns",
            flush=True,
        )
        df = pd.read_parquet(indicator_path, columns=usecols)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read pruned indicator cache: {type(exc).__name__}: {exc}"
        ) from exc

    if df.empty:
        raise RuntimeError(f"Indicator cache is empty: {indicator_path}")

    required = ["symbol", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Indicator cache missing required columns: {missing}\n"
            f"columns={list(df.columns)}"
        )

    df = df.copy(deep=False)

    symbol_sample = df["symbol"].dropna().astype(str).head(2000)
    if not symbol_sample.str.fullmatch(r"\d{6}").all():
        unique_symbols = pd.Series(df["symbol"].dropna().unique())
        symbol_map = {raw: normalize_symbol(raw) for raw in unique_symbols}
        df["symbol"] = df["symbol"].map(symbol_map)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    # Avoid global sort/reset here. Sorting millions of rows can duplicate memory.
    df = df.dropna(subset=["date"])
    df = df[df["symbol"] != ""]
    df = _enforce_canonical_kline_factors(df)

    return df


INDICATOR_CACHE_BASE_REQUIRED_COLUMNS = [
    "symbol",
    "file",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "daily_return_pct",
    "intraday_return_pct",
    "amplitude_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "body_pct",
    "body_abs_pct",
    "is_red_k",
    "is_green_k",
    "is_flat_k",
]

INDICATOR_CACHE_COMMON_REQUIRED_COLUMNS = [
    "ma5",
    "ma10",
    "ma20",
    "ma60",
    "volume_ma5",
    "volume_ma10",
    "volume_ratio_ma5",
    "volume_ratio_ma10",
    "volume_ratio_prev1",
    "macd_dif",
    "macd_dea",
    "macd_hist",
]


def _read_parquet_schema_columns(path: Path) -> List[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    except Exception:
        try:
            return list(pd.read_parquet(path, engine="auto").columns)
        except Exception as exc:
            raise RuntimeError(f"Failed to read parquet schema: {path}: {type(exc).__name__}: {exc}") from exc


def _read_parquet_date_series(path: Path) -> pd.Series:
    try:
        return pd.read_parquet(path, columns=["date"])["date"]
    except Exception as exc:
        raise RuntimeError(f"Failed to read date column from parquet: {path}: {type(exc).__name__}: {exc}") from exc


def _date_only(value: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def _format_date(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "<none>"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _collect_market_cache_dates(market_cache_dir: Path) -> Tuple[pd.DatetimeIndex, int, int]:
    market_cache_dir = Path(market_cache_dir)
    files = sorted(market_cache_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No market cache parquet files found: {market_cache_dir}")

    date_chunks: List[pd.Series] = []
    failed = 0

    for path in files:
        try:
            dates = pd.to_datetime(_read_parquet_date_series(path), errors="coerce").dt.normalize()
            dates = dates.dropna().drop_duplicates()
            if not dates.empty:
                date_chunks.append(dates)
        except Exception as exc:
            failed += 1
            print(f"[WARN] Failed to inspect market cache dates: {path.name}: {exc}", flush=True)

    if not date_chunks:
        raise RuntimeError(f"No usable dates found in market cache: {market_cache_dir}")

    all_dates = pd.concat(date_chunks, ignore_index=True).dropna().drop_duplicates()
    all_dates = pd.to_datetime(all_dates, errors="coerce").dropna().sort_values()
    return pd.DatetimeIndex(all_dates), len(files), failed


def _pool_date_range(pool_path: Path) -> Tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    if not pool_path.exists():
        return None, None, 0

    dates = pd.to_datetime(_read_parquet_date_series(pool_path), errors="coerce").dropna()
    if dates.empty:
        return None, None, 0

    return pd.Timestamp(dates.min()).normalize(), pd.Timestamp(dates.max()).normalize(), int(len(dates))


def resolve_build_window(
    market_dates: pd.DatetimeIndex,
    pool_path: Path,
    start_date: Optional[str],
    end_date: Optional[str],
    incremental: bool,
    incremental_refresh_days: int,
) -> Tuple[pd.Timestamp | None, pd.Timestamp | None, Dict[str, Any]]:
    if len(market_dates) == 0:
        raise RuntimeError("Cannot resolve build window because market_dates is empty.")

    market_min = pd.Timestamp(market_dates.min()).normalize()
    market_max = pd.Timestamp(market_dates.max()).normalize()
    requested_start = _date_only(start_date)
    requested_end = _date_only(end_date)
    effective_end = requested_end or market_max

    if effective_end > market_max:
        raise RuntimeError(
            "Market cache does not contain the requested end date. "
            f"requested_end={_format_date(effective_end)}, market_latest={_format_date(market_max)}"
        )

    if requested_end is not None and requested_end not in market_dates:
        before = market_dates[market_dates < requested_end]
        after = market_dates[market_dates > requested_end]
        nearest_before = pd.Timestamp(before.max()).normalize() if len(before) else None
        nearest_after = pd.Timestamp(after.min()).normalize() if len(after) else None
        raise RuntimeError(
            "Market cache does not contain the exact requested end date. "
            f"requested_end={_format_date(requested_end)}, "
            f"nearest_before={_format_date(nearest_before)}, "
            f"nearest_after={_format_date(nearest_after)}"
        )

    old_min, old_max, old_rows = _pool_date_range(pool_path)

    if incremental:
        if requested_start is not None:
            effective_start = requested_start
            mode = "incremental_explicit_start"
        elif old_max is not None:
            effective_start = old_max - pd.Timedelta(days=int(incremental_refresh_days))
            mode = "incremental_refresh_window"
        else:
            effective_start = market_min
            mode = "incremental_no_existing_pool_full_build"
    else:
        effective_start = requested_start
        mode = "full_build"

    if effective_start is not None and effective_start > effective_end:
        raise RuntimeError(
            "Invalid build window. "
            f"start={_format_date(effective_start)}, end={_format_date(effective_end)}"
        )

    in_window = market_dates[market_dates <= effective_end]
    if effective_start is not None:
        in_window = in_window[in_window >= effective_start]

    if len(in_window) == 0:
        raise RuntimeError(
            "Market cache has no trading dates inside the build window. "
            f"start={_format_date(effective_start)}, end={_format_date(effective_end)}"
        )

    info = {
        "mode": mode,
        "market_min": market_min,
        "market_max": market_max,
        "old_pool_min": old_min,
        "old_pool_max": old_max,
        "old_pool_rows": old_rows,
        "trading_dates_in_window": int(len(in_window)),
    }
    return effective_start, effective_end, info


def required_indicator_cache_columns(strategy_name: str) -> List[str]:
    lower_name = strategy_name.lower()
    required = [
        *INDICATOR_CACHE_BASE_REQUIRED_COLUMNS,
        *INDICATOR_CACHE_COMMON_REQUIRED_COLUMNS,
    ]

    if "renko" in lower_name:
        required.extend(["renko_value"])

    # These strategy-level factor columns are created inside the strategy itself.
    # Do not require them in daily_indicators.parquet.
    seen: set[str] = set()
    result: List[str] = []
    for col in required:
        if col not in seen:
            seen.add(col)
            result.append(col)
    return result


def preflight_check_market_and_indicator_cache(
    market_cache_dir: Path,
    indicator_path: Path,
    strategy_name: str,
    build_start: pd.Timestamp | None,
    build_end: pd.Timestamp | None,
    market_dates: pd.DatetimeIndex,
    market_file_count: int,
    market_failed_count: int,
) -> None:
    print("\n========== PREFLIGHT CHECK ==========")
    print(f"market_cache_dir: {market_cache_dir}")
    print(f"market_files: {market_file_count:,}")
    if market_failed_count:
        print(f"[WARN] market_files_failed_to_inspect: {market_failed_count:,}")

    if not indicator_path.exists():
        raise FileNotFoundError(
            f"Indicator cache not found: {indicator_path}\n"
            f"Please run ops/daily_update/build_indicators.py first."
        )

    market_min = pd.Timestamp(market_dates.min()).normalize()
    market_max = pd.Timestamp(market_dates.max()).normalize()
    print(f"market_date_range: {_format_date(market_min)} -> {_format_date(market_max)}")
    print(f"required_build_window: {_format_date(build_start)} -> {_format_date(build_end)}")

    effective_end = build_end or market_max
    if effective_end > market_max:
        raise RuntimeError(
            "Market cache is stale for this build. "
            f"required_end={_format_date(effective_end)}, market_latest={_format_date(market_max)}"
        )

    window_dates = market_dates[market_dates <= effective_end]
    if build_start is not None:
        window_dates = window_dates[window_dates >= build_start]
    if len(window_dates) == 0:
        raise RuntimeError("No market trading dates are available for the required build window.")

    indicator_columns = _read_parquet_schema_columns(indicator_path)
    required_cols = required_indicator_cache_columns(strategy_name)
    missing_cols = [c for c in required_cols if c not in indicator_columns]
    if missing_cols:
        raise RuntimeError(
            "Indicator cache is missing required columns for build_pool.\n"
            f"indicator_path={indicator_path}\n"
            f"missing_columns={missing_cols}\n"
            "Please run ops/daily_update/build_indicators.py before build_pool."
        )

    indicator_dates = pd.to_datetime(_read_parquet_date_series(indicator_path), errors="coerce").dropna().dt.normalize()
    if indicator_dates.empty:
        raise RuntimeError(f"Indicator cache has no valid date rows: {indicator_path}")

    indicator_date_index = pd.DatetimeIndex(pd.Series(indicator_dates).drop_duplicates().sort_values())
    indicator_min = pd.Timestamp(indicator_date_index.min()).normalize()
    indicator_max = pd.Timestamp(indicator_date_index.max()).normalize()
    print(f"indicator_cache: {indicator_path}")
    print(f"indicator_date_range: {_format_date(indicator_min)} -> {_format_date(indicator_max)}")
    print(f"indicator_required_columns: {len(required_cols):,}")

    if indicator_max < effective_end:
        raise RuntimeError(
            "Indicator cache is stale for this build. "
            f"required_end={_format_date(effective_end)}, indicator_latest={_format_date(indicator_max)}\n"
            "Please run ops/daily_update/build_indicators.py first."
        )

    if effective_end not in indicator_date_index:
        before = indicator_date_index[indicator_date_index < effective_end]
        after = indicator_date_index[indicator_date_index > effective_end]
        nearest_before = pd.Timestamp(before.max()).normalize() if len(before) else None
        nearest_after = pd.Timestamp(after.min()).normalize() if len(after) else None
        raise RuntimeError(
            "Indicator cache does not contain the exact required build end date. "
            f"required_end={_format_date(effective_end)}, "
            f"nearest_before={_format_date(nearest_before)}, "
            f"nearest_after={_format_date(nearest_after)}\n"
            "Please run ops/daily_update/build_indicators.py first."
        )

    if build_start is not None and indicator_max < build_start:
        raise RuntimeError(
            "Indicator cache does not cover the build start date. "
            f"build_start={_format_date(build_start)}, indicator_latest={_format_date(indicator_max)}"
        )

    print("[OK] Market cache date coverage passed.")
    print("[OK] Indicator cache schema/date coverage passed.")


def merge_incremental_pool(
    existing_pool_path: Path,
    new_pool_path: Path,
    strategy_name: str,
    replace_start: pd.Timestamp | None,
    replace_end: pd.Timestamp | None,
) -> pd.DataFrame:
    if not new_pool_path.exists():
        raise FileNotFoundError(f"Incremental build output not found: {new_pool_path}")

    new_pool = pd.read_parquet(new_pool_path)
    if "date" in new_pool.columns:
        new_pool["date"] = pd.to_datetime(new_pool["date"], errors="coerce").dt.normalize()

    if not existing_pool_path.exists():
        final_pool = normalize_pool_schema(new_pool, strategy_name)
        validate_pool_schema(final_pool, strategy_name=strategy_name)
        tmp = existing_pool_path.with_suffix(existing_pool_path.suffix + ".tmp")
        final_pool.to_parquet(tmp, index=False)
        tmp.replace(existing_pool_path)
        return pd.DataFrame({
            "mode": ["incremental_initial_full_write"],
            "old_rows": [0],
            "new_rows": [int(len(new_pool))],
            "final_rows": [int(len(final_pool))],
            "output_path": [str(existing_pool_path)],
        })

    old_pool = pd.read_parquet(existing_pool_path)
    if "date" not in old_pool.columns:
        raise RuntimeError(f"Existing pool has no date column: {existing_pool_path}")

    old_pool["date"] = pd.to_datetime(old_pool["date"], errors="coerce").dt.normalize()

    start = replace_start
    end = replace_end
    if start is None and not new_pool.empty and "date" in new_pool.columns:
        start = pd.Timestamp(new_pool["date"].min()).normalize()
    if end is None and not new_pool.empty and "date" in new_pool.columns:
        end = pd.Timestamp(new_pool["date"].max()).normalize()

    if start is None or end is None:
        raise RuntimeError("Cannot merge incremental pool because replace date window is empty.")

    keep_mask = old_pool["date"].lt(start) | old_pool["date"].gt(end)
    removed_rows = int((~keep_mask).sum())
    kept_old = old_pool.loc[keep_mask].copy(deep=False)

    final_pool = pd.concat([kept_old, new_pool], ignore_index=True)
    final_pool = normalize_pool_schema(final_pool, strategy_name)

    dedupe_cols = [c for c in ["symbol", "date", "selection_strategy"] if c in final_pool.columns]
    if len(dedupe_cols) >= 2:
        before_dedupe = len(final_pool)
        final_pool = final_pool.drop_duplicates(subset=dedupe_cols, keep="last")
        dropped_dupes = before_dedupe - len(final_pool)
    else:
        dropped_dupes = 0

    validate_pool_schema(final_pool, strategy_name=strategy_name)

    tmp = existing_pool_path.with_suffix(existing_pool_path.suffix + ".tmp")
    final_pool.to_parquet(tmp, index=False)
    tmp.replace(existing_pool_path)

    return pd.DataFrame({
        "mode": ["incremental_merge"],
        "replace_start": [_format_date(start)],
        "replace_end": [_format_date(end)],
        "old_rows": [int(len(old_pool))],
        "removed_old_rows": [removed_rows],
        "new_rows": [int(len(new_pool))],
        "dropped_duplicate_rows": [int(dropped_dupes)],
        "final_rows": [int(len(final_pool))],
        "output_path": [str(existing_pool_path)],
    })

def apply_date_filter(
    df: pd.DataFrame,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    out = df

    if start_date:
        start_ts = pd.to_datetime(start_date)
        out = out[out["date"] >= start_ts]

    if end_date:
        end_ts = pd.to_datetime(end_date)
        out = out[out["date"] <= end_ts]

    return out.copy(deep=False)  # avoid duplicating the full indicator cache in memory


def ensure_output_columns(df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """Ensure minimum strategy identity columns without requiring a score system."""
    out = df.copy()

    if TRANSIENT_SELECTION_COLUMN in out.columns:
        out[TRANSIENT_SELECTION_COLUMN] = (
            pd.to_numeric(out[TRANSIENT_SELECTION_COLUMN], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    if "selection_strategy" not in out.columns:
        out["selection_strategy"] = strategy_name

    return out


def _split_strategy_selected_rows(df: pd.DataFrame, keep_all_rows: bool = False) -> pd.DataFrame:
    """Return pool rows from a strategy result.

    Compatibility rule:
    - If selected exists and keep_all_rows is false, keep selected == 1.
    - If selected does not exist, assume the strategy already returned final pool rows.
    """
    if keep_all_rows or TRANSIENT_SELECTION_COLUMN not in df.columns:
        return df.copy()

    selected = pd.to_numeric(df[TRANSIENT_SELECTION_COLUMN], errors="coerce").fillna(0).astype(int)
    return df.loc[selected == 1].copy()


def _drop_duplicate_named_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.is_unique:
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


def _matches_any_pattern(name: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(p.match(name) for p in patterns)


def _drop_old_forward_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if _matches_any_pattern(str(c), OLD_FORWARD_PATTERNS)]
    if not drop_cols:
        return df
    return df.drop(columns=drop_cols, errors="ignore")


def _attach_group_identity(part: pd.DataFrame, group: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = part.copy()

    if "symbol" not in out.columns:
        out["symbol"] = normalize_symbol(symbol)
    else:
        out["symbol"] = out["symbol"].map(normalize_symbol)
        out["symbol"] = out["symbol"].copy()
        out.loc[out["symbol"] == "", "symbol"] = normalize_symbol(symbol)

    if "file" not in out.columns and "file" in group.columns:
        files = group["file"].dropna().astype(str).unique().tolist()
        out["file"] = files[0] if files else pd.NA

    return out


def _attach_missing_indicator_columns(part: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    out = part.copy(deep=False)

    if "date" not in out.columns or "date" not in group.columns:
        return out

    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    add_cols = [
        c for c in REHYDRATE_COLUMNS
        if c in group.columns and c not in out.columns and c != "date"
    ]

    if not add_cols:
        return out

    lookup = group.loc[:, ["date", *add_cols]].copy(deep=False)
    lookup["date"] = pd.to_datetime(lookup["date"], errors="coerce")
    lookup = lookup.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")

    return out.merge(lookup, on="date", how="left", validate="many_to_one")



def _pool_num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pool_safe_pct_distance(close: pd.Series, line: pd.Series) -> pd.Series:
    close_num = _pool_num_series(close)
    line_num = _pool_num_series(line).replace(0, float("nan"))
    return (close_num / line_num - 1.0) * 100.0


def add_trend_distance_factor_columns(part: pd.DataFrame) -> pd.DataFrame:
    out = part.copy(deep=False)

    trend_distance_cols = [
        "t0_close_to_z_short_trend_line_pct",
        "t0_close_to_z_long_trend_line_pct",
    ]

    for col in trend_distance_cols:
        if col not in out.columns:
            out[col] = pd.NA

    if out.empty or "close" not in out.columns:
        return out

    close = _pool_num_series(out["close"])

    if "z_short_trend_line" in out.columns:
        out["t0_close_to_z_short_trend_line_pct"] = _pool_safe_pct_distance(
            close,
            out["z_short_trend_line"],
        )

    if "z_long_trend_line" in out.columns:
        out["t0_close_to_z_long_trend_line_pct"] = _pool_safe_pct_distance(
            close,
            out["z_long_trend_line"],
        )

    if "z_short_trend_line" in out.columns and "z_long_trend_line" in out.columns:
        short_line = _pool_num_series(out["z_short_trend_line"])
        long_line = _pool_num_series(out["z_long_trend_line"])
        valid = short_line.notna() & long_line.notna()

        out["z_short_trend_above_z_long_trend_line"] = pd.NA
        out["z_short_trend_above_z_long_trend_line"] = out["z_short_trend_above_z_long_trend_line"].copy()
        out.loc[valid, "z_short_trend_above_z_long_trend_line"] = (
            short_line.loc[valid] > long_line.loc[valid]
        ).astype(int)

    return out

def _build_forward_lookup(group: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "open", "close"]
    missing = [c for c in required if c not in group.columns]
    if missing:
        raise ValueError(f"Cannot add forward fields. Missing columns in indicator group: {missing}")

    ref = group.loc[:, required].copy(deep=False)
    ref["date"] = pd.to_datetime(ref["date"], errors="coerce")
    ref = ref.dropna(subset=["date"])

    if not ref["date"].is_monotonic_increasing:
        ref = ref.sort_values("date", kind="stable")

    ref = ref.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    lookup = ref.loc[:, ["date"]].copy(deep=False)
    open_num = pd.to_numeric(ref["open"], errors="coerce")
    close_num = pd.to_numeric(ref["close"], errors="coerce")

    for horizon in range(1, FORWARD_HORIZON_MAX + 1):
        lookup[f"t{horizon}_date"] = ref["date"].shift(-horizon)
        lookup[f"t{horizon}_open"] = open_num.shift(-horizon)
        lookup[f"t{horizon}_close"] = close_num.shift(-horizon)

    return lookup


def add_forward_fields_from_t1_open(part: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    """Add T+1 through T+20 prices and returns using T+1 open as buy price."""
    out = _drop_old_forward_columns(part)

    if out.empty:
        for col in FORWARD_COLUMNS:
            out[col] = pd.NA
        return out

    if "date" not in out.columns:
        raise ValueError("Cannot add forward fields. Strategy output missing date column.")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    lookup = _build_forward_lookup(group)
    out = out.merge(lookup, on="date", how="left", validate="many_to_one")

    buy_price = pd.to_numeric(out["t1_open"], errors="coerce")
    valid_buy = buy_price.notna() & (buy_price > 0)

    for horizon in range(1, FORWARD_HORIZON_MAX + 1):
        close_col = f"t{horizon}_close"
        ret_col = f"fwd_return_pct_T{horizon}"
        up_col = f"fwd_up_T{horizon}"

        sell_close = pd.to_numeric(out[close_col], errors="coerce")
        valid = valid_buy & sell_close.notna()
        out[ret_col] = pd.NA
        if ret_col in out.columns:
            out[ret_col] = out[ret_col].copy()
        out.loc[valid, ret_col] = (sell_close.loc[valid] / buy_price.loc[valid] - 1.0) * 100.0
        out[ret_col] = pd.to_numeric(out[ret_col], errors="coerce")

        up = sell_close > buy_price
        out[up_col] = up.where(valid, pd.NA).astype("boolean")

    missing_t1 = (
        out["t1_date"].isna()
        | out["t1_open"].isna()
        | out["t1_close"].isna()
        | ~valid_buy
    )
    missing_after_t1 = pd.Series(False, index=out.index)
    for horizon in range(2, FORWARD_HORIZON_MAX + 1):
        missing_after_t1 = (
            missing_after_t1
            | out[f"t{horizon}_date"].isna()
            | out[f"t{horizon}_close"].isna()
        )

    out["forward_data_status"] = "ok"
    if "forward_data_status" in out.columns:
        out["forward_data_status"] = out["forward_data_status"].copy()
    out.loc[missing_after_t1, "forward_data_status"] = "partial_missing"
    out.loc[missing_t1, "forward_data_status"] = "missing_t1"

    return out


def _strip_strategy_prefix_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=False)  # avoid deep-copying the pool during schema normalization

    drop_cols = [c for c in out.columns if _matches_any_pattern(str(c), DROP_COLUMN_PATTERNS)]
    if drop_cols:
        out = out.drop(columns=drop_cols, errors="ignore")

    for col in list(out.columns):
        col_str = str(col)
        target_name: str | None = None

        for pattern in STRIP_PREFIX_PATTERNS:
            match = pattern.match(col_str)
            if match:
                target_name = match.group(1)
                break

        if target_name is None or target_name == col_str:
            continue

        if target_name in out.columns:
            base = out[target_name]
            prefixed = out[col]
            out[target_name] = base.where(base.notna(), prefixed)
            out = out.drop(columns=[col])
        else:
            out = out.rename(columns={col: target_name})

    return out



def _safe_to_numeric_if_possible(s: pd.Series) -> pd.Series:
    converted = pd.to_numeric(s, errors="coerce")

    original_non_na = s.notna()
    if original_non_na.any() and converted[original_non_na].notna().sum() == 0:
        return s

    return converted

def _coerce_common_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=False)

    if "symbol" in out.columns:
        symbol_sample = out["symbol"].dropna().astype(str).head(2000)
        if not symbol_sample.str.fullmatch(r"\d{6}").all():
            out["symbol"] = out["symbol"].map(normalize_symbol)

    for col in ["date", *[f"t{horizon}_date" for horizon in range(1, FORWARD_HORIZON_MAX + 1)]]:
        if col in out.columns and not pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors="coerce")

    for col in ["selected", "selected_score_base"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    numeric_cols = [
        c for c in out.columns
        if c not in {
            "symbol",
            "file",
            "selection_strategy",
            "risk_tags",
            "j_condition_rule",
            "j_condition_source_col",
            "v4_hint_label",
            "forward_data_status",
        }
        and not str(c).endswith("_date")
        and str(c) not in {"date"}
        and not pd.api.types.is_bool_dtype(out[c])
    ]

    for col in numeric_cols:
        if out[col].dtype == "object":
            out[col] = _safe_to_numeric_if_possible(out[col])

    return out


def _preferred_columns_for_strategy(strategy_name: str) -> List[str]:
    preferred = [
        *CORE_COLUMNS,
        *PRICE_COLUMNS,
        *COMMON_INDICATOR_COLUMNS,
    ]

    lower_name = strategy_name.lower()

    if "b2" in lower_name or "confirm" in lower_name:
        preferred.extend(B2_EXTENSION_COLUMNS)

    if "renko" in lower_name:
        preferred.extend(RENKO_EXTENSION_COLUMNS)

    preferred.extend(FORWARD_COLUMNS)

    seen: set[str] = set()
    result: List[str] = []
    for col in preferred:
        if col not in seen:
            seen.add(col)
            result.append(col)

    return result


def normalize_pool_schema(pool: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """Normalize final pool columns without changing strategy selection results."""
    if pool.empty:
        return pool.copy()

    out = _drop_duplicate_named_columns(pool)
    out = _strip_strategy_prefix_columns(out)
    out = _drop_duplicate_named_columns(out)
    out = _coerce_common_types(out)
    out = out.drop(columns=[c for c in FINAL_POOL_DROP_COLUMNS if c in out.columns], errors="ignore")
    out = out.drop(columns=[c for c in RAW_ABSOLUTE_POOL_DROP_COLUMNS if c in out.columns], errors="ignore")
    out = drop_removed_score_columns(out)

    preferred = _preferred_columns_for_strategy(strategy_name)
    ordered = [c for c in preferred if c in out.columns]
    remaining = [c for c in out.columns if c not in ordered]

    out = out[[*ordered, *remaining]].copy(deep=False)

    # Avoid allocating another full pool copy for large pools.
    if len(out) <= 300_000:
        sort_cols = [c for c in ["date", "symbol"] if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols, ascending=True, kind="stable").reset_index(drop=True)

    return out


def build_pool(
    indicators: pd.DataFrame,
    strategy_name: str,
    strategy_kwargs: Dict[str, Any],
    keep_all_rows: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    select_func = load_strategy_func(strategy_name)

    tmp_dir = output_path.parent / f"_tmp_{output_path.stem}_parts"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    part_paths: List[Path] = []
    errors: List[Dict[str, str]] = []
    selected_rows_total = 0
    part_no = 0

    grouped = indicators.groupby("symbol", sort=True)
    iterator = grouped

    if progress:
        iterator = tqdm(grouped, total=indicators["symbol"].nunique(), desc="Build pool by symbol")

    for symbol, g in iterator:
        g = g.sort_values("date").reset_index(drop=True)

        try:
            selected_df = select_func(g, **strategy_kwargs)

            if selected_df is None or not isinstance(selected_df, pd.DataFrame):
                raise TypeError(
                    f"Strategy returned {type(selected_df)}, expected pandas.DataFrame"
                )

            selected_df = _attach_group_identity(selected_df, g, str(symbol))
            selected_df = _attach_missing_indicator_columns(selected_df, g)
            selected_df = ensure_output_columns(selected_df, strategy_name)

            part = _split_strategy_selected_rows(selected_df, keep_all_rows=keep_all_rows)

            if not part.empty:
                part = add_trend_distance_factor_columns(part)
                part = add_forward_fields_from_t1_open(part, g)
                part = normalize_pool_schema(part, strategy_name)

                # Validate symbol-level output before writing, without keeping all pool rows in RAM.
                validate_pool_schema(part, strategy_name=strategy_name)

                part_no += 1
                part_path = tmp_dir / f"part-{part_no:05d}.parquet"
                part.to_parquet(part_path, index=False)

                part_paths.append(part_path)
                selected_rows_total += int(len(part))

                if part_no % 100 == 0:
                    print(
                        f"[INFO] wrote_pool_parts={part_no:,}, selected_rows={selected_rows_total:,}",
                        flush=True,
                    )

        except Exception as e:
            errors.append({"symbol": str(symbol), "error": repr(e)})
            print(f"[WARN] strategy failed for symbol={symbol}: {e}")

    if errors:
        print(f"[WARN] strategy failed symbols: {len(errors):,}")

    if not parts:
        print("[WARN] No selected rows generated.")
        return pd.DataFrame()

    pool = pd.concat(parts, ignore_index=True)
    pool = normalize_pool_schema(pool, strategy_name)
    report = validate_pool_schema(pool, strategy_name=strategy_name)

    print(f"[OK] Pool schema validation passed: {report.schema_version}")
    print(f"[OK] Dynamic factor columns detected: {len(report.factor_columns)}")
    if report.factor_columns:
        print(report.factor_columns)
    for warning in report.warnings:
        print(f"[WARN] {warning}")

    return pool


def build_pool_streaming_from_market_cache(
    market_cache_dir: Path,
    strategy_name: str,
    strategy_kwargs: Dict[str, Any],
    start_date: Optional[str],
    end_date: Optional[str],
    output_path: Path,
    keep_all_rows: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Low-memory pool builder.

    Reads one market-cache parquet per symbol, computes indicators for that symbol,
    runs the strategy, writes selected rows to temporary parquet parts immediately,
    then merges parquet parts without full pd.concat.
    """
    select_func = load_strategy_func(strategy_name)

    market_cache_dir = Path(market_cache_dir)
    output_path = Path(output_path)

    files = sorted(market_cache_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No market cache parquet files found: {market_cache_dir}")

    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None

    tmp_dir = output_path.parent / f"_tmp_{output_path.stem}_parts"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    part_paths: List[Path] = []
    errors: List[Dict[str, str]] = []
    selected_rows_total = 0
    part_no = 0

    iterator = files
    if progress:
        iterator = tqdm(files, total=len(files), desc="Build pool by symbol file")

    for file_path in iterator:
        symbol = normalize_symbol(file_path.stem)

        try:
            raw = _read_table(file_path)
            if raw is None or raw.empty:
                continue

            base = _standardize_market_df(
                raw,
                fallback_symbol=symbol,
                fallback_file=file_path.name,
            )

            if end_ts is not None:
                base = base[base["date"] <= end_ts]
            if base.empty:
                continue

            g = add_all_indicators(
                base,
                ma_windows=(5, 10, 20, 60),
                volume_windows=(5, 10),
                macd_fast=12,
                macd_slow=26,
                macd_signal=9,
            )

            if start_ts is not None:
                g = g[pd.to_datetime(g["date"], errors="coerce") >= start_ts]
            if end_ts is not None:
                g = g[pd.to_datetime(g["date"], errors="coerce") <= end_ts]

            if g.empty:
                continue

            g = g.sort_values("date", kind="stable").reset_index(drop=True)
            g = _enforce_canonical_kline_factors(g)

            selected_df = select_func(g, **strategy_kwargs)
            if selected_df is None or not isinstance(selected_df, pd.DataFrame):
                raise TypeError(f"Strategy returned {type(selected_df)}, expected pandas.DataFrame")

            selected_df = _attach_group_identity(selected_df, g, symbol)
            selected_df = _attach_missing_indicator_columns(selected_df, g)
            selected_df = ensure_output_columns(selected_df, strategy_name)

            part = _split_strategy_selected_rows(selected_df, keep_all_rows=keep_all_rows)
            if part.empty:
                continue

            part = add_trend_distance_factor_columns(part)
            part = add_forward_fields_from_t1_open(part, g)
            part = normalize_pool_schema(part, strategy_name)

            validate_pool_schema(part, strategy_name=strategy_name)

            part_no += 1
            part_path = tmp_dir / f"part-{part_no:05d}.parquet"
            part.to_parquet(part_path, index=False)
            part_paths.append(part_path)
            selected_rows_total += int(len(part))

            if part_no % 100 == 0:
                print(
                    f"[INFO] wrote_pool_parts={part_no:,}, selected_rows={selected_rows_total:,}",
                    flush=True,
                )

        except Exception as exc:
            errors.append({"symbol": str(symbol), "error": repr(exc)})
            print(f"[WARN] strategy failed for symbol={symbol}: {exc}", flush=True)

    if errors:
        print(f"[WARN] strategy failed symbols: {len(errors):,}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not part_paths:
        print("[WARN] No selected rows generated.", flush=True)
        empty = pd.DataFrame()
        empty.to_parquet(output_path, index=False)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return pd.DataFrame(
            {
                "selection_strategy": [strategy_name],
                "rows_written": [0],
                "output_path": [str(output_path)],
            }
        )

    print(f"[INFO] Merging {len(part_paths):,} pool parts into final parquet...", flush=True)

    try:
        import pyarrow.parquet as pq

        final_tmp = output_path.with_suffix(output_path.suffix + ".tmp")
        if final_tmp.exists():
            final_tmp.unlink()

        writer = None
        try:
            for part_path in part_paths:
                table = pq.read_table(part_path)
                if writer is None:
                    writer = pq.ParquetWriter(final_tmp, table.schema)
                else:
                    table = table.cast(writer.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

        final_tmp.replace(output_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("[OK] Pool parquet written without full pd.concat.", flush=True)
    print(f"[OK] output: {output_path}", flush=True)
    print(f"[OK] selected_rows: {selected_rows_total:,}", flush=True)

    return pd.DataFrame(
        {
            "selection_strategy": [strategy_name],
            "rows_written": [selected_rows_total],
            "output_path": [str(output_path)],
        }
    )


def save_outputs(
    pool: pd.DataFrame,
    output_dir: Path,
    strategy_name: str,
    save_csv: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{strategy_name}_pool.parquet"
    csv_path = output_dir / f"{strategy_name}_pool.csv"
    meta_path = output_dir / f"{strategy_name}_pool.meta.json"

    schema_report = validate_pool_schema(pool, strategy_name=strategy_name)

    pool.to_parquet(parquet_path, index=False)

    if save_csv:
        pool.to_csv(csv_path, index=False, encoding="utf-8-sig")

    meta = {
        "strategy_name": strategy_name,
        "rows": int(len(pool)),
        "symbols": int(pool["symbol"].nunique()) if "symbol" in pool.columns and not pool.empty else 0,
        "date_min": str(pool["date"].min()) if "date" in pool.columns and not pool.empty else None,
        "date_max": str(pool["date"].max()) if "date" in pool.columns and not pool.empty else None,
        "forward_buy_price": "t1_open",
        "forward_return_definition": {
            **{
                f"fwd_return_pct_T{horizon}": f"t{horizon}_close / t1_open - 1"
                for horizon in range(1, FORWARD_HORIZON_MAX + 1)
            },
            **{
                f"fwd_up_T{horizon}": f"t{horizon}_close > t1_open"
                for horizon in range(1, FORWARD_HORIZON_MAX + 1)
            },
        },
        "schema_version": POOL_SCHEMA_VERSION,
        "factor_columns": schema_report.factor_columns,
        "schema_warnings": schema_report.warnings,
        "columns": list(pool.columns),
    }

    if metadata:
        meta.update(metadata)

    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    paths = {
        "parquet": parquet_path,
        "meta": meta_path,
    }

    if save_csv:
        paths["csv"] = csv_path

    return paths


def print_pool_summary(pool: pd.DataFrame, strategy_name: str) -> None:
    print("\n========== POOL SUMMARY ==========")
    print(f"strategy: {strategy_name}")
    print(f"rows:     {len(pool):,}")

    if pool.empty:
        return

    if "symbol" in pool.columns:
        print(f"symbols:  {pool['symbol'].nunique():,}")

    if "date" in pool.columns:
        print(f"dates:    {pool['date'].nunique():,}")
        print(f"range:    {pool['date'].min()} -> {pool['date'].max()}")

    print("\nselected by date sample:")
    if "symbol" in pool.columns:
        by_date = pool.groupby("date")["symbol"].count().reset_index(name="count")
    else:
        by_date = pool.groupby("date").size().reset_index(name="count")
    print(by_date.tail(20).to_string(index=False))

    if "score_rank_key" in pool.columns:
        print("\nscore_rank_key describe:")
        print(pool["score_rank_key"].describe().to_string())

    forward_cols = [f"fwd_return_pct_T{horizon}" for horizon in range(1, FORWARD_HORIZON_MAX + 1) if f"fwd_return_pct_T{horizon}" in pool.columns]
    if forward_cols:
        print("\nforward return columns:")
        print(forward_cols)
        print("\nforward return describe:")
        print(pool[forward_cols].describe().to_string())

    up_cols = [f"fwd_up_T{horizon}" for horizon in range(1, FORWARD_HORIZON_MAX + 1) if f"fwd_up_T{horizon}" in pool.columns]
    if up_cols:
        print("\nforward up ratio:")
        for col in up_cols:
            print(f"{col}: {pool[col].astype('boolean').mean():.4f}")

    if "forward_data_status" in pool.columns:
        print("\nforward_data_status:")
        print(pool["forward_data_status"].value_counts(dropna=False).to_string())

    print("\ncolumns:")
    print(list(pool.columns))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="Strategy module name under strategies/selection, or STRATEGY_NAME registered in strategies/selection/registry.py",
    )

    parser.add_argument(
        "--indicator-path",
        type=str,
        default=str(DEFAULT_INDICATOR_PATH),
        help="Path to daily_indicators.parquet",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_POOL_DIR),
        help="Directory to save pool outputs.",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date, e.g. 2021-01-01",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date, e.g. 2026-05-10",
    )

    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help=(
            "Strategy parameter. Can be repeated. "
            "Formats: key=value, key:int=10, key:float=0.75, key:bool=true"
        ),
    )

    parser.add_argument(
        "--keep-all-rows",
        action="store_true",
        help="Save all strategy output rows instead of selected rows only.",
    )

    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not save CSV, only parquet.",
    )

    parser.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Incrementally update the existing pool. The builder refreshes a recent "
            "date window, deletes old pool rows in that window, appends the new rows, "
            "deduplicates by symbol/date/selection_strategy, then overwrites the pool safely."
        ),
    )

    parser.add_argument(
        "--incremental-refresh-days",
        type=int,
        default=45,
        help=(
            "Calendar-day lookback window to rebuild in incremental mode when --start-date "
            "is not provided. Default 45 to allow T+20 forward fields to be refreshed."
        ),
    )

    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip market-cache and indicator-cache preflight checks.",
    )

    args = parser.parse_args()

    strategy_name = args.strategy
    indicator_path = Path(args.indicator_path)
    output_dir = Path(args.output_dir)
    strategy_kwargs = parse_extra_args(args.param)

    pool_path_arg = (
        getattr(args, "output", None)
        or getattr(args, "output_path", None)
        or getattr(args, "pool_path", None)
    )
    if pool_path_arg:
        pool_path = Path(pool_path_arg)
    else:
        pool_path = POOLS_DIR / f"{strategy_name}_pool.parquet"

    print("[INFO] Building pool in low-memory streaming mode from market cache...")
    print(f"[INFO] market cache dir: {MARKET_CACHE_DIR}")
    print(f"[INFO] indicator cache: {indicator_path}")
    print(f"[INFO] output pool: {pool_path}")
    print(f"[INFO] incremental: {bool(args.incremental)}")
    print("[INFO] Strategy params:")
    print(json.dumps(strategy_kwargs, ensure_ascii=False, indent=2))

    market_dates, market_file_count, market_failed_count = _collect_market_cache_dates(MARKET_CACHE_DIR)
    build_start, build_end, window_info = resolve_build_window(
        market_dates=market_dates,
        pool_path=pool_path,
        start_date=args.start_date,
        end_date=args.end_date,
        incremental=bool(args.incremental),
        incremental_refresh_days=int(args.incremental_refresh_days),
    )

    print("\n========== BUILD WINDOW ==========")
    print(f"mode: {window_info['mode']}")
    print(f"build_start: {_format_date(build_start)}")
    print(f"build_end:   {_format_date(build_end)}")
    print(f"market_range: {_format_date(window_info['market_min'])} -> {_format_date(window_info['market_max'])}")
    print(f"existing_pool_range: {_format_date(window_info['old_pool_min'])} -> {_format_date(window_info['old_pool_max'])}")
    print(f"existing_pool_rows: {window_info['old_pool_rows']:,}")
    print(f"trading_dates_in_window: {window_info['trading_dates_in_window']:,}")

    if not args.skip_preflight:
        preflight_check_market_and_indicator_cache(
            market_cache_dir=MARKET_CACHE_DIR,
            indicator_path=indicator_path,
            strategy_name=strategy_name,
            build_start=build_start,
            build_end=build_end,
            market_dates=market_dates,
            market_file_count=market_file_count,
            market_failed_count=market_failed_count,
        )
    else:
        print("[WARN] Preflight checks skipped by --skip-preflight.")

    if args.incremental:
        tmp_incremental_path = pool_path.with_suffix(pool_path.suffix + ".incremental_build.tmp")
        if tmp_incremental_path.exists():
            tmp_incremental_path.unlink()

        build_pool_streaming_from_market_cache(
            market_cache_dir=MARKET_CACHE_DIR,
            strategy_name=strategy_name,
            strategy_kwargs=strategy_kwargs,
            start_date=_format_date(build_start) if build_start is not None else None,
            end_date=_format_date(build_end) if build_end is not None else None,
            output_path=tmp_incremental_path,
            keep_all_rows=args.keep_all_rows,
            progress=True,
        )

        merge_report = merge_incremental_pool(
            existing_pool_path=pool_path,
            new_pool_path=tmp_incremental_path,
            strategy_name=strategy_name,
            replace_start=build_start,
            replace_end=build_end,
        )
        tmp_incremental_path.unlink(missing_ok=True)

        print("\n========== INCREMENTAL MERGE SUMMARY ==========")
        print(merge_report.to_string(index=False))
        print(f"[DONE] output: {pool_path}")
        return

    build_pool_streaming_from_market_cache(
        market_cache_dir=MARKET_CACHE_DIR,
        strategy_name=strategy_name,
        strategy_kwargs=strategy_kwargs,
        start_date=_format_date(build_start) if build_start is not None else None,
        end_date=_format_date(build_end) if build_end is not None else None,
        output_path=pool_path,
        keep_all_rows=args.keep_all_rows,
        progress=True,
    )
    print("[INFO] Streaming build wrote final parquet already; skip in-memory save_outputs.")
    print(f"[DONE] output: {pool_path}")
    return


    print_pool_summary(pool, strategy_name)

    print("[INFO] Saving outputs...")

    paths = save_outputs(
        pool=pool,
        output_dir=output_dir,
        strategy_name=strategy_name,
        save_csv=not args.no_csv,
        metadata={
            "indicator_path": str(indicator_path),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "strategy_kwargs": strategy_kwargs,
            "keep_all_rows": bool(args.keep_all_rows),
        },
    )

    print("\n========== OUTPUT FILES ==========")
    for k, p in paths.items():
        print(f"[OK] {k}: {p}")


if __name__ == "__main__":
    main()
