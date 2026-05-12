# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Generic pool builder.

Purpose:
- Read indicator cache.
- Dynamically load any selection strategy from selection_strategies/.
- Run strategy per symbol.
- Keep selected rows.
- Add executable forward fields using T+1 open as buy price.
- Normalize pool schema for common columns and strategy-specific extension columns.
- Save pool parquet/csv.

Strategy file requirement:
    selection_strategies/xxx.py

Must provide either:
    def select(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ...
or:
    SELECT_FUNC = select

Strategy output must contain:
    selected

Recommended strategy output:
    selected
    selected_score_base
    score_rank_key
    score_pct
    selection_strategy
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"C:\Users\zyf37\Desktop\BackTest_Data")
DEFAULT_INDICATOR_PATH = DEFAULT_DATA_ROOT / "indicator_cache" / "daily_indicators.parquet"
DEFAULT_POOL_DIR = DEFAULT_DATA_ROOT / "pools"

CORE_COLUMNS = [
    "symbol",
    "file",
    "date",
    "selection_strategy",
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
    "z_short_trend_line",
    "z_long_trend_line",
    "t0_close_to_z_short_trend_line_pct",
    "t0_close_to_z_long_trend_line_pct",
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

FORWARD_COLUMNS = [
    "t1_date",
    "t1_open",
    "t1_close",
    "t2_date",
    "t2_open",
    "t2_close",
    "t3_date",
    "t3_open",
    "t3_close",
    "fwd_return_pct_T1",
    "fwd_return_pct_T2",
    "fwd_return_pct_T3",
    "fwd_up_T1",
    "fwd_up_T2",
    "fwd_up_T3",
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
    strategy_path = PROJECT_ROOT / "selection_strategies" / f"{strategy_name}.py"

    if not strategy_path.exists():
        raise FileNotFoundError(
            f"Strategy file not found: {strategy_path}\n"
            f"Expected: selection_strategies/{strategy_name}.py"
        )

    module_name = f"selection_strategies.{strategy_name}"

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


def load_indicator_cache(indicator_path: Path) -> pd.DataFrame:
    if not indicator_path.exists():
        raise FileNotFoundError(
            f"Indicator cache not found: {indicator_path}\n"
            f"Please run scripts/build_indicators.py first."
        )

    df = pd.read_parquet(indicator_path)

    if df.empty:
        raise RuntimeError(f"Indicator cache is empty: {indicator_path}")

    required = ["symbol", "date"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Indicator cache missing required columns: {missing}\n"
            f"columns={list(df.columns)}"
        )

    df = df.copy()
    df["symbol"] = df["symbol"].map(normalize_symbol)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date"])
    df = df[df["symbol"] != ""]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    return df


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

    return out.copy()


def ensure_output_columns(df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    out = df.copy()

    if "selected" not in out.columns:
        raise ValueError(
            f"Strategy output missing required column: selected. "
            f"Strategy={strategy_name}"
        )

    out["selected"] = pd.to_numeric(out["selected"], errors="coerce").fillna(0).astype(int)

    if "selected_score_base" not in out.columns:
        out["selected_score_base"] = out["selected"]

    if "score_rank_key" not in out.columns:
        out["score_rank_key"] = 0.0

    if "score_pct" not in out.columns:
        out["score_pct"] = 0.0

    if "selection_strategy" not in out.columns:
        out["selection_strategy"] = strategy_name

    return out


def _drop_duplicate_named_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.is_unique:
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


def _matches_any_pattern(name: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(p.match(name) for p in patterns)


def _drop_old_forward_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in df.columns if _matches_any_pattern(str(c), OLD_FORWARD_PATTERNS)]
    if not drop_cols:
        return df.copy()
    return df.drop(columns=drop_cols, errors="ignore").copy()


def _attach_group_identity(part: pd.DataFrame, group: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = part.copy()

    if "symbol" not in out.columns:
        out["symbol"] = normalize_symbol(symbol)
    else:
        out["symbol"] = out["symbol"].map(normalize_symbol)
        out.loc[out["symbol"] == "", "symbol"] = normalize_symbol(symbol)

    if "file" not in out.columns and "file" in group.columns:
        files = group["file"].dropna().astype(str).unique().tolist()
        out["file"] = files[0] if files else pd.NA

    return out


def _attach_missing_indicator_columns(part: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    out = part.copy()

    if "date" not in out.columns or "date" not in group.columns:
        return out

    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    group2 = group.copy()
    group2["date"] = pd.to_datetime(group2["date"], errors="coerce")
    group2 = group2.dropna(subset=["date"])

    add_cols = [
        c for c in REHYDRATE_COLUMNS
        if c in group2.columns and c not in out.columns and c != "date"
    ]

    if not add_cols:
        return out

    lookup = group2[["date", *add_cols]].drop_duplicates(subset=["date"], keep="last")
    out = out.merge(lookup, on="date", how="left", validate="many_to_one")

    return out



def _pool_num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pool_safe_pct_distance(close: pd.Series, line: pd.Series) -> pd.Series:
    close_num = _pool_num_series(close)
    line_num = _pool_num_series(line).replace(0, float("nan"))
    return (close_num / line_num - 1.0) * 100.0


def add_trend_distance_factor_columns(part: pd.DataFrame) -> pd.DataFrame:
    out = part.copy()

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

    return out

def _build_forward_lookup(group: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "open", "close"]
    missing = [c for c in required if c not in group.columns]
    if missing:
        raise ValueError(f"Cannot add forward fields. Missing columns in indicator group: {missing}")

    ref = group.copy()
    ref["date"] = pd.to_datetime(ref["date"], errors="coerce")
    ref = ref.dropna(subset=["date"])
    ref = ref.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    lookup = ref[["date"]].copy()

    for horizon in (1, 2, 3):
        lookup[f"t{horizon}_date"] = ref["date"].shift(-horizon)
        lookup[f"t{horizon}_open"] = pd.to_numeric(ref["open"], errors="coerce").shift(-horizon)
        lookup[f"t{horizon}_close"] = pd.to_numeric(ref["close"], errors="coerce").shift(-horizon)

    return lookup


def add_forward_fields_from_t1_open(part: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    """Add T+1/T+2/T+3 prices and returns using T+1 open as buy price."""
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

    for horizon in (1, 2, 3):
        close_col = f"t{horizon}_close"
        ret_col = f"fwd_return_pct_T{horizon}"
        up_col = f"fwd_up_T{horizon}"

        sell_close = pd.to_numeric(out[close_col], errors="coerce")
        valid = valid_buy & sell_close.notna()
        out[ret_col] = pd.NA
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
    missing_t2_or_t3 = out["t2_date"].isna() | out["t2_close"].isna() | out["t3_date"].isna() | out["t3_close"].isna()

    out["forward_data_status"] = "ok"
    out.loc[missing_t2_or_t3, "forward_data_status"] = "partial_missing"
    out.loc[missing_t1, "forward_data_status"] = "missing_t1"

    return out


def _strip_strategy_prefix_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

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
    out = df.copy()

    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].map(normalize_symbol)

    for col in ["date", "t1_date", "t2_date", "t3_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    for col in ["selected", "selected_score_base"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    numeric_cols = [
        c for c in out.columns
        if c not in {"symbol", "file", "selection_strategy", "risk_tags", "j_condition_rule", "j_condition_source_col", "v4_hint_label", "forward_data_status"}
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

    preferred = _preferred_columns_for_strategy(strategy_name)
    ordered = [c for c in preferred if c in out.columns]
    remaining = [c for c in out.columns if c not in ordered]

    out = out[[*ordered, *remaining]].copy()

    sort_cols = [c for c in ["date", "score_rank_key", "symbol"] if c in out.columns]
    if sort_cols:
        ascending = [True if c != "score_rank_key" else False for c in sort_cols]
        out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    return out


def build_pool(
    indicators: pd.DataFrame,
    strategy_name: str,
    strategy_kwargs: Dict[str, Any],
    keep_all_rows: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    select_func = load_strategy_func(strategy_name)

    parts: List[pd.DataFrame] = []
    errors: List[Dict[str, str]] = []

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

            if keep_all_rows:
                part = selected_df.copy()
            else:
                part = selected_df[selected_df["selected"] == 1].copy()

            if not part.empty:
                part = add_trend_distance_factor_columns(part)
                part = add_forward_fields_from_t1_open(part, g)
                parts.append(part)

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

    return pool


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
            "fwd_return_pct_T1": "t1_close / t1_open - 1",
            "fwd_return_pct_T2": "t2_close / t1_open - 1",
            "fwd_return_pct_T3": "t3_close / t1_open - 1",
            "fwd_up_T1": "t1_close > t1_open",
            "fwd_up_T2": "t2_close > t1_open",
            "fwd_up_T3": "t3_close > t1_open",
        },
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

    forward_cols = [c for c in ["fwd_return_pct_T1", "fwd_return_pct_T2", "fwd_return_pct_T3"] if c in pool.columns]
    if forward_cols:
        print("\nforward return columns:")
        print(forward_cols)
        print("\nforward return describe:")
        print(pool[forward_cols].describe().to_string())

    up_cols = [c for c in ["fwd_up_T1", "fwd_up_T2", "fwd_up_T3"] if c in pool.columns]
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
        help="Strategy module name under selection_strategies, without .py",
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

    args = parser.parse_args()

    strategy_name = args.strategy
    indicator_path = Path(args.indicator_path)
    output_dir = Path(args.output_dir)
    strategy_kwargs = parse_extra_args(args.param)

    print("[INFO] Loading indicator cache...")
    indicators = load_indicator_cache(indicator_path)

    print(f"[INFO] indicator rows: {len(indicators):,}")
    print(f"[INFO] indicator symbols: {indicators['symbol'].nunique():,}")
    print(f"[INFO] indicator date range: {indicators['date'].min()} -> {indicators['date'].max()}")

    indicators = apply_date_filter(indicators, args.start_date, args.end_date)

    if indicators.empty:
        raise RuntimeError("Indicator data is empty after date filtering.")

    print(f"[INFO] rows after date filter: {len(indicators):,}")
    print(f"[INFO] date range after filter: {indicators['date'].min()} -> {indicators['date'].max()}")

    print("[INFO] Strategy params:")
    print(json.dumps(strategy_kwargs, ensure_ascii=False, indent=2))

    print("[INFO] Building pool...")
    pool = build_pool(
        indicators=indicators,
        strategy_name=strategy_name,
        strategy_kwargs=strategy_kwargs,
        keep_all_rows=args.keep_all_rows,
        progress=True,
    )

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
