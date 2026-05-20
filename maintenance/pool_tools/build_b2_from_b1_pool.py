# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.path_manager import INDICATOR_CACHE_PATH, POOLS_DIR, MARKET_CACHE_DIR
from core.indicator_store import _read_table, _standardize_market_df


STRATEGY_NAME = "b2_confirm_from_b1_v0"
DEFAULT_B1_POOL = POOLS_DIR / "b1_stage_low_select_strategy_v1_pool.parquet"
DEFAULT_OUTPUT = POOLS_DIR / f"{STRATEGY_NAME}_pool.parquet"
FORWARD_HORIZON_MAX = 20


BASE_COLUMNS = [
    "symbol", "file", "date",
    "open", "high", "low", "close", "volume", "amount",
    "daily_return_pct", "intraday_return_pct", "amplitude_pct",
    "upper_shadow_pct", "lower_shadow_pct", "body_pct", "body_abs_pct",
    "is_red_k", "is_green_k", "is_flat_k",
    "ma5", "ma10", "ma20", "ma60",
    "volume_ma5", "volume_ma10", "volume_ratio_ma5", "volume_ratio_ma10", "volume_ratio_prev1",
    "kdj_k", "kdj_d", "kdj_j",
    "macd_dif", "macd_dea", "macd_hist",
]


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = _to_num(b).replace(0, np.nan)
    return _to_num(a) / denom


def _tdx_sma(series: pd.Series, period: int, weight: int = 1) -> pd.Series:
    alpha = float(weight) / float(period)
    return _to_num(series).ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def _read_parquet_existing_columns(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    wanted = list(dict.fromkeys(columns))
    try:
        import pyarrow.parquet as pq

        existing = set(pq.read_schema(path).names)
        usecols = [c for c in wanted if c in existing]
        return pd.read_parquet(path, columns=usecols)
    except Exception:
        df = pd.read_parquet(path)
        usecols = [c for c in wanted if c in df.columns]
        return df.loc[:, usecols].copy()


def _normalize_symbol_date(df: pd.DataFrame, name: str) -> pd.DataFrame:
    required = ["symbol", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

    out = df.copy(deep=False)
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["symbol", "date"])
    return out


def _ensure_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"], kind="stable").copy()

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in out.columns:
            out[col] = _to_num(out[col])

    close = _to_num(out["close"])
    open_ = _to_num(out["open"])
    high = _to_num(out["high"])
    low = _to_num(out["low"])
    volume = _to_num(out["volume"])

    prev_close = close.groupby(out["symbol"], sort=False).shift(1)
    prev_close_safe = prev_close.replace(0, np.nan)

    if "daily_return_pct" not in out.columns:
        out["daily_return_pct"] = (close / prev_close_safe - 1.0) * 100.0
    else:
        out["daily_return_pct"] = _to_num(out["daily_return_pct"])

    if "intraday_return_pct" not in out.columns:
        out["intraday_return_pct"] = (close / open_.replace(0, np.nan) - 1.0) * 100.0
    else:
        out["intraday_return_pct"] = _to_num(out["intraday_return_pct"])

    if "amplitude_pct" not in out.columns:
        out["amplitude_pct"] = (high - low) / prev_close_safe * 100.0
    else:
        out["amplitude_pct"] = _to_num(out["amplitude_pct"])

    if "body_pct" not in out.columns:
        out["body_pct"] = (close - open_) / prev_close_safe * 100.0
    else:
        out["body_pct"] = _to_num(out["body_pct"])

    if "body_abs_pct" not in out.columns:
        out["body_abs_pct"] = out["body_pct"].abs()
    else:
        out["body_abs_pct"] = _to_num(out["body_abs_pct"])

    max_oc = pd.concat([open_, close], axis=1).max(axis=1)
    min_oc = pd.concat([open_, close], axis=1).min(axis=1)

    if "upper_shadow_pct" not in out.columns:
        out["upper_shadow_pct"] = (high - max_oc) / prev_close_safe * 100.0
    else:
        out["upper_shadow_pct"] = _to_num(out["upper_shadow_pct"])

    if "lower_shadow_pct" not in out.columns:
        out["lower_shadow_pct"] = (min_oc - low) / prev_close_safe * 100.0
    else:
        out["lower_shadow_pct"] = _to_num(out["lower_shadow_pct"])

    if "volume_ratio_prev1" not in out.columns:
        prev_volume = volume.groupby(out["symbol"], sort=False).shift(1)
        out["volume_ratio_prev1"] = volume / prev_volume.replace(0, np.nan)
    else:
        out["volume_ratio_prev1"] = _to_num(out["volume_ratio_prev1"])

    if "volume_ma5" not in out.columns:
        out["volume_ma5"] = volume.groupby(out["symbol"], sort=False).transform(lambda s: s.rolling(5, min_periods=2).mean())
    if "volume_ratio_ma5" not in out.columns:
        out["volume_ratio_ma5"] = volume / _to_num(out["volume_ma5"]).replace(0, np.nan)

    if "volume_ma10" not in out.columns:
        out["volume_ma10"] = volume.groupby(out["symbol"], sort=False).transform(lambda s: s.rolling(10, min_periods=3).mean())
    if "volume_ratio_ma10" not in out.columns:
        out["volume_ratio_ma10"] = volume / _to_num(out["volume_ma10"]).replace(0, np.nan)

    out["is_red_k"] = (close > open_).astype("int8")
    out["is_green_k"] = (close < open_).astype("int8")
    out["is_flat_k"] = (close == open_).astype("int8")

    if not all(c in out.columns for c in ["kdj_k", "kdj_d", "kdj_j"]):
        high_9 = high.groupby(out["symbol"], sort=False).transform(lambda s: s.rolling(9, min_periods=3).max())
        low_9 = low.groupby(out["symbol"], sort=False).transform(lambda s: s.rolling(9, min_periods=3).min())
        rsv = ((close - low_9) / (high_9 - low_9).replace(0, np.nan) * 100.0).clip(lower=0, upper=100)
        out["kdj_k"] = rsv.groupby(out["symbol"], sort=False).transform(lambda s: _tdx_sma(s, 3, 1))
        out["kdj_d"] = out["kdj_k"].groupby(out["symbol"], sort=False).transform(lambda s: _tdx_sma(s, 3, 1))
        out["kdj_j"] = 3.0 * _to_num(out["kdj_k"]) - 2.0 * _to_num(out["kdj_d"])
    else:
        for col in ["kdj_k", "kdj_d", "kdj_j"]:
            out[col] = _to_num(out[col])

    return out


def _add_forward_fields(part: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    base_out = part.copy(deep=False)
    group = group.sort_values("date", kind="stable").reset_index(drop=True)

    ref = group.loc[:, ["date", "open", "high", "low", "close"]].copy()
    ref["date"] = pd.to_datetime(ref["date"], errors="coerce").dt.normalize()
    ref = ref.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    future_cols: dict[str, pd.Series] = {
        "date": ref["date"],
    }

    for horizon in range(1, FORWARD_HORIZON_MAX + 1):
        future_cols[f"t{horizon}_date"] = ref["date"].shift(-horizon)
        future_cols[f"t{horizon}_open"] = _to_num(ref["open"]).shift(-horizon)
        future_cols[f"t{horizon}_high"] = _to_num(ref["high"]).shift(-horizon)
        future_cols[f"t{horizon}_low"] = _to_num(ref["low"]).shift(-horizon)
        future_cols[f"t{horizon}_close"] = _to_num(ref["close"]).shift(-horizon)

    lookup = pd.DataFrame(future_cols)

    out = base_out.merge(lookup, on="date", how="left", validate="many_to_one")

    buy_price = _to_num(out["t1_open"])
    valid_buy = buy_price.notna() & (buy_price > 0)

    forward_cols: dict[str, pd.Series] = {}

    for horizon in range(1, FORWARD_HORIZON_MAX + 1):
        sell_close = _to_num(out[f"t{horizon}_close"])
        valid = valid_buy & sell_close.notna()

        ret = pd.Series(np.nan, index=out.index, dtype="float64")
        ret.loc[valid] = (sell_close.loc[valid] / buy_price.loc[valid] - 1.0) * 100.0
        forward_cols[f"fwd_return_pct_T{horizon}"] = ret

        up = pd.Series(pd.NA, index=out.index, dtype="boolean")
        up.loc[valid] = (sell_close.loc[valid] > buy_price.loc[valid]).to_numpy(dtype=bool)
        forward_cols[f"fwd_up_T{horizon}"] = up

    status = pd.Series("ok", index=out.index, dtype="object")
    status.loc[out["t1_open"].isna()] = "missing_t1"
    forward_cols["forward_data_status"] = status

    out = pd.concat([out, pd.DataFrame(forward_cols, index=out.index)], axis=1)

    # One final copy de-fragments the per-symbol result.
    return out.copy()


def _build_symbol_b2(
    group: pd.DataFrame,
    b1_dates: set[pd.Timestamp],
    max_days_after_b1: int,
    return_threshold_pct: float,
    min_volume_ratio_prev1: float,
    j_max: float,
    max_upper_shadow_pct: float | None,
) -> pd.DataFrame:
    g = group.sort_values("date", kind="stable").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    if g.empty or not b1_dates:
        return pd.DataFrame()

    date_to_pos = {d: i for i, d in enumerate(g["date"])}
    b1_pos = sorted(date_to_pos[d] for d in b1_dates if d in date_to_pos)
    if not b1_pos:
        return pd.DataFrame()

    n = len(g)
    marker = pd.Series(np.nan, index=range(n), dtype="float64")
    for pos in b1_pos:
        marker.iloc[pos] = float(pos)

    last_b1_pos = marker.ffill().shift(1)
    lag = pd.Series(np.arange(n), index=range(n), dtype="float64") - last_b1_pos

    daily_return = _to_num(g["daily_return_pct"])
    volume_ratio_prev1 = _to_num(g["volume_ratio_prev1"])
    kdj_j = _to_num(g["kdj_j"])
    upper_shadow = _to_num(g["upper_shadow_pct"])

    mask = (
        lag.ge(1)
        & lag.le(int(max_days_after_b1))
        & daily_return.gt(float(return_threshold_pct))
        & volume_ratio_prev1.gt(float(min_volume_ratio_prev1))
        & kdj_j.lt(float(j_max))
    )

    if max_upper_shadow_pct is not None:
        mask = mask & upper_shadow.le(float(max_upper_shadow_pct))

    selected = g.loc[mask].copy()
    if selected.empty:
        return selected

    selected_idx = selected.index.to_numpy()
    b1_idx_for_row = last_b1_pos.loc[selected_idx].astype("int64").to_numpy()

    selected["selection_strategy"] = STRATEGY_NAME
    selected["selected"] = 1
    selected["b1_source_strategy"] = "b1_stage_low_select_strategy_v1"
    selected["b1_date"] = g.loc[b1_idx_for_row, "date"].to_numpy()
    selected["b1_lag_trade_days"] = lag.loc[selected_idx].astype("int64").to_numpy()

    selected["b2_daily_return_gt_threshold"] = 1
    selected["b2_volume_gt_prev1"] = 1
    selected["b2_j_below_max"] = 1
    selected["b2_no_long_upper_shadow"] = (upper_shadow.loc[selected_idx] <= 2.0).astype("int8").to_numpy()
    selected["b2_close_position_pct"] = _safe_div(
        _to_num(selected["close"]) - _to_num(selected["low"]),
        _to_num(selected["high"]) - _to_num(selected["low"]),
    ) * 100.0
    selected["b2_near_limit_up"] = (daily_return.loc[selected_idx] >= 9.0).astype("int8").to_numpy()
    selected["b2_mid_large_positive"] = (daily_return.loc[selected_idx] >= 6.0).astype("int8").to_numpy()

    selected = _add_forward_fields(selected, g)
    return selected


def _symbol_from_market_path(path: Path) -> str:
    m = re.search(r"(\d{6})", path.stem)
    return m.group(1) if m else path.stem


def build_b2_pool(
    b1_pool_path: Path,
    indicator_path: Path,
    output_path: Path,
    max_days_after_b1: int,
    return_threshold_pct: float,
    min_volume_ratio_prev1: float,
    j_max: float,
    max_upper_shadow_pct: float | None,
) -> pd.DataFrame:
    if not b1_pool_path.exists():
        raise FileNotFoundError(f"B1 pool not found: {b1_pool_path}")

    market_cache_dir = Path(MARKET_CACHE_DIR)
    if not market_cache_dir.exists():
        raise FileNotFoundError(f"Market cache dir not found: {market_cache_dir}")

    print("========== B2 pool build started ==========", flush=True)
    print(f"reading_b1_pool: {b1_pool_path}", flush=True)
    print(f"market_cache_dir: {market_cache_dir}", flush=True)
    print("daily_indicators.parquet is not used in this low-memory B2 build.", flush=True)

    b1 = _read_parquet_existing_columns(b1_pool_path, ["symbol", "date", "selected"])
    b1 = _normalize_symbol_date(b1, "B1 pool")

    if "selected" in b1.columns:
        selected_num = pd.to_numeric(b1["selected"], errors="coerce")
        if selected_num.notna().any():
            b1 = b1[selected_num.fillna(0).astype(int).eq(1)]

    b1 = b1.drop_duplicates(subset=["symbol", "date"], keep="last")
    symbols = sorted(b1["symbol"].dropna().astype(str).unique().tolist())

    print(f"b1_rows_after_filter: {len(b1):,}", flush=True)
    print(f"b1_symbol_count: {len(symbols):,}", flush=True)

    if not symbols:
        raise ValueError("No B1 symbols found after filtering.")

    b1_dates_by_symbol = {
        symbol: set(sub["date"].tolist())
        for symbol, sub in b1.groupby("symbol", sort=False)
    }

    print("indexing_market_cache_files...", flush=True)
    market_file_map: dict[str, Path] = {}
    for file_path in market_cache_dir.glob("*.parquet"):
        sym = _symbol_from_market_path(file_path)
        if sym:
            market_file_map[sym] = file_path

    print(f"market_cache_file_count: {len(market_file_map):,}", flush=True)

    parts: list[pd.DataFrame] = []
    missing_market_symbols = 0
    failed_symbols = 0
    total = len(symbols)

    for i, symbol in enumerate(symbols, start=1):
        file_path = market_file_map.get(symbol)

        if file_path is None:
            missing_market_symbols += 1
        else:
            try:
                raw = _read_table(file_path)
                if raw is not None and not raw.empty:
                    group = _standardize_market_df(
                        raw,
                        fallback_symbol=symbol,
                        fallback_file=file_path.name,
                    )
                    group = _normalize_symbol_date(group, f"market cache {symbol}")
                    group = _ensure_base_features(group)

                    part = _build_symbol_b2(
                        group=group,
                        b1_dates=b1_dates_by_symbol.get(symbol, set()),
                        max_days_after_b1=max_days_after_b1,
                        return_threshold_pct=return_threshold_pct,
                        min_volume_ratio_prev1=min_volume_ratio_prev1,
                        j_max=j_max,
                        max_upper_shadow_pct=max_upper_shadow_pct,
                    )

                    if not part.empty:
                        parts.append(part)
            except Exception as exc:
                failed_symbols += 1
                print(f"[WARN] failed symbol={symbol}: {type(exc).__name__}: {exc}", flush=True)

        if i % 50 == 0 or i == total:
            rows = sum(len(p) for p in parts)
            pct = i / max(total, 1) * 100.0
            print(
                f"processed_symbols={i:,}/{total:,} ({pct:.1f}%), "
                f"b2_rows={rows:,}, "
                f"missing_market_symbols={missing_market_symbols:,}, "
                f"failed_symbols={failed_symbols:,}",
                flush=True,
            )

    print("combining_parts...", flush=True)
    if parts:
        pool = pd.concat(parts, ignore_index=True, sort=False)
    else:
        pool = pd.DataFrame()

    if not pool.empty:
        front_cols = [
            "symbol", "file", "date", "selection_strategy", "selected",
            "b1_source_strategy", "b1_date", "b1_lag_trade_days",
            "open", "high", "low", "close", "volume", "amount",
            "daily_return_pct", "intraday_return_pct", "amplitude_pct",
            "upper_shadow_pct", "lower_shadow_pct", "body_pct", "body_abs_pct",
            "volume_ratio_prev1", "volume_ratio_ma5", "volume_ratio_ma10",
            "kdj_k", "kdj_d", "kdj_j",
            "b2_daily_return_gt_threshold", "b2_volume_gt_prev1", "b2_j_below_max",
            "b2_no_long_upper_shadow", "b2_close_position_pct", "b2_near_limit_up", "b2_mid_large_positive",
        ]
        ordered = [c for c in front_cols if c in pool.columns]
        rest = [c for c in pool.columns if c not in ordered]
        pool = pool.loc[:, ordered + rest]
        pool = pool.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing_output: {output_path}", flush=True)
    pool.to_parquet(output_path, index=False)

    print("========== B2 pool build completed ==========", flush=True)
    print(f"b1_pool: {b1_pool_path}", flush=True)
    print(f"output: {output_path}", flush=True)
    print(f"b1_rows: {len(b1):,}", flush=True)
    print(f"b2_rows: {len(pool):,}", flush=True)
    print(f"missing_market_symbols: {missing_market_symbols:,}", flush=True)
    print(f"failed_symbols: {failed_symbols:,}", flush=True)

    if not pool.empty:
        print(f"date_min: {pool['date'].min()}", flush=True)
        print(f"date_max: {pool['date'].max()}", flush=True)
        print(f"symbol_count: {pool['symbol'].nunique():,}", flush=True)

    return pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a B2 confirmation pool from the current B1 v1 pool.")
    parser.add_argument("--b1-pool", type=Path, default=DEFAULT_B1_POOL)
    parser.add_argument("--indicator-cache", type=Path, default=INDICATOR_CACHE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-days-after-b1", type=int, default=5)
    parser.add_argument("--return-threshold-pct", type=float, default=4.0)
    parser.add_argument("--min-volume-ratio-prev1", type=float, default=1.0)
    parser.add_argument("--j-max", type=float, default=55.0)
    parser.add_argument(
        "--max-upper-shadow-pct",
        type=float,
        default=None,
        help="Optional hard filter. Default None keeps upper shadow as factor only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_b2_pool(
        b1_pool_path=args.b1_pool,
        indicator_path=args.indicator_cache,
        output_path=args.output,
        max_days_after_b1=args.max_days_after_b1,
        return_threshold_pct=args.return_threshold_pct,
        min_volume_ratio_prev1=args.min_volume_ratio_prev1,
        j_max=args.j_max,
        max_upper_shadow_pct=args.max_upper_shadow_pct,
    )


if __name__ == "__main__":
    main()
