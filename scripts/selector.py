from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BacktestConfig
from engine import run_selector
from selection_strategies import SELECTION_STRATEGY_REGISTRY


def parse_args():
    default = BacktestConfig()
    parser = argparse.ArgumentParser(description="Unified pool builder entry point.")
    parser.add_argument("--txt-dir", type=Path, default=default.txt_dir)
    parser.add_argument("--market-cache-dir", type=Path, default=default.market_cache_dir)
    parser.add_argument("--indicator-cache-path", type=Path, default=default.indicator_cache_path)
    parser.add_argument("--pools-dir", type=Path, default=default.pools_dir)
    parser.add_argument("--date", type=str, default=None, help="single selection date, for example 2026-03-05")
    parser.add_argument("--start-date", type=str, default=None, help="date")
    parser.add_argument("--end-date", type=str, default=None, help="date")
    parser.add_argument("--strategy", type=str, default=default.selection_strategy, choices=sorted(SELECTION_STRATEGY_REGISTRY))
    parser.add_argument("--n1", type=int, default=default.n1)
    parser.add_argument("--n2", type=int, default=default.n2)
    parser.add_argument("--max-workers", type=int, default=default.max_workers)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--debug-summary",
        action="store_true",
        help="Print path, cache, indicator, and final pool diagnostics after building the pool.",
    )
    return parser.parse_args()


def _count_files(path: Path, pattern: str) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for _ in path.rglob(pattern))


def _safe_read_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"[DEBUG] Failed to read parquet: {path}")
        print(f"[DEBUG] Error: {type(exc).__name__}: {exc}")
        return None


def _as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.fillna(0).astype(bool)


def _print_df_summary(df: pd.DataFrame, name: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> None:
    print(f"\n[DEBUG] {name}")
    print(f"[DEBUG] rows: {len(df):,}")

    if "date" in df.columns:
        date_col = pd.to_datetime(df["date"], errors="coerce")
        print(f"[DEBUG] date range: {date_col.min()} -> {date_col.max()}")
        in_range = df.loc[(date_col >= start_date) & (date_col <= end_date)].copy()
        print(f"[DEBUG] rows in requested range: {len(in_range):,}")
    else:
        in_range = df
        print("[DEBUG] date column: missing")

    symbol_col = None
    for col in ("symbol", "code", "ts_code", "file", "filename"):
        if col in df.columns:
            symbol_col = col
            break

    if symbol_col:
        print(f"[DEBUG] symbol column: {symbol_col}")
        print(f"[DEBUG] unique symbols total: {df[symbol_col].nunique():,}")
        print(f"[DEBUG] unique symbols in range: {in_range[symbol_col].nunique():,}")
    else:
        print("[DEBUG] symbol column: not found")

    for col in ("hard_brick_turn_strong", "selected", "selected_score_base"):
        if col in df.columns:
            total_count = int(_as_bool_series(df[col]).sum())
            range_count = int(_as_bool_series(in_range[col]).sum()) if col in in_range.columns else 0
            print(f"[DEBUG] {col}=true total: {total_count:,}")
            print(f"[DEBUG] {col}=true in range: {range_count:,}")
        else:
            print(f"[DEBUG] {col}: missing")

    if "date" in in_range.columns:
        date_col = pd.to_datetime(in_range["date"], errors="coerce")
        daily_counts = date_col.dt.strftime("%Y-%m-%d").value_counts().sort_index()
        if len(daily_counts) > 0:
            print("[DEBUG] last 20 daily row counts:")
            for day, count in daily_counts.tail(20).items():
                print(f"[DEBUG]   {day}: {count:,}")


def print_debug_summary(config: BacktestConfig) -> None:
    print("\n========== SELECTOR DEBUG SUMMARY ==========")
    print(f"[DEBUG] strategy: {config.selection_strategy}")
    print(f"[DEBUG] date range: {config.start_date} -> {config.end_date}")
    print(f"[DEBUG] n1/n2: {config.n1}/{config.n2}")
    print(f"[DEBUG] max_workers: {config.max_workers}")

    paths = {
        "txt_dir": Path(config.txt_dir),
        "market_cache_dir": Path(config.market_cache_dir),
        "indicator_cache_path": Path(config.indicator_cache_path),
        "pools_dir": Path(config.pools_dir),
    }
    for name, path in paths.items():
        print(f"[DEBUG] {name}: {path}")
        print(f"[DEBUG] {name} exists: {path.exists()}")

    txt_dir = Path(config.txt_dir)
    market_cache_dir = Path(config.market_cache_dir)
    indicator_cache_path = Path(config.indicator_cache_path)
    pools_dir = Path(config.pools_dir)
    pool_path = pools_dir / f"{config.selection_strategy}_pool.parquet"

    print(f"[DEBUG] txt .txt count: {_count_files(txt_dir, '*.txt'):,}")
    print(f"[DEBUG] market cache .parquet count: {_count_files(market_cache_dir, '*.parquet'):,}")
    print(f"[DEBUG] expected pool path: {pool_path}")
    print(f"[DEBUG] expected pool exists: {pool_path.exists()}")

    indicator_df = _safe_read_parquet(indicator_cache_path)
    if indicator_df is not None:
        _print_df_summary(indicator_df, "indicator cache summary", config.start_date, config.end_date)

    pool_df = _safe_read_parquet(pool_path)
    if pool_df is not None:
        _print_df_summary(pool_df, "final pool summary", config.start_date, config.end_date)

    print("========== END SELECTOR DEBUG SUMMARY ==========\n")


def main() -> None:
    args = parse_args()
    if args.date:
        start_date = end_date = pd.Timestamp(args.date)
    else:
        start_date = pd.Timestamp(args.start_date) if args.start_date else BacktestConfig().start_date
        end_date = pd.Timestamp(args.end_date) if args.end_date else BacktestConfig().end_date

    config = replace(
        BacktestConfig(),
        txt_dir=args.txt_dir,
        market_cache_dir=args.market_cache_dir,
        indicator_cache_path=args.indicator_cache_path,
        pools_dir=args.pools_dir,
        start_date=start_date,
        end_date=end_date,
        selection_strategy=args.strategy,
        n1=args.n1,
        n2=args.n2,
        max_workers=args.max_workers,
    )
    run_selector(config, overwrite=args.overwrite)

    if args.debug_summary:
        print_debug_summary(config)


if __name__ == "__main__":
    main()
