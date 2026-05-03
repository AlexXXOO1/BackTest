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

    # IMPORTANT:
    # Do NOT use argparse choices here.
    # New strategy files should be accepted by command line first,
    # then validated against SELECTION_STRATEGY_REGISTRY after parsing.
    # This makes it easier to add new strategies without editing selector.py every time.
    parser.add_argument(
        "--strategy",
        type=str,
        default=default.selection_strategy,
        help=(
            "Selection strategy name. "
            "Example: thunder_bottom_j_strategy_v0. "
            "The strategy must be registered in selection_strategies."
        ),
    )

    parser.add_argument("--n1", type=int, default=default.n1)
    parser.add_argument("--n2", type=int, default=default.n2)
    parser.add_argument("--max-workers", type=int, default=default.max_workers)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--debug-summary",
        action="store_true",
        help="Print path, cache, indicator, and final pool diagnostics after building the pool.",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List registered selection strategies and exit.",
    )
    return parser.parse_args()


def _print_registered_strategies() -> None:
    print("\n========== REGISTERED SELECTION STRATEGIES ==========")
    if not SELECTION_STRATEGY_REGISTRY:
        print("[ERROR] No strategy found in SELECTION_STRATEGY_REGISTRY.")
    else:
        for name in sorted(SELECTION_STRATEGY_REGISTRY):
            print(f"  - {name}")
    print("====================================================\n")


def _validate_strategy_name(strategy_name: str) -> None:
    """
    Validate strategy name after argparse parsing.

    This replaces argparse choices, so command line no longer blocks new names
    before we can print a clearer error message.
    """
    if strategy_name in SELECTION_STRATEGY_REGISTRY:
        return

    print("\n[ERROR] Selection strategy is not registered.")
    print(f"[ERROR] Requested strategy: {strategy_name}")
    print("\n[ERROR] Registered strategies:")
    if SELECTION_STRATEGY_REGISTRY:
        for name in sorted(SELECTION_STRATEGY_REGISTRY):
            print(f"  - {name}")
    else:
        print("  <empty registry>")

    print("\n[FIX SUGGESTIONS]")
    print("1. Make sure your strategy file is located in:")
    print("   selection_strategies/")
    print("2. Make sure the file name matches the strategy name, for example:")
    print("   selection_strategies/thunder_bottom_j_strategy_v0.py")
    print("3. Make sure the strategy file contains:")
    print('   STRATEGY_NAME = "thunder_bottom_j_strategy_v0"')
    print("4. Make sure selection_strategies/__init__.py supports auto-discovery,")
    print("   or manually imports and registers the new strategy.")
    print("5. You can run this command to list registered strategies:")
    print("   python .\\scripts\\selector.py --list-strategies")
    print()

    raise SystemExit(2)


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


def _print_condition_count(
    df: pd.DataFrame,
    in_range: pd.DataFrame,
    col: str,
) -> None:
    if col in df.columns:
        total_count = int(_as_bool_series(df[col]).sum())
        range_count = int(_as_bool_series(in_range[col]).sum()) if col in in_range.columns else 0
        print(f"[DEBUG] {col}=true total: {total_count:,}")
        print(f"[DEBUG] {col}=true in range: {range_count:,}")
    else:
        print(f"[DEBUG] {col}: missing")


def _print_df_summary(
    df: pd.DataFrame,
    name: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> None:
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

    # Common old strategy columns + new thunder bottom strategy columns.
    condition_cols = [
        # Old renko strategy columns.
        "hard_brick_turn_strong",
        "selected_score_base",

        # Generic final column.
        "selected",

        # Thunder bottom J strategy columns.
        "absolute_bottom_event",
        "prior_abs_bottom_seen",
        "has_valid_post_abs_rebound",
        "pullback_after_abs_bottom",
        "relative_bottom_after_abs_bottom",
        "j_low_position",
        "j_near_recent_low",
        "j_turn_up",
        "j_confirm_relative_low",
        "bottom_position_ok",
        "sudden_thunder_move",
        "big_bull_volume_bar",
        "scary_key_position",
        "base_filter_ok",
    ]

    for col in condition_cols:
        _print_condition_count(df, in_range, col)

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

    if args.list_strategies:
        _print_registered_strategies()
        return

    _validate_strategy_name(args.strategy)

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
