from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BacktestConfig
from core.data_store import MarketDataStore
from core.indicator_store import IndicatorStore
from core.storage import read_table, write_table
from core.cache_meta import build_pool_meta, write_json_meta
from scripts.import_tdx_txt import fix_txt_encoding_inplace, print_encoding_fix_report
from scripts.selector import build_pool, get_strategy_func, list_registered_strategies, make_pool_path


def parse_args() -> argparse.Namespace:
    default = BacktestConfig()
    parser = argparse.ArgumentParser(
        description="Daily one-command update: import TDX TXT -> build indicators -> build strategy pools."
    )

    date_group = parser.add_mutually_exclusive_group()
    date_group.add_argument("--date", type=str, default=None, help="Single signal date, e.g. 2026-05-07.")
    date_group.add_argument("--start-date", type=str, default=None, help="Start date for rebuild/backfill.")
    parser.add_argument("--end-date", type=str, default=None, help="End date for rebuild/backfill.")

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["renko_chart_select_strategy_v4"],
        help="Selection strategies to build. Default: renko_chart_select_strategy_v4",
    )
    parser.add_argument("--list-strategies", action="store_true")

    parser.add_argument("--txt-dir", type=Path, default=default.txt_dir)
    parser.add_argument("--market-cache-dir", type=Path, default=default.market_cache_dir)
    parser.add_argument("--indicator-cache-path", type=Path, default=default.indicator_cache_path)
    parser.add_argument("--pools-dir", type=Path, default=default.pools_dir)

    parser.add_argument("--n1", type=int, default=default.n1)
    parser.add_argument("--n2", type=int, default=default.n2)
    parser.add_argument("--lookback-days", type=int, default=250)

    parser.add_argument("--skip-import", action="store_true")
    parser.add_argument("--skip-encoding-fix", action="store_true")
    parser.add_argument("--skip-indicators", action="store_true")
    parser.add_argument("--skip-pools", action="store_true")

    parser.add_argument("--rebuild-indicators", action="store_true")
    parser.add_argument("--rebuild-pools", action="store_true")
    parser.add_argument("--full-rebuild", action="store_true")
    parser.add_argument("--debug-summary", action="store_true")

    return parser.parse_args()


def resolve_date_range(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp]:
    if args.date:
        d = pd.to_datetime(args.date, errors="raise").normalize()
        return d, d

    if not args.start_date or not args.end_date:
        raise ValueError("Use --date YYYY-MM-DD, or use both --start-date and --end-date.")

    start = pd.to_datetime(args.start_date, errors="raise").normalize()
    end = pd.to_datetime(args.end_date, errors="raise").normalize()
    if end < start:
        raise ValueError(f"end_date < start_date: {end} < {start}")
    return start, end


def print_section(title: str) -> None:
    print()
    print("=" * 10, title, "=" * 10)


def write_pool_replace_range(
    *,
    pool: pd.DataFrame,
    pool_path: Path,
    strategy_name: str,
    strategy_func,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    n1: int,
    n2: int,
    market_cache_dir: Path,
    indicator_cache_path: Path,
    pools_dir: Path,
) -> pd.DataFrame:
    """Replace exactly the requested date range inside one strategy pool file.

    This is safer than overwriting the whole pool when you only rebuild one day,
    and safer than append-only when a previously selected stock is no longer selected.
    """
    pool_path.parent.mkdir(parents=True, exist_ok=True)

    old = read_table(pool_path) if pool_path.exists() else pd.DataFrame()
    if not old.empty and "date" in old.columns:
        old = old.copy()
        old_dates = pd.to_datetime(old["date"], errors="coerce").dt.normalize()
        old = old[(old_dates < start_date) | (old_dates > end_date)]

    combined = pd.concat([old, pool], ignore_index=True, sort=False) if not pool.empty else old

    if not combined.empty:
        if "date" in combined.columns:
            combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.normalize()
        dedup_cols = [c for c in ["date", "code", "symbol", "selection_strategy"] if c in combined.columns]
        # Prefer date+code+strategy when code exists; fall back to symbol for older pools.
        if "date" in dedup_cols and "selection_strategy" in dedup_cols:
            if "code" in combined.columns:
                dedup_cols = ["date", "code", "selection_strategy"]
            elif "symbol" in combined.columns:
                dedup_cols = ["date", "symbol", "selection_strategy"]
        if dedup_cols:
            combined = combined.drop_duplicates(dedup_cols, keep="last")

        sort_cols = [c for c in ["date", "score_pct", "score_rank_key", "code", "symbol"] if c in combined.columns]
        if sort_cols:
            ascending = []
            for col in sort_cols:
                ascending.append(False if col in {"score_pct", "score_rank_key"} else True)
            combined = combined.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        else:
            combined = combined.reset_index(drop=True)

    write_table(combined, pool_path)

    meta = build_pool_meta(
        pool=combined,
        strategy_name=strategy_name,
        strategy_func=strategy_func,
        start_date=start_date,
        end_date=end_date,
        n1=n1,
        n2=n2,
        market_cache_dir=market_cache_dir,
        indicator_cache_path=indicator_cache_path,
        pools_dir=pools_dir,
    )
    meta_path = write_json_meta(meta, pool_path)

    print(f"[INFO] Pool saved: {pool_path}")
    print(f"[INFO] Pool meta saved: {meta_path}")
    print(f"[INFO] Rows: old_kept={len(old):,}, rebuilt_range={len(pool):,}, final={len(combined):,}")
    return combined


def main() -> None:
    args = parse_args()

    if args.list_strategies:
        print("========== Registered selection strategies ==========")
        for name in list_registered_strategies():
            print(f"  - {name}")
        return

    start_date, end_date = resolve_date_range(args)

    print_section("DAILY UPDATE CONFIG")
    print(f"date range: {start_date.date()} -> {end_date.date()}")
    print(f"strategies: {', '.join(args.strategies)}")
    print(f"txt_dir: {args.txt_dir}")
    print(f"market_cache_dir: {args.market_cache_dir}")
    print(f"indicator_cache_path: {args.indicator_cache_path}")
    print(f"pools_dir: {args.pools_dir}")
    print(f"n1/n2: {args.n1}/{args.n2}")

    market_store = MarketDataStore(args.txt_dir, args.market_cache_dir)

    if not args.skip_import:
        print_section("STEP 1/3 IMPORT TDX TXT")
        if not args.skip_encoding_fix:
            encoding_report = fix_txt_encoding_inplace(args.txt_dir)
            print_encoding_fix_report(encoding_report)

        import_report = market_store.import_txt_files(
            start_date=start_date,
            end_date=end_date,
            overwrite=bool(args.full_rebuild),
        )
        print("========== Import completed ==========")
        for key, value in import_report.items():
            if key != "failures":
                print(f"{key}: {value}")
        if import_report.get("failures"):
            print("Failures:")
            for item in import_report["failures"][:20]:
                print(f"  - {item['file']}: {item['error']}")
    else:
        print_section("STEP 1/3 IMPORT TDX TXT SKIPPED")

    if not args.skip_indicators:
        print_section("STEP 2/3 BUILD INDICATORS")
        indicator_store = IndicatorStore(args.indicator_cache_path)
        rebuild_indicators = bool(args.full_rebuild or args.rebuild_indicators)

        if rebuild_indicators:
            indicator_start = start_date
            incremental = False
        else:
            # Daily mode: recalculate a rolling window from existing max date.
            # This keeps EMA/KDJ/rolling indicators stable without forcing full rebuild.
            indicator_start = None
            incremental = True

        indicator_df = indicator_store.build(
            market_store=market_store,
            n1=args.n1,
            n2=args.n2,
            start_date=indicator_start,
            end_date=end_date,
            incremental=incremental,
            lookback_days=args.lookback_days,
        )
        print("========== Indicator cache completed ==========")
        print(f"Rows: {len(indicator_df):,}")
        print(f"Path: {args.indicator_cache_path}")
    else:
        print_section("STEP 2/3 BUILD INDICATORS SKIPPED")

    if not args.skip_pools:
        print_section("STEP 3/3 BUILD POOLS")
        for strategy_name in args.strategies:
            print()
            print(f"----- Strategy: {strategy_name} -----")
            strategy_func = get_strategy_func(strategy_name)
            pool = build_pool(
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                market_cache_dir=args.market_cache_dir,
                indicator_cache_path=args.indicator_cache_path,
                n1=args.n1,
                n2=args.n2,
                debug_summary=args.debug_summary,
            )
            pool_path = make_pool_path(args.pools_dir, strategy_name)
            write_pool_replace_range(
                pool=pool,
                pool_path=pool_path,
                strategy_name=strategy_name,
                strategy_func=strategy_func,
                start_date=start_date,
                end_date=end_date,
                n1=args.n1,
                n2=args.n2,
                market_cache_dir=args.market_cache_dir,
                indicator_cache_path=args.indicator_cache_path,
                pools_dir=args.pools_dir,
            )
    else:
        print_section("STEP 3/3 BUILD POOLS SKIPPED")

    print_section("DAILY UPDATE DONE")


if __name__ == "__main__":
    main()
