from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_store import MarketDataStore
from core.indicator_store import IndicatorStore
from core.pool_store import PoolStore, build_pool_from_indicators
from selection_strategies import SELECTION_STRATEGY_REGISTRY

from config import BacktestConfig
def parse_args():
    default = BacktestConfig()
    parser = argparse.ArgumentParser(description="Build a single-file selection pool from the indicator cache.")
    parser.add_argument("--txt-dir", type=Path, default=default.txt_dir)
    parser.add_argument("--market-cache-dir", type=Path, default=default.market_cache_dir)
    parser.add_argument("--indicator-cache-path", type=Path, default=default.indicator_cache_path)
    parser.add_argument("--pools-dir", type=Path, default=default.pools_dir)
    parser.add_argument("--strategy", type=str, default="renko_chart_select_strategy_v0", choices=sorted(SELECTION_STRATEGY_REGISTRY))
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--n1", type=int, default=4)
    parser.add_argument("--n2", type=int, default=6)
    parser.add_argument("--overwrite-range", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    market_store = MarketDataStore(args.txt_dir, args.market_cache_dir)
    indicator_store = IndicatorStore(args.indicator_cache_path)
    if not indicator_store.exists():
        print("Indicator cache is missing. Building it first...")
        if not market_store.list_cached_symbols():
            print("Market cache is empty. Importing TXT files first...")
            market_store.import_txt_files(end_date=args.end_date)
        indicator_store.build(market_store, n1=args.n1, n2=args.n2, end_date=args.end_date)
    indicator_df = indicator_store.read()
    pool_df = build_pool_from_indicators(indicator_df, args.strategy, args.start_date, args.end_date, n1=args.n1, n2=args.n2)
    pool_store = PoolStore(args.pools_dir)
    path = pool_store.write_replace_range(args.strategy, pool_df, args.start_date, args.end_date)
    print("========== Pool build completed ==========")
    print(f"Strategy: {args.strategy}")
    print(f"Rows: {len(pool_df)}")
    print(f"Path: {path}")


if __name__ == "__main__":
    main()
