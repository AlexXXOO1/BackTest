from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_store import MarketDataStore
from indicator_store import IndicatorStore


def parse_args():
    parser = argparse.ArgumentParser(description="Build reusable daily indicator cache.")
    parser.add_argument("--txt-dir", type=Path, default=Path("data"))
    parser.add_argument("--market-cache-dir", type=Path, default=Path("data/market_cache/daily_bars_by_symbol"))
    parser.add_argument("--indicator-cache-path", type=Path, default=Path("data/indicator_cache/daily_indicators.parquet"))
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--n1", type=int, default=4)
    parser.add_argument("--n2", type=int, default=6)
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--lookback-days", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    market_store = MarketDataStore(args.txt_dir, args.market_cache_dir)
    if not market_store.list_cached_symbols():
        print("Market cache is empty. Importing TXT files first...")
        market_store.import_txt_files(end_date=args.end_date)
    indicator_store = IndicatorStore(args.indicator_cache_path)
    df = indicator_store.build(
        market_store=market_store,
        n1=args.n1,
        n2=args.n2,
        start_date=args.start_date,
        end_date=args.end_date,
        incremental=args.incremental,
        lookback_days=args.lookback_days,
    )
    print("========== Indicator cache completed ==========")
    print(f"Rows: {len(df)}")
    print(f"Path: {indicator_store.indicator_cache_path}")


if __name__ == "__main__":
    main()
