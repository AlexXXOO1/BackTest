from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BacktestConfig
from core.pool_store import PoolStore


def main() -> None:
    default = BacktestConfig()

    parser = argparse.ArgumentParser(description="Preview a unified pool parquet file.")
    parser.add_argument("--date", default=None, help="Example: 2026-04-24")
    parser.add_argument("--strategy", default=default.selection_strategy)
    parser.add_argument("--pools-dir", type=Path, default=default.pools_dir)
    parser.add_argument("--export", action="store_true", help="Export filtered result to CSV.")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    pool_store = PoolStore(args.pools_dir)
    pool_path = pool_store.pool_path(args.strategy)

    if not pool_path.exists():
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    df = pool_store.read(args.strategy)

    if args.date:
        target_date = pd.Timestamp(args.date).normalize()
        if "date" not in df.columns:
            raise KeyError("Pool file does not contain a 'date' column.")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df[df["date"] == target_date].copy()

    display_cols = [
        "date",
        "code",
        "symbol",
        "name",
        "score",
        "score_pct",
        "condition_count",
        "j_momentum_or_low",
        "small_rise_long_red_brick",
        "surge_then_shrink_pullback",
        "prior_20d_accelerated_huge_volume_bear",
        "prior_20d_shrink_limit_up",
        "long_lower_shadow_hammer",
        "limit_up_red_brick",
        "close",
        "volume",
        "amount",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]

    print("\nPool file:", pool_path)
    print("Rows:", len(df))
    print("\nPreview:")
    if existing_cols:
        print(df[existing_cols].head(args.limit).to_string(index=False))
    else:
        print(df.head(args.limit).to_string(index=False))

    if args.export:
        out_path = args.pools_dir / (f"pool_{args.date}.csv" if args.date else "pool_preview.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print("\nExported:", out_path)


if __name__ == "__main__":
    main()
