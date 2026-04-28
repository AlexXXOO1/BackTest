import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="Example: 2025-06-05")
    parser.add_argument("--export", action="store_true", help="Export filtered result to CSV")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    pool_path = Path("pools/renko_chart_select_strategy_v1_pool.parquet")

    if not pool_path.exists():
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    df = pd.read_parquet(pool_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    if args.date:
        df = df[df["date"] == args.date]

    print("Pool file:", pool_path)
    print("Rows:", len(df))
    print("Columns:")
    print(df.columns.tolist())

    display_cols = [
        "date",
        "symbol",
        "file_name",
        "score_pct",
        "selected",
        "prior_20d_accelerated_huge_volume_bear"
    ]

    existing_cols = [c for c in display_cols if c in df.columns]

    print("\nPreview:")
    print(df[existing_cols].head(args.limit).to_string(index=False))

    if args.export:
        out_path = Path("pools") / (
            f"pool_{args.date}.csv" if args.date else "pool_preview.csv"
        )
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print("\nExported:", out_path)


if __name__ == "__main__":
    main()