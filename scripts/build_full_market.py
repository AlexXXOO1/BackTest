from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

DEFAULT_DATA_ROOT = Path(r"C:\Users\zyf37\Desktop\BackTest_Data")
DEFAULT_MARKET_CACHE_DIR = DEFAULT_DATA_ROOT / "market_cache" / "daily_bars_by_symbol"
DEFAULT_OUTPUT_PATH = DEFAULT_DATA_ROOT / "pools" / "full_market_pool.parquet"

OUTPUT_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]

COLUMN_ALIASES = {
    "date": ("date", "trade_date", "datetime", "日期"),
    "open": ("open", "open_price", "开盘"),
    "high": ("high", "high_price", "最高"),
    "low": ("low", "low_price", "最低"),
    "close": ("close", "close_price", "收盘"),
    "volume": ("volume", "vol", "成交量"),
    "amount": ("amount", "turnover", "成交额"),
}


def normalize_column_name(value: object) -> str:
    return str(value).strip().lower()


def resolve_column_map(columns: Iterable[object]) -> Dict[str, object]:
    normalized_columns = {normalize_column_name(col): col for col in columns}
    resolved_columns: Dict[str, object] = {}

    for output_col, aliases in COLUMN_ALIASES.items():
        source_col = None

        for alias in aliases:
            normalized_alias = normalize_column_name(alias)
            if normalized_alias in normalized_columns:
                source_col = normalized_columns[normalized_alias]
                break

        if source_col is None:
            raise ValueError(
                f"Missing required source column for {output_col}. "
                f"accepted_aliases={aliases}, available_columns={list(columns)}"
            )

        resolved_columns[output_col] = source_col

    return resolved_columns


def parse_date_series(series: pd.Series) -> pd.Series:
    text_series = series.astype("string").str.strip()
    compact_series = text_series.str.replace(r"\D", "", regex=True)

    compact_dates = pd.to_datetime(
        compact_series.where(compact_series.str.len() == 8),
        format="%Y%m%d",
        errors="coerce",
    )
    generic_dates = pd.to_datetime(text_series, errors="coerce")

    return generic_dates.fillna(compact_dates)


def parse_numeric_series(series: pd.Series) -> pd.Series:
    text_series = series.astype("string").str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(text_series, errors="coerce")


def normalize_frame(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    column_map = resolve_column_map(df.columns)
    output_df = pd.DataFrame(
        {
            output_col: df[source_col]
            for output_col, source_col in column_map.items()
        }
    )

    output_df["date"] = parse_date_series(output_df["date"])

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        output_df[col] = parse_numeric_series(output_df[col])

    output_df = output_df.dropna(subset=["date"]).copy()

    if output_df.empty:
        raise ValueError(f"No valid rows after normalization: {source_path}")

    return output_df[OUTPUT_COLUMNS]


def iter_parquet_files(market_cache_dir: Path, file_glob: str) -> List[Path]:
    if not market_cache_dir.exists():
        raise FileNotFoundError(f"Market cache dir not found: {market_cache_dir}")

    files = sorted(
        path
        for path in market_cache_dir.rglob(file_glob)
        if path.is_file() and path.suffix.lower() == ".parquet"
    )

    if not files:
        raise FileNotFoundError(
            f"No parquet files found under {market_cache_dir} with glob={file_glob}"
        )

    return files


def build_full_market(
    market_cache_dir: Path,
    output_path: Path,
    file_glob: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    files = iter_parquet_files(market_cache_dir, file_glob)
    parts: List[pd.DataFrame] = []

    print(f"[INFO] input_files={len(files):,}")

    for index, path in enumerate(files, start=1):
        source_df = pd.read_parquet(path)
        normalized_df = normalize_frame(source_df, path)
        parts.append(normalized_df)

        if index % 500 == 0 or index == len(files):
            print(f"[INFO] loaded_files={index:,}/{len(files):,}")

    full_market_df = pd.concat(parts, ignore_index=True)

    if start_date:
        full_market_df = full_market_df[
            full_market_df["date"] >= pd.to_datetime(start_date)
        ]

    if end_date:
        full_market_df = full_market_df[
            full_market_df["date"] <= pd.to_datetime(end_date)
        ]

    if full_market_df.empty:
        raise RuntimeError("Full market output is empty after filtering.")

    full_market_df = (
        full_market_df[OUTPUT_COLUMNS]
        .sort_values("date")
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_market_df.to_parquet(output_path, index=False)

    return full_market_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--market-cache-dir",
        type=str,
        default=str(DEFAULT_MARKET_CACHE_DIR),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
    )
    parser.add_argument("--file-glob", type=str, default="*.parquet")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    result = build_full_market(
        market_cache_dir=Path(args.market_cache_dir),
        output_path=Path(args.output_path),
        file_glob=args.file_glob,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print("========== BUILD FULL MARKET DONE ==========")
    print(f"rows={len(result):,}")
    print(f"date_min={result['date'].min()}")
    print(f"date_max={result['date'].max()}")
    print(f"columns={list(result.columns)}")
    print(f"output_path={Path(args.output_path)}")


if __name__ == "__main__":
    main()
