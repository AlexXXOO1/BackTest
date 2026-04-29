from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BacktestConfig
from core.pool_store import PoolStore


DEFAULT_OUTPUT_DIR = Path(r"C:\Users\zyf37\Desktop\Daily_selection")


def normalize_stock_code(value: object) -> str:
    """
    Normalize stock code for matching.

    Examples:
    - SZ#000538 -> 000538
    - SH#600000 -> 600000
    - 000538 -> 000538
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()
    match = re.search(r"(\d{6})", text)
    if match:
        return match.group(1)

    return text


def build_stock_name_map_from_data(data_dir: Path) -> dict[str, str]:
    """
    Build a stock_code -> stock_name mapping from txt files under data_dir.

    Matching priority:
    1. Parse stock name from filename, such as:
       - SZ#000538_云南白药.txt
       - 000538_云南白药.txt
       - 000538 云南白药.txt
    2. Parse stock name from the first few lines of the txt file if possible.

    If your data files are named only like SZ#000538.txt and the file content does not
    contain stock name, this function cannot infer the name.
    """
    name_map: dict[str, str] = {}

    if not data_dir.exists():
        return name_map

    txt_files = list(data_dir.rglob("*.txt"))

    for file_path in txt_files:
        file_name = file_path.stem.strip()
        code = normalize_stock_code(file_name)

        if not code:
            continue

        stock_name = ""

        # Try to parse name from filename.
        # Examples:
        # SZ#000538_云南白药
        # 000538_云南白药
        # 000538 云南白药
        name_part = re.sub(r"^(SH|SZ|BJ)?#?\d{6}", "", file_name, flags=re.IGNORECASE)
        name_part = name_part.strip("_- #")

        if name_part:
            stock_name = name_part

        # Try to parse name from file content if filename did not provide it.
        if not stock_name:
            stock_name = try_read_stock_name_from_txt(file_path)

        if stock_name:
            name_map[code] = stock_name

    return name_map


def try_read_stock_name_from_txt(file_path: Path) -> str:
    """
    Try to read stock name from the first few lines of a txt file.

    This supports common text patterns such as:
    - 股票名称: 云南白药
    - 名称: 云南白药
    - 股票名: 云南白药

    If your txt export does not include stock names, it will return an empty string.
    """
    encodings = ["utf-8-sig", "gbk", "gb2312", "utf-8"]

    for encoding in encodings:
        try:
            with file_path.open("r", encoding=encoding, errors="ignore") as f:
                lines = [next(f, "") for _ in range(10)]
            text = "\n".join(lines)

            patterns = [
                r"股票名称[:：]\s*([^\s,，;；]+)",
                r"股票名[:：]\s*([^\s,，;；]+)",
                r"名称[:：]\s*([^\s,，;；]+)",
            ]

            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1).strip()

        except Exception:
            continue

    return ""


def insert_stock_name_column(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """
    Keep all existing columns unchanged and insert a new column named 股票名
    after the stock code column.

    Priority:
    1. Use existing name column if available.
    2. Use mapping from data txt files.
    3. Empty string if name cannot be found.
    """
    result = df.copy()

    if "股票名" in result.columns:
        result = result.drop(columns=["股票名"])

    code_col = None
    for candidate in ["code", "symbol"]:
        if candidate in result.columns:
            code_col = candidate
            break

    if code_col is None:
        raise KeyError("Pool file does not contain a stock code column. Expected 'code' or 'symbol'.")

    name_map = build_stock_name_map_from_data(data_dir)

    def get_stock_name(row: pd.Series) -> str:
        if "name" in row.index and pd.notna(row["name"]) and str(row["name"]).strip():
            return str(row["name"]).strip()

        code = normalize_stock_code(row[code_col])
        return name_map.get(code, "")

    stock_names = result.apply(get_stock_name, axis=1)

    insert_pos = result.columns.get_loc(code_col) + 1
    result.insert(insert_pos, "股票名", stock_names)

    return result


def main() -> None:
    default = BacktestConfig()

    parser = argparse.ArgumentParser(description="Preview and export a unified pool parquet file.")
    parser.add_argument("--date", default=None, help="Example: 2026-04-24")
    parser.add_argument("--strategy", default=default.selection_strategy)
    parser.add_argument("--pools-dir", type=Path, default=default.pools_dir)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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

    df = insert_stock_name_column(df, args.data_dir)

    print("\nPool file:", pool_path)
    print("Rows:", len(df))

    print("\nPreview:")
    print(df.head(args.limit).to_string(index=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    date_part = args.date if args.date else "all"
    output_name = f"{args.strategy}_{date_part}.csv"
    out_path = args.output_dir / output_name

    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\nExported:", out_path)


if __name__ == "__main__":
    main()