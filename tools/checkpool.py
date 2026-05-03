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


# ============================================================
# Manual print config
# 以后你想控制 terminal 打印内容，只改这里
# CSV 导出不受这里影响，CSV 永远导出全量 df
# ============================================================

PRINT_COLUMNS = [
    "股票代码",
    "股票名",
    "close",
    "T1开盘价",
    "T1收盘价",
    "T2开盘价",
    "T2收盘价",
    "T3开盘价",
    "T3收盘价",
    # "date",
    # "score_pct",
    # "brick_value",
    # "hard_brick_turn_strong",
    # "small_rise_long_red_brick",
    # "j",
    # "J",
    # "open",
    # "pct_chg",
]

# None = terminal 打印全部行
# 50 = terminal 只打印前 50 行
# 注意：CSV 导出不受这个影响，CSV 仍然导出全量
PRINT_LIMIT = None

# False = terminal 不显示 pandas 行号
# True = terminal 显示 pandas 行号
SHOW_INDEX = False


# ============================================================
# Default paths
# ============================================================

DEFAULT_DATA_DIR = Path(r"C:\Users\zyf37\Desktop\BackTest Data\data")
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


def clean_stock_name(value: object) -> str:
    """
    Clean parsed stock name.
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()
    text = text.strip("_- #\t\r\n")

    for bad_word in ["日线", "前复权", "后复权", "不复权"]:
        text = text.replace(bad_word, "").strip()

    return text


def try_read_stock_name_from_txt(file_path: Path) -> str:
    """
    Try to read stock name from a TDX txt file.

    Supported formats:
    1. Label format:
       - 股票名称: 云南白药
       - 股票名: 云南白药
       - 名称: 云南白药

    2. TDX first-line format:
       - 600000 浦发银行 日线 前复权
       - 000538 云南白药 日线 前复权
       - SH#600000 浦发银行 日线 前复权
       - SZ#000538 云南白药 日线 前复权
    """
    encodings = ["utf-8-sig", "gbk", "gb2312", "utf-8"]

    for encoding in encodings:
        try:
            with file_path.open("r", encoding=encoding, errors="ignore") as f:
                lines = [next(f, "") for _ in range(20)]

            lines = [line.strip() for line in lines if str(line).strip()]
            if not lines:
                continue

            text = "\n".join(lines)

            label_patterns = [
                r"股票名称[:：]\s*([^\s,，;；]+)",
                r"股票名[:：]\s*([^\s,，;；]+)",
                r"名称[:：]\s*([^\s,，;；]+)",
            ]

            for pattern in label_patterns:
                match = re.search(pattern, text)
                if match:
                    stock_name = clean_stock_name(match.group(1))
                    if stock_name:
                        return stock_name

            first_line = lines[0]

            tdx_match = re.match(
                r"^(?:SH|SZ|BJ)?#?(\d{6})\s+([^\s,，;；]+)",
                first_line,
                flags=re.IGNORECASE,
            )
            if tdx_match:
                stock_name = clean_stock_name(tdx_match.group(2))
                if stock_name:
                    return stock_name

            parts = re.split(r"\s+", first_line)
            if len(parts) >= 2 and re.search(r"\d{6}", parts[0]):
                stock_name = clean_stock_name(parts[1])
                if stock_name and stock_name not in ["日期", "开盘", "最高", "最低", "收盘"]:
                    return stock_name

        except Exception:
            continue

    return ""


def build_txt_file_map(data_dir: Path, debug: bool = False) -> dict[str, Path]:
    """
    Build stock_code -> txt file path mapping.

    Examples:
    - SH#600000.txt -> 600000
    - SZ#000538.txt -> 000538
    """
    file_map: dict[str, Path] = {}

    if not data_dir.exists():
        if debug:
            print(f"[DEBUG] data_dir does not exist: {data_dir}")
        return file_map

    txt_files = list(data_dir.rglob("*.txt"))

    for file_path in txt_files:
        code = normalize_stock_code(file_path.stem)
        if code:
            file_map[code] = file_path

    if debug:
        print(f"[DEBUG] txt file map size: {len(file_map)}")

    return file_map


def build_stock_name_map_from_data(data_dir: Path, debug: bool = False) -> dict[str, str]:
    """
    Build stock_code -> stock_name mapping from txt files under data_dir.

    Priority:
    1. Parse stock name from filename.
       Examples:
       - SZ#000538_云南白药.txt
       - 000538_云南白药.txt
       - 000538 云南白药.txt

    2. Parse stock name from txt content.
       Example:
       - 600000 浦发银行 日线 前复权
    """
    name_map: dict[str, str] = {}

    if not data_dir.exists():
        if debug:
            print(f"[DEBUG] data_dir does not exist: {data_dir}")
        return name_map

    txt_files = list(data_dir.rglob("*.txt"))

    if debug:
        print(f"[DEBUG] data_dir: {data_dir}")
        print(f"[DEBUG] txt files found: {len(txt_files)}")

    for file_path in txt_files:
        file_stem = file_path.stem.strip()
        code = normalize_stock_code(file_stem)

        if not code:
            continue

        stock_name = ""

        name_part = re.sub(
            r"^(SH|SZ|BJ)?#?\d{6}",
            "",
            file_stem,
            flags=re.IGNORECASE,
        )
        name_part = clean_stock_name(name_part)

        if name_part:
            stock_name = name_part

        if not stock_name:
            stock_name = try_read_stock_name_from_txt(file_path)

        if stock_name:
            name_map[code] = stock_name

    if debug:
        print(f"[DEBUG] stock names parsed: {len(name_map)}")
        sample_items = list(name_map.items())[:10]
        print(f"[DEBUG] sample stock names: {sample_items}")

    return name_map


def read_tdx_txt_ohlc(file_path: Path) -> pd.DataFrame:
    """
    Read one TDX txt file and return daily OHLC data.

    Expected data line:
    02/08/2021,7.57,7.78,7.51,7.67,45713350,416533728.00

    Returned columns:
    - date
    - open
    - high
    - low
    - close
    """
    encodings = ["utf-8-sig", "gbk", "gb2312", "utf-8"]

    for encoding in encodings:
        try:
            rows: list[list[str]] = []

            with file_path.open("r", encoding=encoding, errors="ignore") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue

                    # TDX date format: 02/08/2021
                    if not re.match(r"^\d{1,2}/\d{1,2}/\d{4}", line):
                        continue

                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) < 5:
                        continue

                    rows.append(parts[:5])

            if not rows:
                continue

            df = pd.DataFrame(
                rows,
                columns=["date", "open", "high", "low", "close"],
            )

            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce",
                dayfirst=True,
            ).dt.normalize()

            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["date"])
            df = df.sort_values("date").reset_index(drop=True)

            return df

        except Exception:
            continue

    return pd.DataFrame(columns=["date", "open", "high", "low", "close"])


def add_stock_code_and_name_columns(
    df: pd.DataFrame,
    data_dir: Path,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Add normalized 股票代码 and 股票名 columns.

    These two columns are used for terminal display and are also included
    in the full CSV export.

    Stock code source priority:
    1. code
    2. symbol
    3. 股票代码

    Stock name source priority:
    1. Existing 股票名 column
    2. Existing name column
    3. Existing stock_name column
    4. Mapping parsed from txt files
    5. Empty string
    """
    result = df.copy()

    code_col = None
    for candidate in ["code", "symbol", "股票代码"]:
        if candidate in result.columns:
            code_col = candidate
            break

    if code_col is None:
        raise KeyError(
            "Pool file does not contain a stock code column. "
            "Expected one of: 'code', 'symbol', or '股票代码'."
        )

    name_map = build_stock_name_map_from_data(data_dir, debug=debug)

    def get_stock_code(row: pd.Series) -> str:
        return normalize_stock_code(row[code_col])

    def get_stock_name(row: pd.Series) -> str:
        for name_col in ["股票名", "name", "stock_name"]:
            if name_col in row.index and pd.notna(row[name_col]) and str(row[name_col]).strip():
                return clean_stock_name(row[name_col])

        code = normalize_stock_code(row[code_col])
        return name_map.get(code, "")

    result["股票代码"] = result.apply(get_stock_code, axis=1)
    result["股票名"] = result.apply(get_stock_name, axis=1)

    if debug:
        missing_mask = result["股票名"].astype(str).str.strip() == ""
        missing_count = int(missing_mask.sum())

        print(f"[DEBUG] rows after date filter: {len(result)}")
        print(f"[DEBUG] missing stock name rows: {missing_count}")

        if missing_count > 0:
            missing_codes = (
                result.loc[missing_mask, "股票代码"]
                .astype(str)
                .drop_duplicates()
                .head(30)
                .tolist()
            )
            print(f"[DEBUG] first missing stock-name codes: {missing_codes}")

    return result


def add_future_open_close_columns(
    df: pd.DataFrame,
    data_dir: Path,
    debug: bool = False,
    max_future_days: int = 3,
) -> pd.DataFrame:
    """
    Add T1/T2/T3 open and close price columns.

    T0 is df['date'].
    T1/T2/T3 are the next 1/2/3 available trading dates in the corresponding TDX txt file.

    New columns:
    - T1开盘价
    - T1收盘价
    - T2开盘价
    - T2收盘价
    - T3开盘价
    - T3收盘价

    If future trading data is not available, the value remains 0.0.
    """
    result = df.copy()

    if "date" not in result.columns:
        raise KeyError("Pool file does not contain a 'date' column.")

    if "股票代码" not in result.columns:
        raise KeyError("Dataframe does not contain '股票代码'. Please add stock code column first.")

    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()

    file_map = build_txt_file_map(data_dir, debug=debug)
    ohlc_cache: dict[str, pd.DataFrame] = {}

    # Initialize all future price columns as 0.0.
    for day in range(1, max_future_days + 1):
        result[f"T{day}开盘价"] = 0.0
        result[f"T{day}收盘价"] = 0.0

    missing_file_codes: set[str] = set()
    missing_future_codes: set[str] = set()

    for idx, row in result.iterrows():
        code = normalize_stock_code(row["股票代码"])
        t0_date = row["date"]

        if not code or pd.isna(t0_date):
            continue

        txt_path = file_map.get(code)
        if txt_path is None:
            missing_file_codes.add(code)
            continue

        if code not in ohlc_cache:
            ohlc_cache[code] = read_tdx_txt_ohlc(txt_path)

        price_df = ohlc_cache[code]
        if price_df.empty:
            missing_future_codes.add(code)
            continue

        future_rows = (
            price_df[price_df["date"] > t0_date]
            .sort_values("date")
            .reset_index(drop=True)
        )

        if future_rows.empty:
            missing_future_codes.add(code)
            continue

        for day in range(1, max_future_days + 1):
            pos = day - 1

            # If T1/T2/T3 does not exist, keep 0.0.
            if pos >= len(future_rows):
                missing_future_codes.add(code)
                continue

            future_row = future_rows.iloc[pos]

            open_value = future_row["open"]
            close_value = future_row["close"]

            result.at[idx, f"T{day}开盘价"] = float(open_value) if pd.notna(open_value) else 0.0
            result.at[idx, f"T{day}收盘价"] = float(close_value) if pd.notna(close_value) else 0.0

    if debug:
        print(f"[DEBUG] future price cache loaded: {len(ohlc_cache)} stocks")
        print(f"[DEBUG] missing txt file codes count: {len(missing_file_codes)}")
        print(f"[DEBUG] missing future price codes count: {len(missing_future_codes)}")

        if missing_file_codes:
            print(f"[DEBUG] first missing txt file codes: {sorted(missing_file_codes)[:30]}")

        if missing_future_codes:
            print(f"[DEBUG] first missing future price codes: {sorted(missing_future_codes)[:30]}")

    return result


def build_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build terminal display dataframe according to manual PRINT_COLUMNS config.

    This only affects terminal printing.
    It does not affect CSV export.
    """
    available_print_columns = [col for col in PRINT_COLUMNS if col in df.columns]
    missing_print_columns = [col for col in PRINT_COLUMNS if col not in df.columns]

    if missing_print_columns:
        print("\n[WARN] These PRINT_COLUMNS do not exist in pool file:")
        for col in missing_print_columns:
            print(f"  - {col}")

        print("\n[INFO] Available columns in current pool file:")
        for col in df.columns:
            print(f"  - {col}")

    if not available_print_columns:
        raise ValueError(
            "No valid columns in PRINT_COLUMNS. "
            "Please check PRINT_COLUMNS at the top of this script."
        )

    display_df = df[available_print_columns].copy()

    if PRINT_LIMIT is not None:
        display_df = display_df.head(PRINT_LIMIT)

    return display_df


def main() -> None:
    default = BacktestConfig()

    parser = argparse.ArgumentParser(
        description="Preview selected stocks from a unified pool parquet file."
    )
    parser.add_argument("--date", default=None, help="Example: 2026-04-24")
    parser.add_argument("--strategy", default=default.selection_strategy)
    parser.add_argument("--pools-dir", type=Path, default=default.pools_dir)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="TDX txt data directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Only print in terminal, do not export CSV.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information for stock name and future price parsing.",
    )
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

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df[df["date"] == target_date].copy()

    # Add 股票代码 and 股票名.
    # These columns are used by terminal display and included in full CSV export.
    df = add_stock_code_and_name_columns(
        df=df,
        data_dir=args.data_dir,
        debug=args.debug,
    )

    # Add T1/T2/T3 open and close prices.
    # These columns are also included in full CSV export.
    df = add_future_open_close_columns(
        df=df,
        data_dir=args.data_dir,
        debug=args.debug,
        max_future_days=3,
    )

    # Terminal output only uses PRINT_COLUMNS / PRINT_LIMIT.
    display_df = build_display_df(df)

    # CSV always exports full dataframe after date filtering and enrichment.
    # So CSV includes 股票代码、股票名、T1/T2/T3开盘价、T1/T2/T3收盘价.
    export_df = df.copy()

    print("\n" + "=" * 80)
    print("Pool file:", pool_path)
    print("Data dir:", args.data_dir)
    print("Strategy:", args.strategy)
    print("Date:", args.date if args.date else "all")
    print("Rows total:", len(df))
    print("Rows printed:", len(display_df))
    print("Terminal print columns:", list(display_df.columns))
    print("CSV export columns:", len(export_df.columns))
    print("=" * 80)

    print("\nSelected stocks:")
    if display_df.empty:
        print("No selected stocks.")
    else:
        print(display_df.to_string(index=SHOW_INDEX))

    if not args.no_export:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        date_part = args.date if args.date else "all"
        output_name = f"{args.strategy}_{date_part}.csv"
        out_path = args.output_dir / output_name

        # Important:
        # Export full data, not only PRINT_COLUMNS.
        export_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print("\nExported full CSV:", out_path)


if __name__ == "__main__":
    main()