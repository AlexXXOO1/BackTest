from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BacktestConfig
from core.pool_store import PoolStore


# ============================================================
# Manual print config
# 以后想控制 terminal 打印内容，只改这里
# CSV 导出不受这里影响，CSV 永远导出全量 df
# ============================================================

PRINT_COLUMNS = [
    #"股票代码",
    "股票名",
    #"score",
    #"score_pct",
    #"score_rank_key",
]

# None = terminal 打印全部行
# 50 = terminal 只打印前 50 行
PRINT_LIMIT = None

# False = terminal 不显示 pandas 行号
# True = terminal 显示 pandas 行号
SHOW_INDEX = False


# ============================================================
# Default paths
# ============================================================

DEFAULT_DATA_DIR = Path(r"C:\Users\zyf37\Desktop\BackTest Data\data")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\zyf37\Desktop\Daily_selection")


# ============================================================
# Stock code / stock name helpers
# ============================================================

def normalize_stock_code(value: object) -> str:
    """
    Normalize stock code for display and matching.

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

    Supported examples:
    1. 股票名称: 云南白药
    2. 股票名: 云南白药
    3. 名称: 云南白药
    4. 600000 浦发银行 日线 前复权
    5. SH#600000 浦发银行 日线 前复权
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


def build_stock_name_map_from_data(data_dir: Path, debug: bool = False) -> dict[str, str]:
    """
    Build stock_code -> stock_name mapping from txt files under data_dir.

    Priority:
    1. Parse stock name from filename.
    2. Parse stock name from txt content.
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


def add_stock_code_and_name_columns(
    df: pd.DataFrame,
    data_dir: Path,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Add normalized 股票代码 and 股票名 columns.

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


# ============================================================
# Score helpers
# ============================================================

def ensure_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure score display columns exist.

    If a score column is missing, create it as NaN
    so terminal display will not crash.
    """
    result = df.copy()

    for col in ["score", "score_pct", "score_rank_key"]:
        if col not in result.columns:
            result[col] = pd.NA

    for col in ["score", "score_pct", "score_rank_key"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result


def sort_by_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort selected pool by score.

    Priority:
    1. score_pct high to low
    2. score_rank_key high to low
    3. score high to low
    4. 股票代码 low to high
    """
    result = df.copy()

    sort_cols = []
    ascending = []

    if "score_pct" in result.columns:
        sort_cols.append("score_pct")
        ascending.append(False)

    if "score_rank_key" in result.columns:
        sort_cols.append("score_rank_key")
        ascending.append(False)

    if "score" in result.columns:
        sort_cols.append("score")
        ascending.append(False)

    if "股票代码" in result.columns:
        sort_cols.append("股票代码")
        ascending.append(True)

    if sort_cols:
        result = result.sort_values(
            by=sort_cols,
            ascending=ascending,
            na_position="last",
        ).reset_index(drop=True)

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


# ============================================================
# Terminal table formatter
# 解决中文错位和科学计数法问题
# ============================================================

def display_width(text: object) -> int:
    """
    Calculate display width for terminal alignment.

    Chinese characters count as width 2.
    English letters / digits count as width 1.
    """
    if pd.isna(text):
        s = ""
    else:
        s = str(text)

    width = 0

    for ch in s:
        if unicodedata.east_asian_width(ch) in ("F", "W"):
            width += 2
        else:
            width += 1

    return width


def pad_display(text: object, width: int, align: str = "left") -> str:
    """
    Pad text according to terminal display width.
    """
    if pd.isna(text):
        s = ""
    else:
        s = str(text)

    current_width = display_width(s)
    pad_len = max(width - current_width, 0)

    if align == "right":
        return " " * pad_len + s

    return s + " " * pad_len


def format_number_for_terminal(value: object, col: str) -> str:
    """
    Format numeric columns for clean terminal display.
    """
    if pd.isna(value):
        return ""

    try:
        number = float(value)
    except Exception:
        return str(value)

    if col == "score":
        return f"{number:.1f}"

    if col == "score_pct":
        return f"{number:.2f}"

    if col == "score_rank_key":
        # 不用科学计数法，直接显示整数
        return f"{number:.0f}"

    return str(value)


def format_display_table(df: pd.DataFrame) -> str:
    """
    Format dataframe as an aligned terminal table.

    Handles:
    1. Chinese column names.
    2. Chinese stock names.
    3. score_rank_key scientific notation.
    """
    if df.empty:
        return "No selected stocks."

    display = df.copy()

    for col in display.columns:
        if col in ["score", "score_pct", "score_rank_key"]:
            display[col] = display[col].apply(lambda x: format_number_for_terminal(x, col))
        else:
            display[col] = display[col].fillna("").astype(str)

    right_align_cols = {
        "score",
        "score_pct",
        "score_rank_key",
    }

    col_widths = {}

    for col in display.columns:
        max_data_width = display[col].map(display_width).max()
        header_width = display_width(col)
        col_widths[col] = max(max_data_width, header_width)

    header_parts = []

    for col in display.columns:
        align = "right" if col in right_align_cols else "left"
        header_parts.append(pad_display(col, col_widths[col], align=align))

    header = "  ".join(header_parts)

    separator_parts = []

    for col in display.columns:
        separator_parts.append("-" * col_widths[col])

    separator = "  ".join(separator_parts)

    rows = []

    for _, row in display.iterrows():
        row_parts = []

        for col in display.columns:
            align = "right" if col in right_align_cols else "left"
            row_parts.append(pad_display(row[col], col_widths[col], align=align))

        rows.append("  ".join(row_parts))

    return "\n".join([header, separator] + rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    default = BacktestConfig()

    parser = argparse.ArgumentParser(
        description="Preview selected stocks from a unified pool parquet file."
    )

    parser.add_argument(
        "--date",
        default=None,
        help="Example: 2026-04-30",
    )

    parser.add_argument(
        "--strategy",
        default=default.selection_strategy,
    )

    parser.add_argument(
        "--pools-dir",
        type=Path,
        default=default.pools_dir,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="TDX txt data directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Only print in terminal, do not export CSV.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information for stock name parsing.",
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

    df = add_stock_code_and_name_columns(
        df=df,
        data_dir=args.data_dir,
        debug=args.debug,
    )

    df = ensure_score_columns(df)
    df = sort_by_score(df)

    display_df = build_display_df(df)
    export_df = df.copy()

    print("\n" + "=" * 90)
    print("Pool file:", pool_path)
    print("Data dir:", args.data_dir)
    print("Strategy:", args.strategy)
    print("Date:", args.date if args.date else "all")
    print("Rows total:", len(df))
    print("Rows printed:", len(display_df))
    print("Terminal print columns:", list(display_df.columns))
    print("CSV export columns:", len(export_df.columns))
    print("=" * 90)

    print("\nSelected stocks:")
    print(format_display_table(display_df))

    if not args.no_export:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        date_part = args.date if args.date else "all"
        output_name = f"{args.strategy}_{date_part}.csv"
        out_path = args.output_dir / output_name

        export_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print("\nExported full CSV:", out_path)


if __name__ == "__main__":
    main()