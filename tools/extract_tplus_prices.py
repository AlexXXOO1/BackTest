from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


# ============================================================
# 你每次只需要改这里
# ============================================================

DATA_DIR = r"C:\Users\zyf37\Desktop\BackTest Data\data"

# 支持输入多个股票代码
CODES = [
    "SH#600222",
    "SH#603377",
    "SH#605033",
    "SZ#000553",
    "SZ#000869",
    "SZ#002186",
    "SH#600439",
    "SH#600503",
    "SH#600679",
    "SH#600822",
    "SH#601369",
    "SZ#000952",
    "SZ#002535",
    "SZ#002646",
    "SZ#002869",
    "SZ#000893",
]

# 这个日期是 T0
T0_DATE = "2026-04-24"

# 输出文件路径
OUTPUT_CSV = r"C:\Users\zyf37\Desktop\BackTest Data\output\tplus_prices.csv"


# ============================================================
# 下面代码一般不用改
# ============================================================

def read_text_with_fallback(path: Path) -> str:
    """
    Read TXT file with multiple encoding fallbacks.
    TongDaXin exported TXT files may use GBK/GB18030.
    """
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]

    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue

    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_code(code: str) -> str:
    """
    Normalize stock code.

    Example:
    sz#000538 -> SZ#000538
    """
    return code.strip().upper()


def find_stock_file(data_dir: Path, stock_code: str) -> Optional[Path]:
    """
    Find stock TXT file by code.

    Supports:
    - SZ#000538.txt
    - 000538.txt
    - files whose name contains 000538
    """
    stock_code = normalize_code(stock_code)
    pure_code = stock_code.split("#")[-1]

    direct_candidates = [
        data_dir / f"{stock_code}.txt",
        data_dir / f"{pure_code}.txt",
    ]

    for path in direct_candidates:
        if path.exists():
            return path

    matches = list(data_dir.glob(f"*{pure_code}*.txt"))
    if matches:
        return matches[0]

    return None


def parse_stock_txt(path: Path) -> tuple[str, str, pd.DataFrame]:
    """
    Parse TongDaXin-style TXT file.

    Expected format:
    First line:
        BJ#920000 安徽凤凰 日线 前复权

    Second line:
        日期 开盘 最高 最低 收盘 成交量 成交额

    Data lines:
        02/08/2021,5.51,5.60,5.51,5.60,54900,327821.59
    """
    text = read_text_with_fallback(path)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) < 3:
        raise ValueError(f"File has too few lines: {path}")

    first_line = lines[0]
    first_parts = first_line.split()

    file_code = first_parts[0] if len(first_parts) >= 1 else path.stem
    stock_name = first_parts[1] if len(first_parts) >= 2 else ""

    data_lines = lines[2:]

    rows = []

    for line in data_lines:
        parts = [x.strip() for x in line.split(",")]

        if len(parts) < 5:
            continue

        while len(parts) < 7:
            parts.append("")

        rows.append(parts[:7])

    if not rows:
        raise ValueError(f"No valid data rows found: {path}")

    df = pd.DataFrame(
        rows,
        columns=[
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ],
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    df = df.reset_index(drop=True)

    return file_code, stock_name, df


def parse_t0_date(date_str: str) -> pd.Timestamp:
    """
    Parse T0 date.

    Supports:
    - 2026-04-24
    - 20260424
    - 24/04/2026
    """
    date_str = date_str.strip()

    if len(date_str) == 8 and date_str.isdigit():
        return pd.to_datetime(date_str, format="%Y%m%d")

    return pd.to_datetime(date_str, errors="raise")


def get_price_at_offset(
    df: pd.DataFrame,
    t0_idx: int,
    offset: int,
) -> tuple[object, object]:
    """
    Get open and close price at T+offset.

    T+1/T+2/T+3 are based on trading days, not calendar days.
    If the target trading day does not exist, return 0.
    """
    target_idx = t0_idx + offset

    if target_idx >= len(df):
        return 0, 0

    row = df.iloc[target_idx]
    return row["open"], row["close"]


def extract_one_stock(
    data_dir: Path,
    stock_code: str,
    t0_date: pd.Timestamp,
) -> dict:
    stock_code = normalize_code(stock_code)
    path = find_stock_file(data_dir, stock_code)

    result = {
        "股票代码": stock_code,
        "股票名": "",
        "T0日期": t0_date.strftime("%Y-%m-%d"),
        "T0开盘价": "",
        "T0收盘价": "",
        "T+1开盘价": "",
        "T+1收盘价": "",
        "T+2开盘价": "",
        "T+2收盘价": "",
        "T+3开盘价": "",
        "T+3收盘价": "",
        "状态": "",
    }

    if path is None:
        result["状态"] = "未找到股票TXT文件"
        return result

    try:
        file_code, stock_name, df = parse_stock_txt(path)
    except Exception as e:
        result["状态"] = f"读取失败: {e}"
        return result

    result["股票代码"] = file_code
    result["股票名"] = stock_name

    match = df.index[df["date"] == t0_date]

    if len(match) == 0:
        result["状态"] = "未找到T0日期，可能该日不是交易日或数据未更新"
        return result

    t0_idx = int(match[0])
    t0_row = df.iloc[t0_idx]

    result["T0开盘价"] = t0_row["open"]
    result["T0收盘价"] = t0_row["close"]

    for offset in [1, 2, 3]:
        open_price, close_price = get_price_at_offset(df, t0_idx, offset)
        result[f"T+{offset}开盘价"] = open_price
        result[f"T+{offset}收盘价"] = close_price

    result["状态"] = "OK"
    return result


def main() -> None:
    data_dir = Path(DATA_DIR)
    output_path = Path(OUTPUT_CSV)
    t0_date = parse_t0_date(T0_DATE)

    rows = []

    for code in CODES:
        rows.append(
            extract_one_stock(
                data_dir=data_dir,
                stock_code=code,
                t0_date=t0_date,
            )
        )

    out_df = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("提取完成")
    print(f"T0日期: {t0_date.strftime('%Y-%m-%d')}")
    print(f"股票数量: {len(CODES)}")
    print(f"输出文件: {output_path}")
    print("=" * 80)
    print(out_df)


if __name__ == "__main__":
    main()