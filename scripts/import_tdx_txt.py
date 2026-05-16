# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.path_manager import DATA_ROOT, RAW_TDX_TXT_DIR, MARKET_CACHE_DIR

DEFAULT_DATA_ROOT = DATA_ROOT
DEFAULT_TXT_DIR = RAW_TDX_TXT_DIR
DEFAULT_MARKET_CACHE_DIR = MARKET_CACHE_DIR


ENCODING_CANDIDATES = [
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "gbk",
    "gb2312",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "big5",
]

TDX_SOURCE_TAIL_LINE = "#数据来源:通达信"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import raw TDX TXT files into market_cache/daily_bars_by_symbol."
    )

    parser.add_argument(
        "--txt-dir",
        type=Path,
        default=DEFAULT_TXT_DIR,
        help="Raw TDX TXT directory.",
    )

    parser.add_argument(
        "--market-cache-dir",
        type=Path,
        default=DEFAULT_MARKET_CACHE_DIR,
        help="Output market cache directory.",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date, e.g. 2021-01-01.",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date, e.g. 2026-05-10.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parquet files.",
    )

    parser.add_argument(
        "--clear-market-cache",
        action="store_true",
        help="Clear existing parquet files in market cache before import.",
    )

    parser.add_argument(
        "--fix-encoding",
        action="store_true",
        default=True,
        help="Fix TXT encoding before importing. Default: enabled.",
    )

    parser.add_argument(
        "--no-fix-encoding",
        dest="fix_encoding",
        action="store_false",
        help="Disable TXT encoding fixing.",
    )

    parser.add_argument(
        "--fixed-encoding",
        type=str,
        default="utf-8-sig",
        help="Encoding used when overwriting fixed TXT files. Default: utf-8-sig.",
    )

    return parser.parse_args()


def read_text_with_best_encoding(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()

    for encoding in ENCODING_CANDIDATES:
        try:
            return raw.decode(encoding), encoding
        except Exception:
            continue

    return raw.decode("gb18030", errors="replace"), "gb18030_replace"


def normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\n", "\r\n")


def remove_tdx_source_tail_line_after_converted(text: str) -> tuple[str, bool]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    last_real_idx = len(lines) - 1
    while last_real_idx >= 0 and lines[last_real_idx].strip() == "":
        last_real_idx -= 1

    if last_real_idx >= 0 and lines[last_real_idx].strip() == TDX_SOURCE_TAIL_LINE:
        del lines[last_real_idx]
        cleaned = "\n".join(lines).rstrip() + "\n"
        return cleaned, True

    return normalized, False


def fix_txt_encoding_inplace(txt_dir: Path, output_encoding: str = "utf-8-sig") -> dict:
    report = {
        "txt_dir": str(txt_dir),
        "total_txt_files": 0,
        "fixed_files": 0,
        "skipped_files": 0,
        "removed_tdx_source_tail_files": 0,
        "failures": [],
        "encoding_count": {},
    }

    if not txt_dir.exists():
        report["failures"].append(
            {
                "file": str(txt_dir),
                "error": "txt_dir does not exist",
            }
        )
        return report

    txt_files = sorted(txt_dir.rglob("*.txt"))
    report["total_txt_files"] = len(txt_files)

    for file_path in tqdm(txt_files, desc="Fix TXT encoding"):
        try:
            text, detected_encoding = read_text_with_best_encoding(file_path)

            text = normalize_newlines(text)
            file_path.write_text(text, encoding=output_encoding, newline="")

            converted_text = file_path.read_text(encoding=output_encoding)

            converted_text, removed_tail = remove_tdx_source_tail_line_after_converted(
                converted_text
            )

            converted_text = normalize_newlines(converted_text)
            file_path.write_text(converted_text, encoding=output_encoding, newline="")

            if removed_tail:
                report["removed_tdx_source_tail_files"] += 1

            report["fixed_files"] += 1
            report["encoding_count"][detected_encoding] = (
                report["encoding_count"].get(detected_encoding, 0) + 1
            )

        except Exception as exc:
            report["skipped_files"] += 1
            report["failures"].append(
                {
                    "file": str(file_path),
                    "error": repr(exc),
                }
            )

    return report


def print_encoding_fix_report(report: dict) -> None:
    print("========== TXT encoding fix completed ==========")
    print(f"txt_dir: {report['txt_dir']}")
    print(f"total_txt_files: {report['total_txt_files']}")
    print(f"fixed_files: {report['fixed_files']}")
    print(f"skipped_files: {report['skipped_files']}")
    print(
        "removed_tdx_source_tail_files: "
        f"{report.get('removed_tdx_source_tail_files', 0)}"
    )

    if report["encoding_count"]:
        print("detected_encoding_count:")
        for encoding, count in sorted(report["encoding_count"].items()):
            print(f"  {encoding}: {count}")

    if report["failures"]:
        print("Encoding fix failures:")
        for item in report["failures"][:20]:
            print(f"  - {item['file']}: {item['error']}")


def normalize_symbol_from_path(path: Path) -> Optional[str]:
    """
    Use filename as symbol.

    Examples:
        SH#600000.txt -> SH#600000
        SZ#003816.txt -> SZ#003816
        600000.txt    -> 600000

    不做 A 股前缀过滤，因为用户确认 raw_tdx_data 下都是备选票数据。
    """
    stem = path.stem.strip().upper()

    if not stem:
        return None

    stem = stem.replace(" ", "")

    if "#" in stem:
        left, right = stem.split("#", 1)
        code_match = re.search(r"(\d{6})", right)
        if code_match:
            market = left.strip().upper()
            code = code_match.group(1)
            if market in {"SH", "SZ"}:
                return f"{market}#{code}"
            return f"{market}#{code}"

    code_match = re.search(r"(\d{6})", stem)
    if not code_match:
        return None

    code = code_match.group(1)

    if code.startswith(("600", "601", "603", "605", "688")):
        return f"SH#{code}"

    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return f"SZ#{code}"

    return code


def read_tdx_txt_table(path: Path) -> pd.DataFrame:
    """
    Robust reader for TDX TXT daily files.

    Handles this mixed delimiter format:
        003816 中国广核 日线 前复权
        日期\\t    开盘\\t    最高\\t    最低\\t    收盘\\t    成交量\\t    成交额
        02/08/2021,2.17,2.22,2.17,2.21,126580018,279129504.00

    Also handles:
    - gbk / gb18030 / utf-8-sig / utf-16
    - 通达信页眉
    - 尾部 #数据来源:通达信
    - comma / tab / whitespace mixed delimiters
    - only-header files: return empty DataFrame
    """
    raw = path.read_bytes()

    text = None
    used_encoding = None
    decode_errors = []

    for enc in ENCODING_CANDIDATES:
        try:
            text = raw.decode(enc)
            used_encoding = enc
            break
        except Exception as exc:
            decode_errors.append(f"{enc}: {repr(exc)}")

    if text is None:
        text = raw.decode("gb18030", errors="ignore")
        used_encoding = "gb18030_ignore"

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        if line.startswith("#数据来源"):
            continue

        if "数据来源" in line:
            continue

        lines.append(line)

    if not lines:
        return pd.DataFrame()

    header_idx = None

    for i, line in enumerate(lines):
        if "日期" in line and ("开盘" in line or "收盘" in line):
            header_idx = i
            break

    if header_idx is None:
        for i, line in enumerate(lines):
            if re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", line):
                header_idx = i
                break

            if re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", line):
                header_idx = i
                break

            if re.search(r"\b\d{8}\b", line):
                header_idx = i
                break

    if header_idx is None:
        raise RuntimeError(
            f"Cannot find header/data start in TXT: {path}, "
            f"used_encoding={used_encoding}, first_lines={lines[:5]}, "
            f"decode_errors={decode_errors[:3]}"
        )

    usable_lines = lines[header_idx:]

    if not usable_lines:
        return pd.DataFrame()

    first_line = usable_lines[0]

    first_is_data = bool(
        re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", first_line)
        or re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", first_line)
        or re.search(r"\b\d{8}\b", first_line)
    )

    if first_is_data:
        header = ["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额"]
        data_lines = usable_lines
    else:
        header = [
            x.strip()
            for x in re.split(r"[\t,\s]+", first_line)
            if x.strip()
        ]
        data_lines = usable_lines[1:]

    if len(header) < 6:
        raise RuntimeError(
            f"Cannot parse header in TXT: {path}, "
            f"used_encoding={used_encoding}, header={header}, first_line={first_line}"
        )

    # 统一只保留前 7 列：日期、开盘、最高、最低、收盘、成交量、成交额
    if len(header) >= 7:
        header = header[:7]
    else:
        header = ["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额"][: len(header)]

    rows = []

    for line in data_lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("#数据来源") or "数据来源" in line:
            continue

        if "," in line:
            parts = [x.strip() for x in line.split(",")]
        else:
            parts = [x.strip() for x in re.split(r"[\t,\s]+", line) if x.strip()]

        if len(parts) < 6:
            continue

        if len(parts) >= 7:
            parts = parts[:7]
        else:
            parts = parts + [""] * (7 - len(parts))

        rows.append(parts)

    if not rows:
        return pd.DataFrame(columns=["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额"])

    return pd.DataFrame(
        rows,
        columns=["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额"],
    )


def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = list(df.columns)
    exact = {str(c).strip(): c for c in cols}
    lower = {str(c).strip().lower(): c for c in cols}

    for name in candidates:
        if name in exact:
            return exact[name]

        key = name.lower()
        if key in lower:
            return lower[key]

    return None


def parse_tdx_date_series(s: pd.Series) -> pd.Series:
    """
    通达信样本里日期为 02/08/2021，表示 2021-08-02。
    因此优先 dayfirst=True。
    """
    raw = s.astype(str).str.strip()

    dt = pd.to_datetime(raw, errors="coerce", dayfirst=True)

    # 兜底处理 YYYYMMDD
    mask = dt.isna() & raw.str.match(r"^\d{8}$", na=False)
    if mask.any():
        dt.loc[mask] = pd.to_datetime(raw.loc[mask], format="%Y%m%d", errors="coerce")

    return dt


def normalize_market_df(raw: pd.DataFrame, symbol: str, file_path: Path) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "file",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
            ]
        )

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    col_date = find_col(df, ["日期", "date", "Date", "交易日期", "时间"])
    col_open = find_col(df, ["开盘", "open", "Open", "OPEN"])
    col_high = find_col(df, ["最高", "high", "High", "HIGH"])
    col_low = find_col(df, ["最低", "low", "Low", "LOW"])
    col_close = find_col(df, ["收盘", "close", "Close", "CLOSE"])
    col_volume = find_col(df, ["成交量", "volume", "Volume", "VOL", "vol"])
    col_amount = find_col(df, ["成交额", "amount", "Amount", "成交金额", "AMOUNT"])

    if not all([col_date, col_open, col_high, col_low, col_close, col_volume, col_amount]):
        if len(df.columns) >= 7:
            col_date = df.columns[0]
            col_open = df.columns[1]
            col_high = df.columns[2]
            col_low = df.columns[3]
            col_close = df.columns[4]
            col_volume = df.columns[5]
            col_amount = df.columns[6]
        else:
            raise ValueError(
                f"Cannot infer required columns from {file_path}. "
                f"columns={list(df.columns)}"
            )

    # 关键修复：用 df.index 初始化，避免 symbol/file 变成 NaN
    out = pd.DataFrame(index=df.index)

    out["symbol"] = symbol
    out["file"] = file_path.name
    out["date"] = parse_tdx_date_series(df[col_date])

    out["open"] = pd.to_numeric(df[col_open], errors="coerce")
    out["high"] = pd.to_numeric(df[col_high], errors="coerce")
    out["low"] = pd.to_numeric(df[col_low], errors="coerce")
    out["close"] = pd.to_numeric(df[col_close], errors="coerce")
    out["volume"] = pd.to_numeric(df[col_volume], errors="coerce").fillna(0).astype("int64")
    out["amount"] = pd.to_numeric(df[col_amount], errors="coerce")

    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out[out["close"] > 0]

    if out.empty:
        return out.reset_index(drop=True)

    # 再赋一次，确保不会因为 dropna/index 对齐异常导致 NaN
    out["symbol"] = symbol
    out["file"] = file_path.name

    out = out.sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    out = out.reset_index(drop=True)

    return out


def discover_txt_files(txt_dir: Path) -> list[Path]:
    if not txt_dir.exists():
        return []

    files = []
    files.extend(txt_dir.rglob("*.txt"))
    files.extend(txt_dir.rglob("*.TXT"))

    return sorted(set(files), key=lambda x: str(x))


def clear_market_cache_dir(market_cache_dir: Path) -> None:
    market_cache_dir.mkdir(parents=True, exist_ok=True)

    for p in market_cache_dir.glob("*.parquet"):
        p.unlink()

    for p in market_cache_dir.glob("*.csv"):
        p.unlink()

    for p in market_cache_dir.glob("*.json"):
        p.unlink()


def import_txt_files(
    txt_dir: Path,
    market_cache_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    report = {
        "txt_dir": str(txt_dir),
        "market_cache_dir": str(market_cache_dir),
        "total_txt_files": 0,
        "imported_files": 0,
        "skipped_files": 0,
        "failed_files": 0,
        "empty_raw_files": 0,
        "empty_normalized_files": 0,
        "sz_003_files": 0,
        "failures": [],
    }

    files = discover_txt_files(txt_dir)
    report["total_txt_files"] = len(files)

    market_cache_dir.mkdir(parents=True, exist_ok=True)

    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None

    for path in tqdm(files, desc="Import TXT to market cache"):
        try:
            symbol = normalize_symbol_from_path(path)

            if symbol is None:
                report["skipped_files"] += 1
                continue

            out_path = market_cache_dir / f"{symbol}.parquet"

            if out_path.exists() and not overwrite:
                report["skipped_files"] += 1
                continue

            raw = read_tdx_txt_table(path)

            if raw.empty:
                report["empty_raw_files"] += 1
                report["skipped_files"] += 1
                continue

            df = normalize_market_df(raw, symbol=symbol, file_path=path)

            if start_ts is not None:
                df = df[df["date"] >= start_ts]

            if end_ts is not None:
                df = df[df["date"] <= end_ts]

            if df.empty:
                report["empty_normalized_files"] += 1
                report["skipped_files"] += 1
                continue

            df.to_parquet(out_path, index=False)

            report["imported_files"] += 1

            if symbol.startswith("SZ#003"):
                report["sz_003_files"] += 1

        except Exception as exc:
            report["failed_files"] += 1
            report["failures"].append(
                {
                    "file": str(path),
                    "error": repr(exc),
                }
            )

    return report


def print_import_report(report: dict) -> None:
    print("========== Import completed ==========")

    for key, value in report.items():
        if key != "failures":
            print(f"{key}: {value}")

    if report["failures"]:
        print("Failures:")
        for item in report["failures"][:30]:
            print(f"  - {item['file']}: {item['error']}")


def print_market_cache_check(market_cache_dir: Path) -> None:
    print("========== CHECK MARKET CACHE ==========")

    parquet_files = sorted(market_cache_dir.glob("*.parquet"))
    print(f"market parquet count: {len(parquet_files)}")

    check_003 = sorted(market_cache_dir.glob("SZ#003*.parquet"))
    print(f"SZ#003 parquet count: {len(check_003)}")

    for p in check_003[:20]:
        print(" ", p.name)

    if parquet_files:
        sample = parquet_files[0]
        try:
            df = pd.read_parquet(sample)
            print("sample:", sample)
            print(df.head(3).to_string(index=False))
            print("sample rows:", len(df))
            print("sample date range:", df["date"].min(), "->", df["date"].max())
            print("sample symbol head:", df["symbol"].head(3).tolist())
        except Exception as exc:
            print("sample read failed:", repr(exc))


def main() -> None:
    args = parse_args()

    if args.clear_market_cache:
        print("========== CLEAR MARKET CACHE ==========")
        print(f"clear dir: {args.market_cache_dir}")
        clear_market_cache_dir(args.market_cache_dir)

    if args.fix_encoding:
        encoding_report = fix_txt_encoding_inplace(
            txt_dir=args.txt_dir,
            output_encoding=args.fixed_encoding,
        )
        print_encoding_fix_report(encoding_report)

    report = import_txt_files(
        txt_dir=args.txt_dir,
        market_cache_dir=args.market_cache_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        overwrite=args.overwrite,
    )

    print_import_report(report)
    print_market_cache_check(args.market_cache_dir)


if __name__ == "__main__":
    main()