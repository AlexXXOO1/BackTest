from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BacktestConfig
from core.data_store import MarketDataStore


# Common encodings for TDX / Windows exported TXT files.
# Order matters: try Chinese Windows encodings before UTF-8 fallbacks.
ENCODING_CANDIDATES = [
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "gbk",
    "gb2312",
    "big5",
]


def parse_args():
    default = BacktestConfig()
    parser = argparse.ArgumentParser(description="Import TDX TXT files into the standard market cache.")
    parser.add_argument("--txt-dir", type=Path, default=default.txt_dir)
    parser.add_argument("--market-cache-dir", type=Path, default=default.market_cache_dir)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")

    # New options
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
    """
    Read a text file using several common encodings.

    Returns:
        tuple[str, str]: text content and detected encoding.

    Raises:
        UnicodeDecodeError: if none of the candidate encodings can decode the file.
    """
    raw = path.read_bytes()
    last_error: Optional[UnicodeDecodeError] = None

    for encoding in ENCODING_CANDIDATES:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError as exc:
            last_error = exc

    # Last fallback: decode with replacement to avoid import interruption.
    # This is not perfect, but prevents one bad file from stopping the whole import.
    return raw.decode("gb18030", errors="replace"), "gb18030_replace"


def normalize_newlines(text: str) -> str:
    """
    Normalize line endings to Windows-friendly CRLF.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\n", "\r\n")


def fix_txt_encoding_inplace(txt_dir: Path, output_encoding: str = "utf-8-sig") -> dict:
    """
    Fix all .txt files under txt_dir and overwrite original files.

    This function:
    1. Recursively scans txt_dir for .txt files.
    2. Tries to read each file using common TDX/Windows encodings.
    3. Overwrites the source file using output_encoding.

    Args:
        txt_dir: Directory containing TXT files.
        output_encoding: Encoding used for rewritten files.

    Returns:
        dict: Summary report.
    """
    report = {
        "txt_dir": str(txt_dir),
        "total_txt_files": 0,
        "fixed_files": 0,
        "skipped_files": 0,
        "failures": [],
        "encoding_count": {},
    }

    if not txt_dir.exists():
        report["failures"].append({
            "file": str(txt_dir),
            "error": "txt_dir does not exist",
        })
        return report

    txt_files = sorted(txt_dir.rglob("*.txt"))
    report["total_txt_files"] = len(txt_files)

    for file_path in txt_files:
        try:
            text, detected_encoding = read_text_with_best_encoding(file_path)
            text = normalize_newlines(text)

            file_path.write_text(text, encoding=output_encoding, newline="")
            report["fixed_files"] += 1
            report["encoding_count"][detected_encoding] = report["encoding_count"].get(detected_encoding, 0) + 1

        except Exception as exc:
            report["skipped_files"] += 1
            report["failures"].append({
                "file": str(file_path),
                "error": repr(exc),
            })

    return report


def print_encoding_fix_report(report: dict) -> None:
    print("========== TXT encoding fix completed ==========")
    print(f"txt_dir: {report['txt_dir']}")
    print(f"total_txt_files: {report['total_txt_files']}")
    print(f"fixed_files: {report['fixed_files']}")
    print(f"skipped_files: {report['skipped_files']}")

    if report["encoding_count"]:
        print("detected_encoding_count:")
        for encoding, count in sorted(report["encoding_count"].items()):
            print(f"  {encoding}: {count}")

    if report["failures"]:
        print("Encoding fix failures:")
        for item in report["failures"][:20]:
            print(f"  - {item['file']}: {item['error']}")


def main() -> None:
    args = parse_args()

    if args.fix_encoding:
        encoding_report = fix_txt_encoding_inplace(
            txt_dir=args.txt_dir,
            output_encoding=args.fixed_encoding,
        )
        print_encoding_fix_report(encoding_report)

    store = MarketDataStore(args.txt_dir, args.market_cache_dir)
    report = store.import_txt_files(
        start_date=args.start_date,
        end_date=args.end_date,
        overwrite=args.overwrite,
    )

    print("========== Import completed ==========")
    for key, value in report.items():
        if key != "failures":
            print(f"{key}: {value}")

    if report["failures"]:
        print("Failures:")
        for item in report["failures"][:20]:
            print(f"  - {item['file']}: {item['error']}")


if __name__ == "__main__":
    main()