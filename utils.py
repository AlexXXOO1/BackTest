from __future__ import annotations

from pathlib import Path

import pandas as pd


TDX_TXT_ENCODINGS = (
    "utf-8-sig",
    "gb18030",
    "gbk",
    "cp936",
    "big5",
    "latin1",
)


def _score_decoded_sample(text: str) -> int:
    """Score decoded text by how much it looks like a TDX daily export file."""
    if not text:
        return -10_000

    score = 0
    score += text.count(",") * 3
    score += text.count("/") * 2
    score += text.count("-")
    score += sum(ch.isdigit() for ch in text)
    score -= text.count("�") * 100
    score -= text.count("\x00") * 50
    return score


def detect_tdx_txt_encoding(file_path: str | Path, sample_size: int = 65536) -> str:
    """
    Detect a readable encoding for a TDX TXT export.

    TDX exports are often saved as ANSI/GBK/GB18030 on Windows. This helper
    tries several common encodings and returns the one that best preserves the
    numeric CSV-like rows. It is intentionally tolerant because header lines may
    contain Chinese text while the useful data rows are numeric.
    """
    file_path = Path(file_path)
    raw = file_path.read_bytes()[:sample_size]

    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    best_encoding = "gb18030"
    best_score = -10_000

    for encoding in TDX_TXT_ENCODINGS:
        try:
            text = raw.decode(encoding, errors="strict")
        except UnicodeDecodeError:
            continue

        score = _score_decoded_sample(text)
        if score > best_score:
            best_score = score
            best_encoding = encoding

    return best_encoding


def clean_tdx_txt_line(line: str) -> str:
    """Normalize one decoded TDX TXT line before parsing."""
    return (
        line.replace("\ufeff", "")
        .replace("\x00", "")
        .replace("�", "")
        .strip()
    )


def read_tdx_export_txt(file_path: str | Path, end_date=None, encoding: str | None = None) -> pd.DataFrame:
    """
    Read a daily TXT file exported by TDX.

    Expected data columns are date, open, high, low, close, volume, and amount.
    The reader automatically handles common TDX encodings such as GBK, GB18030,
    ANSI/CP936, UTF-8 BOM, and falls back with tolerant decoding.

    When end_date is provided, rows after that date are not read.
    """
    file_path = Path(file_path)
    rows = []

    if end_date is not None:
        end_date = pd.Timestamp(end_date)

    if encoding is None:
        encoding = detect_tdx_txt_encoding(file_path)

    with open(file_path, "rb") as f:
        for raw_line in f:
            try:
                line = raw_line.decode(encoding, errors="ignore")
            except Exception:
                line = raw_line.decode("gb18030", errors="ignore")

            line = clean_tdx_txt_line(line)
            if not line:
                continue

            lower_line = line.lower()
            if "date" in lower_line or "open" in lower_line or "high" in lower_line or "close" in lower_line:
                continue

            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 7:
                continue

            dt = None
            for fmt in ("%d/%m/%Y", "%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y"):
                try:
                    dt = pd.to_datetime(parts[0], format=fmt)
                    break
                except Exception:
                    pass
            if dt is None:
                continue

            if end_date is not None and dt > end_date:
                break

            try:
                rows.append(
                    {
                        "date": dt,
                        "open": float(parts[1]),
                        "high": float(parts[2]),
                        "low": float(parts[3]),
                        "close": float(parts[4]),
                        "volume": float(parts[5]),
                        "amount": float(parts[6]),
                    }
                )
            except Exception:
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])
    return df.sort_values("date").reset_index(drop=True)


MAIN_BOARD_PREFIXES = (
    "SH#600",
    "SH#601",
    "SH#603",
    "SH#605",
    "SZ#000",
    "SZ#001",
    "SZ#002",
)

EXCLUDED_PREFIXES = (
    "SZ#300",
    "SZ#301",
    "SH#688",
    "SH#689",
    "BJ#",
)


def is_main_board_txt(file_path: str | Path) -> bool:
    """Return true when the file name belongs to A-share main-board symbols."""
    name = Path(file_path).name.upper()
    if any(name.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False
    return any(name.startswith(prefix) for prefix in MAIN_BOARD_PREFIXES)


def is_st_txt(file_path: str | Path) -> bool:
    """Return true when the path or file name contains the ST marker."""
    name = Path(file_path).stem.upper()
    return "ST" in name
