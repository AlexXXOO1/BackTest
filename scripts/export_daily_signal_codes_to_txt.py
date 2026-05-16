# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def normalize_code(value: object) -> str:
    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return ""

    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]

    if text.isdigit():
        return text.zfill(6)

    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the code column from a CSV to a TXT file.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Input CSV path.")
    parser.add_argument("--code-column", type=str, default="code", help="Column name containing stock codes.")
    parser.add_argument("--output-txt", type=Path, default=None, help="Optional output TXT path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, dtype={args.code_column: "string"})

    if args.code_column not in df.columns:
        raise ValueError(f"Missing required column: {args.code_column}. Available columns: {list(df.columns)}")

    codes = (
        df[args.code_column]
        .map(normalize_code)
        .dropna()
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .tolist()
    )

    output_txt = args.output_txt or input_csv.with_name(f"{input_csv.stem}_codes.txt")
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(codes) + ("\n" if codes else ""), encoding="utf-8")

    print(f"Saved codes: {len(codes)}")
    print(f"Output txt: {output_txt}")


if __name__ == "__main__":
    main()
