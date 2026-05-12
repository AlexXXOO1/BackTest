from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_CSV = Path(
    r"C:\Users\zyf37\Desktop\BackTest_System\daily_signal_score_2026-05-11_fwd_return_pct_T1.csv"
)


def normalize_code(value: object) -> str:
    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return ""

    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]

    if text.isdigit():
        return text.zfill(6)

    return text


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, dtype={"code": "string"})

    if "code" not in df.columns:
        raise ValueError(f"Missing required column: code. Available columns: {list(df.columns)}")

    codes = (
        df["code"]
        .map(normalize_code)
        .dropna()
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .tolist()
    )

    output_txt = INPUT_CSV.with_name(f"{INPUT_CSV.stem}_codes.txt")
    output_txt.write_text("\n".join(codes) + ("\n" if codes else ""), encoding="utf-8")

    print(f"Saved codes: {len(codes)}")
    print(f"Output txt: {output_txt}")


if __name__ == "__main__":
    main()
