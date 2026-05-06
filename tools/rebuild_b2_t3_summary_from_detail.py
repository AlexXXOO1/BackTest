from __future__ import annotations

r"""
Rebuild B2 T+3 quality summary from detail CSV.

Use this when summary CSV is incomplete or contains a literal "..." row.

It reads:
    b2_confirm_select_strategy_v0_t3_quality_detail.csv

And writes:
    b2_confirm_select_strategy_v0_t3_quality_summary_fixed.csv
"""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DETAIL_PATH = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\output\pool_quality_b2_t3\b2_confirm_select_strategy_v0_t3_quality_detail.csv"
)


def bool_mean_pct(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").mean()) * 100.0


def num_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").mean())


def num_median(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").median())


def num_max(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").max())


def num_min(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").min())


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    rows.append({"metric": "trade_count", "value": len(detail)})

    sell_modes = [
        ("t2_open", "win_t2_open", "ret_t1_open_to_t2_open_pct"),
        ("t2_close", "win_t2_close", "ret_t1_open_to_t2_close_pct"),
        ("t3_open", "win_t3_open", "ret_t1_open_to_t3_open_pct"),
        ("t3_close", "win_t3_close", "ret_t1_open_to_t3_close_pct"),
    ]

    for label, win_col, ret_col in sell_modes:
        rows.extend(
            [
                {"metric": f"{label}_win_rate_pct", "value": bool_mean_pct(detail, win_col)},
                {"metric": f"{label}_avg_ret_pct", "value": num_mean(detail, ret_col)},
                {"metric": f"{label}_median_ret_pct", "value": num_median(detail, ret_col)},
                {"metric": f"{label}_best_ret_pct", "value": num_max(detail, ret_col)},
                {"metric": f"{label}_worst_ret_pct", "value": num_min(detail, ret_col)},
            ]
        )

    horizons = [
        ("t1_t2", "max_opportunity_t1_t2_pct", "max_drawdown_t1_t2_pct"),
        ("t1_t3", "max_opportunity_t1_t3_pct", "max_drawdown_t1_t3_pct"),
    ]

    for horizon, opp_col, dd_col in horizons:
        rows.extend(
            [
                {"metric": f"{horizon}_hit_plus_2pct_rate_pct", "value": bool_mean_pct(detail, f"hit_plus_2pct_{horizon}")},
                {"metric": f"{horizon}_hit_plus_3pct_rate_pct", "value": bool_mean_pct(detail, f"hit_plus_3pct_{horizon}")},
                {"metric": f"{horizon}_hit_plus_5pct_rate_pct", "value": bool_mean_pct(detail, f"hit_plus_5pct_{horizon}")},
                {"metric": f"{horizon}_hit_minus_2pct_rate_pct", "value": bool_mean_pct(detail, f"hit_minus_2pct_{horizon}")},
                {"metric": f"{horizon}_hit_minus_3pct_rate_pct", "value": bool_mean_pct(detail, f"hit_minus_3pct_{horizon}")},
                {"metric": f"{horizon}_hit_minus_5pct_rate_pct", "value": bool_mean_pct(detail, f"hit_minus_5pct_{horizon}")},
                {"metric": f"{horizon}_avg_max_opportunity_pct", "value": num_mean(detail, opp_col)},
                {"metric": f"{horizon}_avg_max_drawdown_pct", "value": num_mean(detail, dd_col)},
            ]
        )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild fixed B2 T+3 summary from detail CSV.")
    parser.add_argument("--detail-path", type=Path, default=DEFAULT_DETAIL_PATH)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.detail_path.exists():
        raise FileNotFoundError(f"Detail CSV not found: {args.detail_path}")

    detail = pd.read_csv(args.detail_path)

    # Rebuild win columns if missing but return columns exist.
    ret_cols = {
        "win_t2_open": "ret_t1_open_to_t2_open_pct",
        "win_t2_close": "ret_t1_open_to_t2_close_pct",
        "win_t3_open": "ret_t1_open_to_t3_open_pct",
        "win_t3_close": "ret_t1_open_to_t3_close_pct",
    }
    for win_col, ret_col in ret_cols.items():
        if win_col not in detail.columns and ret_col in detail.columns:
            detail[win_col] = pd.to_numeric(detail[ret_col], errors="coerce") > 0

    summary = build_summary(detail)

    output_path = args.output_path
    if output_path is None:
        output_path = args.detail_path.with_name(
            args.detail_path.stem.replace("_detail", "_summary_fixed") + ".csv"
        )

    summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=" * 88)
    print("Fixed T+3 summary generated")
    print("=" * 88)
    print(f"Detail : {args.detail_path}")
    print(f"Output : {output_path}")
    print("-" * 88)
    for _, r in summary.iterrows():
        print(f"{str(r['metric']).ljust(44)} : {r['value']}")


if __name__ == "__main__":
    main()
