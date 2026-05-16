
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Extreme bucket refinement analyzer.

Purpose:
- Do not modify Analyze Pool Indicator outputs.
- Recompute the original quantile buckets for one numeric factor.
- Take only the lowest original bucket and the highest original bucket.
- Split each selected extreme bucket into 5 sub-buckets by default.
- Save standalone CSV outputs for a separate dashboard page.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.path_manager import DATA_ROOT, POOLS_DIR


TARGET_COLS_DEFAULT = [
    "fwd_return_pct_T1",
    "fwd_return_pct_T2",
    "fwd_return_pct_T3",
    "fwd_return_pct_T4",
]


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _rank_quantile_bucket(s: pd.Series, bucket_count: int) -> pd.Series:
    bucket_count = int(bucket_count)
    if bucket_count < 2:
        raise ValueError("bucket_count must be >= 2")

    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    valid = s.notna()
    n = int(valid.sum())
    if n == 0:
        return out

    q = min(bucket_count, n)
    ranks = s.loc[valid].rank(method="first", ascending=True)
    labels = pd.qcut(ranks, q=q, labels=False, duplicates="drop") + 1
    out.loc[valid] = labels.astype("int64").to_numpy()
    return out


def _metric_row(df: pd.DataFrame, target_col: str) -> dict:
    y = _safe_numeric(df[target_col])
    valid_y = y.dropna()
    sample_count = int(valid_y.shape[0])
    win_count = int((valid_y > 0).sum())
    loss_count = int((valid_y <= 0).sum())

    return {
        "sample_count": sample_count,
        "mean_return": float(valid_y.mean()) if sample_count else np.nan,
        "median_return": float(valid_y.median()) if sample_count else np.nan,
        "up_ratio": float((valid_y > 0).mean()) if sample_count else np.nan,
        "win_count": win_count,
        "loss_count": loss_count,
    }


def _factor_interval(df: pd.DataFrame, factor: str) -> dict:
    x = _safe_numeric(df[factor])
    return {
        "min_factor": float(x.min()) if x.notna().any() else np.nan,
        "max_factor": float(x.max()) if x.notna().any() else np.nan,
        "mean_factor": float(x.mean()) if x.notna().any() else np.nan,
        "median_factor": float(x.median()) if x.notna().any() else np.nan,
    }


def _analyze_one_target(
    df: pd.DataFrame,
    factor: str,
    target_col: str,
    bucket_count: int,
    sub_bucket_count: int,
    min_samples: int,
    include_lowest: bool,
    include_highest: bool,
) -> tuple[list[dict], list[dict]]:
    work = df.loc[:, [factor, target_col]].copy(deep=False)
    work[factor] = _safe_numeric(work[factor])
    work[target_col] = _safe_numeric(work[target_col])
    work = work.dropna(subset=[factor, target_col])

    if work.empty:
        return [], []

    work["source_bucket"] = _rank_quantile_bucket(work[factor], bucket_count=bucket_count)
    work = work.dropna(subset=["source_bucket"])
    work["source_bucket"] = work["source_bucket"].astype(int)

    highest_bucket = int(work["source_bucket"].max())
    selected_sources: list[tuple[str, int]] = []

    if include_lowest:
        selected_sources.append(("lowest", 1))

    if include_highest and highest_bucket != 1:
        selected_sources.append(("highest", highest_bucket))

    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    for source_type, source_bucket in selected_sources:
        source_df = work.loc[work["source_bucket"] == source_bucket].copy(deep=False)
        if source_df.empty:
            continue

        source_metrics = _metric_row(source_df, target_col)
        source_interval = _factor_interval(source_df, factor)

        summary_rows.append(
            {
                "target_col": target_col,
                "factor": factor,
                "bucket_count": int(bucket_count),
                "sub_bucket_count": int(sub_bucket_count),
                "source_bucket_type": source_type,
                "source_bucket": int(source_bucket),
                "source_min_factor": source_interval["min_factor"],
                "source_max_factor": source_interval["max_factor"],
                "source_mean_factor": source_interval["mean_factor"],
                "source_median_factor": source_interval["median_factor"],
                "source_sample_count": source_metrics["sample_count"],
                "source_mean_return": source_metrics["mean_return"],
                "source_median_return": source_metrics["median_return"],
                "source_up_ratio": source_metrics["up_ratio"],
                "source_win_count": source_metrics["win_count"],
                "source_loss_count": source_metrics["loss_count"],
            }
        )

        source_df["sub_bucket"] = _rank_quantile_bucket(
            source_df[factor],
            bucket_count=sub_bucket_count,
        )
        source_df = source_df.dropna(subset=["sub_bucket"])
        source_df["sub_bucket"] = source_df["sub_bucket"].astype(int)

        for sub_bucket, sub_df in source_df.groupby("sub_bucket", sort=True):
            sub_metrics = _metric_row(sub_df, target_col)
            sub_interval = _factor_interval(sub_df, factor)

            detail_rows.append(
                {
                    "target_col": target_col,
                    "factor": factor,
                    "bucket_count": int(bucket_count),
                    "sub_bucket_count": int(sub_bucket_count),
                    "source_bucket_type": source_type,
                    "source_bucket": int(source_bucket),
                    "sub_bucket": int(sub_bucket),
                    "sub_min_factor": sub_interval["min_factor"],
                    "sub_max_factor": sub_interval["max_factor"],
                    "sub_mean_factor": sub_interval["mean_factor"],
                    "sub_median_factor": sub_interval["median_factor"],
                    "sample_count": sub_metrics["sample_count"],
                    "mean_return": sub_metrics["mean_return"],
                    "median_return": sub_metrics["median_return"],
                    "up_ratio": sub_metrics["up_ratio"],
                    "win_count": sub_metrics["win_count"],
                    "loss_count": sub_metrics["loss_count"],
                    "mean_return_vs_source": (
                        sub_metrics["mean_return"] - source_metrics["mean_return"]
                        if pd.notna(sub_metrics["mean_return"]) and pd.notna(source_metrics["mean_return"])
                        else np.nan
                    ),
                    "sample_ok": bool(sub_metrics["sample_count"] >= int(min_samples)),
                    "interpretation_note": (
                        "For lowest source bucket, sub_bucket 1 is the most extreme low value. "
                        "For highest source bucket, the largest sub_bucket is the most extreme high value."
                    ),
                }
            )

    return summary_rows, detail_rows


def analyze_extreme_bucket_refinement(
    pool_path: str | Path,
    factor: str,
    target_cols: Iterable[str],
    output_dir: str | Path,
    bucket_count: int = 10,
    sub_bucket_count: int = 5,
    min_samples: int = 10000,
    include_lowest: bool = True,
    include_highest: bool = True,
) -> dict[str, Path]:
    pool_path = Path(pool_path)
    output_dir = Path(output_dir)

    target_cols = [str(c) for c in target_cols if str(c).strip()]
    if not target_cols:
        raise ValueError("target_cols is empty")

    needed_cols = list(dict.fromkeys([factor, *target_cols]))
    df = pd.read_parquet(pool_path, columns=needed_cols)

    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Pool missing required columns: {missing}")

    if _safe_numeric(df[factor]).dropna().empty:
        raise ValueError(f"Factor is not numeric or has no valid numeric values: {factor}")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary: list[dict] = []
    all_detail: list[dict] = []

    for target_col in target_cols:
        summary_rows, detail_rows = _analyze_one_target(
            df=df,
            factor=factor,
            target_col=target_col,
            bucket_count=bucket_count,
            sub_bucket_count=sub_bucket_count,
            min_samples=min_samples,
            include_lowest=include_lowest,
            include_highest=include_highest,
        )
        all_summary.extend(summary_rows)
        all_detail.extend(detail_rows)

    summary = pd.DataFrame(all_summary)
    detail = pd.DataFrame(all_detail)

    summary_path = output_dir / "extreme_bucket_refinement_summary.csv"
    detail_path = output_dir / "extreme_bucket_refinement_detail.csv"
    meta_path = output_dir / "extreme_bucket_refinement.meta.json"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    meta = {
        "pool_path": str(pool_path),
        "factor": factor,
        "target_cols": target_cols,
        "bucket_count": int(bucket_count),
        "sub_bucket_count": int(sub_bucket_count),
        "min_samples": int(min_samples),
        "include_lowest": bool(include_lowest),
        "include_highest": bool(include_highest),
        "outputs": {
            "summary": str(summary_path),
            "detail": str(detail_path),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "summary": summary_path,
        "detail": detail_path,
        "meta": meta_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pool-path",
        type=str,
        default=str(POOLS_DIR / "b1_stage_low_select_strategy_v0_pool.parquet"),
    )
    parser.add_argument("--factor", type=str, required=True)
    parser.add_argument(
        "--target-col",
        action="append",
        default=[],
        help="Target column. Can be repeated. Defaults to T1-T4 if omitted.",
    )
    parser.add_argument("--bucket-count", type=int, default=10)
    parser.add_argument("--sub-bucket-count", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=10000)
    parser.add_argument(
        "--side",
        type=str,
        default="both",
        choices=["both", "lowest", "highest"],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_ROOT / "output" / "extreme_bucket_refinement"),
    )

    args = parser.parse_args()

    target_cols = args.target_col or TARGET_COLS_DEFAULT

    paths = analyze_extreme_bucket_refinement(
        pool_path=args.pool_path,
        factor=args.factor,
        target_cols=target_cols,
        output_dir=Path(args.output_dir) / Path(args.pool_path).stem / args.factor,
        bucket_count=args.bucket_count,
        sub_bucket_count=args.sub_bucket_count,
        min_samples=args.min_samples,
        include_lowest=args.side in {"both", "lowest"},
        include_highest=args.side in {"both", "highest"},
    )

    print("========== Extreme bucket refinement completed ==========")
    for k, p in paths.items():
        print(f"{k}: {p}")


if __name__ == "__main__":
    main()
