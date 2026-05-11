# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


EXCLUDE_COLS = {
    "date",
    "symbol",
    "code",
    "name",
    "stock_name",
    "selection_strategy",
    "selected",
    "selected_score_base",
    "score_rank_key",
    "score_pct",
}


FORWARD_LABEL_PREFIXES = (
    "fwd_close_T",
    "fwd_return_pct_T",
    "fwd_up_T",
)


INVALID_FORWARD_TARGET_PREFIXES = (
    "fwd_close_T",
    "fwd_up_T",
)


@dataclass
class IndicatorDirectionResult:
    summary: pd.DataFrame
    bucket_detail: pd.DataFrame


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s,
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)


def _starts_with_any(col: str, prefixes: tuple[str, ...]) -> bool:
    col = str(col)

    return any(
        col.startswith(prefix)
        for prefix in prefixes
    )


def _is_forward_label_col(col: str) -> bool:
    return _starts_with_any(
        col,
        FORWARD_LABEL_PREFIXES,
    )


def _is_invalid_forward_target_col(col: str) -> bool:
    return _starts_with_any(
        col,
        INVALID_FORWARD_TARGET_PREFIXES,
    )


def _validate_target_col(df: pd.DataFrame, target_col: str) -> None:
    if target_col not in df.columns:
        raise ValueError(f"target_col not found: {target_col}")

    if _is_invalid_forward_target_col(target_col):
        raise ValueError(
            f"Invalid target_col: {target_col}. "
            f"不要用 fwd_close_T* 或 fwd_up_T* 做 target。"
            f"请使用 fwd_return_pct_T1 / fwd_return_pct_T2 / fwd_return_pct_T3。"
        )


def _is_numeric_factor(df: pd.DataFrame, col: str) -> bool:
    if col in EXCLUDE_COLS:
        return False

    # 防未来函数：
    # 所有 fwd_* 未来标签都禁止作为 factor。
    # fwd_return_pct_T1/T2/T3 只能作为 target。
    if _is_forward_label_col(col):
        return False

    if pd.api.types.is_bool_dtype(df[col]):
        return False

    s = _safe_numeric(df[col])
    valid_ratio = s.notna().mean()

    if valid_ratio < 0.3:
        return False

    if s.nunique(dropna=True) < 5:
        return False

    return True


def _detect_factor_cols(
    df: pd.DataFrame,
    target_col: str,
    factor_cols: Iterable[str] | None = None,
) -> list[str]:
    if factor_cols is not None:
        return [
            c for c in factor_cols
            if c in df.columns
            and c != target_col
            and _is_numeric_factor(df, c)
        ]

    return [
        c for c in df.columns
        if c != target_col
        and _is_numeric_factor(df, c)
    ]


def _calc_up_ratio(target: pd.Series) -> float:
    target = _safe_numeric(target).dropna()

    if target.empty:
        return np.nan

    return float((target > 0).mean())


def analyze_indicator_direction(
    df: pd.DataFrame,
    target_col: str,
    factor_cols: Iterable[str] | None = None,
    n_bins: int = 5,
    min_samples: int = 30,
) -> IndicatorDirectionResult:
    """
    分析初筛池中哪些指标是正向指标。

    推荐 target_col：
    - fwd_return_pct_T1
    - fwd_return_pct_T2
    - fwd_return_pct_T3

    禁止：
    - fwd_close_T1 / fwd_close_T2 / fwd_close_T3
    - fwd_up_T1 / fwd_up_T2 / fwd_up_T3

    防未来函数规则：
    - fwd_* 未来标签不能作为 factor。
    - fwd_return_pct_T* 只能作为 target。
    """

    _validate_target_col(df, target_col)

    work = df.copy()
    work[target_col] = _safe_numeric(work[target_col])

    factors = _detect_factor_cols(
        work,
        target_col=target_col,
        factor_cols=factor_cols,
    )

    summary_rows = []
    bucket_rows = []

    for factor in factors:
        tmp = work[[factor, target_col]].copy()

        tmp[factor] = _safe_numeric(tmp[factor])
        tmp[target_col] = _safe_numeric(tmp[target_col])

        tmp = tmp.dropna()

        if len(tmp) < int(min_samples):
            continue

        if tmp[factor].nunique(dropna=True) < int(n_bins):
            continue

        pearson_ic = tmp[factor].corr(
            tmp[target_col],
            method="pearson",
        )

        spearman_ic = tmp[factor].corr(
            tmp[target_col],
            method="spearman",
        )

        try:
            tmp["bucket"] = pd.qcut(
                tmp[factor],
                q=int(n_bins),
                labels=False,
                duplicates="drop",
            ) + 1
        except Exception:
            continue

        bucket = (
            tmp.groupby("bucket", as_index=False)
            .agg(
                sample_count=(target_col, "size"),
                mean_factor=(factor, "mean"),
                mean_return=(target_col, "mean"),
                median_return=(target_col, "median"),
                up_ratio=(target_col, _calc_up_ratio),
            )
            .sort_values("bucket")
        )

        if bucket.empty or bucket["bucket"].nunique() < 2:
            continue

        bottom = bucket.iloc[0]
        top = bucket.iloc[-1]

        bottom_mean_return = float(bottom["mean_return"])
        top_mean_return = float(top["mean_return"])

        bottom_up_ratio = float(bottom["up_ratio"])
        top_up_ratio = float(top["up_ratio"])

        top_minus_bottom_return = top_mean_return - bottom_mean_return
        top_minus_bottom_up_ratio = top_up_ratio - bottom_up_ratio

        positive_score = 0

        if pd.notna(spearman_ic) and spearman_ic > 0:
            positive_score += 1

        if pd.notna(pearson_ic) and pearson_ic > 0:
            positive_score += 1

        if top_minus_bottom_return > 0:
            positive_score += 1

        if top_minus_bottom_up_ratio > 0:
            positive_score += 1

        if positive_score >= 3:
            direction = "positive"
        elif positive_score <= 1:
            direction = "negative"
        else:
            direction = "unclear"

        summary_rows.append(
            {
                "factor": factor,
                "direction": direction,
                "positive_score": positive_score,
                "sample_count": len(tmp),
                "pearson_ic": pearson_ic,
                "spearman_ic": spearman_ic,
                "bottom_mean_return": bottom_mean_return,
                "top_mean_return": top_mean_return,
                "top_minus_bottom_return": top_minus_bottom_return,
                "bottom_up_ratio": bottom_up_ratio,
                "top_up_ratio": top_up_ratio,
                "top_minus_bottom_up_ratio": top_minus_bottom_up_ratio,
            }
        )

        bucket["factor"] = factor
        bucket["target_col"] = target_col

        bucket_rows.append(bucket)

    summary = pd.DataFrame(summary_rows)

    if not summary.empty:
        direction_order = {
            "positive": 0,
            "unclear": 1,
            "negative": 2,
        }

        summary["_direction_order"] = summary["direction"].map(direction_order)

        summary = (
            summary.sort_values(
                [
                    "_direction_order",
                    "positive_score",
                    "top_minus_bottom_return",
                    "spearman_ic",
                ],
                ascending=[True, False, False, False],
            )
            .drop(columns=["_direction_order"])
            .reset_index(drop=True)
        )

    if bucket_rows:
        bucket_detail = pd.concat(bucket_rows, ignore_index=True)
    else:
        bucket_detail = pd.DataFrame()

    return IndicatorDirectionResult(
        summary=summary,
        bucket_detail=bucket_detail,
    )

# ======================================================================================
# CLI runner used by analyze_tools/pool_dashboard.py
# ======================================================================================


def _read_pool_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix in {".csv", ".txt"}:
        for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        return pd.read_csv(path)

    raise ValueError(f"Unsupported pool file type: {path}")


def _normalize_primary_horizon(raw: str) -> str:
    s = str(raw).strip()

    if not s:
        raise ValueError("primary horizon is empty")

    if s.lower().startswith("fwd_return_pct_t"):
        t = s.split("_")[-1].upper()
        return f"fwd_return_pct_{t}"

    s = s.upper()
    if not s.startswith("T"):
        s = f"T{s}"

    return f"fwd_return_pct_{s}"


def _resolve_target_col(df: pd.DataFrame, raw: str) -> str:
    target = _normalize_primary_horizon(raw)

    if target in df.columns:
        return target

    lower_map = {str(c).lower(): c for c in df.columns}
    if target.lower() in lower_map:
        return str(lower_map[target.lower()])

    available = [str(c) for c in df.columns if str(c).lower().startswith("fwd_return_pct_t")]
    raise ValueError(
        f"target_col not found: {target}. "
        f"Available forward return targets: {available}"
    )


def _parse_factor_cols(raw: str | None) -> list[str] | None:
    if raw is None:
        return None

    cols = [x.strip() for x in str(raw).replace(";", ",").split(",") if x.strip()]
    return cols or None


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze whether pool indicator columns are positive / negative factors."
    )

    parser.add_argument("--pool-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-horizon", default="T2")
    parser.add_argument("--bucket-count", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--factor-cols", default=None, help="Optional comma/semicolon separated factor columns.")
    parser.add_argument("--include-unselected", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pool_path = Path(args.pool_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _read_pool_table(pool_path)

    if df.empty:
        raise RuntimeError(f"pool is empty: {pool_path}")

    if not args.include_unselected and "selected" in df.columns:
        selected_num = pd.to_numeric(df["selected"], errors="coerce")
        if selected_num.notna().any():
            df = df[selected_num.fillna(0).astype(int) == 1].copy()
        else:
            df = df[df["selected"].astype(bool)].copy()

    target_col = _resolve_target_col(df, args.primary_horizon)
    factor_cols = _parse_factor_cols(args.factor_cols)

    print("========== Analyze pool indicator direction ==========")
    print(f"pool_path      : {pool_path}")
    print(f"rows           : {len(df):,}")
    print(f"target_col     : {target_col}")
    print(f"bucket_count   : {args.bucket_count}")
    print(f"min_samples    : {args.min_samples}")
    print(f"selected_only  : {not args.include_unselected}")

    result = analyze_indicator_direction(
        df=df,
        target_col=target_col,
        factor_cols=factor_cols,
        n_bins=int(args.bucket_count),
        min_samples=int(args.min_samples),
    )

    summary = result.summary.copy()
    bucket_detail = result.bucket_detail.copy()

    if not summary.empty:
        summary.insert(0, "pool_name", pool_path.stem)
        summary.insert(1, "target_col", target_col)

    if not bucket_detail.empty:
        bucket_detail.insert(0, "pool_name", pool_path.stem)

    _save_csv(summary, output_dir / "indicator_direction_summary.csv")
    _save_csv(bucket_detail, output_dir / "indicator_bucket_detail.csv")

    if not summary.empty and "direction" in summary.columns:
        _save_csv(
            summary[summary["direction"].astype(str) == "positive"].copy(),
            output_dir / "indicator_positive_factors.csv",
        )
        _save_csv(
            summary[summary["direction"].astype(str) == "negative"].copy(),
            output_dir / "indicator_negative_factors.csv",
        )

    print("\n========== Result overview ==========")
    if summary.empty:
        print("No valid factors. Try reducing --min-samples or checking target columns.")
    else:
        print(summary.head(30).to_string(index=False))

    print(f"\nOutput dir: {output_dir}")


if __name__ == "__main__":
    main()

