# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ANALYSIS_SCHEMA_VERSION = "factor_shape_v11_pool_trend_distance_only"

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


ABSOLUTE_MARKET_VALUE_COLS = {
    "amount",
    "close",
    "high",
    "low",
    "open",
    "short_trend",
    "t1_open",
    "tdx_long_trend_line",
    "tdx_short_trend_line",
    "trend_line",
    "volume",
    "z_long_trend_line",
    "z_short_trend_line",
}

ABSOLUTE_MARKET_VALUE_PREFIX_NUMBER_COLS = (
    "ma",
    "volume_ma",
    "amount_ma",
)

FORWARD_LABEL_PREFIXES = (
    "fwd_close_T",
    "fwd_return_pct_T",
    "fwd_up_T",
)

INVALID_FORWARD_TARGET_PREFIXES = (
    "fwd_close_T",
    "fwd_up_T",
)


FUTURE_RAW_EXCLUDE_COLS = {
    "t1_date",
    "t1_close",
    "t2_date",
    "t2_open",
    "t2_close",
    "t3_date",
    "t3_open",
    "t3_close",
}

FUTURE_RAW_PREFIXES = (
    "t1_",
    "t2_",
    "t3_",
)

DETAIL_ID_COLS = (
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
    "close",
    "daily_return_pct",
    "market_regime",
    "v4_hint_label",
    "hint_label",
)

IC_EPS = 0.005
RETURN_EPS = 0.05
UP_RATIO_EPS = 0.005


@dataclass
class IndicatorDirectionResult:
    summary: pd.DataFrame
    bucket_detail: pd.DataFrame
    bucket_member_detail: pd.DataFrame


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _safe_float(x) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def _add_executable_entry_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "t1_open" in out.columns and "close" in out.columns:
        t1_open = _safe_numeric(out["t1_open"])
        close = _safe_numeric(out["close"])
        out["t1_open_gap_pct"] = np.where(close > 0, (t1_open / close - 1.0) * 100.0, np.nan)

    return out



def _signed_vote(value: float, eps: float) -> int:
    value = _safe_float(value)
    if pd.isna(value):
        return 0
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _starts_with_any(col: str, prefixes: tuple[str, ...]) -> bool:
    col = str(col)
    return any(col.startswith(prefix) for prefix in prefixes)


def _dedupe_keep_order(cols: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for col in cols:
        c = str(col)
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _member_detail_cols(df: pd.DataFrame, factor: str, target_col: str) -> list[str]:
    forward_return_cols = sorted(
        [str(c) for c in df.columns if str(c).lower().startswith("fwd_return_pct_t")],
        key=lambda x: (
            int(x.lower().split("fwd_return_pct_t")[-1])
            if x.lower().split("fwd_return_pct_t")[-1].isdigit()
            else 999,
            x,
        ),
    )
    candidate_cols = _dedupe_keep_order([*DETAIL_ID_COLS, *forward_return_cols])
    return [c for c in candidate_cols if c in df.columns and c not in {factor, target_col}]


def _is_forward_label_col(col: str) -> bool:
    return _starts_with_any(col, FORWARD_LABEL_PREFIXES)


def _is_invalid_forward_target_col(col: str) -> bool:
    return _starts_with_any(col, INVALID_FORWARD_TARGET_PREFIXES)


def _is_future_raw_col(col: str) -> bool:
    c = str(col).strip().lower()
    return c in FUTURE_RAW_EXCLUDE_COLS


def _is_absolute_market_value_col(col: str) -> bool:
    c = str(col).strip().lower()

    if c in ABSOLUTE_MARKET_VALUE_COLS:
        return True

    for prefix in ABSOLUTE_MARKET_VALUE_PREFIX_NUMBER_COLS:
        if c.startswith(prefix) and c[len(prefix):].isdigit():
            return True

    return False


def _validate_target_col(df: pd.DataFrame, target_col: str) -> None:
    if target_col not in df.columns:
        raise ValueError(f"target_col not found: {target_col}")

    if _is_invalid_forward_target_col(target_col):
        raise ValueError(
            f"Invalid target_col: {target_col}. Use fwd_return_pct_T1/T2/T3 as target. "
            f"Do not use fwd_close_T* or fwd_up_T*."
        )


def _is_numeric_factor(df: pd.DataFrame, col: str) -> bool:
    if col in EXCLUDE_COLS:
        return False

    if _is_forward_label_col(col):
        return False

    if _is_future_raw_col(col):
        return False

    if _is_absolute_market_value_col(col):
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
            c
            for c in factor_cols
            if c in df.columns and c != target_col and _is_numeric_factor(df, c)
        ]

    return [c for c in df.columns if c != target_col and _is_numeric_factor(df, c)]


def _calc_up_ratio(target: pd.Series) -> float:
    target = _safe_numeric(target).dropna()
    if target.empty:
        return np.nan
    return float((target > 0).mean())


def _middle_mean(bucket: pd.DataFrame, col: str) -> float:
    if len(bucket) >= 3:
        middle = bucket.iloc[1:-1]
    else:
        middle = bucket
    return _safe_float(middle[col].mean())


def _build_monotonic_bias(monotonic_votes: int) -> str:
    if monotonic_votes >= 3:
        return "higher_better"
    if monotonic_votes <= -3:
        return "lower_better"
    return "mixed_or_weak"


def _classify_bucket_pattern(
    monotonic_votes: int,
    top_mean_return: float,
    bottom_mean_return: float,
    middle_mean_return: float,
    top_up_ratio: float,
    bottom_up_ratio: float,
    middle_up_ratio: float,
    best_bucket: int,
    worst_bucket: int,
    bucket_count_actual: int,
    best_minus_worst_return: float,
) -> tuple[str, str, str, str]:
    top_vs_middle_return = top_mean_return - middle_mean_return
    top_vs_middle_up_ratio = top_up_ratio - middle_up_ratio
    bottom_vs_middle_return = bottom_mean_return - middle_mean_return
    bottom_vs_middle_up_ratio = bottom_up_ratio - middle_up_ratio

    has_clear_spread = best_minus_worst_return > RETURN_EPS
    best_is_middle = 1 < best_bucket < bucket_count_actual

    top_extreme_bad = (
        top_vs_middle_return < -RETURN_EPS
        or top_vs_middle_up_ratio < -UP_RATIO_EPS
        or worst_bucket == bucket_count_actual
    )
    bottom_extreme_bad = (
        bottom_vs_middle_return < -RETURN_EPS
        or bottom_vs_middle_up_ratio < -UP_RATIO_EPS
        or worst_bucket == 1
    )

    top_is_good = (
        top_vs_middle_return >= -RETURN_EPS
        and top_vs_middle_up_ratio >= -UP_RATIO_EPS
        and worst_bucket != bucket_count_actual
    )
    bottom_is_good = (
        bottom_vs_middle_return >= -RETURN_EPS
        and bottom_vs_middle_up_ratio >= -UP_RATIO_EPS
        and worst_bucket != 1
    )

    if monotonic_votes >= 3 and top_is_good:
        return (
            "higher_better",
            "prefer_high_values",
            "none",
            "Higher factor values have stronger monotonic evidence and no high-tail warning.",
        )

    if monotonic_votes <= -3 and bottom_is_good:
        return (
            "lower_better",
            "prefer_low_values",
            "none",
            "Lower factor values have stronger monotonic evidence and no low-tail warning.",
        )

    if has_clear_spread and best_is_middle and top_extreme_bad and bottom_extreme_bad:
        return (
            "middle_range_best",
            "use_middle_range",
            "both_tails",
            "The best bucket is in the middle and both extreme sides are weaker.",
        )

    if has_clear_spread and top_extreme_bad:
        return (
            "high_extreme_risk",
            "cap_high_values",
            "high",
            "The highest bucket is weaker than the middle buckets or is the worst bucket.",
        )

    if has_clear_spread and bottom_extreme_bad:
        return (
            "low_extreme_risk",
            "floor_low_values",
            "low",
            "The lowest bucket is weaker than the middle buckets or is the worst bucket.",
        )

    if has_clear_spread and best_is_middle:
        return (
            "middle_range_best",
            "use_middle_range",
            "none",
            "The best bucket is in the middle range; a simple higher/lower rule is not enough.",
        )

    return (
        "weak_or_unclear",
        "ignore_for_now",
        "none",
        "No stable monotonic or bucket-shape edge is detected.",
    )


def analyze_indicator_direction(
    df: pd.DataFrame,
    target_col: str,
    factor_cols: Iterable[str] | None = None,
    n_bins: int = 10,
    min_samples: int = 1000,
    export_member_detail: bool = False,
) -> IndicatorDirectionResult:
    """Analyze factor bucket shape without treating forward, future raw, or absolute market value columns as factors."""
    df = _add_executable_entry_features(df.copy())

    _validate_target_col(df, target_col)

    work = df.copy()
    work = _add_executable_entry_features(work)
    work[target_col] = _safe_numeric(work[target_col])

    factors = _detect_factor_cols(work, target_col=target_col, factor_cols=factor_cols)

    summary_rows = []
    bucket_rows = []
    member_rows = []

    for factor in factors:
        if export_member_detail:
            detail_cols = _member_detail_cols(work, factor=factor, target_col=target_col)
            tmp_cols = _dedupe_keep_order([factor, target_col, *detail_cols])
        else:
            tmp_cols = [factor, target_col]
        tmp = work[tmp_cols].copy()
        tmp[factor] = _safe_numeric(tmp[factor])
        tmp[target_col] = _safe_numeric(tmp[target_col])
        tmp = tmp.dropna(subset=[factor, target_col])

        if len(tmp) < int(min_samples):
            continue

        if tmp[factor].nunique(dropna=True) < int(n_bins):
            continue

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
                min_factor=(factor, "min"),
                max_factor=(factor, "max"),
                mean_factor=(factor, "mean"),
                mean_return=(target_col, "mean"),
                median_return=(target_col, "median"),
                up_ratio=(target_col, _calc_up_ratio),
            )
            .sort_values("bucket")
            .reset_index(drop=True)
        )

        if bucket.empty or bucket["bucket"].nunique() < 2:
            continue

        bucket_count_actual = int(bucket["bucket"].nunique())
        bottom = bucket.iloc[0]
        top = bucket.iloc[-1]

        bottom_bucket = int(bottom["bucket"])
        top_bucket = int(top["bucket"])
        bottom_mean_return = _safe_float(bottom["mean_return"])
        top_mean_return = _safe_float(top["mean_return"])
        bottom_up_ratio = _safe_float(bottom["up_ratio"])
        top_up_ratio = _safe_float(top["up_ratio"])

        middle_mean_return = _middle_mean(bucket, "mean_return")
        middle_up_ratio = _middle_mean(bucket, "up_ratio")

        top_minus_bottom_return = top_mean_return - bottom_mean_return
        top_minus_bottom_up_ratio = top_up_ratio - bottom_up_ratio
        top_vs_middle_return = top_mean_return - middle_mean_return
        top_vs_middle_up_ratio = top_up_ratio - middle_up_ratio
        bottom_vs_middle_return = bottom_mean_return - middle_mean_return
        bottom_vs_middle_up_ratio = bottom_up_ratio - middle_up_ratio

        best_row = bucket.loc[bucket["mean_return"].idxmax()]
        worst_row = bucket.loc[bucket["mean_return"].idxmin()]
        best_bucket = int(best_row["bucket"])
        worst_bucket = int(worst_row["bucket"])
        best_bucket_mean_return = _safe_float(best_row["mean_return"])
        worst_bucket_mean_return = _safe_float(worst_row["mean_return"])
        best_bucket_up_ratio = _safe_float(best_row["up_ratio"])
        worst_bucket_up_ratio = _safe_float(worst_row["up_ratio"])
        best_minus_worst_return = best_bucket_mean_return - worst_bucket_mean_return
        best_minus_worst_up_ratio = best_bucket_up_ratio - worst_bucket_up_ratio

        spearman_ic = tmp[factor].corr(tmp[target_col], method="spearman")
        pearson_ic = tmp[factor].corr(tmp[target_col], method="pearson")

        vote_spearman = _signed_vote(spearman_ic, IC_EPS)
        vote_pearson = _signed_vote(pearson_ic, IC_EPS)
        vote_tail_return = _signed_vote(top_minus_bottom_return, RETURN_EPS)
        vote_tail_up_ratio = _signed_vote(top_minus_bottom_up_ratio, UP_RATIO_EPS)
        monotonic_votes = vote_spearman + vote_pearson + vote_tail_return + vote_tail_up_ratio
        monotonic_bias = _build_monotonic_bias(monotonic_votes)

        bucket_pattern, action_hint, risk_side, pattern_reason = _classify_bucket_pattern(
            monotonic_votes=monotonic_votes,
            top_mean_return=top_mean_return,
            bottom_mean_return=bottom_mean_return,
            middle_mean_return=middle_mean_return,
            top_up_ratio=top_up_ratio,
            bottom_up_ratio=bottom_up_ratio,
            middle_up_ratio=middle_up_ratio,
            best_bucket=best_bucket,
            worst_bucket=worst_bucket,
            bucket_count_actual=bucket_count_actual,
            best_minus_worst_return=best_minus_worst_return,
        )

        summary_rows.append(
            {
                "schema_version": ANALYSIS_SCHEMA_VERSION,
                "factor": factor,
                "bucket_pattern": bucket_pattern,
                "action_hint": action_hint,
                "risk_side": risk_side,
                "pattern_reason": pattern_reason,
                "monotonic_bias": monotonic_bias,
                "monotonic_votes": monotonic_votes,
                "vote_spearman": vote_spearman,
                "vote_pearson": vote_pearson,
                "vote_tail_return": vote_tail_return,
                "vote_tail_up_ratio": vote_tail_up_ratio,
                "sample_count": len(tmp),
                "bucket_count_actual": bucket_count_actual,
                "spearman_ic": spearman_ic,
                "pearson_ic": pearson_ic,
                "bottom_bucket": bottom_bucket,
                "top_bucket": top_bucket,
                "bottom_mean_return": bottom_mean_return,
                "middle_mean_return": middle_mean_return,
                "top_mean_return": top_mean_return,
                "top_minus_bottom_return": top_minus_bottom_return,
                "top_vs_middle_return": top_vs_middle_return,
                "bottom_vs_middle_return": bottom_vs_middle_return,
                "bottom_up_ratio": bottom_up_ratio,
                "middle_up_ratio": middle_up_ratio,
                "top_up_ratio": top_up_ratio,
                "top_minus_bottom_up_ratio": top_minus_bottom_up_ratio,
                "top_vs_middle_up_ratio": top_vs_middle_up_ratio,
                "bottom_vs_middle_up_ratio": bottom_vs_middle_up_ratio,
                "best_bucket": best_bucket,
                "worst_bucket": worst_bucket,
                "best_bucket_mean_return": best_bucket_mean_return,
                "worst_bucket_mean_return": worst_bucket_mean_return,
                "best_minus_worst_return": best_minus_worst_return,
                "best_bucket_up_ratio": best_bucket_up_ratio,
                "worst_bucket_up_ratio": worst_bucket_up_ratio,
                "best_minus_worst_up_ratio": best_minus_worst_up_ratio,
            }
        )

        bucket["factor"] = factor
        bucket["target_col"] = target_col
        bucket_rows.append(bucket)

        if export_member_detail:
            member_detail = tmp.copy()
            member_detail.insert(0, "factor", factor)
            member_detail.insert(1, "target_col", target_col)
            member_detail = member_detail.rename(
                columns={
                    factor: "factor_value",
                    target_col: "target_value",
                }
            )
            member_rows.append(member_detail)

    summary = pd.DataFrame(summary_rows)

    if not summary.empty:
        pattern_order = {
            "high_extreme_risk": 0,
            "low_extreme_risk": 1,
            "middle_range_best": 2,
            "higher_better": 3,
            "lower_better": 4,
            "weak_or_unclear": 5,
        }
        summary["_pattern_order"] = summary["bucket_pattern"].map(pattern_order).fillna(99)
        summary["_abs_monotonic_votes"] = summary["monotonic_votes"].abs()
        summary["_abs_spearman_ic"] = summary["spearman_ic"].abs()
        summary = (
            summary.sort_values(
                [
                    "_pattern_order",
                    "best_minus_worst_return",
                    "_abs_monotonic_votes",
                    "_abs_spearman_ic",
                ],
                ascending=[True, False, False, False],
            )
            .drop(columns=["_pattern_order", "_abs_monotonic_votes", "_abs_spearman_ic"])
            .reset_index(drop=True)
        )

    if bucket_rows:
        bucket_detail = pd.concat(bucket_rows, ignore_index=True)
    else:
        bucket_detail = pd.DataFrame()

    if member_rows:
        bucket_member_detail = pd.concat(member_rows, ignore_index=True)
        sort_cols = [c for c in ["factor", "bucket", "date", "score_rank_key", "symbol", "code"] if c in bucket_member_detail.columns]
        if sort_cols:
            ascending = [True] * len(sort_cols)
            if "score_rank_key" in sort_cols:
                ascending[sort_cols.index("score_rank_key")] = False
            bucket_member_detail = bucket_member_detail.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    else:
        bucket_member_detail = pd.DataFrame()

    return IndicatorDirectionResult(
        summary=summary,
        bucket_detail=bucket_detail,
        bucket_member_detail=bucket_member_detail,
    )


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
        f"target_col not found: {target}. Available forward return targets: {available}"
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
        description="Analyze selected-pool factor bucket shape and suggested factor usage."
    )
    parser.add_argument("--pool-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-horizon", default="T2")
    parser.add_argument("--bucket-count", type=int, default=10)
    parser.add_argument("--min-samples", type=int, default=1000)
    parser.add_argument("--factor-cols", default=None, help="Optional comma/semicolon separated factor columns.")
    parser.add_argument("--export-member-detail", action="store_true", help="Export full bucket member detail. Disabled by default to avoid memory blow-up.")
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

    print("========== Analyze pool factor bucket shape ==========")
    print(f"schema_version : {ANALYSIS_SCHEMA_VERSION}")
    print(f"pool_path      : {pool_path}")
    print(f"rows           : {len(df):,}")
    print(f"target_col     : {target_col}")
    print(f"bucket_count   : {args.bucket_count}")
    print(f"min_samples    : {args.min_samples}")
    print(f"selected_only  : {not args.include_unselected}")
    print(f"export_detail  : {args.export_member_detail}")

    result = analyze_indicator_direction(
        df=df,
        target_col=target_col,
        factor_cols=factor_cols,
        n_bins=int(args.bucket_count),
        min_samples=int(args.min_samples),
        export_member_detail=bool(args.export_member_detail),
    )

    summary = result.summary.copy()
    bucket_detail = result.bucket_detail.copy()
    bucket_member_detail = result.bucket_member_detail.copy()

    if not summary.empty:
        summary.insert(0, "pool_name", pool_path.stem)
        summary.insert(1, "target_col", target_col)

    if not bucket_detail.empty:
        bucket_detail.insert(0, "pool_name", pool_path.stem)
        bucket_detail.insert(1, "schema_version", ANALYSIS_SCHEMA_VERSION)

    if not bucket_member_detail.empty:
        bucket_member_detail.insert(0, "pool_name", pool_path.stem)
        bucket_member_detail.insert(1, "schema_version", ANALYSIS_SCHEMA_VERSION)

    _save_csv(summary, output_dir / "indicator_direction_summary.csv")
    _save_csv(bucket_detail, output_dir / "indicator_bucket_detail.csv")
    if not bucket_member_detail.empty:
        _save_csv(bucket_member_detail, output_dir / "indicator_bucket_member_detail.csv")
    else:
        print("[SKIP] indicator_bucket_member_detail.csv disabled or empty")

    if not summary.empty and "action_hint" in summary.columns:
        for name in sorted(summary["action_hint"].dropna().astype(str).unique()):
            file_name = f"indicator_{name}_factors.csv"
            _save_csv(summary[summary["action_hint"].astype(str) == name].copy(), output_dir / file_name)

    print("\n========== Result overview ==========")
    if summary.empty:
        print("No valid factors. Try reducing --min-samples or checking target columns.")
    else:
        print(summary.head(30).to_string(index=False))

    print(f"\nOutput dir: {output_dir}")


if __name__ == "__main__":
    main()
