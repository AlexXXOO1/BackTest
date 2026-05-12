"""Validate whether a pool score is useful before comparing the pool with the market."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_TARGET_PREFIX = "fwd_return_pct_"
DEFAULT_THRESHOLD_TEXT = "0,1,2,3,4,5"
DEFAULT_TOP_N = 5
DEFAULT_BUCKET_COUNT = 10
DEFAULT_MIN_DAILY_SAMPLES = 5


class ScoreValidationError(RuntimeError):
    """Raised when score validation cannot run with the provided data."""


def _to_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(str(value)).expanduser().resolve()


def _project_root() -> Path:
    return Path.cwd().resolve()


def _infer_data_root(project_root: Path) -> Path:
    candidates = [
        Path.cwd().resolve() / "BackTest_Data",
        project_root.parent / "BackTest_Data",
        Path.home() / "Desktop" / "BackTest_Data",
    ]
    for item in candidates:
        if item.exists():
            return item
    return project_root.parent / "BackTest_Data"


def _infer_latest_pool_path(data_root: Path) -> Path:
    pool_dir = data_root / "pools"
    if not pool_dir.exists():
        raise ScoreValidationError(f"Pool directory not found: {pool_dir}")
    files = sorted(pool_dir.glob("*_pool.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise ScoreValidationError(f"No *_pool.parquet files found in: {pool_dir}")
    return files[0]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ScoreValidationError(f"Unsupported input file type: {path.suffix}")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _parse_csv_text(text: str | None) -> list[str]:
    if text is None or str(text).strip() == "":
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _parse_thresholds(text: str | None) -> list[float]:
    values = []
    for item in _parse_csv_text(text or DEFAULT_THRESHOLD_TEXT):
        values.append(float(item))
    if not values:
        values = [0.0]
    return sorted(set(values))


def _infer_date_col(df: pd.DataFrame, user_value: str | None) -> str:
    if user_value:
        if user_value not in df.columns:
            raise ScoreValidationError(f"date column not found: {user_value}")
        return user_value
    for col in ["date", "trade_date", "datetime", "dt"]:
        if col in df.columns:
            return col
    raise ScoreValidationError("Cannot infer date column. Please pass --date-col.")


def _infer_code_col(df: pd.DataFrame, user_value: str | None) -> str | None:
    if user_value:
        if user_value not in df.columns:
            raise ScoreValidationError(f"code column not found: {user_value}")
        return user_value
    for col in ["code", "symbol", "ts_code", "ticker"]:
        if col in df.columns:
            return col
    return None


def _infer_score_col(df: pd.DataFrame, user_value: str | None) -> str:
    if user_value:
        if user_value not in df.columns:
            raise ScoreValidationError(f"score column not found: {user_value}")
        return user_value
    candidates = [
        "score_rank_key",
        "selected_score_total",
        "selected_score",
        "selected_score_base",
        "score",
        "rank_score",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    score_like = [col for col in df.columns if "score" in str(col).lower() and not str(col).lower().startswith("fwd_")]
    if len(score_like) == 1:
        return score_like[0]
    raise ScoreValidationError(
        "Cannot infer score column. Please pass --score-col. Available score-like columns: "
        + ", ".join(score_like[:30])
    )


def _infer_target_cols(df: pd.DataFrame, user_text: str | None) -> list[str]:
    if user_text:
        cols = _parse_csv_text(user_text)
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ScoreValidationError(f"target columns not found: {missing}")
        return cols
    preferred = ["fwd_return_pct_T1", "fwd_return_pct_T2", "fwd_return_pct_T3"]
    cols = [col for col in preferred if col in df.columns]
    if cols:
        return cols
    cols = [col for col in df.columns if str(col).startswith(DEFAULT_TARGET_PREFIX)]
    if cols:
        return sorted(cols)
    raise ScoreValidationError("No forward-return target columns found. Please pass --target-cols.")


def _clean_data(df: pd.DataFrame, date_col: str, score_col: str, target_cols: Iterable[str]) -> pd.DataFrame:
    need_cols = [date_col, score_col, *target_cols]
    missing = [col for col in need_cols if col not in df.columns]
    if missing:
        raise ScoreValidationError(f"Missing required columns: {missing}")
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    for col in target_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[date_col, score_col])
    out = out.sort_values([date_col, score_col]).reset_index(drop=True)
    if out.empty:
        raise ScoreValidationError("No usable rows after cleaning date and score columns.")
    return out


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    sub = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 3:
        return np.nan
    if sub["x"].nunique(dropna=True) < 2 or sub["y"].nunique(dropna=True) < 2:
        return np.nan
    return float(sub["x"].corr(sub["y"], method=method))


def _mean(values: pd.Series | list[float]) -> float:
    s = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return np.nan
    return float(s.mean())


def _median(values: pd.Series | list[float]) -> float:
    s = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return np.nan
    return float(s.median())


def _std(values: pd.Series | list[float]) -> float:
    s = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) <= 1:
        return np.nan
    return float(s.std(ddof=1))


def _t_stat(values: pd.Series | list[float]) -> float:
    s = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) <= 1:
        return np.nan
    sd = s.std(ddof=1)
    if sd == 0 or pd.isna(sd):
        return np.nan
    return float(s.mean() / (sd / math.sqrt(len(s))))


def _make_bucket_labels(df: pd.DataFrame, score_col: str, bucket_count: int) -> pd.Series:
    score = pd.to_numeric(df[score_col], errors="coerce")
    if score.nunique(dropna=True) <= 1:
        return pd.Series(np.nan, index=df.index)
    ranked = score.rank(method="first")
    try:
        labels = pd.qcut(ranked, q=min(bucket_count, int(ranked.nunique())), labels=False, duplicates="drop")
        return labels.astype("float64") + 1
    except ValueError:
        return pd.Series(np.nan, index=df.index)


def _score_level_summary(df: pd.DataFrame, score_col: str, target_cols: list[str]) -> pd.DataFrame:
    rows = []
    unique_scores = pd.Series(df[score_col].dropna().unique()).sort_values().tolist()
    for target in target_cols:
        sub = df[[score_col, target]].replace([np.inf, -np.inf], np.nan).dropna()
        for score_value, group in sub.groupby(score_col, sort=True):
            rows.append(
                {
                    "target": target,
                    "score_value": score_value,
                    "sample_count": int(len(group)),
                    "mean_return": _mean(group[target]),
                    "median_return": _median(group[target]),
                    "up_ratio": _mean(group[target] > 0),
                }
            )
    out = pd.DataFrame(rows)
    if len(unique_scores) > 100:
        return out.sort_values(["target", "sample_count"], ascending=[True, False]).head(1000)
    return out


def _bucket_summary(df: pd.DataFrame, score_col: str, target_cols: list[str], bucket_count: int) -> pd.DataFrame:
    labels = _make_bucket_labels(df, score_col, bucket_count)
    rows = []
    base = df.copy()
    base["score_bucket"] = labels
    base = base.dropna(subset=["score_bucket"])
    if base.empty:
        return pd.DataFrame()
    base["score_bucket"] = base["score_bucket"].astype(int)
    for target in target_cols:
        sub = base[["score_bucket", score_col, target]].replace([np.inf, -np.inf], np.nan).dropna()
        for bucket, group in sub.groupby("score_bucket", sort=True):
            rows.append(
                {
                    "target": target,
                    "score_bucket": int(bucket),
                    "sample_count": int(len(group)),
                    "score_min": float(group[score_col].min()),
                    "score_max": float(group[score_col].max()),
                    "score_mean": _mean(group[score_col]),
                    "mean_return": _mean(group[target]),
                    "median_return": _median(group[target]),
                    "up_ratio": _mean(group[target] > 0),
                }
            )
    return pd.DataFrame(rows)


def _daily_ic(df: pd.DataFrame, date_col: str, score_col: str, target_cols: list[str], min_daily_samples: int) -> pd.DataFrame:
    rows = []
    for target in target_cols:
        sub = df[[date_col, score_col, target]].replace([np.inf, -np.inf], np.nan).dropna()
        for date_value, group in sub.groupby(date_col, sort=True):
            if len(group) < min_daily_samples:
                continue
            rows.append(
                {
                    "date": date_value,
                    "target": target,
                    "sample_count": int(len(group)),
                    "spearman_ic": _safe_corr(group[score_col], group[target], "spearman"),
                    "pearson_ic": _safe_corr(group[score_col], group[target], "pearson"),
                }
            )
    return pd.DataFrame(rows)


def _daily_ic_summary(daily_ic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if daily_ic.empty:
        return pd.DataFrame()
    for target, group in daily_ic.groupby("target", sort=True):
        rows.append(
            {
                "target": target,
                "ic_days": int(len(group)),
                "spearman_ic_mean": _mean(group["spearman_ic"]),
                "spearman_ic_median": _median(group["spearman_ic"]),
                "spearman_ic_std": _std(group["spearman_ic"]),
                "spearman_ic_t_stat": _t_stat(group["spearman_ic"]),
                "spearman_ic_positive_day_ratio": _mean(group["spearman_ic"] > 0),
                "pearson_ic_mean": _mean(group["pearson_ic"]),
                "pearson_ic_median": _median(group["pearson_ic"]),
                "pearson_ic_t_stat": _t_stat(group["pearson_ic"]),
                "pearson_ic_positive_day_ratio": _mean(group["pearson_ic"] > 0),
            }
        )
    return pd.DataFrame(rows)


def _topn_for_one_day(group: pd.DataFrame, score_col: str, target: str, top_n: int, threshold: float) -> pd.DataFrame:
    sub = group[group[score_col] >= threshold].copy()
    if sub.empty:
        return sub
    return sub.sort_values([score_col, target], ascending=[False, False]).head(top_n)


def _threshold_sweep(
    df: pd.DataFrame,
    date_col: str,
    score_col: str,
    target_cols: list[str],
    thresholds: list[float],
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows = []
    summary_rows = []
    all_dates = pd.Index(sorted(df[date_col].dropna().unique()))
    total_days = len(all_dates)
    for target in target_cols:
        usable = df[[date_col, score_col, target]].replace([np.inf, -np.inf], np.nan).dropna()
        full_daily = usable.groupby(date_col, sort=True)[target].agg(full_pool_mean_return="mean", full_pool_count="count")
        for threshold in thresholds:
            selected_parts = []
            for date_value, group in usable.groupby(date_col, sort=True):
                top_part = _topn_for_one_day(group, score_col, target, top_n, threshold)
                if not top_part.empty:
                    top_part = top_part.copy()
                    top_part["date"] = date_value
                    selected_parts.append(top_part)
            if selected_parts:
                selected = pd.concat(selected_parts, ignore_index=True)
                top_daily = selected.groupby(date_col, sort=True).agg(
                    topn_mean_return=(target, "mean"),
                    topn_count=(target, "count"),
                    topn_mean_score=(score_col, "mean"),
                    topn_min_score=(score_col, "min"),
                    topn_max_score=(score_col, "max"),
                )
            else:
                selected = pd.DataFrame(columns=usable.columns)
                top_daily = pd.DataFrame(
                    columns=["topn_mean_return", "topn_count", "topn_mean_score", "topn_min_score", "topn_max_score"]
                )
            joined = full_daily.join(top_daily, how="left")
            joined["is_active_day"] = joined["topn_count"].notna()
            joined["excess_vs_full_pool"] = joined["topn_mean_return"] - joined["full_pool_mean_return"]
            joined["beat_full_pool"] = joined["topn_mean_return"] > joined["full_pool_mean_return"]
            joined = joined.reset_index().rename(columns={date_col: "date"})
            active = joined[joined["is_active_day"]].copy()
            for _, row in joined.iterrows():
                daily_rows.append(
                    {
                        "target": target,
                        "threshold": threshold,
                        "date": row["date"],
                        "is_active_day": bool(row["is_active_day"]),
                        "topn_count": row.get("topn_count", np.nan),
                        "topn_mean_score": row.get("topn_mean_score", np.nan),
                        "topn_mean_return": row.get("topn_mean_return", np.nan),
                        "full_pool_count": row.get("full_pool_count", np.nan),
                        "full_pool_mean_return": row.get("full_pool_mean_return", np.nan),
                        "excess_vs_full_pool": row.get("excess_vs_full_pool", np.nan),
                        "beat_full_pool": row.get("beat_full_pool", np.nan),
                    }
                )
            if active.empty:
                summary_rows.append(
                    {
                        "target": target,
                        "threshold": threshold,
                        "top_n": top_n,
                        "total_days": total_days,
                        "active_days": 0,
                        "empty_days": total_days,
                        "active_day_ratio": 0.0,
                        "avg_topn_count_on_active_days": np.nan,
                        "topn_daily_avg_return": np.nan,
                        "full_pool_same_days_avg_return": np.nan,
                        "daily_excess_vs_full_pool": np.nan,
                        "beat_full_pool_day_ratio": np.nan,
                        "weighted_topn_return": np.nan,
                        "weighted_full_pool_same_days_return": np.nan,
                        "weighted_excess_vs_full_pool": np.nan,
                    }
                )
                continue
            active_dates = pd.Index(active["date"].unique())
            full_same_days_rows = usable[usable[date_col].isin(active_dates)]
            summary_rows.append(
                {
                    "target": target,
                    "threshold": threshold,
                    "top_n": top_n,
                    "total_days": total_days,
                    "active_days": int(len(active)),
                    "empty_days": int(total_days - len(active)),
                    "active_day_ratio": float(len(active) / total_days) if total_days else np.nan,
                    "avg_topn_count_on_active_days": _mean(active["topn_count"]),
                    "topn_daily_avg_return": _mean(active["topn_mean_return"]),
                    "full_pool_same_days_avg_return": _mean(active["full_pool_mean_return"]),
                    "daily_excess_vs_full_pool": _mean(active["excess_vs_full_pool"]),
                    "beat_full_pool_day_ratio": _mean(active["beat_full_pool"]),
                    "weighted_topn_return": _mean(selected[target]) if not selected.empty else np.nan,
                    "weighted_full_pool_same_days_return": _mean(full_same_days_rows[target]),
                    "weighted_excess_vs_full_pool": _mean(selected[target]) - _mean(full_same_days_rows[target]) if not selected.empty else np.nan,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(daily_rows)


def _quality_summary(
    df: pd.DataFrame,
    score_col: str,
    target_cols: list[str],
    bucket_summary: pd.DataFrame,
    daily_ic_summary: pd.DataFrame,
    threshold_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for target in target_cols:
        sub = df[[score_col, target]].replace([np.inf, -np.inf], np.nan).dropna()
        bucket = bucket_summary[bucket_summary["target"] == target].sort_values("score_bucket") if not bucket_summary.empty else pd.DataFrame()
        ic = daily_ic_summary[daily_ic_summary["target"] == target] if not daily_ic_summary.empty else pd.DataFrame()
        sweep = threshold_summary[threshold_summary["target"] == target] if not threshold_summary.empty else pd.DataFrame()
        pooled_spearman = _safe_corr(sub[score_col], sub[target], "spearman")
        pooled_pearson = _safe_corr(sub[score_col], sub[target], "pearson")
        top_bucket_return = np.nan
        bottom_bucket_return = np.nan
        top_minus_bottom = np.nan
        monotonic_up_ratio = np.nan
        best_bucket = np.nan
        top_bucket_is_best = np.nan
        if not bucket.empty:
            bottom = bucket.iloc[0]
            top = bucket.iloc[-1]
            bottom_bucket_return = bottom["mean_return"]
            top_bucket_return = top["mean_return"]
            top_minus_bottom = top_bucket_return - bottom_bucket_return
            diffs = bucket["mean_return"].diff().dropna()
            monotonic_up_ratio = _mean(diffs > 0)
            best_bucket = int(bucket.sort_values("mean_return", ascending=False).iloc[0]["score_bucket"])
            top_bucket_is_best = bool(best_bucket == int(bucket["score_bucket"].max()))
        best_threshold = np.nan
        best_threshold_excess = np.nan
        best_threshold_beat_ratio = np.nan
        if not sweep.empty:
            ranked = sweep.sort_values(
                ["daily_excess_vs_full_pool", "beat_full_pool_day_ratio", "active_days"],
                ascending=[False, False, False],
            )
            best = ranked.iloc[0]
            best_threshold = best["threshold"]
            best_threshold_excess = best["daily_excess_vs_full_pool"]
            best_threshold_beat_ratio = best["beat_full_pool_day_ratio"]
        row = {
            "target": target,
            "sample_count": int(len(sub)),
            "score_col": score_col,
            "pooled_spearman_ic": pooled_spearman,
            "pooled_pearson_ic": pooled_pearson,
            "daily_spearman_ic_mean": ic.iloc[0]["spearman_ic_mean"] if not ic.empty else np.nan,
            "daily_spearman_ic_t_stat": ic.iloc[0]["spearman_ic_t_stat"] if not ic.empty else np.nan,
            "daily_spearman_ic_positive_day_ratio": ic.iloc[0]["spearman_ic_positive_day_ratio"] if not ic.empty else np.nan,
            "bottom_bucket_return": bottom_bucket_return,
            "top_bucket_return": top_bucket_return,
            "top_minus_bottom_return": top_minus_bottom,
            "bucket_monotonic_up_ratio": monotonic_up_ratio,
            "best_bucket": best_bucket,
            "top_bucket_is_best": top_bucket_is_best,
            "best_threshold": best_threshold,
            "best_threshold_daily_excess": best_threshold_excess,
            "best_threshold_beat_full_pool_day_ratio": best_threshold_beat_ratio,
        }
        row["score_quality_label"] = _score_quality_label(row)
        rows.append(row)
    return pd.DataFrame(rows)


def _score_quality_label(row: dict) -> str:
    ic_mean = row.get("daily_spearman_ic_mean", np.nan)
    ic_pos = row.get("daily_spearman_ic_positive_day_ratio", np.nan)
    spread = row.get("top_minus_bottom_return", np.nan)
    mono = row.get("bucket_monotonic_up_ratio", np.nan)
    best_excess = row.get("best_threshold_daily_excess", np.nan)
    beat_ratio = row.get("best_threshold_beat_full_pool_day_ratio", np.nan)
    strong_votes = 0
    if pd.notna(ic_mean) and ic_mean > 0:
        strong_votes += 1
    if pd.notna(ic_pos) and ic_pos >= 0.55:
        strong_votes += 1
    if pd.notna(spread) and spread > 0:
        strong_votes += 1
    if pd.notna(mono) and mono >= 0.55:
        strong_votes += 1
    if pd.notna(best_excess) and best_excess > 0:
        strong_votes += 1
    if pd.notna(beat_ratio) and beat_ratio >= 0.5:
        strong_votes += 1
    if strong_votes >= 5:
        return "strong_score"
    if strong_votes >= 4:
        return "usable_score"
    if strong_votes >= 3:
        return "weak_score"
    return "not_supported"


def _report_text(args: argparse.Namespace, files: dict[str, Path], quality: pd.DataFrame) -> str:
    lines = []
    lines.append("Score Quality Validation Report")
    lines.append("=" * 80)
    lines.append(f"pool_path: {args.pool_path}")
    lines.append(f"score_col: {args.score_col}")
    lines.append(f"date_col: {args.date_col}")
    lines.append(f"target_cols: {', '.join(args.target_cols)}")
    lines.append(f"top_n: {args.top_n}")
    lines.append(f"thresholds: {', '.join(str(x) for x in args.thresholds)}")
    lines.append("")
    if quality.empty:
        lines.append("No quality summary generated.")
    else:
        keep = [
            "target",
            "score_quality_label",
            "daily_spearman_ic_mean",
            "daily_spearman_ic_positive_day_ratio",
            "top_minus_bottom_return",
            "bucket_monotonic_up_ratio",
            "best_threshold",
            "best_threshold_daily_excess",
            "best_threshold_beat_full_pool_day_ratio",
        ]
        lines.append(quality[keep].to_string(index=False))
    lines.append("")
    lines.append("Output files:")
    for name, path in files.items():
        lines.append(f"- {name}: {path}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate whether score ranking inside a selected pool is useful.")
    parser.add_argument("--pool-path", default=None, help="Pool parquet/csv path. If omitted, use latest *_pool.parquet under BackTest_Data/pools.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: BackTest_Data/output/score_quality_validation")
    parser.add_argument("--score-col", default=None, help="Score column. Default auto: score_rank_key, selected_score_total, selected_score, selected_score_base, score.")
    parser.add_argument("--target-cols", default=None, help="Comma separated target columns. Default auto: fwd_return_pct_T1,T2,T3.")
    parser.add_argument("--date-col", default=None, help="Date column. Default auto: date/trade_date/datetime/dt.")
    parser.add_argument("--code-col", default=None, help="Optional code column. Default auto: code/symbol/ts_code/ticker.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Top N per day for score validation.")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLD_TEXT, help="Comma separated score thresholds, e.g. 0,1,2,3,4,5.")
    parser.add_argument("--bucket-count", type=int, default=DEFAULT_BUCKET_COUNT, help="Score bucket count for pooled bucket validation.")
    parser.add_argument("--min-daily-samples", type=int, default=DEFAULT_MIN_DAILY_SAMPLES, help="Minimum rows per day for daily IC.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    ns = parser.parse_args(argv)
    try:
        project_root = _project_root()
        data_root = _infer_data_root(project_root)
        pool_path = _to_path(ns.pool_path) or _infer_latest_pool_path(data_root)
        output_dir = _to_path(ns.output_dir) or (data_root / "output" / "score_quality_validation")
        if not pool_path.exists():
            raise ScoreValidationError(f"pool path not found: {pool_path}")
        raw = _read_table(pool_path)
        date_col = _infer_date_col(raw, ns.date_col)
        code_col = _infer_code_col(raw, ns.code_col)
        score_col = _infer_score_col(raw, ns.score_col)
        if str(score_col).startswith(DEFAULT_TARGET_PREFIX):
            raise ScoreValidationError(f"score_col cannot be a future return column: {score_col}")
        target_cols = _infer_target_cols(raw, ns.target_cols)
        thresholds = _parse_thresholds(ns.thresholds)
        df = _clean_data(raw, date_col, score_col, target_cols)
        ns.pool_path = str(pool_path)
        ns.output_dir = str(output_dir)
        ns.score_col = score_col
        ns.date_col = date_col
        ns.code_col = code_col
        ns.target_cols = target_cols
        ns.thresholds = thresholds
        score_level = _score_level_summary(df, score_col, target_cols)
        bucket = _bucket_summary(df, score_col, target_cols, ns.bucket_count)
        daily_ic = _daily_ic(df, date_col, score_col, target_cols, ns.min_daily_samples)
        daily_ic_sum = _daily_ic_summary(daily_ic)
        threshold_sum, threshold_daily = _threshold_sweep(df, date_col, score_col, target_cols, thresholds, ns.top_n)
        quality = _quality_summary(df, score_col, target_cols, bucket, daily_ic_sum, threshold_sum)
        files = {
            "score_quality_summary": output_dir / "score_quality_summary.csv",
            "score_level_summary": output_dir / "score_level_summary.csv",
            "score_bucket_summary": output_dir / "score_bucket_summary.csv",
            "daily_ic": output_dir / "score_daily_ic.csv",
            "daily_ic_summary": output_dir / "score_daily_ic_summary.csv",
            "threshold_sweep_summary": output_dir / "score_threshold_sweep_summary.csv",
            "threshold_sweep_daily": output_dir / "score_threshold_sweep_daily.csv",
            "report": output_dir / "score_quality_report.txt",
        }
        _write_table(quality, files["score_quality_summary"])
        _write_table(score_level, files["score_level_summary"])
        _write_table(bucket, files["score_bucket_summary"])
        _write_table(daily_ic, files["daily_ic"])
        _write_table(daily_ic_sum, files["daily_ic_summary"])
        _write_table(threshold_sum, files["threshold_sweep_summary"])
        _write_table(threshold_daily, files["threshold_sweep_daily"])
        report = _report_text(ns, files, quality)
        files["report"].parent.mkdir(parents=True, exist_ok=True)
        files["report"].write_text(report, encoding="utf-8")
        print(report)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
