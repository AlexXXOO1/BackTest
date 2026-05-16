from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.path_manager import DATA_ROOT, POOLS_DIR, MARKET_CACHE_DIR

DEFAULT_DATA_ROOT = DATA_ROOT
DEFAULT_POOL_PATH = POOLS_DIR / "renko_chart_select_strategy_v0_pool.parquet"
DEFAULT_MARKET_CACHE_DIR = MARKET_CACHE_DIR

DATE_CANDIDATES = ("date", "trade_date", "datetime")
SYMBOL_CANDIDATES = ("symbol", "code", "ticker", "ts_code")
OPEN_CANDIDATES = ("open", "open_price")
CLOSE_CANDIDATES = ("close", "close_price")

T2_RETURN_COL = "_t1_open_to_t2_close_pct"
T3_RETURN_COL = "_t1_open_to_t3_close_pct"

FACTOR_EXCLUDE_COLUMNS = {
    "date",
    "trade_date",
    "datetime",
    "symbol",
    "code",
    "ticker",
    "ts_code",
    "file",
    "selection_strategy",
    "selected",
    "selected_score_base",
    "score_rank_key",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
}

FACTOR_EXCLUDE_PATTERNS = (
    "fwd",
    "forward",
    "future",
    "target",
    "next_",
    "t1_",
    "t2_",
    "t3_",
    "t4_",
)


def render_run_progress(slot, percent: int, label: str) -> None:
    value = max(0, min(100, int(percent)))
    degrees = value * 3.6
    safe_label = html.escape(str(label))

    slot.markdown(
        f"""
<style>
[data-testid="stToolbar"] {{
    visibility: hidden;
}}

.run-progress-widget {{
    position: fixed;
    top: 12px;
    right: 24px;
    z-index: 999999;
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.94);
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08);
    backdrop-filter: blur(8px);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

.run-progress-ring {{
    width: 58px;
    height: 58px;
    border-radius: 50%;
    background: conic-gradient(#59636e {degrees}deg, #edf0f2 0deg);
    position: relative;
}}

.run-progress-ring::after {{
    content: "";
    position: absolute;
    inset: 7px;
    border-radius: 50%;
    background: #ffffff;
}}

.run-progress-percent {{
    position: absolute;
    inset: 0;
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #2f3437;
    font-size: 16px;
    font-weight: 700;
}}

.run-progress-meta {{
    min-width: 116px;
    line-height: 1.2;
}}

.run-progress-title {{
    color: #2f3437;
    font-size: 15px;
    font-weight: 700;
}}

.run-progress-label {{
    margin-top: 3px;
    color: #6b7280;
    font-size: 12px;
    white-space: nowrap;
}}
</style>

<div class="run-progress-widget">
    <div class="run-progress-ring">
        <div class="run-progress-percent">{value}%</div>
    </div>
    <div class="run-progress-meta">
        <div class="run-progress-title">Run Progress</div>
        <div class="run-progress-label">{safe_label}</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def normalize_column_name(value: object) -> str:
    return str(value).strip().lower()


def find_column(columns: Iterable[object], candidates: Iterable[str]) -> Optional[str]:
    normalized = {normalize_column_name(col): str(col) for col in columns}
    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def normalize_symbol(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().upper()
    if not text:
        return ""

    parts = re.split(r"[._\- ]+", text)
    for part in parts:
        digits = re.sub(r"\D", "", part)
        if len(digits) >= 6:
            return digits[-6:]

    digits = re.sub(r"\D", "", text)
    if len(digits) >= 6:
        return digits[-6:]

    return text


def parse_date_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    compact = text.str.replace(r"\D", "", regex=True)

    compact_dates = pd.to_datetime(
        compact.where(compact.str.len() == 8),
        format="%Y%m%d",
        errors="coerce",
    )
    generic_dates = pd.to_datetime(text, errors="coerce")

    return generic_dates.fillna(compact_dates).dt.normalize()


def parse_numeric_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


def parse_boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    text = series.astype("string").str.strip().str.lower()
    mapped = text.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "y": True,
            "n": False,
        }
    )

    numeric = pd.to_numeric(text, errors="coerce")
    numeric_mask = numeric.isin([0, 1])
    fill_mask = mapped.isna() & numeric_mask
    mapped.loc[fill_mask] = numeric.loc[fill_mask].astype(bool)

    return mapped.astype("boolean")


def infer_factor_kind(series: pd.Series) -> Optional[str]:
    raw_count = int(series.notna().sum())
    if raw_count == 0:
        return None

    bool_series = parse_boolean_series(series)
    bool_count = int(bool_series.notna().sum())
    bool_unique_count = int(bool_series.dropna().nunique())

    if bool_count > 0 and bool_unique_count <= 2 and bool_count >= max(1, int(raw_count * 0.9)):
        return "boolean"

    numeric = pd.to_numeric(series, errors="coerce")
    if int(numeric.notna().sum()) > 0:
        return "numeric"

    return None


@st.cache_data(show_spinner=False)
def load_pool(path_text: str, use_selected_only: bool) -> pd.DataFrame:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Pool file not found: {path}")

    df = pd.read_parquet(path)
    date_col = find_column(df.columns, DATE_CANDIDATES)
    symbol_col = find_column(df.columns, SYMBOL_CANDIDATES)

    if date_col is None:
        raise ValueError(f"Pool date column not found. available_columns={list(df.columns)}")
    if symbol_col is None:
        raise ValueError(f"Pool symbol column not found. available_columns={list(df.columns)}")

    if use_selected_only and "selected" in df.columns:
        selected_numeric = pd.to_numeric(df["selected"], errors="coerce")
        df = df[selected_numeric.fillna(0) > 0].copy()

    df = df.copy()
    df["_signal_date"] = parse_date_series(df[date_col])
    df["_symbol_norm"] = df[symbol_col].map(normalize_symbol)
    df = df.dropna(subset=["_signal_date"])
    df = df[df["_symbol_norm"] != ""].copy()

    return df


def get_factor_metadata(df: pd.DataFrame) -> Dict[str, str]:
    metadata: Dict[str, str] = {}

    for col in df.columns:
        col_text = str(col)
        lower_col = normalize_column_name(col_text)

        if lower_col.startswith("_"):
            continue
        if lower_col in FACTOR_EXCLUDE_COLUMNS:
            continue
        if any(pattern in lower_col for pattern in FACTOR_EXCLUDE_PATTERNS):
            continue

        factor_kind = infer_factor_kind(df[col])
        if factor_kind is not None:
            metadata[col_text] = factor_kind

    return dict(sorted(metadata.items(), key=lambda item: item[0]))


def normalize_market_frame(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    date_col = find_column(df.columns, DATE_CANDIDATES)
    open_col = find_column(df.columns, OPEN_CANDIDATES)
    close_col = find_column(df.columns, CLOSE_CANDIDATES)
    symbol_col = find_column(df.columns, SYMBOL_CANDIDATES)

    if date_col is None or open_col is None or close_col is None:
        raise ValueError(
            f"Market file missing required columns: {source_path}, available_columns={list(df.columns)}"
        )

    out = pd.DataFrame()
    out["_signal_date"] = parse_date_series(df[date_col])
    out["_open"] = parse_numeric_series(df[open_col])
    out["_close"] = parse_numeric_series(df[close_col])

    if symbol_col is not None:
        out["_symbol_norm"] = df[symbol_col].map(normalize_symbol)
    else:
        out["_symbol_norm"] = normalize_symbol(source_path.stem)

    out = out.dropna(subset=["_signal_date", "_open", "_close"])
    out = out[out["_symbol_norm"] != ""].copy()

    return out[["_signal_date", "_symbol_norm", "_open", "_close"]]


@st.cache_data(show_spinner=False)
def load_market_forward_returns(market_cache_dir_text: str) -> pd.DataFrame:
    market_cache_dir = Path(market_cache_dir_text)
    if not market_cache_dir.exists():
        raise FileNotFoundError(f"Market cache dir not found: {market_cache_dir}")

    files = sorted(path for path in market_cache_dir.rglob("*.parquet") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {market_cache_dir}")

    parts: List[pd.DataFrame] = []

    for path in files:
        source_df = pd.read_parquet(path)
        if source_df.empty:
            continue

        market_df = normalize_market_frame(source_df, path)
        if market_df.empty:
            continue

        market_df = market_df.sort_values(["_symbol_norm", "_signal_date"]).copy()
        grouped = market_df.groupby("_symbol_norm", group_keys=False)

        market_df["_t1_open"] = grouped["_open"].shift(-1)
        market_df["_t2_close"] = grouped["_close"].shift(-2)
        market_df["_t3_close"] = grouped["_close"].shift(-3)

        market_df[T2_RETURN_COL] = (market_df["_t2_close"] / market_df["_t1_open"] - 1.0) * 100.0
        market_df[T3_RETURN_COL] = (market_df["_t3_close"] / market_df["_t1_open"] - 1.0) * 100.0

        market_df = market_df.dropna(subset=["_t1_open"]).copy()

        if not market_df.empty:
            parts.append(
                market_df[
                    [
                        "_signal_date",
                        "_symbol_norm",
                        "_t1_open",
                        "_t2_close",
                        "_t3_close",
                        T2_RETURN_COL,
                        T3_RETURN_COL,
                    ]
                ]
            )

    if not parts:
        raise RuntimeError("No valid market forward return rows were built.")

    return pd.concat(parts, ignore_index=True)


def assign_quantile_bucket(series: pd.Series, bucket_count: int) -> pd.Series:
    result = pd.Series(pd.NA, index=series.index, dtype="string")
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()

    if valid.empty:
        return result

    actual_bucket_count = min(int(bucket_count), int(valid.nunique()))

    if actual_bucket_count <= 1:
        result.loc[valid.index] = "Q1"
        return result

    ranked = valid.rank(method="first")
    bucket_codes = pd.qcut(ranked, q=actual_bucket_count, labels=False, duplicates="drop")
    labels = bucket_codes.astype(int).map(lambda value: f"Q{int(value) + 1}")
    result.loc[valid.index] = labels.astype("string")

    return result


def add_factor_buckets(
    df: pd.DataFrame,
    selected_factors: List[str],
    metadata: Dict[str, str],
    bucket_count: int,
) -> pd.DataFrame:
    work_df = df.copy()
    bucket_cols: List[str] = []

    for factor in selected_factors:
        factor_kind = metadata[factor]
        bucket_col = f"__bucket__{factor}"
        bucket_cols.append(bucket_col)

        if factor_kind == "boolean":
            bool_series = parse_boolean_series(work_df[factor])
            work_df[bucket_col] = bool_series.map({True: "True", False: "False"}).astype("string")
        else:
            work_df[f"__numeric__{factor}"] = pd.to_numeric(work_df[factor], errors="coerce")
            work_df[bucket_col] = assign_quantile_bucket(work_df[f"__numeric__{factor}"], bucket_count)

    if bucket_cols:
        work_df["_bucket_group"] = work_df[bucket_cols].apply(
            lambda row: " | ".join(
                [
                    f"{selected_factors[index]}={row.iloc[index]}"
                    for index in range(len(selected_factors))
                    if pd.notna(row.iloc[index])
                ]
            ),
            axis=1,
        )
        work_df.loc[work_df["_bucket_group"] == "", "_bucket_group"] = pd.NA

    return work_df


def add_return_metrics(out: pd.DataFrame, df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    for label, return_col in [("t2", T2_RETURN_COL), ("t3", T3_RETURN_COL)]:
        grouped = df.groupby(group_cols, dropna=False)[return_col]
        metrics = grouped.agg(
            **{
                f"{label}_sample_count": "count",
                f"{label}_mean_return_pct": "mean",
                f"{label}_median_return_pct": "median",
            }
        ).reset_index()

        win_rate = (
            df.assign(_win=df[return_col] > 0)
            .groupby(group_cols, dropna=False)["_win"]
            .mean()
            .reset_index(name=f"{label}_win_rate_pct")
        )
        win_rate[f"{label}_win_rate_pct"] = win_rate[f"{label}_win_rate_pct"] * 100.0

        metrics = metrics.merge(win_rate, on=group_cols, how="left")
        out = out.merge(metrics, on=group_cols, how="left")

    return out


def aggregate_returns(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    base = df[group_cols].drop_duplicates().reset_index(drop=True)
    return add_return_metrics(base, df, group_cols)


def aggregate_overall(df: pd.DataFrame) -> Dict[str, float]:
    result: Dict[str, float] = {}

    for label, return_col in [("t2", T2_RETURN_COL), ("t3", T3_RETURN_COL)]:
        value = df[return_col]
        result[f"{label}_sample_count"] = float(value.notna().sum())
        result[f"{label}_mean_return_pct"] = float(value.mean())
        result[f"{label}_median_return_pct"] = float(value.median())
        result[f"{label}_win_rate_pct"] = float((value > 0).mean() * 100.0)

    return result


def build_daily_comparison(selected_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    selected_daily = aggregate_returns(selected_df, ["_signal_date"])
    baseline_daily = aggregate_returns(baseline_df, ["_signal_date"])

    comparison = selected_daily.merge(
        baseline_daily,
        on="_signal_date",
        how="inner",
        suffixes=("_bucket", "_baseline"),
    )

    for label in ["t2", "t3"]:
        comparison[f"{label}_excess_mean_return_pct"] = (
            comparison[f"{label}_mean_return_pct_bucket"] - comparison[f"{label}_mean_return_pct_baseline"]
        )
        comparison[f"{label}_excess_win_rate_pct"] = (
            comparison[f"{label}_win_rate_pct_bucket"] - comparison[f"{label}_win_rate_pct_baseline"]
        )

    return comparison.sort_values("_signal_date").reset_index(drop=True)


def bucket_sort_number(value: object) -> int:
    text = str(value)
    if text.startswith("Q") and text[1:].isdigit():
        return int(text[1:])
    if text == "False":
        return 1001
    if text == "True":
        return 1002
    return 9999


def sort_bucket_values(values: List[str]) -> List[str]:
    return sorted(values, key=lambda value: (bucket_sort_number(value), value))


def build_bucket_range_summary(
    df: pd.DataFrame,
    selected_factors: List[str],
    metadata: Dict[str, str],
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []

    for factor in selected_factors:
        bucket_col = f"__bucket__{factor}"
        working = df.dropna(subset=[bucket_col]).copy()
        if working.empty:
            continue

        if metadata[factor] == "numeric":
            value_col = f"__numeric__{factor}"
            working["_range_value"] = pd.to_numeric(working[value_col], errors="coerce")
            spec = (
                working.groupby(bucket_col, dropna=False)
                .agg(
                    factor_min=("_range_value", "min"),
                    factor_max=("_range_value", "max"),
                    factor_median=("_range_value", "median"),
                    sample_count=(bucket_col, "count"),
                    date_count=("_signal_date", "nunique"),
                )
                .reset_index()
                .rename(columns={bucket_col: "bucket"})
            )
            spec["bucket_label"] = spec.apply(
                lambda row: f"{row['bucket']} [{row['factor_min']:.6g}, {row['factor_max']:.6g}]",
                axis=1,
            )
        else:
            spec = (
                working.groupby(bucket_col, dropna=False)
                .agg(
                    sample_count=(bucket_col, "count"),
                    date_count=("_signal_date", "nunique"),
                )
                .reset_index()
                .rename(columns={bucket_col: "bucket"})
            )
            spec["factor_min"] = spec["bucket"]
            spec["factor_max"] = spec["bucket"]
            spec["factor_median"] = spec["bucket"]
            spec["bucket_label"] = spec["bucket"]

        metrics = aggregate_returns(working, [bucket_col]).rename(columns={bucket_col: "bucket"})
        summary = spec.merge(metrics, on="bucket", how="left")
        summary.insert(0, "factor", factor)
        summary.insert(1, "factor_type", metadata[factor])
        parts.append(summary)

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    result["_bucket_order"] = result["bucket"].astype(str).map(bucket_sort_number)
    result = result.sort_values(["factor", "_bucket_order"]).drop(columns=["_bucket_order"])
    return result.reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="Single Factor Validation", layout="wide")
    st.title("Single Factor Validation")

    progress_slot = st.empty()
    render_run_progress(progress_slot, 5, "Initializing")

    st.caption("Numeric factors use global quantile buckets over the loaded strategy pool sample.")
    st.caption("Forward returns are calculated from T+1 open to T+2 close and from T+1 open to T+3 close.")

    with st.sidebar:
        st.header("Input")
        pool_path = st.text_input("Strategy pool parquet", value=str(DEFAULT_POOL_PATH))
        market_cache_dir = st.text_input("Market cache dir", value=str(DEFAULT_MARKET_CACHE_DIR))
        use_selected_only = st.checkbox("Use selected rows only when selected column exists", value=True)

        st.header("Bucket")
        bucket_count = st.selectbox("Numeric bucket count", [5, 10, 20], index=1)

        st.header("Benchmark")
        benchmark_mode = st.selectbox("Compare with", ["Strategy pool", "Full market"])

    try:
        render_run_progress(progress_slot, 15, "Loading pool")
        pool_df = load_pool(pool_path, use_selected_only)

        render_run_progress(progress_slot, 35, "Loading market data")
        market_forward_df = load_market_forward_returns(market_cache_dir)

        render_run_progress(progress_slot, 50, "Building forward returns")
    except Exception as exc:
        render_run_progress(progress_slot, 100, "Failed")
        st.error(str(exc))
        return

    merged_pool_df = pool_df.merge(
        market_forward_df,
        on=["_signal_date", "_symbol_norm"],
        how="inner",
    )

    render_run_progress(progress_slot, 60, "Matching pool and market data")

    if merged_pool_df.empty:
        render_run_progress(progress_slot, 100, "No matched rows")
        st.error("No matched pool rows with forward returns.")
        return

    render_run_progress(progress_slot, 68, "Scanning factors")
    factor_metadata = get_factor_metadata(merged_pool_df)
    if not factor_metadata:
        render_run_progress(progress_slot, 100, "No factors")
        st.error("No numeric or boolean factor columns were found.")
        return

    factor_options = list(factor_metadata.keys())
    default_factors = ["amplitude_pct"] if "amplitude_pct" in factor_options else factor_options[:1]

    selected_factors = st.multiselect(
        "Factors",
        factor_options,
        default=default_factors,
        format_func=lambda value: f"{value} ({factor_metadata[value]})",
    )

    if not selected_factors:
        render_run_progress(progress_slot, 100, "Waiting for factors")
        st.warning("Select at least one factor.")
        return

    render_run_progress(progress_slot, 76, "Building factor buckets")
    bucketed_df = add_factor_buckets(
        merged_pool_df,
        selected_factors,
        factor_metadata,
        int(bucket_count),
    )
    bucketed_df = bucketed_df.dropna(subset=[T2_RETURN_COL, T3_RETURN_COL], how="all").copy()

    if bucketed_df.empty:
        render_run_progress(progress_slot, 100, "No bucket rows")
        st.error("No rows after bucket calculation.")
        return

    render_run_progress(progress_slot, 82, "Rendering bucket ranges")
    bucket_range_summary = build_bucket_range_summary(bucketed_df, selected_factors, factor_metadata)

    st.subheader("Global Quantile Bucket Ranges")
    st.dataframe(bucket_range_summary, use_container_width=True, hide_index=True, height=360)

    st.subheader("Bucket Selection")

    selected_bucket_filters: Dict[str, List[str]] = {}
    bucket_filter_cols = st.columns(min(len(selected_factors), 4))

    for index, factor in enumerate(selected_factors):
        bucket_col = f"__bucket__{factor}"
        available_buckets = sort_bucket_values(
            [str(value) for value in bucketed_df[bucket_col].dropna().unique()]
        )

        with bucket_filter_cols[index % len(bucket_filter_cols)]:
            selected_bucket_filters[factor] = st.multiselect(
                f"{factor} buckets",
                available_buckets,
                default=available_buckets,
                key=f"bucket_filter_{index}_{factor}",
            )

    selected_df = bucketed_df.copy()
    for factor, selected_buckets in selected_bucket_filters.items():
        bucket_col = f"__bucket__{factor}"
        selected_df = selected_df[selected_df[bucket_col].isin(selected_buckets)].copy()

    selected_df = selected_df.dropna(subset=[T2_RETURN_COL, T3_RETURN_COL], how="all").copy()

    if selected_df.empty:
        render_run_progress(progress_slot, 100, "No selected rows")
        st.error("Selected buckets have no valid rows.")
        return

    signal_dates = sorted(selected_df["_signal_date"].dropna().unique())

    if benchmark_mode == "Strategy pool":
        baseline_df = merged_pool_df[merged_pool_df["_signal_date"].isin(signal_dates)].copy()
    else:
        baseline_df = market_forward_df[market_forward_df["_signal_date"].isin(signal_dates)].copy()

    baseline_df = baseline_df.dropna(subset=[T2_RETURN_COL, T3_RETURN_COL], how="all").copy()

    if baseline_df.empty:
        render_run_progress(progress_slot, 100, "No benchmark rows")
        st.error("Benchmark has no valid rows.")
        return

    render_run_progress(progress_slot, 88, "Computing metrics")
    selected_metrics = aggregate_overall(selected_df)
    baseline_metrics = aggregate_overall(baseline_df)

    st.subheader("Selected Bucket Result")

    metric_cols = st.columns(6)
    metric_cols[0].metric("T2 sample count", f"{selected_metrics['t2_sample_count']:,.0f}")
    metric_cols[1].metric(
        "T2 mean return pct",
        f"{selected_metrics['t2_mean_return_pct']:.3f}",
        f"{selected_metrics['t2_mean_return_pct'] - baseline_metrics['t2_mean_return_pct']:.3f}",
    )
    metric_cols[2].metric(
        "T2 win rate pct",
        f"{selected_metrics['t2_win_rate_pct']:.2f}",
        f"{selected_metrics['t2_win_rate_pct'] - baseline_metrics['t2_win_rate_pct']:.2f}",
    )
    metric_cols[3].metric("T3 sample count", f"{selected_metrics['t3_sample_count']:,.0f}")
    metric_cols[4].metric(
        "T3 mean return pct",
        f"{selected_metrics['t3_mean_return_pct']:.3f}",
        f"{selected_metrics['t3_mean_return_pct'] - baseline_metrics['t3_mean_return_pct']:.3f}",
    )
    metric_cols[5].metric(
        "T3 win rate pct",
        f"{selected_metrics['t3_win_rate_pct']:.2f}",
        f"{selected_metrics['t3_win_rate_pct'] - baseline_metrics['t3_win_rate_pct']:.2f}",
    )

    if len(selected_factors) > 1:
        combined_summary = aggregate_returns(bucketed_df.dropna(subset=["_bucket_group"]), ["_bucket_group"])
        combined_summary = combined_summary.rename(columns={"_bucket_group": "bucket_group"})
        combined_summary = combined_summary.sort_values(
            ["t3_mean_return_pct", "t2_mean_return_pct"],
            ascending=[False, False],
        )

        st.subheader("Combined Bucket Group Summary")
        st.dataframe(combined_summary, use_container_width=True, hide_index=True, height=420)

    daily_comparison = build_daily_comparison(selected_df, baseline_df)

    st.subheader("Daily Comparison")
    chart_columns = [
        "t2_mean_return_pct_bucket",
        "t2_mean_return_pct_baseline",
        "t2_excess_mean_return_pct",
        "t3_mean_return_pct_bucket",
        "t3_mean_return_pct_baseline",
        "t3_excess_mean_return_pct",
    ]
    chart_df = daily_comparison.rename(columns={"_signal_date": "date"}).set_index("date")[chart_columns]
    st.line_chart(chart_df)

    st.dataframe(daily_comparison, use_container_width=True, hide_index=True, height=420)

    st.subheader("Selected Bucket Detail")
    bucket_cols = [f"__bucket__{factor}" for factor in selected_factors]
    numeric_cols = [
        f"__numeric__{factor}"
        for factor in selected_factors
        if f"__numeric__{factor}" in selected_df.columns
    ]

    detail_columns = (
        ["_signal_date", "_symbol_norm"]
        + selected_factors
        + numeric_cols
        + bucket_cols
        + ["_bucket_group", "_t1_open", "_t2_close", "_t3_close", T2_RETURN_COL, T3_RETURN_COL]
    )
    detail_columns = [col for col in detail_columns if col in selected_df.columns]
    detail_df = selected_df[detail_columns].sort_values(["_signal_date", "_symbol_norm"])

    render_run_progress(progress_slot, 96, "Rendering output")
    st.dataframe(detail_df, use_container_width=True, hide_index=True, height=420)

    st.download_button(
        "Download bucket range CSV",
        bucket_range_summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="single_factor_quantile_bucket_range_summary.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download daily comparison CSV",
        daily_comparison.to_csv(index=False).encode("utf-8-sig"),
        file_name="single_factor_bucket_daily_comparison.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download selected bucket detail CSV",
        detail_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="single_factor_bucket_detail.csv",
        mime="text/csv",
    )

    render_run_progress(progress_slot, 100, "Ready")


if __name__ == "__main__":
    main()
