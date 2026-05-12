from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st

DEFAULT_DATA_ROOT = Path(r"C:\Users\zyf37\Desktop\BackTest_Data")
DEFAULT_POOL_PATH = DEFAULT_DATA_ROOT / "pools" / "renko_chart_select_strategy_v4_pool.parquet"
DEFAULT_MARKET_CACHE_DIR = DEFAULT_DATA_ROOT / "market_cache" / "daily_bars_by_symbol"

DATE_CANDIDATES = ("date", "trade_date", "datetime")
SYMBOL_CANDIDATES = ("symbol", "code", "ticker", "ts_code")
OPEN_CANDIDATES = ("open", "open_price")
CLOSE_CANDIDATES = ("close", "close_price")

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


def get_factor_candidates(df: pd.DataFrame) -> List[str]:
    result: List[str] = []

    for col in df.columns:
        col_text = str(col)
        lower_col = normalize_column_name(col_text)

        if lower_col.startswith("_"):
            continue
        if lower_col in FACTOR_EXCLUDE_COLUMNS:
            continue
        if any(pattern in lower_col for pattern in FACTOR_EXCLUDE_PATTERNS):
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() > 0:
            result.append(col_text)

    return sorted(result)


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
        market_df["_t3_close"] = grouped["_close"].shift(-3)
        market_df["_t1_open_to_t3_close_pct"] = (
            market_df["_t3_close"] / market_df["_t1_open"] - 1.0
        ) * 100.0

        market_df = market_df.dropna(
            subset=["_t1_open", "_t3_close", "_t1_open_to_t3_close_pct"]
        ).copy()

        if not market_df.empty:
            parts.append(
                market_df[
                    [
                        "_signal_date",
                        "_symbol_norm",
                        "_t1_open",
                        "_t3_close",
                        "_t1_open_to_t3_close_pct",
                    ]
                ]
            )

    if not parts:
        raise RuntimeError("No valid market forward return rows were built.")

    return pd.concat(parts, ignore_index=True)


def apply_factor_selection(
    df: pd.DataFrame,
    factor_directions: Dict[str, str],
    selection_mode: str,
    top_n: int,
    top_pct: float,
) -> pd.DataFrame:
    work_df = df.copy()
    score_cols: List[str] = []

    for factor, direction in factor_directions.items():
        numeric = pd.to_numeric(work_df[factor], errors="coerce")
        work_df[f"__factor_value__{factor}"] = numeric

        ascending = direction == "higher"
        score_col = f"__factor_score__{factor}"

        work_df[score_col] = work_df.groupby("_signal_date")[f"__factor_value__{factor}"].rank(
            pct=True,
            ascending=ascending,
            method="average",
        )
        score_cols.append(score_col)

    work_df["_factor_score"] = work_df[score_cols].mean(axis=1, skipna=True)
    work_df = work_df.dropna(subset=["_factor_score"]).copy()

    work_df["_factor_rank"] = work_df.groupby("_signal_date")["_factor_score"].rank(
        ascending=False,
        method="first",
    )

    daily_count = work_df.groupby("_signal_date")["_factor_score"].transform("count")

    if selection_mode == "Top N per date":
        work_df["_factor_selected"] = work_df["_factor_rank"] <= int(top_n)
    else:
        selected_count = (daily_count * float(top_pct) / 100.0).clip(lower=1).round()
        work_df["_factor_selected"] = work_df["_factor_rank"] <= selected_count

    return work_df[work_df["_factor_selected"]].copy()


def aggregate_daily(df: pd.DataFrame, label: str) -> pd.DataFrame:
    value_col = "_t1_open_to_t3_close_pct"

    grouped = df.groupby("_signal_date")[value_col]
    out = grouped.agg(
        sample_count="count",
        mean_return_pct="mean",
        median_return_pct="median",
    ).reset_index()

    out["win_rate_pct"] = (
        df.assign(_win=df[value_col] > 0)
        .groupby("_signal_date")["_win"]
        .mean()
        .reindex(out["_signal_date"])
        .to_numpy()
        * 100.0
    )
    out["group"] = label
    return out


def aggregate_overall(df: pd.DataFrame) -> Dict[str, float]:
    value = df["_t1_open_to_t3_close_pct"]
    return {
        "sample_count": float(value.notna().sum()),
        "mean_return_pct": float(value.mean()),
        "median_return_pct": float(value.median()),
        "win_rate_pct": float((value > 0).mean() * 100.0),
    }


def build_comparison_table(selected_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    selected_daily = aggregate_daily(selected_df, "factor_selection")
    baseline_daily = aggregate_daily(baseline_df, "baseline")

    comparison = selected_daily.merge(
        baseline_daily,
        on="_signal_date",
        how="inner",
        suffixes=("_factor", "_baseline"),
    )

    comparison["excess_mean_return_pct"] = (
        comparison["mean_return_pct_factor"] - comparison["mean_return_pct_baseline"]
    )
    comparison["excess_win_rate_pct"] = (
        comparison["win_rate_pct_factor"] - comparison["win_rate_pct_baseline"]
    )

    return comparison.sort_values("_signal_date").reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="Single Factor Validation", layout="wide")
    st.title("Single Factor Validation")

    with st.sidebar:
        st.header("Input")
        pool_path = st.text_input("Strategy pool parquet", value=str(DEFAULT_POOL_PATH))
        market_cache_dir = st.text_input("Market cache dir", value=str(DEFAULT_MARKET_CACHE_DIR))
        use_selected_only = st.checkbox("Use selected rows only when selected column exists", value=True)

        st.header("Selection")
        selection_mode = st.selectbox("Factor selection mode", ["Top N per date", "Top percent per date"])
        top_n = st.number_input("Top N per date", min_value=1, max_value=5000, value=20, step=1)
        top_pct = st.number_input("Top percent per date", min_value=0.1, max_value=100.0, value=20.0, step=0.5)

        st.header("Benchmark")
        benchmark_mode = st.selectbox("Compare with", ["Strategy pool", "Full market"])

    try:
        pool_df = load_pool(pool_path, use_selected_only)
        market_forward_df = load_market_forward_returns(market_cache_dir)
    except Exception as exc:
        st.error(str(exc))
        return

    merged_pool_df = pool_df.merge(
        market_forward_df,
        on=["_signal_date", "_symbol_norm"],
        how="inner",
    )

    if merged_pool_df.empty:
        st.error("No matched pool rows with forward returns.")
        return

    factor_candidates = get_factor_candidates(merged_pool_df)
    default_factors = ["amplitude_pct"] if "amplitude_pct" in factor_candidates else factor_candidates[:1]

    selected_factors = st.multiselect(
        "Factors",
        factor_candidates,
        default=default_factors,
    )

    if not selected_factors:
        st.warning("Select at least one factor.")
        return

    factor_directions: Dict[str, str] = {}
    direction_cols = st.columns(min(len(selected_factors), 4))

    for index, factor in enumerate(selected_factors):
        with direction_cols[index % len(direction_cols)]:
            direction_label = st.selectbox(
                f"{factor} direction",
                ["higher", "lower"],
                index=0,
                key=f"direction_{factor}",
            )
            factor_directions[factor] = direction_label

    selected_df = apply_factor_selection(
        merged_pool_df,
        factor_directions=factor_directions,
        selection_mode=selection_mode,
        top_n=int(top_n),
        top_pct=float(top_pct),
    )

    signal_dates = sorted(selected_df["_signal_date"].dropna().unique())

    if benchmark_mode == "Strategy pool":
        baseline_df = merged_pool_df[merged_pool_df["_signal_date"].isin(signal_dates)].copy()
    else:
        baseline_df = market_forward_df[market_forward_df["_signal_date"].isin(signal_dates)].copy()

    selected_df = selected_df.dropna(subset=["_t1_open_to_t3_close_pct"]).copy()
    baseline_df = baseline_df.dropna(subset=["_t1_open_to_t3_close_pct"]).copy()

    if selected_df.empty:
        st.error("Factor selection has no rows after forward return filtering.")
        return

    if baseline_df.empty:
        st.error("Benchmark has no rows after forward return filtering.")
        return

    selected_metrics = aggregate_overall(selected_df)
    baseline_metrics = aggregate_overall(baseline_df)

    st.subheader("Overall Result")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Factor sample count", f"{selected_metrics['sample_count']:,.0f}")
    metric_cols[1].metric(
        "Mean return pct",
        f"{selected_metrics['mean_return_pct']:.3f}",
        f"{selected_metrics['mean_return_pct'] - baseline_metrics['mean_return_pct']:.3f}",
    )
    metric_cols[2].metric(
        "Median return pct",
        f"{selected_metrics['median_return_pct']:.3f}",
        f"{selected_metrics['median_return_pct'] - baseline_metrics['median_return_pct']:.3f}",
    )
    metric_cols[3].metric(
        "Win rate pct",
        f"{selected_metrics['win_rate_pct']:.2f}",
        f"{selected_metrics['win_rate_pct'] - baseline_metrics['win_rate_pct']:.2f}",
    )

    comparison = build_comparison_table(selected_df, baseline_df)

    st.subheader("Daily Comparison")
    chart_df = comparison.rename(
        columns={
            "_signal_date": "date",
            "mean_return_pct_factor": "factor_mean_return_pct",
            "mean_return_pct_baseline": "baseline_mean_return_pct",
        }
    ).set_index("date")[
        ["factor_mean_return_pct", "baseline_mean_return_pct", "excess_mean_return_pct"]
    ]
    st.line_chart(chart_df)

    st.dataframe(comparison, use_container_width=True, hide_index=True, height=420)

    st.subheader("Factor Selection Detail")
    detail_columns = (
        ["_signal_date", "_symbol_norm"]
        + selected_factors
        + ["_factor_score", "_factor_rank", "_t1_open", "_t3_close", "_t1_open_to_t3_close_pct"]
    )
    detail_columns = [col for col in detail_columns if col in selected_df.columns]
    detail_df = selected_df[detail_columns].sort_values(["_signal_date", "_factor_rank"])
    st.dataframe(detail_df, use_container_width=True, hide_index=True, height=420)

    st.download_button(
        "Download daily comparison CSV",
        comparison.to_csv(index=False).encode("utf-8-sig"),
        file_name="single_factor_daily_comparison.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download factor selection detail CSV",
        detail_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="single_factor_selection_detail.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
