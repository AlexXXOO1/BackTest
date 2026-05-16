
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Standalone Streamlit page for extreme bucket refinement.

This page intentionally does not modify or depend on the main Pool Dashboard page.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import platform as _platform
    _platform.system = lambda: "Windows"
    _platform.machine = lambda: "AMD64"
    _platform.processor = lambda: "AMD64"
except Exception:
    pass

import pandas as pd
import streamlit as st

from core.path_manager import DATA_ROOT, POOLS_DIR
from analysis.analyze_extreme_bucket_refinement import (
    TARGET_COLS_DEFAULT,
    analyze_extreme_bucket_refinement,
)


IDENTITY_OR_NON_FACTOR_COLUMNS = {
    "symbol",
    "code",
    "file",
    "date",
    "selection_strategy",
    "risk_tags",
    "j_condition_rule",
    "j_condition_source_col",
    "v4_hint_label",
    "forward_data_status",
    "selected",
    "selected_score_base",
    "score",
    "score_rank_key",
    "score_pct",
}


def _list_pool_files() -> list[Path]:
    if not POOLS_DIR.exists():
        return []
    return sorted(POOLS_DIR.glob("*_pool.parquet"))


def _read_pool_columns(pool_path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(pool_path).schema_arrow.names)
    except Exception:
        return list(pd.read_parquet(pool_path).columns)


def _candidate_factor_columns(columns: list[str]) -> list[str]:
    result = []
    for col in columns:
        col_str = str(col)
        if col_str in IDENTITY_OR_NON_FACTOR_COLUMNS:
            continue
        if col_str.startswith("fwd_return_pct_") or col_str.startswith("fwd_up_"):
            continue
        if col_str.endswith("_date"):
            continue
        if col_str in {
            "t1_open", "t1_close",
            "t2_open", "t2_close",
            "t3_open", "t3_close",
            "t4_open", "t4_close",
        }:
            continue
        result.append(col_str)
    return result


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    st.set_page_config(page_title="Extreme Bucket Refinement", layout="wide")
    st.title("Extreme Bucket Refinement")
    st.caption(
        "Standalone tool: split only the lowest original bucket and the highest original bucket into smaller buckets. "
        "This does not write into Analyze Pool Indicator outputs."
    )

    pool_files = _list_pool_files()
    if not pool_files:
        st.error(f"No pool parquet files found under: {POOLS_DIR}")
        return

    pool_labels = [p.name for p in pool_files]
    selected_pool_label = st.sidebar.selectbox("Pool", pool_labels, index=0)
    pool_path = pool_files[pool_labels.index(selected_pool_label)]

    columns = _read_pool_columns(pool_path)
    target_options = [c for c in TARGET_COLS_DEFAULT if c in columns]
    if not target_options:
        target_options = [c for c in columns if str(c).startswith("fwd_return_pct_")]

    factor_options = _candidate_factor_columns(columns)
    if not factor_options:
        st.error("No candidate factor columns found.")
        return

    default_factor_index = factor_options.index("amplitude_pct") if "amplitude_pct" in factor_options else 0

    factor = st.sidebar.selectbox("Factor", factor_options, index=default_factor_index)
    target_cols = st.sidebar.multiselect("Target horizons", target_options, default=target_options)
    side = st.sidebar.selectbox("Extreme side", ["both", "lowest", "highest"], index=0)
    bucket_count = st.sidebar.number_input("Original bucket count", min_value=2, max_value=100, value=10, step=1)
    sub_bucket_count = st.sidebar.number_input("Sub bucket count", min_value=2, max_value=20, value=5, step=1)
    min_samples = st.sidebar.number_input("Min samples", min_value=1, max_value=1_000_000, value=10000, step=1000)

    output_dir = DATA_ROOT / "output" / "extreme_bucket_refinement" / pool_path.stem / factor

    st.write("Pool:", str(pool_path))
    st.write("Output:", str(output_dir))

    run = st.button("Run extreme bucket refinement", type="primary")

    if run:
        with st.spinner("Running extreme bucket refinement..."):
            paths = analyze_extreme_bucket_refinement(
                pool_path=pool_path,
                factor=factor,
                target_cols=target_cols,
                output_dir=output_dir,
                bucket_count=int(bucket_count),
                sub_bucket_count=int(sub_bucket_count),
                min_samples=int(min_samples),
                include_lowest=side in {"both", "lowest"},
                include_highest=side in {"both", "highest"},
            )
        st.success("Completed.")
        st.write(paths)

    summary_path = output_dir / "extreme_bucket_refinement_summary.csv"
    detail_path = output_dir / "extreme_bucket_refinement_detail.csv"

    summary = _load_csv(summary_path)
    detail = _load_csv(detail_path)

    if summary.empty and detail.empty:
        st.info("No output yet. Click Run to generate standalone refinement results.")
        return

    st.subheader("Source extreme bucket summary")
    st.dataframe(summary, width="stretch", hide_index=True)

    st.subheader("5-sub-bucket detail")
    st.dataframe(detail, width="stretch", hide_index=True, height=520)

    st.download_button(
        "Download summary CSV",
        data=summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="extreme_bucket_refinement_summary.csv",
        mime="text/csv",
        disabled=summary.empty,
    )

    st.download_button(
        "Download detail CSV",
        data=detail.to_csv(index=False).encode("utf-8-sig"),
        file_name="extreme_bucket_refinement_detail.csv",
        mime="text/csv",
        disabled=detail.empty,
    )


if __name__ == "__main__":
    main()
