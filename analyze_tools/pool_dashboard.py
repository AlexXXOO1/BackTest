# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_POOL_PATH = Path(
    r"C:\Users\zyf37\Desktop\BackTest_Data\pools\renko_chart_select_strategy_v4_pool.parquet"
)


st.set_page_config(
    page_title="Pool Dashboard",
    layout="wide",
)


@st.cache_data(show_spinner=True)
def load_pool(pool_path: str) -> pd.DataFrame:
    path = Path(pool_path)

    if not path.exists():
        raise FileNotFoundError(f"Pool file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "selected" in df.columns:
        df = df[df["selected"].fillna(0).astype(int) == 1].copy()

    df = df.sort_values(["date", "symbol"], ascending=[False, True]).reset_index(drop=True)
    return df


def fmt_num(x, ndigits: int = 3):
    try:
        if pd.isna(x):
            return ""
        return round(float(x), ndigits)
    except Exception:
        return x


st.title("Renko Pool Dashboard")

with st.sidebar:
    st.header("Data")
    pool_path = st.text_input("Pool path", value=str(DEFAULT_POOL_PATH))

    reload_btn = st.button("Reload data")
    if reload_btn:
        st.cache_data.clear()

try:
    df = load_pool(pool_path)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if df.empty:
    st.warning("Pool is empty.")
    st.stop()

min_date = df["date"].min()
max_date = df["date"].max()

st.caption(f"Pool path: `{pool_path}`")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Trading days", f"{df['date'].nunique():,}")
c3.metric("Symbols", f"{df['symbol'].nunique():,}" if "symbol" in df.columns else "-")
c4.metric("Date range", f"{min_date.date()} → {max_date.date()}")

st.divider()

# Daily count
st.subheader("Daily pool count")

daily = (
    df.groupby("date")
    .size()
    .reset_index(name="pool_count")
    .sort_values("date")
)

st.line_chart(daily.set_index("date")["pool_count"])

# Filters
st.subheader("Filter")

fc1, fc2, fc3, fc4 = st.columns(4)

all_dates = sorted(df["date"].dropna().dt.date.unique(), reverse=True)
target_date = fc1.selectbox("Date", all_dates, index=0)

symbol_query = fc2.text_input("Symbol contains", value="").strip()

if "v4_hint_label" in df.columns:
    hint_options = ["ALL"] + sorted(df["v4_hint_label"].dropna().astype(str).unique().tolist())
    hint_label = fc3.selectbox("Hint label", hint_options, index=0)
else:
    hint_label = "ALL"
    fc3.write("No v4_hint_label")

sort_candidates = [
    "score_rank_key",
    "score_pct",
    "v4_net_hint_score",
    "v4_up_hint_score",
    "v4_risk_hint_score",
    "daily_return_pct",
    "v4_close_to_ma5",
    "v4_brk",
    "v4_crh",
    "v4_pgh",
    "volume_ratio_prev1",
]
sort_candidates = [c for c in sort_candidates if c in df.columns]
sort_col = fc4.selectbox("Sort by", sort_candidates if sort_candidates else ["date"])

view = df[df["date"].dt.date == target_date].copy()

if symbol_query and "symbol" in view.columns:
    view = view[view["symbol"].astype(str).str.contains(symbol_query, case=False, na=False)].copy()

if hint_label != "ALL" and "v4_hint_label" in view.columns:
    view = view[view["v4_hint_label"].astype(str) == hint_label].copy()

if sort_col in view.columns:
    view = view.sort_values(sort_col, ascending=False)

st.divider()

# Target date summary
st.subheader(f"Selected rows on {target_date}")

sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Rows", f"{len(view):,}")
if "v4_hint_label" in view.columns:
    sc2.metric("Up potential", f"{(view['v4_hint_label'] == 'up_potential').sum():,}")
    sc3.metric("Risk", f"{(view['v4_hint_label'] == 'risk').sum():,}")
    sc4.metric("Neutral", f"{(view['v4_hint_label'] == 'neutral').sum():,}")
else:
    sc2.metric("Hint labels", "-")
    sc3.metric("Risk", "-")
    sc4.metric("Neutral", "-")

if "v4_hint_label" in view.columns and not view.empty:
    st.bar_chart(view["v4_hint_label"].value_counts())

default_cols = [
    "date",
    "symbol",
    "close",
    "daily_return_pct",
    "v4_close_to_ma5",
    "v4_brk",
    "v4_crh",
    "v4_pgh",
    "v4_up_hint_score",
    "v4_risk_hint_score",
    "v4_net_hint_score",
    "v4_hint_label",
    "volume_ratio_prev1",
    "score_rank_key",
    "score_pct",
]
default_cols = [c for c in default_cols if c in view.columns]

with st.expander("Choose columns", expanded=False):
    selected_cols = st.multiselect(
        "Columns",
        options=view.columns.tolist(),
        default=default_cols,
    )

if not selected_cols:
    selected_cols = default_cols if default_cols else view.columns.tolist()

display_df = view[selected_cols].copy()

for c in display_df.columns:
    if pd.api.types.is_float_dtype(display_df[c]):
        display_df[c] = display_df[c].map(lambda x: fmt_num(x, 4))

st.dataframe(
    display_df,
    use_container_width=True,
    height=520,
)

st.download_button(
    label="Download current view as CSV",
    data=display_df.to_csv(index=False, encoding="utf-8-sig"),
    file_name=f"pool_view_{target_date}.csv",
    mime="text/csv",
)

st.divider()

# Single symbol detail
st.subheader("Symbol detail")

if "symbol" in df.columns:
    symbols = sorted(df["symbol"].dropna().astype(str).unique().tolist())
    selected_symbol = st.selectbox("Symbol", symbols, index=0)

    one = df[df["symbol"].astype(str) == selected_symbol].copy()
    one = one.sort_values("date")

    chart_cols = [
        "close",
        "renko_value",
        "v4_brk",
        "daily_return_pct",
        "v4_close_to_ma5",
        "v4_net_hint_score",
    ]
    chart_cols = [c for c in chart_cols if c in one.columns]

    if chart_cols:
        st.line_chart(one.set_index("date")[chart_cols])

    detail_cols = [
        "date",
        "close",
        "daily_return_pct",
        "v4_close_to_ma5",
        "v4_brk",
        "v4_crh",
        "v4_pgh",
        "v4_up_hint_score",
        "v4_risk_hint_score",
        "v4_net_hint_score",
        "v4_hint_label",
    ]
    detail_cols = [c for c in detail_cols if c in one.columns]

    st.dataframe(one[detail_cols].sort_values("date", ascending=False), use_container_width=True, height=360)
else:
    st.info("No symbol column found.")
