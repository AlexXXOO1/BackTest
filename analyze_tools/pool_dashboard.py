# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import html
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(r"C:\Users\zyf37\Desktop\BackTest_System")
DATA_ROOT = Path(r"C:\Users\zyf37\Desktop\BackTest_Data")

DEFAULT_POOLS_DIR = DATA_ROOT / "pools"
DEFAULT_POOL_PATH = DEFAULT_POOLS_DIR / "renko_chart_select_strategy_v0_pool.parquet"
DEFAULT_MARKET_CACHE_DIR = DATA_ROOT / "market_cache" / "daily_bars_by_symbol"
DEFAULT_SH_INDEX_DIR = DATA_ROOT / "raw_SH_index"

DEFAULT_ANALYZE_POOL_SCRIPT = PROJECT_ROOT / "analyze_tools" / "analyze_pool_indicator_direction.py"
DEFAULT_ANALYZE_POOL_OUTPUT_DIR = DATA_ROOT / "output" / "analyze_pool_indicator_dashboard_v3"

st.set_page_config(
    page_title="Pool Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================================================
# UI helpers
# ======================================================================================


def inject_style() -> None:
    st.markdown(
        """
        <style>

            section[data-testid="stSidebar"] {
                background: #f8fafc;
                border-right: 1px solid rgba(148, 163, 184, 0.22);
            }
            .page-hero {
                display: flex;
                justify-content: space-between;
                align-items: flex-end;
                gap: 18px;
            }
            .pool-pill {
                display: inline-flex;
                align-items: center;
                padding: 8px 13px;
                border-radius: 999px;
                background: #fff7ed;
                border: 1px solid #fed7aa;
                color: #9a3412;
                font-size: 0.88rem;
                font-weight: 650;
                white-space: nowrap;
            }
            .block-container {
                padding-top: 3rem !important;
                padding-bottom: 2.25rem;
                max-width: 1500px;
            }
            h1, h2, h3 {
                letter-spacing: -0.02em;
            }
            div[data-testid="stMetric"] {
                background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.96));
                border: 1px solid rgba(148, 163, 184, 0.28);
                border-radius: 16px;
                padding: 14px 16px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            }
            .auto-card {
                min-height: 108px;
                border-radius: 18px;
                padding: 18px 20px;
                background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
                border: 1px solid rgba(148, 163, 184, 0.30);
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.055);
                overflow: visible;
            }
            .auto-card-label {
                color: #475569;
                font-size: 0.92rem;
                font-weight: 620;
                margin-bottom: 10px;
            }
            .auto-card-value {
                color: #0f172a;
                font-weight: 780;
                line-height: 1.22;
                font-size: clamp(0.78rem, 1.15vw, 1.18rem);
                white-space: normal;
                overflow-wrap: anywhere;
                word-break: break-word;
            }
            div[data-testid="stMetricLabel"] {
                color: #64748b;
                font-size: 0.82rem;
            }
            div[data-testid="stMetricValue"] {
                font-size: 1.28rem;
                font-weight: 700;
            }
            .page-hero {
                padding: 12px 0 18px 0;
                margin-top: 12px;
                overflow: visible;
                border-bottom: 1px solid rgba(148, 163, 184, 0.22);
                margin-bottom: 16px;
            }
            .page-title {
                font-size: 2.0rem;
                font-weight: 760;
                line-height: 1.35;
                padding-top: 6px;
                overflow: visible;
            }
            .page-caption {
                color: #64748b;
                font-size: 0.98rem;
                margin-top: 6px;
            }
            .section-title {
                font-size: 1.18rem;
                font-weight: 720;
                margin: 4px 0 2px 0;
            }
            .section-caption {
                color: #64748b;
                font-size: 0.90rem;
                margin-bottom: 8px;
            }
            .soft-card {
                border: 1px solid rgba(148, 163, 184, 0.25);
                border-radius: 16px;
                padding: 12px 16px;
                background: rgba(248,250,252,0.68);
                margin: 8px 0 12px 0;
            }
            .pill {
                display: inline-block;
                padding: 3px 10px;
                margin: 0 6px 6px 0;
                border-radius: 999px;
                background: #f1f5f9;
                border: 1px solid rgba(148, 163, 184, 0.28);
                color: #334155;
                font-size: 0.82rem;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 999px;
                padding: 8px 14px;
                background: #f8fafc;
                border: 1px solid rgba(148, 163, 184, 0.22);
            }
            .stTabs [aria-selected="true"] {
                background: #e2e8f0;
                color: #0f172a;
            }
            div[data-testid="stDataFrame"] {
                border-radius: 14px;
                overflow: hidden;
                border: 1px solid rgba(148, 163, 184, 0.18);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, caption: str = "", pool_path: Path | None = None) -> None:
    pool_html = ""
    if pool_path is not None:
        pool_html = f'<div class="pool-pill">{safe_pool_name(pool_path)} {pool_path.suffix.lower().replace(".", "")}</div>'
    st.markdown(
        f"""
        <div class="page-hero">
          <div class="page-title">{title}</div>
          {pool_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, caption: str | None = None) -> None:
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def auto_card(label: str, value: Any) -> str:
    label_s = html.escape(str(label))
    value_s = html.escape(str(value))
    return f"""
    <div class="auto-card">
        <div class="auto-card-label">{label_s}</div>
        <div class="auto-card-value">{value_s}</div>
    </div>
    """

def render_pills(items: list[str], max_items: int = 8) -> None:
    if not items:
        return

    shown = items[:max_items]
    html = "".join(f'<span class="pill">{x}</span>' for x in shown)
    if len(items) > max_items:
        html += f'<span class="pill">+{len(items) - max_items}</span>'
    st.markdown(html, unsafe_allow_html=True)


def clean_display_df(df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows).copy()

    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda x: fmt_num(x, 4))
    return out


def apply_plotly_layout(
    fig: go.Figure,
    title: str,
    yaxis_title: str | None = None,
    height: int = 460,
    legend_top: bool = True,
) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=18)),
        hovermode="x unified",
        height=height,
        template="plotly_white",
        margin=dict(l=20, r=24, t=68, b=28),
        legend=dict(
            orientation="h" if legend_top else "v",
            yanchor="bottom" if legend_top else "top",
            y=1.02 if legend_top else 1,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_xaxes(showgrid=False, rangeslider=dict(visible=False))
    fig.update_yaxes(
        title_text=yaxis_title,
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.22)",
        zerolinecolor="rgba(15, 23, 42, 0.18)",
    )
    return fig


def build_line_chart_from_pivot(
    pivot: pd.DataFrame,
    title: str,
    yaxis_title: str,
    height: int = 420,
) -> go.Figure:
    fig = go.Figure()
    plot = pivot.copy().sort_index()

    for col in plot.columns:
        fig.add_trace(
            go.Scatter(
                x=plot.index,
                y=plot[col],
                mode="lines+markers",
                name=str(col),
                line=dict(width=2),
                marker=dict(size=4),
                hovertemplate="%{x|%Y-%m-%d}<br>" + str(col) + "=%{y:.4f}<extra></extra>",
            )
        )

    return apply_plotly_layout(fig, title=title, yaxis_title=yaxis_title, height=height)


inject_style()


# ======================================================================================
# Common helpers
# ======================================================================================


def normalize_date_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


@st.cache_data(show_spinner=True)
def load_pool(pool_path: str, selected_only: bool = True) -> pd.DataFrame:
    path = Path(pool_path)

    if not path.exists():
        raise FileNotFoundError(f"Pool file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported pool file type: {path.suffix}")

    df = normalize_date_col(df)

    if selected_only and "selected" in df.columns:
        selected_num = pd.to_numeric(df["selected"], errors="coerce")
        if selected_num.notna().any():
            df = df[selected_num.fillna(0).astype(int) == 1].copy()
        else:
            df = df[df["selected"].astype(bool)].copy()

    sort_cols = [c for c in ["date", "symbol", "code"] if c in df.columns]
    if sort_cols:
        ascending = [False] + [True] * (len(sort_cols) - 1)
        df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    return df


@st.cache_data(show_spinner=False)
def load_csv_if_exists(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)
    df = normalize_date_col(df)
    return df


@st.cache_data(show_spinner=False)
def load_sh_index(index_dir: str) -> pd.DataFrame:
    """
    Load SH index TXT/CSV files from the configured index directory.
    Prefer files that contain 999999, for example SH#999999.txt.
    """
    index_dir_path = Path(index_dir)
    if not index_dir_path.exists():
        return pd.DataFrame()

    files = list(index_dir_path.glob("*999999*.txt"))
    if not files:
        files = list(index_dir_path.glob("*999999*.csv"))
    if not files:
        files = list(index_dir_path.glob("*.txt")) + list(index_dir_path.glob("*.csv"))
    if not files:
        return pd.DataFrame()

    file_path = sorted(files)[0]

    text = None
    for enc in ["utf-8-sig", "gb18030", "gbk", "utf-8"]:
        try:
            text = file_path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except Exception:
            continue

    if text is None:
        return pd.DataFrame()

    rows = []
    date_patterns = [
        re.compile(r"^\d{2}/\d{2}/\d{4},"),
        re.compile(r"^\d{4}-\d{2}-\d{2},"),
        re.compile(r"^\d{8},"),
    ]

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if not any(p.match(line) for p in date_patterns):
            continue

        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5:
            continue

        while len(parts) < 7:
            parts.append("")

        rows.append(parts[:7])

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "volume", "amount"],
    )

    parsed = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    if parsed.isna().mean() > 0.8:
        parsed = pd.to_datetime(df["date"], errors="coerce")
    if parsed.isna().mean() > 0.8:
        parsed = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    df["date"] = parsed

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df["index_return_pct"] = df["close"].pct_change() * 100
    df["source_file"] = str(file_path)

    return df.reset_index(drop=True)


def fmt_num(x: Any, ndigits: int = 4) -> Any:
    try:
        if pd.isna(x):
            return ""
        return round(float(x), ndigits)
    except Exception:
        return x


def show_download(df: pd.DataFrame, file_name: str, label: str = "Download CSV") -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False, encoding="utf-8-sig"),
        file_name=file_name,
        mime="text/csv",
    )


def list_pool_files(pools_dir: Path) -> list[Path]:
    if not pools_dir.exists():
        return []

    files: list[Path] = []
    files.extend(sorted(pools_dir.glob("*.parquet")))
    files.extend(sorted(pools_dir.glob("*.csv")))
    return files


def safe_pool_name(path: Path) -> str:
    name = path.stem
    name = name.replace("_pool", "")
    name = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", name)
    return name


def format_pool_option(path: Path, include_file_name: bool = False) -> str:
    suffix = path.suffix.lower().replace(".", "")
    name = safe_pool_name(path)
    if include_file_name:
        return f"{name}  |  {path.name}  |  {suffix}"
    return f"{name} {suffix}"


def list_forward_return_targets(df: pd.DataFrame) -> list[str]:
    targets = [
        str(c)
        for c in df.columns
        if str(c).lower().startswith("fwd_return_pct_t")
    ]

    def target_sort_key(x: str) -> tuple[int, str]:
        m = re.search(r"T(\d+)$", x, flags=re.IGNORECASE)
        if m:
            return int(m.group(1)), x
        return 999, x

    return sorted(targets, key=target_sort_key)


def select_default_index(options: list[Any], preferred: Any) -> int:
    try:
        return list(options).index(preferred)
    except Exception:
        return 0


def build_pool_count_with_index_chart(
    daily: pd.DataFrame,
    sh_index_df: pd.DataFrame,
    show_sh_index: bool,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    plot_daily = daily.copy().sort_values("date")

    fig.add_trace(
        go.Bar(
            x=plot_daily["date"],
            y=plot_daily["pool_count"],
            name="All",
            opacity=0.46,
            marker_line_width=0,
            hovertemplate="%{x|%Y-%m-%d}<br>All=%{y}<extra></extra>",
        ),
        secondary_y=False,
    )

    if show_sh_index and not sh_index_df.empty and not daily.empty:
        min_date = plot_daily["date"].min()
        max_date = plot_daily["date"].max()

        sh_plot = sh_index_df[
            (sh_index_df["date"] >= min_date)
            & (sh_index_df["date"] <= max_date)
        ].copy()

        if not sh_plot.empty:
            fig.add_trace(
                go.Scatter(
                    x=sh_plot["date"],
                    y=sh_plot["close"],
                    mode="lines",
                    name="SH Index Close",
                    line=dict(color="red", width=1.8),
                    hovertemplate="%{x|%Y-%m-%d}<br>SH Index=%{y:.2f}<extra></extra>",
                ),
                secondary_y=True,
            )

    fig.update_layout(
        title=dict(text="Daily Signals + SH Index", x=0.01, xanchor="left", font=dict(size=18)),
        hovermode="x unified",
        barmode="overlay",
        bargap=0.06,
        height=580,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=28, t=72, b=30),
    )
    fig.update_xaxes(
        showgrid=False,
        rangeslider=dict(visible=False),
    )
    fig.update_yaxes(
        title_text="Pool count",
        secondary_y=False,
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.22)",
        zerolinecolor="rgba(15, 23, 42, 0.18)",
    )
    fig.update_yaxes(title_text="SH Index Close", secondary_y=True, showgrid=False)

    return fig


def build_bucket_factor_chart(bucket_df: pd.DataFrame, factor: str) -> go.Figure:
    plot = bucket_df[bucket_df["factor"].astype(str) == str(factor)].copy()
    plot = plot.sort_values("bucket")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=plot["bucket"],
            y=plot["mean_return"],
            name="Mean return",
            opacity=0.55,
            hovertemplate="bucket=%{x}<br>mean_return=%{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=plot["bucket"],
            y=plot["up_ratio"],
            name="Up ratio",
            mode="lines+markers",
            line=dict(width=2.4),
            marker=dict(size=7),
            hovertemplate="bucket=%{x}<br>up_ratio=%{y:.4f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title=dict(text=f"Bucket aggregate: {factor}", x=0.01, xanchor="left", font=dict(size=18)),
        hovermode="x unified",
        height=420,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=24, t=68, b=30),
    )
    fig.update_yaxes(title_text="Mean return", secondary_y=False, showgrid=True, gridcolor="rgba(148, 163, 184, 0.22)")
    fig.update_yaxes(title_text="Up ratio", secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Bucket", showgrid=False)
    return fig


def preferred_bucket_member_cols(df: pd.DataFrame) -> list[str]:
    base_cols = [
        "factor",
        "bucket",
        "factor_value",
        "target_col",
        "target_value",
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
    ]
    fwd_cols = sorted(
        [str(c) for c in df.columns if str(c).lower().startswith("fwd_return_pct_t")],
        key=lambda x: (
            int(re.search(r"T(\d+)$", x, flags=re.IGNORECASE).group(1))
            if re.search(r"T(\d+)$", x, flags=re.IGNORECASE)
            else 999,
            x,
        ),
    )
    preferred = [c for c in [*base_cols, *fwd_cols] if c in df.columns]
    return preferred + [c for c in df.columns if c not in preferred]


def run_subprocess(cmd: list[str], cwd: Path) -> tuple[int, str, str, float]:
    t0 = time.time()
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - t0
    return result.returncode, result.stdout, result.stderr, elapsed


# ======================================================================================
# Page 1: Single Pool Viewer
# ======================================================================================


def render_pool_viewer(pool_path_obj: Path | None) -> None:
    page_header(
        "Single Pool Viewer",
        "View one pool: daily signal count, SH index trend, and row details.",
    )

    if pool_path_obj is None:
        st.error("No pool file selected.")
        return

    control_col1, control_col2 = st.columns([1, 1])
    with control_col1:
        selected_only = st.checkbox("Only selected == 1", value=True, key="single_selected_only")
    with control_col2:
        show_sh_index = st.checkbox("Overlay SH index", value=True, key="single_show_sh_index")

    sh_index_dir = str(DEFAULT_SH_INDEX_DIR)

    try:
        df = load_pool(str(pool_path_obj), selected_only=selected_only)
    except Exception as exc:
        st.error(str(exc))
        return

    if df.empty:
        st.warning("Pool is empty.")
        return

    if "date" not in df.columns:
        st.error("Pool missing date column.")
        return

    min_date = df["date"].min()
    max_date = df["date"].max()
    symbol_col = "symbol" if "symbol" in df.columns else "code" if "code" in df.columns else None

    st.markdown(
        f"""
        <div class="soft-card">
            <b>Current pool:</b> {safe_pool_name(pool_path_obj)}<br/>
            <span style="color:#64748b;">File name: {pool_path_obj.name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Trading days", f"{df['date'].nunique():,}")
    c3.metric("Symbols", f"{df[symbol_col].nunique():,}" if symbol_col else "-")
    c4.metric("Date range", f"{min_date.date()} → {max_date.date()}")

    st.divider()
    section_header(
        "Daily pool count",
        "Daily count uses all selected rows in the current pool.",
    )

    daily = df.groupby("date").size().reset_index(name="pool_count").sort_values("date")
    sh_index_df = load_sh_index(sh_index_dir) if show_sh_index else pd.DataFrame()
    fig = build_pool_count_with_index_chart(daily, sh_index_df, show_sh_index)
    st.plotly_chart(fig, use_container_width=True)

    if show_sh_index:
        if sh_index_df.empty:
            st.warning(f"No SH index data found in: {sh_index_dir}")
        else:
            sh_plot = sh_index_df[
                (sh_index_df["date"] >= daily["date"].min())
                & (sh_index_df["date"] <= daily["date"].max())
            ].copy()
            if sh_plot.empty:
                st.warning("SH index data exists, but it does not overlap the current pool date range.")
            else:
                st.caption(
                    f"SH index date range: {sh_plot['date'].min().date()} → {sh_plot['date'].max().date()}, "
                    f"rows: {len(sh_plot)} rows"
                )

    st.divider()
    section_header("Daily rows", "Choose a date, then filter by code, label, and sort column.")
    fc1, fc2, fc3, fc4 = st.columns([1.2, 1.2, 1.2, 1.2])

    all_dates = sorted(df["date"].dropna().dt.date.unique(), reverse=True)
    target_date = fc1.selectbox("Date", all_dates, index=0)
    symbol_query = fc2.text_input("Symbol/code contains", value="").strip()

    label_col = None
    for c in ["v4_hint_label", "hint_label", "market_regime", "selection_strategy"]:
        if c in df.columns:
            label_col = c
            break

    if label_col:
        label_options = ["ALL"] + sorted(df[label_col].dropna().astype(str).unique().tolist())
        label_value = fc3.selectbox(label_col, label_options, index=0)
    else:
        label_value = "ALL"
        fc3.write("No label column")

    sort_candidates = [
        "score_rank_key",
        "score_pct",
        "selected_score_base",
        "v4_net_hint_score",
        "v4_up_hint_score",
        "v4_risk_hint_score",
        "daily_return_pct",
        "v4_close_to_ma5",
        "long_pos_21",
        "volume_ratio_ma5",
        "volume_ratio_prev1",
        "date",
    ]
    sort_candidates = [c for c in sort_candidates if c in df.columns]
    sort_col = fc4.selectbox("Sort by", sort_candidates if sort_candidates else ["date"])

    view = df[df["date"].dt.date == target_date].copy()

    if symbol_query:
        matched = pd.Series(False, index=view.index)
        for c in ["symbol", "code"]:
            if c in view.columns:
                matched = matched | view[c].astype(str).str.contains(symbol_query, case=False, na=False)
        view = view[matched].copy()

    if label_col and label_value != "ALL":
        view = view[view[label_col].astype(str) == label_value].copy()

    if sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=False)

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Rows", f"{len(view):,}")
    sc2.metric("Columns", f"{len(view.columns):,}")
    sc3.metric("Symbols", f"{view[symbol_col].nunique():,}" if symbol_col and symbol_col in view.columns else "-")
    sc4.metric("Sort", sort_col)

    default_cols = [
        "date",
        "symbol",
        "code",
        "close",
        "daily_return_pct",
        "long_pos_21",
        "volume_ratio_ma5",
        "volume_ratio_prev1",
        "v4_close_to_ma5",
        "v4_brk",
        "v4_crh",
        "v4_pgh",
        "v4_up_hint_score",
        "v4_risk_hint_score",
        "v4_net_hint_score",
        "v4_hint_label",
        "selected_score_base",
        "score_rank_key",
        "score_pct",
    ]
    default_cols = [c for c in default_cols if c in view.columns]

    with st.expander("Choose columns", expanded=False):
        selected_cols = st.multiselect(
            "Columns",
            options=view.columns.tolist(),
            default=default_cols if default_cols else view.columns.tolist()[:20],
        )

    if not selected_cols:
        selected_cols = default_cols if default_cols else view.columns.tolist()

    display_df = clean_display_df(view[selected_cols])
    st.dataframe(display_df, use_container_width=True, height=560)
    show_download(display_df, file_name=f"pool_view_{target_date}.csv", label="Download current view as CSV")


# ======================================================================================
# Page 2: Analyze Pool Indicator Runner
# ======================================================================================


def render_analyze_pool_indicator(default_pool_path: Path | None) -> None:
    page_header("Analyze Pool Indicator", pool_path=default_pool_path)

    if default_pool_path is None:
        st.error("No pool file selected.")
        return

    pool_path = default_pool_path

    try:
        target_source_df = load_pool(str(pool_path), selected_only=False)
        target_options = list_forward_return_targets(target_source_df)
    except Exception as exc:
        st.error(f"Unable to read pool columns: {exc}")
        return

    if not target_options:
        st.error("No fwd_return_pct_T* target column found in this pool.")
        return

    default_target = "fwd_return_pct_T2" if "fwd_return_pct_T2" in target_options else target_options[0]
    target_col = st.session_state.get("indicator_target_col", default_target)
    if target_col not in target_options:
        target_col = default_target

    bucket_count = int(st.session_state.get("indicator_bucket_count", 10))
    min_samples = int(st.session_state.get("indicator_min_samples", 5000))
    output_dir = DEFAULT_ANALYZE_POOL_OUTPUT_DIR / safe_pool_name(pool_path) / str(target_col)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(auto_card("Pool", format_pool_option(pool_path)), unsafe_allow_html=True)
    c2.markdown(auto_card("Target", target_col), unsafe_allow_html=True)
    c3.markdown(auto_card("Buckets", int(bucket_count)), unsafe_allow_html=True)
    c4.markdown(auto_card("Min samples", f"{int(min_samples):,}"), unsafe_allow_html=True)

    cmd = [
        sys.executable,
        str(DEFAULT_ANALYZE_POOL_SCRIPT),
        "--pool-path",
        str(pool_path),
        "--output-dir",
        str(output_dir),
        "--primary-horizon",
        str(target_col),
        "--bucket-count",
        str(int(bucket_count)),
        "--min-samples",
        str(int(min_samples)),
    ]

    run_btn = bool(st.session_state.pop("run_indicator_summary", False))
    if run_btn:
        if not DEFAULT_ANALYZE_POOL_SCRIPT.exists():
            st.error(f"Script not found: {DEFAULT_ANALYZE_POOL_SCRIPT}")
            return
        if not pool_path.exists():
            st.error(f"Pool not found: {pool_path}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        for old_file in output_dir.glob("indicator_*.csv"):
            try:
                old_file.unlink()
            except Exception:
                pass

        with st.spinner("Running analyze pool indicator..."):
            code, stdout, stderr, elapsed = run_subprocess(cmd, cwd=PROJECT_ROOT)

        st.session_state["indicator_last_output_dir"] = str(output_dir)

        if code == 0:
            st.success(f"Finished. elapsed={elapsed:.1f}s")
        else:
            st.error(f"Failed. returncode={code}")

        if stderr.strip():
            with st.expander("stderr", expanded=True):
                st.code(stderr[-12000:], language="text")

    last_output_dir = Path(st.session_state.get("indicator_last_output_dir", str(output_dir)))
    summary_path = last_output_dir / "indicator_direction_summary.csv"
    summary_df = load_csv_if_exists(str(summary_path)) if summary_path.exists() else pd.DataFrame()

    st.divider()
    section_header("Summary")

    if summary_df.empty:
        st.info("No summary output yet. Use Run in the left sidebar.")
        return

    new_schema = "bucket_pattern" in summary_df.columns and "action_hint" in summary_df.columns
    if not new_schema:
        st.error("Old indicator output schema detected. Click Run to regenerate with the new bucket-shape logic.")
        st.dataframe(clean_display_df(summary_df.head(80)), use_container_width=True, height=420)
        return

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Factors", f"{len(summary_df):,}")
    d2.metric(
        "Prefer high",
        f"{(summary_df['action_hint'].astype(str) == 'prefer_high_values').sum():,}"
        if "action_hint" in summary_df.columns
        else "-",
    )
    d3.metric(
        "Cap high",
        f"{(summary_df['action_hint'].astype(str) == 'cap_high_values').sum():,}"
        if "action_hint" in summary_df.columns
        else "-",
    )
    d4.metric(
        "Middle range",
        f"{(summary_df['action_hint'].astype(str) == 'use_middle_range').sum():,}"
        if "action_hint" in summary_df.columns
        else "-",
    )

    f1, f2, f3 = st.columns([1.1, 1.1, 2.0])
    pattern_options = ["ALL"] + sorted(summary_df["bucket_pattern"].dropna().astype(str).unique().tolist()) if "bucket_pattern" in summary_df.columns else ["ALL"]
    action_options = ["ALL"] + sorted(summary_df["action_hint"].dropna().astype(str).unique().tolist()) if "action_hint" in summary_df.columns else ["ALL"]
    pattern_filter = f1.selectbox("Bucket pattern", pattern_options, index=0)
    action_filter = f2.selectbox("Action hint", action_options, index=0)
    factor_query = f3.text_input("Search factor", value="").strip()

    view = summary_df.copy()
    if pattern_filter != "ALL" and "bucket_pattern" in view.columns:
        view = view[view["bucket_pattern"].astype(str) == pattern_filter].copy()
    if action_filter != "ALL" and "action_hint" in view.columns:
        view = view[view["action_hint"].astype(str) == action_filter].copy()
    if factor_query and "factor" in view.columns:
        view = view[view["factor"].astype(str).str.contains(factor_query, case=False, na=False)].copy()

    preferred_cols = [
        "factor",
        "bucket_pattern",
        "action_hint",
        "risk_side",
        "pattern_reason",
        "sample_count",
        "bucket_count_actual",
        "spearman_ic",
        "pearson_ic",
        "best_bucket",
        "worst_bucket",
        "best_bucket_mean_return",
        "worst_bucket_mean_return",
        "best_minus_worst_return",
        "best_bucket_up_ratio",
        "worst_bucket_up_ratio",
        "best_minus_worst_up_ratio",
        "bottom_mean_return",
        "middle_mean_return",
        "top_mean_return",
        "top_minus_bottom_return",
        "bottom_up_ratio",
        "middle_up_ratio",
        "top_up_ratio",
        "top_minus_bottom_up_ratio",
    ]
    preferred_cols = [c for c in preferred_cols if c in view.columns]
    if preferred_cols:
        view = view[preferred_cols + [c for c in view.columns if c not in preferred_cols]]

    st.dataframe(clean_display_df(view), use_container_width=True, height=650)
    show_download(view, "indicator_direction_summary_view.csv", "Download summary")


def render_single_factor_analysis(default_pool_path: Path | None) -> None:
    page_header("Single Factor Analysis", pool_path=default_pool_path)

    if default_pool_path is None:
        st.error("No pool file selected.")
        return

    pool_path = default_pool_path

    try:
        target_source_df = load_pool(str(pool_path), selected_only=False)
        target_options = list_forward_return_targets(target_source_df)
    except Exception as exc:
        st.error(f"Unable to read pool columns: {exc}")
        return

    if not target_options:
        st.error("No fwd_return_pct_T* target column found in this pool.")
        return

    default_target = "fwd_return_pct_T2" if "fwd_return_pct_T2" in target_options else target_options[0]

    with st.sidebar:
        st.divider()
        st.header("Single factor config")
        target_col = st.selectbox(
            "Target",
            target_options,
            index=select_default_index(target_options, default_target),
            key="single_factor_target_col",
        )

        if st.button("Reload factor output", use_container_width=True, key="single_factor_reload"):
            st.cache_data.clear()
            st.rerun()

    output_dir = DEFAULT_ANALYZE_POOL_OUTPUT_DIR / safe_pool_name(pool_path) / str(target_col)
    summary_path = output_dir / "indicator_direction_summary.csv"
    bucket_path = output_dir / "indicator_bucket_detail.csv"
    member_path = output_dir / "indicator_bucket_member_detail.csv"

    summary_df = load_csv_if_exists(str(summary_path)) if summary_path.exists() else pd.DataFrame()
    bucket_df = load_csv_if_exists(str(bucket_path)) if bucket_path.exists() else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(auto_card("Pool", format_pool_option(pool_path)), unsafe_allow_html=True)
    c2.markdown(auto_card("Target", target_col), unsafe_allow_html=True)
    c3.markdown(auto_card("Summary rows", f"{len(summary_df):,}"), unsafe_allow_html=True)
    c4.markdown(auto_card("Bucket rows", f"{len(bucket_df):,}"), unsafe_allow_html=True)

    if summary_df.empty or bucket_df.empty:
        st.warning("No factor bucket output found. Please run Analyze Pool Indicator first.")
        st.code(str(output_dir), language="text")
        return

    if "factor" not in summary_df.columns or "factor" not in bucket_df.columns:
        st.error("Required column missing: factor")
        st.dataframe(clean_display_df(summary_df.head(50)), use_container_width=True)
        return

    factor_options = sorted(summary_df["factor"].dropna().astype(str).unique().tolist())
    if not factor_options:
        st.warning("No factor found in summary output.")
        return

    def _factor_label(factor: str) -> str:
        row = summary_df[summary_df["factor"].astype(str) == factor].head(1)
        if row.empty:
            return factor
        parts = [factor]
        for col in ["action_hint", "bucket_pattern", "risk_side"]:
            if col in row.columns:
                val = str(row.iloc[0].get(col, "")).strip()
                if val and val.lower() != "nan":
                    parts.append(val)
        return "  |  ".join(parts)

    label_to_factor = {_factor_label(f): f for f in factor_options}
    factor_labels = list(label_to_factor.keys())

    with st.sidebar:
        selected_factor_label = st.selectbox(
            "Factor",
            factor_labels,
            index=0,
            key="single_factor_selected_factor",
        )

    factor = label_to_factor[selected_factor_label]

    st.divider()
    section_header("Factor summary")

    summary_row = summary_df[summary_df["factor"].astype(str) == factor].head(1).copy()
    if summary_row.empty:
        st.error(f"Factor not found in summary: {factor}")
        return

    row = summary_row.iloc[0]

    card_specs = [
        ("Factor", factor),
        ("Pattern", row.get("bucket_pattern", "-")),
        ("Action", row.get("action_hint", "-")),
        ("Risk side", row.get("risk_side", "-")),
        ("Samples", row.get("sample_count", "-")),
    ]
    cards = st.columns(len(card_specs))
    for col, (label, value) in zip(cards, card_specs):
        col.markdown(auto_card(label, value), unsafe_allow_html=True)

    ic_specs = [
        ("Spearman IC", row.get("spearman_ic", "-")),
        ("Pearson IC", row.get("pearson_ic", "-")),
        ("Best bucket", row.get("best_bucket", "-")),
        ("Worst bucket", row.get("worst_bucket", "-")),
        ("Best-Worst return", row.get("best_minus_worst_return", "-")),
    ]
    ic_cards = st.columns(len(ic_specs))
    for col, (label, value) in zip(ic_cards, ic_specs):
        col.markdown(auto_card(label, value), unsafe_allow_html=True)

    if "pattern_reason" in summary_row.columns:
        reason = str(row.get("pattern_reason", "")).strip()
        if reason and reason.lower() != "nan":
            st.info(reason)

    st.divider()
    section_header("Bucket performance")

    factor_bucket = bucket_df[bucket_df["factor"].astype(str) == factor].copy()
    if factor_bucket.empty:
        st.warning(f"No bucket detail found for factor: {factor}")
        return

    if "bucket" in factor_bucket.columns:
        factor_bucket = factor_bucket.sort_values("bucket")

    st.divider()
    section_header("Bucket intervals")

    def _bucket_interval_view(df_bucket: pd.DataFrame) -> pd.DataFrame:
        view = df_bucket.copy()

        if "bucket" not in view.columns:
            return pd.DataFrame()

        min_col = "min_factor" if "min_factor" in view.columns else "factor_min" if "factor_min" in view.columns else None
        max_col = "max_factor" if "max_factor" in view.columns else "factor_max" if "factor_max" in view.columns else None
        mean_col = "mean_factor" if "mean_factor" in view.columns else "factor_mean" if "factor_mean" in view.columns else None

        cols = ["bucket"]

        if min_col is not None and max_col is not None:
            min_v = pd.to_numeric(view[min_col], errors="coerce")
            max_v = pd.to_numeric(view[max_col], errors="coerce")
            view["factor_interval"] = [
                f"[{lo:.6f}, {hi:.6f}]"
                if pd.notna(lo) and pd.notna(hi)
                else ""
                for lo, hi in zip(min_v, max_v)
            ]
            cols.extend(["factor_interval", min_col, max_col])

        if mean_col is not None:
            cols.append(mean_col)

        for c in [
            "sample_count",
            "mean_return",
            "median_return",
            "up_ratio",
            "win_count",
            "loss_count",
        ]:
            if c in view.columns:
                cols.append(c)

        cols = [c for c in cols if c in view.columns]
        return view[cols].copy()

    interval_view = _bucket_interval_view(factor_bucket)

    if interval_view.empty:
        st.warning("No bucket interval columns found. Expected min_factor/max_factor or factor_min/factor_max.")
    else:
        st.dataframe(clean_display_df(interval_view), use_container_width=True, hide_index=True, height=360)
        show_download(interval_view, f"single_factor_{factor}_{target_col}_bucket_intervals.csv", "Download bucket intervals")

    required_chart_cols = {"bucket", "mean_return", "up_ratio"}
    if required_chart_cols.issubset(set(factor_bucket.columns)):
        st.plotly_chart(build_bucket_factor_chart(factor_bucket, factor), use_container_width=True)
    else:
        st.warning("Bucket chart skipped because required columns are missing: bucket / mean_return / up_ratio")

    value_cols = [
        c for c in factor_bucket.columns
        if str(c).lower() in {
            "factor_min",
            "factor_max",
            "factor_mean",
            "factor_median",
            "factor_q25",
            "factor_q75",
            "min_factor_value",
            "max_factor_value",
        }
    ]

    if "bucket" in factor_bucket.columns and value_cols:
        value_fig = go.Figure()
        for c in value_cols:
            y = pd.to_numeric(factor_bucket[c], errors="coerce")
            if y.notna().any():
                value_fig.add_trace(
                    go.Scatter(
                        x=factor_bucket["bucket"],
                        y=y,
                        mode="lines+markers",
                        name=str(c),
                        line=dict(width=2),
                        marker=dict(size=6),
                    )
                )
        value_fig = apply_plotly_layout(
            value_fig,
            title=f"Factor value by bucket: {factor}",
            yaxis_title="Factor value",
            height=420,
        )
        st.plotly_chart(value_fig, use_container_width=True)

    preferred_cols = [
        "factor",
        "target_col",
        "bucket",
        "bucket_label",
        "sample_count",
        "factor_min",
        "factor_max",
        "factor_mean",
        "factor_median",
        "mean_return",
        "median_return",
        "up_ratio",
        "win_count",
        "loss_count",
        "best_bucket",
        "worst_bucket",
    ]
    preferred_cols = [c for c in preferred_cols if c in factor_bucket.columns]
    table_view = factor_bucket[preferred_cols + [c for c in factor_bucket.columns if c not in preferred_cols]].copy()

    st.dataframe(clean_display_df(table_view), use_container_width=True, height=520)
    show_download(table_view, f"single_factor_{factor}_{target_col}_bucket_detail.csv", "Download bucket detail")

    st.divider()
    section_header("Member detail")

    if not member_path.exists():
        st.caption("Member detail is not generated by default. Use --export-member-detail only for small factor sets.")
        return

    with st.expander("Load member detail for this factor", expanded=False):
        st.warning("This file can be large. Load only when you really need row-level bucket members.")
        max_rows = st.number_input(
            "Max rows to display",
            min_value=100,
            max_value=100000,
            value=5000,
            step=100,
            key="single_factor_member_max_rows",
        )
        if st.button("Load member detail", key="single_factor_load_member"):
            member_df = load_csv_if_exists(str(member_path))
            if member_df.empty or "factor" not in member_df.columns:
                st.info("No member detail available.")
            else:
                member_view = member_df[member_df["factor"].astype(str) == factor].copy()
                if member_view.empty:
                    st.info(f"No member rows found for factor: {factor}")
                else:
                    member_view = member_view.head(int(max_rows)).copy()
                    ordered_cols = preferred_bucket_member_cols(member_view)
                    member_view = member_view[ordered_cols]
                    st.dataframe(clean_display_df(member_view), use_container_width=True, height=520)
                    show_download(member_view, f"single_factor_{factor}_{target_col}_member_detail.csv", "Download member detail")




def render_signal_analysis(default_pool_path: Path | None) -> None:
    page_header("Signal Analysis", pool_path=default_pool_path)

    if default_pool_path is None:
        st.error("No pool file selected.")
        return

    pool_path = default_pool_path

    factors = [
        "volume_ratio_prev1",
        "amplitude_pct",
        "daily_return_pct",
        "body_abs_pct",
        "volume_ratio_ma10",
        "volume_ratio_ma5",
        "red_vs_prev_green_ratio",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "t0_close_to_z_short_trend_line_pct",
        "t0_close_to_z_long_trend_line_pct",
        "t1_open_gap_pct",
        "renko_value",
        "macd_dif",
        "macd_dea",
        "macd_hist",
        "intraday_return_pct",
        "body_pct",
    ]

    try:
        pool_df = load_pool(str(pool_path), selected_only=True)
        all_pool_df = load_pool(str(pool_path), selected_only=False)
        target_options = list_forward_return_targets(all_pool_df)
    except Exception as exc:
        st.error(f"Unable to read pool: {exc}")
        return

    if pool_df.empty:
        st.warning("Selected pool is empty.")
        return

    if "date" not in pool_df.columns:
        st.error("Pool missing date column.")
        return

    if not target_options:
        st.error("No fwd_return_pct_T* target column found in this pool.")
        return

    pool_df = normalize_date_col(pool_df)

    available_dates = sorted(pool_df["date"].dropna().dt.date.unique(), reverse=True)
    if not available_dates:
        st.warning("No valid signal date found.")
        return

    default_target = "fwd_return_pct_T1" if "fwd_return_pct_T1" in target_options else target_options[0]

    with st.sidebar:
        st.divider()
        st.header("Signal config")

        target_col = st.selectbox(
            "Bucket target",
            target_options,
            index=select_default_index(target_options, default_target),
            key="signal_target_col",
        )

        signal_date = st.date_input(
            "T0 date",
            value=available_dates[0],
            min_value=available_dates[-1],
            max_value=available_dates[0],
            key="signal_date",
        )

        code_query = st.text_input(
            "Stock code / symbol",
            value="",
            placeholder="?? 002595",
            key="signal_code_query",
        ).strip()

        if st.button("Reload signal data", use_container_width=True, key="signal_reload"):
            st.cache_data.clear()
            st.rerun()

    output_dir = DEFAULT_ANALYZE_POOL_OUTPUT_DIR / safe_pool_name(pool_path) / str(target_col)
    bucket_path = output_dir / "indicator_bucket_detail.csv"
    summary_path = output_dir / "indicator_direction_summary.csv"

    bucket_df = load_csv_if_exists(str(bucket_path)) if bucket_path.exists() else pd.DataFrame()
    summary_df = load_csv_if_exists(str(summary_path)) if summary_path.exists() else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(auto_card("Pool", format_pool_option(pool_path)), unsafe_allow_html=True)
    c2.markdown(auto_card("Target", target_col), unsafe_allow_html=True)
    c3.markdown(auto_card("Bucket rows", f"{len(bucket_df):,}"), unsafe_allow_html=True)
    c4.markdown(auto_card("T0 date", signal_date), unsafe_allow_html=True)

    if bucket_df.empty:
        st.warning("No bucket detail output found. Run Analyze Pool Indicator first.")
        st.code(str(output_dir), language="text")
        return

    if "factor" not in bucket_df.columns or "bucket" not in bucket_df.columns:
        st.error("indicator_bucket_detail.csv must contain factor and bucket columns.")
        st.dataframe(clean_display_df(bucket_df.head(50)), use_container_width=True)
        return

    if not code_query:
        st.info("Enter a stock code / symbol in the left sidebar.")
        return

    def _norm_code(x: Any) -> str:
        s = str(x).strip().lower()
        s = s.replace(".sz", "").replace(".sh", "")
        s = s.replace("sz.", "").replace("sh.", "")
        s = s.replace("sz", "").replace("sh", "")
        return s

    view = pool_df[pool_df["date"].dt.date == signal_date].copy()

    code_cols = [c for c in ["symbol", "code"] if c in view.columns]
    if not code_cols:
        st.error("Pool has no symbol/code column.")
        return

    q = _norm_code(code_query)
    mask = pd.Series(False, index=view.index)

    for col in code_cols:
        normalized = view[col].map(_norm_code)
        raw = view[col].astype(str).str.strip().str.lower()
        mask = mask | (normalized == q) | raw.eq(code_query.strip().lower()) | raw.str.endswith(code_query.strip().lower())

    hit = view.loc[mask].copy()

    if hit.empty:
        st.warning(f"No selected signal found for {code_query} on {signal_date}.")
        return

    signal_row = hit.iloc[0].copy()

    symbol_display = "-"
    for col in ["symbol", "code"]:
        if col in signal_row.index and pd.notna(signal_row.get(col)):
            symbol_display = str(signal_row.get(col))
            break

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Code", symbol_display)
    d2.metric("Rows matched", f"{len(hit):,}")
    d3.metric("Date", str(signal_date))
    d4.metric("Factor count", str(len(factors)))

    def _ensure_signal_factor(row: pd.Series, factor: str) -> Any:
        if factor in row.index:
            return row.get(factor)

        if factor == "t1_open_gap_pct":
            try:
                t1_open = pd.to_numeric(pd.Series([row.get("t1_open")]), errors="coerce").iloc[0]
                close = pd.to_numeric(pd.Series([row.get("close")]), errors="coerce").iloc[0]
                if pd.notna(t1_open) and pd.notna(close) and float(close) != 0:
                    return (float(t1_open) / float(close) - 1.0) * 100.0
            except Exception:
                return pd.NA

        return pd.NA

    def _range_cols(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
        min_col = None
        max_col = None
        mean_col = None

        for c in ["min_factor", "factor_min", "min_factor_value", "bucket_min", "value_min", "min_value"]:
            if c in df.columns:
                min_col = c
                break

        for c in ["max_factor", "factor_max", "max_factor_value", "bucket_max", "value_max", "max_value"]:
            if c in df.columns:
                max_col = c
                break

        for c in ["mean_factor", "factor_mean", "mean_factor_value"]:
            if c in df.columns:
                mean_col = c
                break

        return min_col, max_col, mean_col

    def _bucket_for_value(factor: str, value: Any) -> dict[str, Any]:
        factor_bucket = bucket_df[bucket_df["factor"].astype(str) == str(factor)].copy()

        if factor_bucket.empty:
            return {
                "bucket": "",
                "bucket_interval": "",
                "bucket_status": "no_bucket_output",
                "bucket_mean_return": "",
                "bucket_up_ratio": "",
                "bucket_sample_count": "",
            }

        min_col, max_col, _ = _range_cols(factor_bucket)
        if min_col is None or max_col is None:
            return {
                "bucket": "",
                "bucket_interval": "",
                "bucket_status": "missing_bucket_range",
                "bucket_mean_return": "",
                "bucket_up_ratio": "",
                "bucket_sample_count": "",
            }

        try:
            v = float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])
        except Exception:
            v = float("nan")

        if pd.isna(v):
            return {
                "bucket": "",
                "bucket_interval": "",
                "bucket_status": "missing_factor_value",
                "bucket_mean_return": "",
                "bucket_up_ratio": "",
                "bucket_sample_count": "",
            }

        tmp = factor_bucket.copy()
        tmp["_min"] = pd.to_numeric(tmp[min_col], errors="coerce")
        tmp["_max"] = pd.to_numeric(tmp[max_col], errors="coerce")
        tmp = tmp.dropna(subset=["_min", "_max"]).copy()

        if tmp.empty:
            return {
                "bucket": "",
                "bucket_interval": "",
                "bucket_status": "empty_bucket_range",
                "bucket_mean_return": "",
                "bucket_up_ratio": "",
                "bucket_sample_count": "",
            }

        lo = tmp[["_min", "_max"]].min(axis=1)
        hi = tmp[["_min", "_max"]].max(axis=1)
        matched = tmp[(v >= lo) & (v <= hi)].copy()

        status = "ok"
        if matched.empty:
            tmp["_distance"] = tmp.apply(
                lambda r: min(abs(v - float(r["_min"])), abs(v - float(r["_max"]))),
                axis=1,
            )
            matched = tmp.sort_values("_distance").head(1).copy()
            status = "nearest_out_of_range"

        selected_bucket = matched.iloc[0]

        interval = ""
        try:
            interval = f"[{float(selected_bucket['_min']):.6f}, {float(selected_bucket['_max']):.6f}]"
        except Exception:
            interval = ""

        return {
            "bucket": selected_bucket.get("bucket", ""),
            "bucket_interval": interval,
            "bucket_status": status,
            "bucket_mean_return": selected_bucket.get("mean_return", ""),
            "bucket_up_ratio": selected_bucket.get("up_ratio", ""),
            "bucket_sample_count": selected_bucket.get("sample_count", ""),
        }

    rows = []

    for factor in factors:
        value = _ensure_signal_factor(signal_row, factor)
        bucket_info = _bucket_for_value(factor, value)

        summary_row = summary_df[summary_df["factor"].astype(str) == factor].head(1) if not summary_df.empty and "factor" in summary_df.columns else pd.DataFrame()
        action_hint = ""
        bucket_pattern = ""

        if not summary_row.empty:
            action_hint = summary_row.iloc[0].get("action_hint", "")
            bucket_pattern = summary_row.iloc[0].get("bucket_pattern", "")

        rows.append(
            {
                "factor": factor,
                "factor_value": value,
                "bucket": bucket_info["bucket"],
                "bucket_interval": bucket_info["bucket_interval"],
                "bucket_status": bucket_info["bucket_status"],
                "bucket_mean_return": bucket_info["bucket_mean_return"],
                "bucket_up_ratio": bucket_info["bucket_up_ratio"],
                "bucket_sample_count": bucket_info["bucket_sample_count"],
                "action_hint": action_hint,
                "bucket_pattern": bucket_pattern,
            }
        )

    result_df = pd.DataFrame(rows)

    st.divider()
    section_header("Signal factor bucket mapping")

    st.dataframe(clean_display_df(result_df), use_container_width=True, hide_index=True, height=680)
    show_download(
        result_df,
        f"signal_analysis_{symbol_display}_{signal_date}_{target_col}.csv",
        "Download signal bucket mapping",
    )



def render_daily_signal_score_analysis(default_pool_path: Path | None) -> None:
    page_header("Daily Signal Score", pool_path=default_pool_path)

    if default_pool_path is None:
        st.error("No pool file selected.")
        return

    pool_path = default_pool_path

    factors = [
        "volume_ratio_prev1",
        "amplitude_pct",
        "daily_return_pct",
        "body_abs_pct",
        "volume_ratio_ma10",
        "volume_ratio_ma5",
        "red_vs_prev_green_ratio",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "t0_close_to_z_short_trend_line_pct",
        "t0_close_to_z_long_trend_line_pct",
        "t1_open_gap_pct",
        "renko_value",
        "macd_dif",
        "macd_dea",
        "macd_hist",
        "intraday_return_pct",
        "body_pct",
    ]

    try:
        pool_df = load_pool(str(pool_path), selected_only=True)
        all_pool_df = load_pool(str(pool_path), selected_only=False)
        target_options = list_forward_return_targets(all_pool_df)
    except Exception as exc:
        st.error(f"Unable to read pool: {exc}")
        return

    if pool_df.empty:
        st.warning("Selected pool is empty.")
        return

    if "date" not in pool_df.columns:
        st.error("Pool missing date column.")
        return

    if not target_options:
        st.error("No fwd_return_pct_T* target column found in this pool.")
        return

    pool_df = normalize_date_col(pool_df)

    available_dates = sorted(pool_df["date"].dropna().dt.date.unique(), reverse=True)
    if not available_dates:
        st.warning("No valid signal date found.")
        return

    default_target = "fwd_return_pct_T1" if "fwd_return_pct_T1" in target_options else target_options[0]

    with st.sidebar:
        st.divider()
        st.header("Daily score config")

        target_col = st.selectbox(
            "Bucket target",
            target_options,
            index=select_default_index(target_options, default_target),
            key="daily_signal_score_target_col",
        )

        signal_date = st.date_input(
            "T0 date",
            value=available_dates[0],
            min_value=available_dates[-1],
            max_value=available_dates[0],
            key="daily_signal_score_date",
        )

        min_total_score = st.number_input(
            "Min total score",
            min_value=-500,
            max_value=500,
            value=-500,
            step=1,
            key="daily_signal_score_min_total",
        )

        search_code = st.text_input(
            "Search code",
            value="",
            placeholder="optional",
            key="daily_signal_score_search_code",
        ).strip()

        if st.button("Reload daily score data", use_container_width=True, key="daily_signal_score_reload"):
            st.cache_data.clear()
            st.rerun()

    output_dir = DEFAULT_ANALYZE_POOL_OUTPUT_DIR / safe_pool_name(pool_path) / str(target_col)
    bucket_path = output_dir / "indicator_bucket_detail.csv"

    bucket_df = load_csv_if_exists(str(bucket_path)) if bucket_path.exists() else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(auto_card("Pool", format_pool_option(pool_path)), unsafe_allow_html=True)
    c2.markdown(auto_card("Target", target_col), unsafe_allow_html=True)
    c3.markdown(auto_card("Bucket rows", f"{len(bucket_df):,}"), unsafe_allow_html=True)
    c4.markdown(auto_card("T0 date", signal_date), unsafe_allow_html=True)

    if bucket_df.empty:
        st.warning("No bucket detail output found. Run Analyze Pool Indicator first.")
        st.code(str(output_dir), language="text")
        return

    if "factor" not in bucket_df.columns or "bucket" not in bucket_df.columns:
        st.error("indicator_bucket_detail.csv must contain factor and bucket columns.")
        st.dataframe(clean_display_df(bucket_df.head(50)), use_container_width=True)
        return

    day_df = pool_df[pool_df["date"].dt.date == signal_date].copy()

    if day_df.empty:
        st.warning(f"No selected pool rows found on {signal_date}.")
        return

    code_col = "symbol" if "symbol" in day_df.columns else "code" if "code" in day_df.columns else None
    if code_col is None:
        st.error("Pool has no symbol/code column.")
        return

    def _norm_code(x: Any) -> str:
        s = str(x).strip().lower()
        s = s.replace(".sz", "").replace(".sh", "")
        s = s.replace("sz.", "").replace("sh.", "")
        s = s.replace("sz", "").replace("sh", "")
        return s

    def _num(x: Any) -> float:
        try:
            return float(pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0])
        except Exception:
            return float("nan")

    def _ensure_factor_value(row: pd.Series, factor: str) -> Any:
        if factor in row.index:
            return row.get(factor)

        if factor == "t1_open_gap_pct":
            try:
                t1_open = _num(row.get("t1_open"))
                close = _num(row.get("close"))
                if pd.notna(t1_open) and pd.notna(close) and float(close) != 0:
                    return (float(t1_open) / float(close) - 1.0) * 100.0
            except Exception:
                return pd.NA

        return pd.NA

    def _range_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
        min_col = None
        max_col = None

        for c in ["min_factor", "factor_min", "min_factor_value", "bucket_min", "value_min", "min_value"]:
            if c in df.columns:
                min_col = c
                break

        for c in ["max_factor", "factor_max", "max_factor_value", "bucket_max", "value_max", "max_value"]:
            if c in df.columns:
                max_col = c
                break

        return min_col, max_col

    bucket_maps: dict[str, pd.DataFrame] = {}

    for factor in factors:
        fb = bucket_df[bucket_df["factor"].astype(str) == factor].copy()

        if fb.empty:
            continue

        min_col, max_col = _range_cols(fb)
        if min_col is None or max_col is None:
            continue

        fb["_min"] = pd.to_numeric(fb[min_col], errors="coerce")
        fb["_max"] = pd.to_numeric(fb[max_col], errors="coerce")
        fb["_lo"] = fb[["_min", "_max"]].min(axis=1)
        fb["_hi"] = fb[["_min", "_max"]].max(axis=1)
        fb["_mean_return_num"] = pd.to_numeric(fb.get("mean_return", pd.Series(index=fb.index)), errors="coerce")
        fb["_bucket_num"] = pd.to_numeric(fb["bucket"], errors="coerce")

        fb = fb.dropna(subset=["_lo", "_hi", "_bucket_num"]).copy()

        if fb.empty:
            continue

        fb = fb.sort_values(["_mean_return_num", "_bucket_num"], ascending=[False, True]).reset_index(drop=True)
        fb["_bucket_rank"] = range(1, len(fb) + 1)
        fb["_factor_score"] = 6 - fb["_bucket_rank"]

        bucket_maps[factor] = fb

    def _bucket_match(factor: str, value: Any) -> dict[str, Any]:
        if factor not in bucket_maps:
            return {
                "bucket": "",
                "bucket_rank": "",
                "score": 0,
                "bucket_interval": "",
                "mean_return": "",
                "up_ratio": "",
                "sample_count": "",
                "status": "no_bucket_output",
            }

        v = _num(value)

        if pd.isna(v):
            return {
                "bucket": "",
                "bucket_rank": "",
                "score": 0,
                "bucket_interval": "",
                "mean_return": "",
                "up_ratio": "",
                "sample_count": "",
                "status": "missing_factor_value",
            }

        fb = bucket_maps[factor]
        matched = fb[(v >= fb["_lo"]) & (v <= fb["_hi"])].copy()

        status = "ok"

        if matched.empty:
            tmp = fb.copy()
            tmp["_distance"] = tmp.apply(
                lambda r: min(abs(v - float(r["_lo"])), abs(v - float(r["_hi"]))),
                axis=1,
            )
            matched = tmp.sort_values("_distance").head(1).copy()
            status = "nearest_out_of_range"

        row = matched.iloc[0]

        return {
            "bucket": int(row["_bucket_num"]) if pd.notna(row["_bucket_num"]) else "",
            "bucket_rank": int(row["_bucket_rank"]) if pd.notna(row["_bucket_rank"]) else "",
            "score": int(row["_factor_score"]) if pd.notna(row["_factor_score"]) else 0,
            "bucket_interval": f"[{float(row['_lo']):.6f}, {float(row['_hi']):.6f}]",
            "mean_return": row.get("mean_return", ""),
            "up_ratio": row.get("up_ratio", ""),
            "sample_count": row.get("sample_count", ""),
            "status": status,
        }

    summary_rows = []
    detail_by_code: dict[str, pd.DataFrame] = {}

    for idx, signal_row in day_df.iterrows():
        code_value = str(signal_row.get(code_col, ""))

        detail_rows = []
        total_score = 0
        valid_factor_count = 0
        missing_factor_count = 0

        for factor in factors:
            factor_value = _ensure_factor_value(signal_row, factor)
            bucket_info = _bucket_match(factor, factor_value)

            score = int(bucket_info["score"])
            total_score += score

            if bucket_info["status"] == "ok":
                valid_factor_count += 1
            else:
                missing_factor_count += 1

            detail_rows.append(
                {
                    "factor": factor,
                    "factor_value": factor_value,
                    "bucket": bucket_info["bucket"],
                    "bucket_rank_by_mean_return": bucket_info["bucket_rank"],
                    "score": score,
                    "bucket_interval": bucket_info["bucket_interval"],
                    "bucket_mean_return": bucket_info["mean_return"],
                    "bucket_up_ratio": bucket_info["up_ratio"],
                    "bucket_sample_count": bucket_info["sample_count"],
                    "status": bucket_info["status"],
                }
            )

        detail_df = pd.DataFrame(detail_rows)
        detail_by_code[code_value] = detail_df

        summary_rows.append(
            {
                "code": code_value,
                "date": signal_date,
                "total_score": total_score,
                "valid_factor_count": valid_factor_count,
                "missing_factor_count": missing_factor_count,
            }
        )

    score_df = pd.DataFrame(summary_rows)

    if search_code:
        q = _norm_code(search_code)
        score_df = score_df[
            score_df["code"].map(_norm_code).eq(q)
            | score_df["code"].astype(str).str.lower().str.endswith(search_code.lower())
        ].copy()

    score_df = score_df[pd.to_numeric(score_df["total_score"], errors="coerce") >= int(min_total_score)].copy()
    score_df = score_df.sort_values(["total_score", "valid_factor_count", "code"], ascending=[False, False, True]).reset_index(drop=True)
    score_df.insert(0, "rank", range(1, len(score_df) + 1))

    st.divider()
    section_header("Daily pool score")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Rows", f"{len(score_df):,}")
    s2.metric("Factors", f"{len(factors):,}")
    s3.metric("Max score", f"{score_df['total_score'].max() if not score_df.empty else '-'}")
    s4.metric("Min score", f"{score_df['total_score'].min() if not score_df.empty else '-'}")

    if score_df.empty:
        st.info("No rows after filter.")
        return

    summary_download_df = score_df.copy()
    show_download(
        summary_download_df,
        f"daily_signal_score_{signal_date}_{target_col}.csv",
        "Download daily score summary",
    )

    st.caption("Click the arrow on each stock row to expand factor-level bucket scoring detail.")

    for _, row in score_df.iterrows():
        code_value = str(row["code"])
        rank_value = int(row["rank"])
        total_score = row["total_score"]
        valid_count = row["valid_factor_count"]
        missing_count = row["missing_factor_count"]

        title = (
            f"#{rank_value}  {code_value}  "
            f"| total_score={total_score}  "
            f"| valid={valid_count}  "
            f"| missing={missing_count}"
        )

        with st.expander(title, expanded=False):
            head_df = pd.DataFrame(
                [
                    {
                        "rank": rank_value,
                        "code": code_value,
                        "date": row["date"],
                        "total_score": total_score,
                        "valid_factor_count": valid_count,
                        "missing_factor_count": missing_count,
                    }
                ]
            )
            st.dataframe(clean_display_df(head_df), use_container_width=True, hide_index=True, height=86)

            detail_df = detail_by_code.get(code_value, pd.DataFrame())
            if detail_df.empty:
                st.info("No factor detail.")
            else:
                detail_view = detail_df.copy()

                display_cols = [
                    "factor",
                    "factor_value",
                    "bucket",
                    "bucket_rank_by_mean_return",
                    "score",
                    "bucket_interval",
                    "bucket_mean_return",
                    "bucket_up_ratio",
                    "bucket_sample_count",
                    "status",
                ]

                for numeric_col in [
                    "factor_value",
                    "bucket",
                    "bucket_rank_by_mean_return",
                    "score",
                    "bucket_mean_return",
                    "bucket_up_ratio",
                    "bucket_sample_count",
                ]:
                    if numeric_col in detail_view.columns:
                        detail_view[numeric_col] = pd.to_numeric(detail_view[numeric_col], errors="coerce")

                if "score" in detail_view.columns:
                    detail_view = detail_view.sort_values(
                        ["score", "bucket_rank_by_mean_return", "factor"],
                        ascending=[False, True, True],
                    )

                display_cols = [c for c in display_cols if c in detail_view.columns]
                detail_view = detail_view[
                    display_cols + [c for c in detail_view.columns if c not in display_cols]
                ].copy()

                detail_view = detail_view.rename(
                    columns={
                        "factor": "Factor",
                        "factor_value": "Factor Value",
                        "bucket": "Bucket",
                        "bucket_rank_by_mean_return": "Bucket Rank",
                        "score": "Factor Score",
                        "bucket_interval": "Bucket Interval",
                        "bucket_mean_return": "Bucket Mean Return",
                        "bucket_up_ratio": "Bucket Up Ratio",
                        "bucket_sample_count": "Bucket Sample Count",
                        "status": "Match Status",
                    }
                )

                def _make_unique_columns(cols: list[str]) -> list[str]:
                    seen: dict[str, int] = {}
                    unique_cols: list[str] = []

                    for col in cols:
                        base = str(col)
                        count = seen.get(base, 0)

                        if count == 0:
                            unique_cols.append(base)
                        else:
                            unique_cols.append(f"{base}_{count + 1}")

                        seen[base] = count + 1

                    return unique_cols

                detail_view.columns = _make_unique_columns([str(c) for c in detail_view.columns])

                st.caption(
                    "Bucket Rank is sorted by bucket mean_return descending: "
                    "rank 1 = 5 points, rank 2 = 4 points, each lower rank subtracts 1 point."
                )
                st.dataframe(clean_display_df(detail_view), use_container_width=True, hide_index=True, height=520)

                show_download(
                    detail_view,
                    f"daily_signal_score_detail_{code_value}_{signal_date}_{target_col}.csv",
                    f"Download {code_value} factor detail",
                )


# ======================================================================================
# App router
# ======================================================================================


pool_files_for_sidebar = sorted(list_pool_files(DEFAULT_POOLS_DIR), key=lambda p: p.stat().st_mtime, reverse=True)
selected_pool_path: Path | None = None

with st.sidebar:
    st.title("BackTest Dashboard")

    page = st.radio(
        "Page",
        [
            "Single Pool Viewer",
            "Analyze Pool Indicator",
            "Single Factor Analysis",
            "Signal Analysis",
            "Daily Signal Score",
        ],
        index=1,
    )

    st.divider()
    st.header("Pool")
    if pool_files_for_sidebar:
        sidebar_label_map = {format_pool_option(p): p for p in pool_files_for_sidebar}
        sidebar_labels = list(sidebar_label_map.keys())
        default_sidebar_index = 0
        if DEFAULT_POOL_PATH.exists():
            for i, label in enumerate(sidebar_labels):
                if sidebar_label_map[label].name == DEFAULT_POOL_PATH.name:
                    default_sidebar_index = i
                    break
        selected_sidebar_label = st.selectbox(
            "Current pool",
            sidebar_labels,
            index=default_sidebar_index,
            key="sidebar_selected_pool",
        )
        selected_pool_path = sidebar_label_map[selected_sidebar_label]
    else:
        st.info("No pool file found.")

    if page == "Analyze Pool Indicator" and selected_pool_path is not None:
        st.divider()
        st.header("Analyze config")
        try:
            target_source_df = load_pool(str(selected_pool_path), selected_only=False)
            target_options = list_forward_return_targets(target_source_df)
        except Exception:
            target_options = []

        if target_options:
            default_target = "fwd_return_pct_T2" if "fwd_return_pct_T2" in target_options else target_options[0]
            st.selectbox(
                "Target",
                target_options,
                index=select_default_index(target_options, default_target),
                key="indicator_target_col",
            )

        st.number_input("Bucket count", min_value=2, max_value=50, value=10, step=1, key="indicator_bucket_count")
        st.number_input("Min samples", min_value=1, max_value=100000, value=5000, step=100, key="indicator_min_samples")

        if st.button("Run", type="primary", use_container_width=True):
            st.session_state["run_indicator_summary"] = True

if page == "Single Pool Viewer":
    render_pool_viewer(selected_pool_path)
elif page == "Analyze Pool Indicator":
    render_analyze_pool_indicator(selected_pool_path)
elif page == "Single Factor Analysis":
    render_single_factor_analysis(selected_pool_path)
elif page == "Signal Analysis":
    render_signal_analysis(selected_pool_path)
elif page == "Daily Signal Score":
    render_daily_signal_score_analysis(selected_pool_path)
