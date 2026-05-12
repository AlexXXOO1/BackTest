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
    st.markdown(
        """
        <div class="soft-card">
            <b>??????</b><br/>
            <span style="color:#64748b;">????????????????????</span>
        </div>
        """,
        unsafe_allow_html=True,
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
