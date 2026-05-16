# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import html
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Work around Python 3.14 on Windows: platform.machine() may hang in WMI query during pandas import.
import platform as _platform
_platform.machine = lambda: "AMD64"

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.path_manager import DATA_ROOT, POOLS_DIR, MARKET_CACHE_DIR, RAW_SH_INDEX_DIR, OUTPUT_DIR

DEFAULT_POOLS_DIR = POOLS_DIR
DEFAULT_POOL_PATH = DEFAULT_POOLS_DIR / "renko_chart_select_strategy_v0_pool.parquet"
DEFAULT_MARKET_CACHE_DIR = MARKET_CACHE_DIR
DEFAULT_SH_INDEX_DIR = RAW_SH_INDEX_DIR

DEFAULT_ANALYZE_POOL_SCRIPT = PROJECT_ROOT / "analysis" / "analyze_pool_indicator_direction.py"
DEFAULT_ANALYZE_POOL_OUTPUT_DIR = OUTPUT_DIR / "analyze_pool_indicator_dashboard_v3"
DEFAULT_TARGET_COLS = ("fwd_return_pct_T1", "fwd_return_pct_T2", "fwd_return_pct_T3", "fwd_return_pct_T4")


def target_cols_to_show(target_options: list[str]) -> list[str]:
    shown = [c for c in DEFAULT_TARGET_COLS if c in target_options]
    return shown if shown else list(target_options)



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


def clean_display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        series = out[col]

        if pd.api.types.is_datetime64_any_dtype(series):
            out[col] = series.dt.strftime("%Y-%m-%d")
            continue

        if pd.api.types.is_object_dtype(series):
            cleaned = series.replace("", pd.NA)
            numeric_probe = pd.to_numeric(cleaned, errors="coerce")
            non_empty_count = cleaned.notna().sum()

            if non_empty_count > 0 and numeric_probe.notna().sum() / non_empty_count >= 0.8:
                out[col] = numeric_probe
            else:
                out[col] = series.astype("string").fillna("")

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

    return df.reset_index(drop=True)


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
        "View one formal pool: daily signal count, SH index trend, and row details.",
    )

    if pool_path_obj is None:
        st.error("No pool file selected.")
        return

    show_sh_index = st.checkbox("Overlay SH index", value=True, key="single_show_sh_index")
    st.caption("Formal pools should contain selected == 1 rows only. If a selected column exists, this page filters selected == 1 automatically.")

    sh_index_dir = str(DEFAULT_SH_INDEX_DIR)

    try:
        df = load_pool(str(pool_path_obj), selected_only=True)
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
        "Daily count uses all selected rows in the current formal pool.",
    )

    daily = df.groupby("date").size().reset_index(name="pool_count").sort_values("date")
    sh_index_df = load_sh_index(sh_index_dir) if show_sh_index else pd.DataFrame()
    fig = build_pool_count_with_index_chart(daily, sh_index_df, show_sh_index)
    st.plotly_chart(fig, width='stretch')

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
    section_header("Daily rows", "Choose a date, then filter by code or label. Default order preserves the pool file order.")
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
    sort_options = ["Original order"] + sort_candidates
    sort_col = fc4.selectbox("Sort by", sort_options, index=0)

    view = df[df["date"].dt.date == target_date].copy()

    if symbol_query:
        matched = pd.Series(False, index=view.index)
        for c in ["symbol", "code"]:
            if c in view.columns:
                matched = matched | view[c].astype(str).str.contains(symbol_query, case=False, na=False)
        view = view[matched].copy()

    if label_col and label_value != "ALL":
        view = view[view[label_col].astype(str) == label_value].copy()

    if sort_col != "Original order" and sort_col in view.columns:
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
    st.dataframe(display_df, width='stretch', height=560)
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

    display_targets = target_cols_to_show(target_options)
    if not display_targets:
        st.error("No fwd_return_pct_T* target column found in this pool.")
        return

    bucket_count = int(st.session_state.get("indicator_bucket_count", 10))
    min_samples = int(st.session_state.get("indicator_min_samples", 10000))

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(auto_card("Pool", format_pool_option(pool_path)), unsafe_allow_html=True)
    c2.markdown(auto_card("Targets", ", ".join(display_targets)), unsafe_allow_html=True)
    c3.markdown(auto_card("Buckets", int(bucket_count)), unsafe_allow_html=True)
    c4.markdown(auto_card("Min samples", f"{int(min_samples):,}"), unsafe_allow_html=True)

    run_btn = bool(st.session_state.pop("run_indicator_summary", False))
    if run_btn:
        if not DEFAULT_ANALYZE_POOL_SCRIPT.exists():
            st.error(f"Script not found: {DEFAULT_ANALYZE_POOL_SCRIPT}")
            return
        if not pool_path.exists():
            st.error(f"Pool not found: {pool_path}")
            return

        run_rows: list[dict[str, Any]] = []
        for target_col in display_targets:
            output_dir = DEFAULT_ANALYZE_POOL_OUTPUT_DIR / safe_pool_name(pool_path) / str(target_col)
            output_dir.mkdir(parents=True, exist_ok=True)
            for old_file in output_dir.glob("indicator_*.csv"):
                try:
                    old_file.unlink()
                except Exception:
                    pass

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

            with st.spinner(f"Running Analyze Pool Indicator for {target_col}..."):
                code, stdout, stderr, elapsed = run_subprocess(cmd, cwd=PROJECT_ROOT)

            run_rows.append(
                {
                    "target_col": target_col,
                    "returncode": code,
                    "elapsed_sec": round(elapsed, 2),
                    "output_dir": str(output_dir),
                    "stderr_tail": stderr[-6000:] if stderr.strip() else "",
                }
            )

        st.cache_data.clear()
        run_df = pd.DataFrame(run_rows)
        if (run_df["returncode"] == 0).all():
            st.success("Finished all target horizons.")
        else:
            st.error("At least one target horizon failed.")
        st.dataframe(run_df.drop(columns=["stderr_tail"]), width='stretch', hide_index=True)
        for row in run_rows:
            if row["stderr_tail"]:
                with st.expander(f"stderr: {row['target_col']}", expanded=True):
                    st.code(row["stderr_tail"], language="text")

    st.divider()
    section_header("Summary by target horizon")

    for target_col in display_targets:
        output_dir = DEFAULT_ANALYZE_POOL_OUTPUT_DIR / safe_pool_name(pool_path) / str(target_col)
        summary_path = output_dir / "indicator_direction_summary.csv"
        summary_df = load_csv_if_exists(str(summary_path)) if summary_path.exists() else pd.DataFrame()

        with st.expander(target_col, expanded=False):
            if summary_df.empty:
                st.info(f"No summary output yet for {target_col}. Use Run in the left sidebar.")
                st.code(str(output_dir), language="text")
                continue

            new_schema = "bucket_pattern" in summary_df.columns and "action_hint" in summary_df.columns
            if not new_schema:
                st.error("Old indicator output schema detected. Click Run to regenerate with the new bucket-shape logic.")
                st.dataframe(clean_display_df(summary_df.head(80)), width='stretch', height=420)
                continue

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
            pattern_filter = f1.selectbox("Bucket pattern", pattern_options, index=0, key=f"indicator_pattern_{target_col}")
            action_filter = f2.selectbox("Action hint", action_options, index=0, key=f"indicator_action_{target_col}")
            factor_query = f3.text_input("Search factor", value="", key=f"indicator_factor_query_{target_col}").strip()

            view = summary_df.copy()
            if pattern_filter != "ALL" and "bucket_pattern" in view.columns:
                view = view[view["bucket_pattern"].astype(str) == pattern_filter].copy()
            if action_filter != "ALL" and "action_hint" in view.columns:
                view = view[view["action_hint"].astype(str) == action_filter].copy()
            if factor_query and "factor" in view.columns:
                view = view[view["factor"].astype(str).str.contains(factor_query, case=False, na=False)].copy()

            preferred_cols = [
                "factor",
                "target_col",
                "bucket_pattern",
                "action_hint",
                "risk_side",
                "pattern_reason",
                "sample_count",
                "bucket_count_actual",
                "spearman_ic",
                "pearson_ic",
                "best_minus_worst_return",
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
            remove_cols = {
                "best_bucket",
                "worst_bucket",
                "best_bucket_mean_return",
                "worst_bucket_mean_return",
                "best_bucket_up_ratio",
                "worst_bucket_up_ratio",
            }
            view = view.drop(columns=[c for c in remove_cols if c in view.columns])
            preferred_cols = [c for c in preferred_cols if c in view.columns]
            if preferred_cols:
                view = view[preferred_cols + [c for c in view.columns if c not in preferred_cols]]

            st.dataframe(clean_display_df(view), width='stretch', height=520)
            show_download(view, f"indicator_direction_summary_{target_col}.csv", f"Download {target_col} summary")



def _single_factor_formula_map() -> dict[str, dict[str, str]]:
    return {
        "daily_return_pct": {
            "formula": "(close / prev_close - 1) * 100",
            "meaning": "Daily return from previous close to current close.",
        },
        "intraday_return_pct": {
            "formula": "(close / open - 1) * 100",
            "meaning": "Intraday return from current open to current close.",
        },
        "amplitude_pct": {
            "formula": "(high - low) / prev_close * 100",
            "meaning": "Daily high-low range relative to previous close.",
        },
        "upper_shadow_pct": {
            "formula": "(high - max(open, close)) / prev_close * 100",
            "meaning": "Upper shadow length relative to previous close.",
        },
        "lower_shadow_pct": {
            "formula": "(min(open, close) - low) / prev_close * 100",
            "meaning": "Lower shadow length relative to previous close.",
        },
        "body_pct": {
            "formula": "(close - open) / prev_close * 100",
            "meaning": "Signed candle body length relative to previous close.",
        },
        "body_abs_pct": {
            "formula": "abs(close - open) / prev_close * 100",
            "meaning": "Absolute candle body length relative to previous close.",
        },
        "volume_ratio_ma5": {
            "formula": "volume / volume_ma5",
            "meaning": "Current volume divided by 5-day average volume.",
        },
        "volume_ratio_ma10": {
            "formula": "volume / volume_ma10",
            "meaning": "Current volume divided by 10-day average volume.",
        },
        "volume_ratio_prev1": {
            "formula": "volume / prev_volume",
            "meaning": "Current volume divided by previous trading day's volume.",
        },
        "t1_open_gap_pct": {
            "formula": "(t1_open / close - 1) * 100",
            "meaning": "T+1 open gap relative to T0 close.",
        },
        "t0_close_to_z_short_trend_line_pct": {
            "formula": "(close / z_short_trend_line - 1) * 100",
            "meaning": "T0 close distance from the short trend line.",
        },
        "t0_close_to_z_long_trend_line_pct": {
            "formula": "(close / z_long_trend_line - 1) * 100",
            "meaning": "T0 close distance from the long trend line.",
        },
        "macd_dif": {
            "formula": "EMA(close, 12) - EMA(close, 26)",
            "meaning": "MACD DIF line.",
        },
        "macd_dea": {
            "formula": "EMA(macd_dif, 9)",
            "meaning": "MACD DEA signal line.",
        },
        "macd_hist": {
            "formula": "macd_dif - macd_dea",
            "meaning": "MACD histogram value.",
        },
        "renko_value": {
            "formula": "strategy-specific Renko state value",
            "meaning": "Renko-derived factor generated by the selection strategy.",
        },
        "b1_tminus1_j": {
            "formula": "shift(kdj_j, 1)",
            "meaning": "Previous trading day KDJ J value retained as a continuous numeric factor for bucket analysis.",
        },
    }


def _render_single_factor_formula(factor: str) -> None:
    st.markdown("#### Factor formula")
    info = _single_factor_formula_map().get(str(factor))

    if info is None:
        st.info(f"No formula has been registered for `{factor}` yet. Add it to `_single_factor_formula_map()` in `app/ui/pool_dashboard.py`.")
        return

    formula_df = pd.DataFrame(
        [
            {
                "factor": factor,
                "formula": info["formula"],
                "meaning": info["meaning"],
            }
        ]
    )
    st.dataframe(clean_display_df(formula_df), width="stretch", hide_index=True)


def _pick_first_existing_col(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {str(col).lower(): col for col in columns}
    for candidate in candidates:
        found = lower_map.get(str(candidate).lower())
        if found is not None:
            return found
    return None


def _build_extreme_lookup_pool(pool_df: pd.DataFrame, factor: str, target_col: str) -> pd.DataFrame:
    if factor not in pool_df.columns or target_col not in pool_df.columns:
        return pd.DataFrame()

    sample_cols = [
        "symbol",
        "code",
        "date",
        "file",
        "selection_strategy",
        target_col,
        "fwd_return_pct_T1",
        "fwd_return_pct_T2",
        "fwd_return_pct_T3",
        "fwd_return_pct_T4",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]

    cols: list[str] = []
    for col in [factor, target_col, *sample_cols]:
        if col in pool_df.columns and col not in cols:
            cols.append(col)

    work = pool_df[cols].copy()
    work["_factor_value_numeric"] = pd.to_numeric(work[factor], errors="coerce")
    work["_target_value_numeric"] = pd.to_numeric(work[target_col], errors="coerce")
    return work.dropna(subset=["_factor_value_numeric", "_target_value_numeric"])


def _first_extreme_sample_from_work(
    work: pd.DataFrame,
    factor: str,
    target_col: str,
    bucket_value,
    side: str,
    boundary_value,
) -> dict | None:
    import numpy as np

    boundary = pd.to_numeric(pd.Series([boundary_value]), errors="coerce").iloc[0]
    if pd.isna(boundary) or work.empty:
        return None

    tolerance = max(abs(float(boundary)) * 1e-9, 1e-10)
    exact = work[np.isclose(work["_factor_value_numeric"], float(boundary), rtol=0, atol=tolerance)]

    match_type = "exact"
    if exact.empty:
        distances = (work["_factor_value_numeric"] - float(boundary)).abs()
        if distances.empty:
            return None
        exact = work.loc[[distances.idxmin()]].copy()
        match_type = "nearest"

    sample = exact.iloc[0]

    row = {
        "target_col": target_col,
        "factor": factor,
        "bucket": bucket_value,
        "extreme_side": side,
        "bucket_extreme_value": float(boundary),
        "sample_factor_value": sample.get(factor),
        "match_type": match_type,
    }

    for sample_col in [
        "symbol",
        "code",
        "date",
        "file",
        "selection_strategy",
        target_col,
        "fwd_return_pct_T1",
        "fwd_return_pct_T2",
        "fwd_return_pct_T3",
        "fwd_return_pct_T4",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]:
        if sample_col in work.columns and sample_col not in row:
            row[sample_col] = sample.get(sample_col)

    return row


def _render_single_factor_extreme_samples(
    factor: str,
    pool_path: Path,
    output_map: dict,
    display_targets: list[str],
) -> None:
    st.markdown("---")
    st.subheader("Extreme value samples")
    st.caption(
        "For numeric factors, this looks up the first selected-pool row matching each bucket min/max boundary. "
        "If no exact floating-point match exists, the nearest factor value is shown."
    )

    try:
        pool_df = load_pool(str(pool_path), selected_only=True)
    except Exception as exc:
        st.warning(f"Unable to load pool for extreme sample lookup: {exc}")
        return

    if factor not in pool_df.columns:
        st.info(f"The selected factor `{factor}` does not exist in the pool file, so sample lookup is unavailable.")
        return

    factor_numeric = pd.to_numeric(pool_df[factor], errors="coerce")
    if factor_numeric.dropna().empty:
        st.info("The selected factor is not numeric, so extreme value samples are not available.")
        return

    all_rows: list[dict] = []

    for target_col in display_targets:
        data = output_map.get(target_col, {})
        bucket_df = data.get("bucket_df", pd.DataFrame())

        if bucket_df.empty or "factor" not in bucket_df.columns:
            continue

        factor_bucket = bucket_df[bucket_df["factor"].astype(str) == str(factor)].copy()
        if factor_bucket.empty:
            continue

        target_work = _build_extreme_lookup_pool(pool_df, factor=factor, target_col=str(target_col))
        if target_work.empty:
            continue

        bucket_col = _pick_first_existing_col(
            list(factor_bucket.columns),
            ["bucket", "bucket_id", "bucket_no", "bucket_index", "bucket_label"],
        )
        min_col = _pick_first_existing_col(
            list(factor_bucket.columns),
            ["min_factor", "factor_min", "min_value", "bucket_min", "value_min", "left", "lower_bound"],
        )
        max_col = _pick_first_existing_col(
            list(factor_bucket.columns),
            ["max_factor", "factor_max", "max_value", "bucket_max", "value_max", "right", "upper_bound"],
        )

        if bucket_col is None or min_col is None or max_col is None:
            continue

        factor_bucket["_bucket_sort_key"] = pd.to_numeric(factor_bucket[bucket_col], errors="coerce")
        factor_bucket = factor_bucket.sort_values(["_bucket_sort_key", bucket_col], kind="stable")

        for _, bucket_row in factor_bucket.iterrows():
            bucket_value = bucket_row[bucket_col]

            for side, boundary_col in (("min", min_col), ("max", max_col)):
                sample_row = _first_extreme_sample_from_work(
                    work=target_work,
                    factor=factor,
                    target_col=str(target_col),
                    bucket_value=bucket_value,
                    side=side,
                    boundary_value=bucket_row[boundary_col],
                )
                if sample_row is not None:
                    all_rows.append(sample_row)

    if not all_rows:
        st.info("No extreme value samples were found for the selected factor.")
        return

    extreme_df = pd.DataFrame(all_rows)

    preferred_cols = [
        "target_col",
        "factor",
        "bucket",
        "extreme_side",
        "bucket_extreme_value",
        "sample_factor_value",
        "match_type",
        "symbol",
        "code",
        "date",
        "file",
        "selection_strategy",
        "fwd_return_pct_T1",
        "fwd_return_pct_T2",
        "fwd_return_pct_T3",
        "fwd_return_pct_T4",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    preferred_cols = [c for c in preferred_cols if c in extreme_df.columns]
    extreme_df = extreme_df[preferred_cols + [c for c in extreme_df.columns if c not in preferred_cols]].copy()

    st.dataframe(clean_display_df(extreme_df), width="stretch", hide_index=True, height=520)

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

    display_targets = target_cols_to_show(target_options)
    if not display_targets:
        st.error("No fwd_return_pct_T* target column found in this pool.")
        return

    with st.sidebar:
        st.divider()
        st.header("Single factor config")
        if st.button("Reload factor output", width='stretch', key="single_factor_reload"):
            st.cache_data.clear()
            st.rerun()

    output_map: dict[str, dict[str, Any]] = {}
    factor_set: set[str] = set()
    for target_col in display_targets:
        output_dir = DEFAULT_ANALYZE_POOL_OUTPUT_DIR / safe_pool_name(pool_path) / str(target_col)
        summary_path = output_dir / "indicator_direction_summary.csv"
        bucket_path = output_dir / "indicator_bucket_detail.csv"
        member_path = output_dir / "indicator_bucket_member_detail.csv"
        summary_df = load_csv_if_exists(str(summary_path)) if summary_path.exists() else pd.DataFrame()
        bucket_df = load_csv_if_exists(str(bucket_path)) if bucket_path.exists() else pd.DataFrame()
        output_map[target_col] = {
            "output_dir": output_dir,
            "summary_df": summary_df,
            "bucket_df": bucket_df,
            "member_path": member_path,
        }
        if not summary_df.empty and "factor" in summary_df.columns:
            factor_set.update(summary_df["factor"].dropna().astype(str).unique().tolist())

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(auto_card("Pool", format_pool_option(pool_path)), unsafe_allow_html=True)
    c2.markdown(auto_card("Targets", ", ".join(display_targets)), unsafe_allow_html=True)
    c3.markdown(auto_card("Factors", f"{len(factor_set):,}"), unsafe_allow_html=True)
    c4.markdown(auto_card("Output root", DEFAULT_ANALYZE_POOL_OUTPUT_DIR.name), unsafe_allow_html=True)

    if not factor_set:
        st.warning("No factor bucket output found. Please run Analyze Pool Indicator first.")
        st.code(str(DEFAULT_ANALYZE_POOL_OUTPUT_DIR / safe_pool_name(pool_path)), language="text")
        return

    factor_options = sorted(factor_set)

    def _factor_label(factor: str) -> str:
        for target_col in display_targets:
            summary_df = output_map[target_col]["summary_df"]
            if summary_df.empty or "factor" not in summary_df.columns:
                continue
            row = summary_df[summary_df["factor"].astype(str) == factor].head(1)
            if row.empty:
                continue
            parts = [factor]
            for col in ["action_hint", "bucket_pattern", "risk_side"]:
                if col in row.columns:
                    val = str(row.iloc[0].get(col, "")).strip()
                    if val and val.lower() != "nan":
                        parts.append(val)
            return "  |  ".join(parts)
        return factor

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

    _render_single_factor_formula(factor)

    st.divider()
    st.info(
        "Spearman IC measures the monotonic rank relationship between factor values and forward returns. "
        "Pearson IC measures the linear correlation between factor values and forward returns."
    )

    for target_col in display_targets:
        data = output_map[target_col]
        summary_df = data["summary_df"]
        bucket_df = data["bucket_df"]
        member_path = data["member_path"]
        output_dir = data["output_dir"]

        with st.expander(target_col, expanded=True):
            if summary_df.empty or bucket_df.empty:
                st.warning("No factor bucket output found for this target. Please run Analyze Pool Indicator first.")
                st.code(str(output_dir), language="text")
                continue

            if "factor" not in summary_df.columns or "factor" not in bucket_df.columns:
                st.error("Required column missing: factor")
                continue

            summary_row = summary_df[summary_df["factor"].astype(str) == factor].head(1).copy()
            if summary_row.empty:
                st.warning(f"Factor not found for {target_col}: {factor}")
                continue

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
            ]
            ic_cards = st.columns(len(ic_specs))
            for col, (label, value) in zip(ic_cards, ic_specs):
                col.markdown(auto_card(label, value), unsafe_allow_html=True)

            if "pattern_reason" in summary_row.columns:
                reason = str(row.get("pattern_reason", "")).strip()
                if reason and reason.lower() != "nan":
                    st.info(reason)

            factor_bucket = bucket_df[bucket_df["factor"].astype(str) == factor].copy()
            if factor_bucket.empty:
                st.warning(f"No bucket detail found for factor: {factor}")
                continue

            if "bucket" in factor_bucket.columns:
                factor_bucket = factor_bucket.sort_values("bucket")

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
                        f"[{lo:.6f}, {hi:.6f}]" if pd.notna(lo) and pd.notna(hi) else ""
                        for lo, hi in zip(min_v, max_v)
                    ]
                    cols.extend(["factor_interval", min_col, max_col])
                if mean_col is not None:
                    cols.append(mean_col)
                for c in ["sample_count", "mean_return", "median_return", "up_ratio", "win_count", "loss_count"]:
                    if c in view.columns:
                        cols.append(c)
                cols = [c for c in cols if c in view.columns]
                return view[cols].copy()

            st.markdown("**Bucket intervals and performance**")
            interval_view = _bucket_interval_view(factor_bucket)
            if interval_view.empty:
                st.warning("No bucket interval columns found. Expected min_factor/max_factor or factor_min/factor_max.")
            else:
                st.dataframe(clean_display_df(interval_view), width='stretch', hide_index=True, height=320)
                show_download(interval_view, f"single_factor_{factor}_{target_col}_bucket_intervals.csv", f"Download {target_col} bucket intervals")

            required_chart_cols = {"bucket", "mean_return", "up_ratio"}
            if required_chart_cols.issubset(set(factor_bucket.columns)):
                st.plotly_chart(build_bucket_factor_chart(factor_bucket, factor), width='stretch')
            else:
                st.warning("Bucket chart skipped because required columns are missing: bucket / mean_return / up_ratio")

            value_cols = [
                c for c in factor_bucket.columns
                if str(c).lower() in {
                    "min_factor",
                    "max_factor",
                    "mean_factor",
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
                    title=f"Factor value by bucket: {factor} / {target_col}",
                    yaxis_title="Factor value",
                    height=420,
                )
                st.plotly_chart(value_fig, width='stretch')

            preferred_cols = [
                "factor",
                "target_col",
                "bucket",
                "bucket_label",
                "sample_count",
                "min_factor",
                "max_factor",
                "mean_factor",
                "median_factor",
                "factor_min",
                "factor_max",
                "factor_mean",
                "factor_median",
                "mean_return",
                "median_return",
                "up_ratio",
                "win_count",
                "loss_count",
            ]
            remove_cols = {"best_bucket", "worst_bucket", "best_minus_worst_return"}
            factor_bucket = factor_bucket.drop(columns=[c for c in remove_cols if c in factor_bucket.columns])
            preferred_cols = [c for c in preferred_cols if c in factor_bucket.columns]
            table_view = factor_bucket[preferred_cols + [c for c in factor_bucket.columns if c not in preferred_cols]].copy()

            st.dataframe(clean_display_df(table_view), width='stretch', height=420)
            show_download(table_view, f"single_factor_{factor}_{target_col}_bucket_detail.csv", f"Download {target_col} bucket detail")

            st.markdown("**Member detail**")
            if not member_path.exists():
                st.caption("Member detail is not generated by default. Use --export-member-detail only for small factor sets.")
            else:
                with st.expander(f"Load member detail for {target_col}", expanded=False):
                    st.warning("This file can be large. Load only when you need row-level bucket members.")
                    max_rows = st.number_input(
                        "Max rows to display",
                        min_value=100,
                        max_value=100000,
                        value=5000,
                        step=100,
                        key=f"single_factor_member_max_rows_{target_col}",
                    )
                    if st.button("Load member detail", key=f"single_factor_load_member_{target_col}"):
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
                                st.dataframe(clean_display_df(member_view), width='stretch', height=520)
                                show_download(member_view, f"single_factor_{factor}_{target_col}_member_detail.csv", f"Download {target_col} member detail")

    try:
        _render_single_factor_extreme_samples(
            factor=factor,
            pool_path=pool_path,
            output_map=output_map,
            display_targets=display_targets,
        )
    except Exception as exc:
        st.warning(f"Extreme value samples section failed: {exc}")



# ======================================================================================
# Page 4: Multi-Factor Combination Test
# ======================================================================================


def render_multi_factor_combination_test(default_pool_path: Path | None) -> None:
    page_header("Multi-Factor Combination Test", pool_path=default_pool_path)

    if default_pool_path is None:
        st.error("No pool file selected.")
        return

    pool_path = default_pool_path

    try:
        pool_df = load_pool(str(pool_path), selected_only=True)
        all_pool_df = load_pool(str(pool_path), selected_only=False)
        target_options = list_forward_return_targets(all_pool_df)
    except Exception as exc:
        st.error(f"Unable to read pool: {exc}")
        return

    display_targets = target_cols_to_show(target_options)
    if pool_df.empty:
        st.warning("Selected pool is empty.")
        return
    if not display_targets:
        st.error("No fwd_return_pct_T* target column found in this pool.")
        return

    analyze_root = DEFAULT_ANALYZE_POOL_OUTPUT_DIR / safe_pool_name(pool_path)

    target_bucket_frames: list[pd.DataFrame] = []
    searched_paths: list[Path] = []
    for target_col in display_targets:
        bucket_path = analyze_root / target_col / "indicator_bucket_detail.csv"
        searched_paths.append(bucket_path)
        if bucket_path.exists():
            part = load_csv_if_exists(str(bucket_path))
            if not part.empty:
                target_bucket_frames.append(part)

    historical_bucket_df = pd.concat(target_bucket_frames, ignore_index=True) if target_bucket_frames else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(auto_card("Benchmark", "Strategy Pool Baseline"), unsafe_allow_html=True)
    c2.markdown(auto_card("Bucket definition", "T0 factor distribution"), unsafe_allow_html=True)
    c3.markdown(auto_card("Pool rows", f"{len(pool_df):,}"), unsafe_allow_html=True)
    c4.markdown(auto_card("Targets", ", ".join(display_targets)), unsafe_allow_html=True)

    st.caption("Combination logic is AND. Bucket intervals are defined only by T0 factor values, then evaluated against T+1, T+2, and T+3 forward returns.")

    if historical_bucket_df.empty or "factor" not in historical_bucket_df.columns or "bucket" not in historical_bucket_df.columns:
        st.warning("No prior bucket detail output found. Run Analyze Pool Indicator first so the page can infer available factors and bucket counts.")
        st.code("\n".join(str(p) for p in searched_paths), language="text")
        return

    factor_options = sorted(historical_bucket_df["factor"].dropna().astype(str).unique().tolist())
    if not factor_options:
        st.warning("No factor found in prior bucket detail output.")
        return

    selected_factors = st.multiselect(
        "Factors",
        options=factor_options,
        default=[],
        key="multi_factor_selected_factors",
    )

    def _range_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
        min_col = "min_factor" if "min_factor" in df.columns else "factor_min" if "factor_min" in df.columns else None
        max_col = "max_factor" if "max_factor" in df.columns else "factor_max" if "factor_max" in df.columns else None
        return min_col, max_col

    def _safe_key(text: str) -> str:
        return re.sub(r"[^0-9A-Za-z_]+", "_", str(text))

    def _ensure_factor_value(df: pd.DataFrame, factor: str) -> pd.Series:
        if factor in df.columns:
            return pd.to_numeric(df[factor], errors="coerce")
        if factor == "t1_open_gap_pct" and "t1_open" in df.columns and "close" in df.columns:
            t1_open = pd.to_numeric(df["t1_open"], errors="coerce")
            close = pd.to_numeric(df["close"], errors="coerce")
            return (t1_open / close - 1.0) * 100.0
        return pd.Series(pd.NA, index=df.index)

    def _infer_bucket_count(factor: str) -> int:
        fb = historical_bucket_df[historical_bucket_df["factor"].astype(str) == factor].copy()
        if fb.empty or "bucket" not in fb.columns:
            return 10
        bucket_num = pd.to_numeric(fb["bucket"], errors="coerce").dropna()
        if bucket_num.empty:
            return 10
        return max(2, int(bucket_num.max()))

    def _build_t0_bucket_definition(factor: str) -> pd.DataFrame:
        values = _ensure_factor_value(pool_df, factor).dropna()
        values = pd.to_numeric(values, errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
        if values.empty:
            return pd.DataFrame(columns=["factor", "bucket", "sample_count", "min_factor", "max_factor", "mean_factor"])

        unique_count = int(values.nunique(dropna=True))
        if unique_count < 2:
            return pd.DataFrame(columns=["factor", "bucket", "sample_count", "min_factor", "max_factor", "mean_factor"])

        n_bins = min(_infer_bucket_count(factor), unique_count)
        tmp = pd.DataFrame({"factor_value": values})

        if unique_count <= 2 or factor == "z_short_trend_above_z_long_trend_line":
            ordered_values = sorted(tmp["factor_value"].dropna().unique())
            bucket_map = {value: idx + 1 for idx, value in enumerate(ordered_values)}
            tmp["bucket"] = tmp["factor_value"].map(bucket_map).astype("Int64")
        else:
            try:
                tmp["bucket"] = pd.qcut(
                    tmp["factor_value"],
                    q=int(n_bins),
                    labels=False,
                    duplicates="drop",
                ) + 1
            except Exception:
                return pd.DataFrame(columns=["factor", "bucket", "sample_count", "min_factor", "max_factor", "mean_factor"])

        out = (
            tmp.groupby("bucket", as_index=False)
            .agg(
                sample_count=("factor_value", "size"),
                min_factor=("factor_value", "min"),
                max_factor=("factor_value", "max"),
                mean_factor=("factor_value", "mean"),
            )
            .sort_values("bucket")
            .reset_index(drop=True)
        )
        out.insert(0, "factor", factor)
        return out

    condition_map: dict[str, list[Any]] = {}
    bucket_definition_map: dict[str, pd.DataFrame] = {}

    if selected_factors:
        st.markdown("**Bucket conditions**")

    for factor in selected_factors:
        fb = _build_t0_bucket_definition(factor)
        bucket_definition_map[factor] = fb
        if "bucket" in fb.columns:
            fb = fb.sort_values("bucket")

        if fb.empty:
            st.warning(f"No T0 bucket definition can be built for factor: {factor}")
            condition_map[factor] = []
            continue

        min_col, max_col = _range_cols(fb)
        label_to_bucket: dict[str, Any] = {}
        labels: list[str] = []
        for _, row in fb.iterrows():
            bucket_value = row.get("bucket")
            if min_col is not None and max_col is not None:
                lo = pd.to_numeric(pd.Series([row.get(min_col)]), errors="coerce").iloc[0]
                hi = pd.to_numeric(pd.Series([row.get(max_col)]), errors="coerce").iloc[0]
                interval = f" [{float(lo):.6f}, {float(hi):.6f}]" if pd.notna(lo) and pd.notna(hi) else ""
            else:
                interval = ""
            label = f"Bucket {bucket_value}{interval}"
            labels.append(label)
            label_to_bucket[label] = bucket_value

        chosen_labels = st.multiselect(
            f"{factor}: T0 buckets",
            options=labels,
            default=[],
            key=f"multi_factor_buckets_{_safe_key(factor)}",
        )
        condition_map[factor] = [label_to_bucket[x] for x in chosen_labels]

    run_clicked = st.button("Run", type="primary", width='stretch', key="multi_factor_run")

    if not run_clicked:
        st.info("Select factors and buckets, then click Run. The analysis does not run automatically.")
        return

    if not selected_factors:
        st.error("Select at least one factor.")
        return

    missing_bucket_factors = [factor for factor, buckets in condition_map.items() if not buckets]
    if missing_bucket_factors:
        st.error("Select at least one bucket for every selected factor: " + ", ".join(missing_bucket_factors))
        return

    work = pool_df.copy()
    mask = pd.Series(True, index=work.index)

    def _assign_bucket(values: pd.Series, fb: pd.DataFrame) -> pd.Series:
        min_col, max_col = _range_cols(fb)
        assigned = pd.Series(pd.NA, index=values.index, dtype="object")
        if min_col is None or max_col is None:
            return assigned
        for _, row in fb.iterrows():
            bucket_value = row.get("bucket")
            lo = pd.to_numeric(pd.Series([row.get(min_col)]), errors="coerce").iloc[0]
            hi = pd.to_numeric(pd.Series([row.get(max_col)]), errors="coerce").iloc[0]
            if pd.isna(lo) or pd.isna(hi):
                continue
            low = min(float(lo), float(hi))
            high = max(float(lo), float(hi))
            assigned.loc[(values >= low) & (values <= high)] = bucket_value
        return assigned

    for factor in selected_factors:
        fb = bucket_definition_map.get(factor, pd.DataFrame()).copy()
        values = _ensure_factor_value(work, factor)
        bucket_values = _assign_bucket(values, fb)
        value_col = f"{factor}__value"
        bucket_col = f"{factor}__bucket"
        work[value_col] = values
        work[bucket_col] = bucket_values
        selected_buckets = [str(x) for x in condition_map[factor]]
        mask = mask & bucket_values.astype(str).isin(selected_buckets)

    detail_df = work.loc[mask].copy()

    def _perf_stats(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
        if target_col not in df.columns:
            return {"sample_count": 0, "mean_return": pd.NA, "median_return": pd.NA, "up_ratio": pd.NA}
        s = pd.to_numeric(df[target_col], errors="coerce").dropna()
        if s.empty:
            return {"sample_count": 0, "mean_return": pd.NA, "median_return": pd.NA, "up_ratio": pd.NA}
        return {
            "sample_count": int(s.size),
            "mean_return": float(s.mean()),
            "median_return": float(s.median()),
            "up_ratio": float((s > 0).mean()),
        }

    summary_rows: list[dict[str, Any]] = []
    for target_col in display_targets:
        base = _perf_stats(pool_df, target_col)
        combo = _perf_stats(detail_df, target_col)
        summary_rows.append(
            {
                "target_col": target_col,
                "benchmark": "Strategy Pool Baseline",
                "benchmark_sample_count": base["sample_count"],
                "combo_sample_count": combo["sample_count"],
                "coverage_ratio": (combo["sample_count"] / base["sample_count"]) if base["sample_count"] else pd.NA,
                "benchmark_mean_return": base["mean_return"],
                "combo_mean_return": combo["mean_return"],
                "excess_mean_return": (combo["mean_return"] - base["mean_return"]) if pd.notna(combo["mean_return"]) and pd.notna(base["mean_return"]) else pd.NA,
                "benchmark_median_return": base["median_return"],
                "combo_median_return": combo["median_return"],
                "benchmark_up_ratio": base["up_ratio"],
                "combo_up_ratio": combo["up_ratio"],
                "excess_up_ratio": (combo["up_ratio"] - base["up_ratio"]) if pd.notna(combo["up_ratio"]) and pd.notna(base["up_ratio"]) else pd.NA,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    condition_rows = []
    for factor, buckets in condition_map.items():
        condition_rows.append({"factor": factor, "selected_buckets": ", ".join(map(str, buckets))})
    condition_df = pd.DataFrame(condition_rows)

    st.divider()
    section_header("Combination conditions")
    st.dataframe(condition_df, width='stretch', hide_index=True)

    bucket_definition_df = pd.concat(
        [df.assign(selected_for_condition=factor in condition_map) for factor, df in bucket_definition_map.items() if not df.empty],
        ignore_index=True,
    ) if bucket_definition_map else pd.DataFrame()
    if not bucket_definition_df.empty:
        with st.expander("T0 bucket definitions", expanded=False):
            st.dataframe(clean_display_df(bucket_definition_df), width='stretch', hide_index=True)
            show_download(bucket_definition_df, "multi_factor_t0_bucket_definitions.csv", "Download T0 bucket definitions")

    section_header("Summary vs strategy pool baseline")
    st.dataframe(clean_display_df(summary_df), width='stretch', hide_index=True)
    show_download(summary_df, "multi_factor_combination_summary.csv", "Download summary")

    st.divider()
    section_header("Matched detail")

    value_bucket_cols = []
    for factor in selected_factors:
        value_bucket_cols.extend([f"{factor}__value", f"{factor}__bucket"])

    base_cols = [c for c in ["date", "symbol", "code", "name", "stock_name", "close", "daily_return_pct"] if c in detail_df.columns]
    target_cols = [c for c in display_targets if c in detail_df.columns]
    detail_cols = base_cols + value_bucket_cols + target_cols
    detail_cols = [c for c in detail_cols if c in detail_df.columns]
    detail_view = detail_df[detail_cols + [c for c in detail_df.columns if c not in detail_cols]].copy()

    st.metric("Matched rows", f"{len(detail_view):,}")
    if detail_view.empty:
        st.info("No rows matched all AND conditions.")
    else:
        st.dataframe(clean_display_df(detail_view), width='stretch', height=620)
        show_download(detail_view, "multi_factor_combination_detail.csv", "Download detail")

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
            "Multi-Factor Combination Test",
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
        st.caption("Run analyzes T+1 through T+4 together when those target columns exist.")
        st.number_input("Bucket count", min_value=2, max_value=50, value=10, step=1, key="indicator_bucket_count")
        st.number_input("Min samples", min_value=10000, max_value=100000, value=10000, step=100, key="indicator_min_samples")

        if st.button("Run", type="primary", width='stretch'):
            st.session_state["run_indicator_summary"] = True

if page == "Single Pool Viewer":
    render_pool_viewer(selected_pool_path)
elif page == "Analyze Pool Indicator":
    render_analyze_pool_indicator(selected_pool_path)
elif page == "Single Factor Analysis":
    render_single_factor_analysis(selected_pool_path)
elif page == "Multi-Factor Combination Test":
    render_multi_factor_combination_test(selected_pool_path)
