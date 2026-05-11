# -*- coding: utf-8 -*-
from __future__ import annotations

import re
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
DEFAULT_POOL_PATH = DEFAULT_POOLS_DIR / "renko_chart_select_strategy_v4_pool.parquet"
DEFAULT_MARKET_CACHE_DIR = DATA_ROOT / "market_cache" / "daily_bars_by_symbol"
DEFAULT_SH_INDEX_DIR = DATA_ROOT / "raw_SH_index"

DEFAULT_MULTI_POOL_OUTPUT_ROOT = DATA_ROOT / "output" / "multi_pool_compare_dashboard"
DEFAULT_MULTI_POOL_OUTPUT_DIR = DEFAULT_MULTI_POOL_OUTPUT_ROOT / "latest"
MULTI_POOL_ANALYZE_SCRIPT = PROJECT_ROOT / "analyze_tools" / "analyze_multi_pool_compare.py"

DEFAULT_ANALYZE_POOL_SCRIPT = PROJECT_ROOT / "analyze_tools" / "analyze_pool_indicator_direction.py"
DEFAULT_ANALYZE_POOL_OUTPUT_DIR = DATA_ROOT / "output" / "analyze_pool_indicator_dashboard"


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
            .block-container {
                padding-top: 1.35rem;
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
            div[data-testid="stMetricLabel"] {
                color: #64748b;
                font-size: 0.82rem;
            }
            div[data-testid="stMetricValue"] {
                font-size: 1.28rem;
                font-weight: 700;
            }
            .page-hero {
                padding: 6px 0 14px 0;
                border-bottom: 1px solid rgba(148, 163, 184, 0.22);
                margin-bottom: 16px;
            }
            .page-title {
                font-size: 2.0rem;
                font-weight: 760;
                line-height: 1.15;
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


def page_header(title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="page-hero">
          <div class="page-title">{title}</div>
          <div class="page-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, caption: str | None = None) -> None:
    caption_html = f'<div class="section-caption">{caption}</div>' if caption else ""
    st.markdown(
        f'<div class="section-title">{title}</div>{caption_html}',
        unsafe_allow_html=True,
    )


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
    读取 C:\\Users\\zyf37\\Desktop\\BackTest_Data\\raw_SH_index 下的上证指数 TXT。
    优先读取包含 999999 的文件，例如 SH#999999.txt。
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
    try:
        mtime = pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
    except Exception:
        mtime = ""

    suffix = path.suffix.lower().replace(".", "")
    name = safe_pool_name(path)
    if include_file_name:
        return f"{name}  |  {path.name}  |  {suffix}  |  {mtime}"
    return f"{name}  |  {suffix}  |  {mtime}"


def build_pool_count_with_index_chart(
    daily: pd.DataFrame,
    sh_index_df: pd.DataFrame,
    show_sh_index: bool,
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    plot_daily = daily.copy().sort_values("date")
    plot_daily["pool_count_ma5"] = plot_daily["pool_count"].rolling(5, min_periods=1).mean()

    fig.add_trace(
        go.Bar(
            x=plot_daily["date"],
            y=plot_daily["pool_count"],
            name="Pool count",
            opacity=0.46,
            marker_line_width=0,
            hovertemplate="%{x|%Y-%m-%d}<br>Pool count=%{y}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=plot_daily["date"],
            y=plot_daily["pool_count_ma5"],
            mode="lines",
            name="Pool count MA5",
            line=dict(width=2.4),
            hovertemplate="%{x|%Y-%m-%d}<br>MA5=%{y:.1f}<extra></extra>",
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
                    name="上证指数收盘价",
                    line=dict(color="red", width=1.8),
                    hovertemplate="%{x|%Y-%m-%d}<br>上证指数=%{y:.2f}<extra></extra>",
                ),
                secondary_y=True,
            )

    fig.update_layout(
        title=dict(text="Daily pool count + 上证指数", x=0.01, xanchor="left", font=dict(size=18)),
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
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]
            )
        ),
    )
    fig.update_yaxes(
        title_text="Pool count",
        secondary_y=False,
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.22)",
        zerolinecolor="rgba(15, 23, 42, 0.18)",
    )
    fig.update_yaxes(title_text="上证指数收盘价", secondary_y=True, showgrid=False)

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
        title=dict(text=f"Bucket detail: {factor}", x=0.01, xanchor="left", font=dict(size=18)),
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


def render_pool_viewer() -> None:
    page_header("Single Pool Viewer", "查看单个 pool 的出票数量、上证指数走势、当日明细和个股历史。")

    with st.sidebar:
        st.header("Single Pool")

        if st.button("刷新 pool 列表", key="single_refresh_pool_list", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        pool_files = list_pool_files(DEFAULT_POOLS_DIR)
        if not pool_files:
            st.error("没有找到 pool 文件。")
            return

        pool_files = sorted(pool_files, key=lambda p: p.stat().st_mtime, reverse=True)
        pool_option_map = {format_pool_option(p): p for p in pool_files}

        default_index = 0
        if DEFAULT_POOL_PATH.exists():
            for i, p in enumerate(pool_files):
                if p.name == DEFAULT_POOL_PATH.name:
                    default_index = i
                    break

        selected_pool_label = st.selectbox(
            "选择 pool",
            options=list(pool_option_map.keys()),
            index=default_index,
        )
        pool_path_obj = pool_option_map[selected_pool_label]
        pool_path = str(pool_path_obj)

        selected_only = st.checkbox("Only selected == 1", value=True, key="single_selected_only")
        show_sh_index = st.checkbox("Daily pool count 叠加上证指数", value=True)

    sh_index_dir = str(DEFAULT_SH_INDEX_DIR)

    try:
        df = load_pool(pool_path, selected_only=selected_only)
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
            <b>当前 pool：</b>{safe_pool_name(pool_path_obj)}<br/>
            <span style="color:#64748b;">文件名：{pool_path_obj.name}</span>
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
        "Pool count MA5 = 最近 5 个交易日出票数量的移动平均，只用于看出票数量趋势，不参与选股。",
    )

    daily = df.groupby("date").size().reset_index(name="pool_count").sort_values("date")
    sh_index_df = load_sh_index(sh_index_dir) if show_sh_index else pd.DataFrame()
    fig = build_pool_count_with_index_chart(daily, sh_index_df, show_sh_index)
    st.plotly_chart(fig, use_container_width=True)

    if show_sh_index:
        if sh_index_df.empty:
            st.warning(f"没有从目录读取到上证指数数据：{sh_index_dir}")
        else:
            sh_plot = sh_index_df[
                (sh_index_df["date"] >= daily["date"].min())
                & (sh_index_df["date"] <= daily["date"].max())
            ].copy()
            if sh_plot.empty:
                st.warning("上证指数数据存在，但和当前 pool 日期范围没有交集。")
            else:
                st.caption(
                    f"上证指数数据范围：{sh_plot['date'].min().date()} → {sh_plot['date'].max().date()}，"
                    f"共 {len(sh_plot)} 条"
                )

    st.divider()
    tab_rows, tab_symbol = st.tabs(["当日明细", "个股历史"])

    with tab_rows:
        section_header("Filter", "先选日期，再按代码、标签和排序字段查看出票明细。")
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
        st.dataframe(display_df, use_container_width=True, height=520)
        show_download(display_df, file_name=f"pool_view_{target_date}.csv", label="Download current view as CSV")

    with tab_symbol:
        section_header("Symbol detail", "查看单只股票在当前 pool 中出现时的指标变化。")
        if symbol_col and symbol_col in df.columns:
            symbols = sorted(df[symbol_col].dropna().astype(str).unique().tolist())
            if symbols:
                selected_symbol = st.selectbox("Symbol/code", symbols, index=0)
                one = df[df[symbol_col].astype(str) == selected_symbol].copy().sort_values("date")

                chart_cols = [
                    "close",
                    "renko_value",
                    "v4_brk",
                    "daily_return_pct",
                    "v4_close_to_ma5",
                    "v4_net_hint_score",
                    "long_pos_21",
                    "volume_ratio_ma5",
                ]
                chart_cols = [c for c in chart_cols if c in one.columns]

                if chart_cols:
                    fig_symbol = build_line_chart_from_pivot(
                        one.set_index("date")[chart_cols],
                        title=f"{selected_symbol} indicators",
                        yaxis_title="value",
                        height=430,
                    )
                    st.plotly_chart(fig_symbol, use_container_width=True)

                detail_cols = [
                    "date",
                    "close",
                    "daily_return_pct",
                    "long_pos_21",
                    "volume_ratio_ma5",
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
                ]
                detail_cols = [c for c in detail_cols if c in one.columns]

                st.dataframe(
                    clean_display_df(one[detail_cols].sort_values("date", ascending=False)),
                    use_container_width=True,
                    height=360,
                )
        else:
            st.info("No symbol/code column found.")


# ======================================================================================
# Page 2: Analyze Pool Indicator Runner
# ======================================================================================


def render_analyze_pool_indicator() -> None:
    page_header(
        "Analyze Pool Indicator",
        "调用 analyze_tools\\analyze_pool_indicator_direction.py，分析指标分桶方向；fwd_* 未来列只作为 target，不作为 factor。",
    )

    with st.sidebar:
        st.header("Indicator Config")

        if st.button("刷新 pool 列表", key="indicator_refresh_pool_list", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        pool_files = sorted(list_pool_files(DEFAULT_POOLS_DIR), key=lambda p: p.stat().st_mtime, reverse=True)
        if pool_files:
            label_map = {format_pool_option(p): p for p in pool_files}
            pool_label = st.selectbox("选择 pool", list(label_map.keys()), index=0)
            pool_path = label_map[pool_label]
        else:
            pool_path = Path(st.text_input("Pool path", value=str(DEFAULT_POOL_PATH)))

        primary_horizon = st.selectbox("Primary horizon", ["T1", "T2", "T3", "T4", "T5"], index=1)
        bucket_count = st.number_input("Bucket count", min_value=2, max_value=50, value=10, step=1)
        min_samples = st.number_input("Min samples", min_value=1, max_value=100000, value=1000, step=100)
        selected_only = st.checkbox("Only selected == 1", value=True, key="indicator_selected_only")

        with st.expander("Advanced path config", expanded=False):
            script_path = Path(st.text_input("Script path", value=str(DEFAULT_ANALYZE_POOL_SCRIPT)))
            output_dir = Path(st.text_input("Output dir", value=str(DEFAULT_ANALYZE_POOL_OUTPUT_DIR)))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pool", safe_pool_name(pool_path))
    c2.metric("Target", f"fwd_return_pct_{primary_horizon}")
    c3.metric("Buckets", int(bucket_count))
    c4.metric("Min samples", f"{int(min_samples):,}")

    cmd = [
        sys.executable,
        str(script_path),
        "--pool-path",
        str(pool_path),
        "--output-dir",
        str(output_dir),
        "--primary-horizon",
        str(primary_horizon),
        "--bucket-count",
        str(int(bucket_count)),
        "--min-samples",
        str(int(min_samples)),
    ]

    if not selected_only:
        cmd.append("--include-unselected")

    with st.expander("Command", expanded=False):
        st.code(" ".join(f'\"{x}\"' if " " in x else x for x in cmd), language="powershell")

    run_col, path_col = st.columns([1, 4])
    with run_col:
        run_btn = st.button("Run", type="primary", use_container_width=True)
    with path_col:
        st.caption(f"输出目录：`{output_dir}`")

    if run_btn:
        if not script_path.exists():
            st.error(f"Script not found: {script_path}")
            return
        if not pool_path.exists():
            st.error(f"Pool not found: {pool_path}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner("Running analyze pool indicator..."):
            code, stdout, stderr, elapsed = run_subprocess(cmd, cwd=PROJECT_ROOT)

        st.session_state["indicator_last_output_dir"] = str(output_dir)

        if code == 0:
            st.success(f"Finished. elapsed={elapsed:.1f}s")
        else:
            st.error(f"Failed. returncode={code}")

        with st.expander("stdout", expanded=False):
            st.code(stdout[-12000:], language="text")

        if stderr.strip():
            with st.expander("stderr", expanded=True):
                st.code(stderr[-12000:], language="text")

    last_output_dir = Path(st.session_state.get("indicator_last_output_dir", str(output_dir)))

    st.divider()
    section_header("Output files", "运行后会读取输出目录中的 CSV，并优先展示 summary 与 bucket detail。")

    if not last_output_dir.exists():
        st.info("Output dir does not exist yet.")
        return

    files = sorted(last_output_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        st.info("No CSV output found yet.")
        return

    summary_path = last_output_dir / "indicator_direction_summary.csv"
    bucket_path = last_output_dir / "indicator_bucket_detail.csv"
    summary_df = load_csv_if_exists(str(summary_path)) if summary_path.exists() else pd.DataFrame()
    bucket_df = load_csv_if_exists(str(bucket_path)) if bucket_path.exists() else pd.DataFrame()

    tabs = st.tabs(["Summary", "Bucket detail", "CSV files"])

    with tabs[0]:
        if summary_df.empty:
            st.info("没有找到 indicator_direction_summary.csv，下面可在 CSV files 中查看原始输出。")
        else:
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Factors", f"{len(summary_df):,}")
            d2.metric("Positive", f"{(summary_df['direction'].astype(str) == 'positive').sum():,}" if "direction" in summary_df.columns else "-")
            d3.metric("Unclear", f"{(summary_df['direction'].astype(str) == 'unclear').sum():,}" if "direction" in summary_df.columns else "-")
            d4.metric("Negative", f"{(summary_df['direction'].astype(str) == 'negative').sum():,}" if "direction" in summary_df.columns else "-")

            f1, f2 = st.columns([1, 2])
            if "direction" in summary_df.columns:
                direction_filter = f1.multiselect(
                    "Direction",
                    options=sorted(summary_df["direction"].dropna().astype(str).unique().tolist()),
                    default=sorted(summary_df["direction"].dropna().astype(str).unique().tolist()),
                )
            else:
                direction_filter = []
            factor_query = f2.text_input("Search factor", value="").strip()

            view = summary_df.copy()
            if direction_filter and "direction" in view.columns:
                view = view[view["direction"].astype(str).isin(direction_filter)].copy()
            if factor_query and "factor" in view.columns:
                view = view[view["factor"].astype(str).str.contains(factor_query, case=False, na=False)].copy()

            preferred_cols = [
                "factor",
                "direction",
                "positive_score",
                "sample_count",
                "spearman_ic",
                "pearson_ic",
                "top_minus_bottom_return",
                "top_minus_bottom_up_ratio",
                "bottom_mean_return",
                "top_mean_return",
                "bottom_up_ratio",
                "top_up_ratio",
            ]
            preferred_cols = [c for c in preferred_cols if c in view.columns]
            if preferred_cols:
                view = view[preferred_cols + [c for c in view.columns if c not in preferred_cols]]

            st.dataframe(clean_display_df(view), use_container_width=True, height=520)
            show_download(view, "indicator_direction_summary_view.csv", "Download summary view")

    with tabs[1]:
        if bucket_df.empty:
            st.info("没有找到 indicator_bucket_detail.csv。")
        else:
            factors = sorted(bucket_df["factor"].dropna().astype(str).unique().tolist()) if "factor" in bucket_df.columns else []
            selected_factor = st.selectbox("Factor", factors, index=0) if factors else None
            if selected_factor:
                fig_bucket = build_bucket_factor_chart(bucket_df, selected_factor)
                st.plotly_chart(fig_bucket, use_container_width=True)
                factor_bucket = bucket_df[bucket_df["factor"].astype(str) == str(selected_factor)].copy()
                st.dataframe(clean_display_df(factor_bucket), use_container_width=True, height=320)
                show_download(factor_bucket, f"bucket_detail_{selected_factor}.csv", "Download factor bucket")

    with tabs[2]:
        selected_file = st.selectbox("CSV output", files, format_func=lambda p: p.name)
        out_df = load_csv_if_exists(str(selected_file))
        st.dataframe(clean_display_df(out_df.head(1000)), use_container_width=True, height=520)
        show_download(out_df, selected_file.name, "Download selected CSV")


# ======================================================================================
# Page 3: N Pool Compare Runner
# ======================================================================================


def run_multi_pool_compare_script(
    pool_paths: list[Path],
    pool_names: list[str],
    market_cache_dir: Path,
    output_dir: Path,
    start_date,
    end_date,
    horizons: str,
    max_workers: int,
    selected_only: bool,
) -> tuple[int, str, str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(MULTI_POOL_ANALYZE_SCRIPT),
        "--pool-paths",
        ";".join(str(p) for p in pool_paths),
        "--pool-names",
        ";".join(pool_names),
        "--market-cache-dir",
        str(market_cache_dir),
        "--output-dir",
        str(output_dir),
        "--start-date",
        str(start_date),
        "--end-date",
        str(end_date),
        "--horizons",
        horizons,
        "--max-workers",
        str(max_workers),
    ]

    if not selected_only:
        cmd.append("--no-selected-only")

    return run_subprocess(cmd, cwd=PROJECT_ROOT)


def render_multi_pool_compare_runner() -> None:
    page_header(
        "N Pool Compare Runner",
        "手动选择 2-5 个 pool，点击 Run 后展示出票数量、收益、上涨率、超额收益和覆盖情况。",
    )

    with st.sidebar:
        st.header("N Pool Compare Config")

        if st.button("刷新 pool 列表", key="multi_refresh_pool_list", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        pools_dir = Path(st.text_input("Pools dir", value=str(DEFAULT_POOLS_DIR)))
        market_cache_dir = Path(st.text_input("Market cache dir", value=str(DEFAULT_MARKET_CACHE_DIR)))
        output_root = Path(st.text_input("Output root", value=str(DEFAULT_MULTI_POOL_OUTPUT_ROOT)))

        start_date = st.date_input("Start date", value=pd.Timestamp("2024-01-01").date())
        end_date = st.date_input("End date", value=pd.Timestamp.today().date())
        horizons = st.text_input("Horizons", value="1,2,3")

        max_workers = st.number_input("Max workers", min_value=1, max_value=32, value=8, step=1)
        selected_only = st.checkbox("Only selected == 1", value=True, key="multi_selected_only")

    pool_files = sorted(list_pool_files(pools_dir), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pool_files:
        st.error(f"No pool parquet/csv found in: {pools_dir}")
        return

    pool_label_map = {format_pool_option(p, include_file_name=True): p for p in pool_files}

    selected_labels = st.multiselect(
        "Select 2-5 pools",
        options=list(pool_label_map.keys()),
        default=list(pool_label_map.keys())[:2],
    )

    selected_paths = [pool_label_map[x] for x in selected_labels]

    if len(selected_paths) < 2 or len(selected_paths) > 5:
        st.warning("请选择 2-5 个 pool。")
        return

    selected_df = pd.DataFrame(
        [
            {
                "pool_name": safe_pool_name(p),
                "file_name": p.name,
                "modified_time": pd.Timestamp(p.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
            }
            for p in selected_paths
        ]
    )

    section_header("Selected pools")
    st.dataframe(selected_df, use_container_width=True, height=150)
    render_pills([safe_pool_name(p) for p in selected_paths])

    pool_names_text = st.text_input(
        "Pool names, separated by semicolon",
        value=";".join(safe_pool_name(p) for p in selected_paths),
    )
    pool_names = [x.strip() for x in pool_names_text.split(";") if x.strip()]

    if len(pool_names) != len(selected_paths):
        st.error("Pool names 数量必须和选择的 pool 数量一致。")
        return
    if len(set(pool_names)) != len(pool_names):
        st.error("Pool names 不能重复。")
        return

    default_run_name = "multi_pool_" + "_vs_".join(pool_names)
    run_name = st.text_input("Run name", value=default_run_name)
    output_dir = output_root / run_name

    c1, c2 = st.columns([1, 4])
    with c1:
        run_btn = st.button("Run analysis", type="primary", use_container_width=True)
    with c2:
        st.caption(f"输出目录：`{output_dir}`")

    if run_btn:
        if not MULTI_POOL_ANALYZE_SCRIPT.exists():
            st.error(f"分析脚本不存在: {MULTI_POOL_ANALYZE_SCRIPT}")
            return

        with st.spinner("Running multi-pool analysis..."):
            code, stdout, stderr, elapsed = run_multi_pool_compare_script(
                pool_paths=selected_paths,
                pool_names=pool_names,
                market_cache_dir=market_cache_dir,
                output_dir=output_dir,
                start_date=start_date,
                end_date=end_date,
                horizons=horizons,
                max_workers=int(max_workers),
                selected_only=selected_only,
            )

        st.session_state["multi_pool_last_output_dir"] = str(output_dir)

        if code == 0:
            st.success(f"Analysis finished. elapsed={elapsed:.1f}s")
        else:
            st.error(f"Analysis failed. returncode={code}")

        with st.expander("stdout", expanded=False):
            st.code(stdout[-12000:], language="text")

        if stderr.strip():
            with st.expander("stderr", expanded=True):
                st.code(stderr[-12000:], language="text")

    last_output_dir = Path(st.session_state.get("multi_pool_last_output_dir", str(output_dir)))

    st.divider()
    section_header("Analysis result")

    if not last_output_dir.exists():
        st.info("还没有运行结果。选择 pool 后点击 Run。")
        return

    pool_daily_path = last_output_dir / "1_pool_daily_metrics.csv"
    pool_summary_path = last_output_dir / "2_pool_summary.csv"
    pairwise_daily_path = last_output_dir / "3_pairwise_daily_compare.csv"
    pairwise_summary_path = last_output_dir / "4_pairwise_summary.csv"
    daily_coverage_path = last_output_dir / "5_daily_coverage.csv"
    pairwise_coverage_path = last_output_dir / "6_pairwise_coverage_daily.csv"

    expected_files = [
        pool_daily_path,
        pool_summary_path,
        pairwise_daily_path,
        pairwise_summary_path,
        daily_coverage_path,
        pairwise_coverage_path,
    ]
    missing = [p for p in expected_files if not p.exists()]

    if missing:
        st.warning("结果文件不完整。")
        st.code("\n".join(str(p) for p in missing), language="text")
        return

    pool_daily = load_csv_if_exists(str(pool_daily_path))
    pool_summary = load_csv_if_exists(str(pool_summary_path))
    pairwise_daily = load_csv_if_exists(str(pairwise_daily_path))
    pairwise_summary = load_csv_if_exists(str(pairwise_summary_path))
    daily_coverage = load_csv_if_exists(str(daily_coverage_path))
    pairwise_coverage = load_csv_if_exists(str(pairwise_coverage_path))

    if pool_summary.empty:
        st.warning("pool_summary is empty.")
        return

    all_horizons = sorted(pool_summary["horizon"].dropna().astype(str).unique().tolist())
    all_pools = sorted(pool_summary["pool_name"].dropna().astype(str).unique().tolist())
    all_pairs = sorted(pairwise_summary["pair"].dropna().astype(str).unique().tolist()) if "pair" in pairwise_summary.columns else []

    f1, f2, f3 = st.columns([1, 2, 2])
    selected_horizon = f1.selectbox("Horizon", options=all_horizons, index=0, key="multi_pool_horizon")
    selected_pool_filter = f2.multiselect("Display pools", options=all_pools, default=all_pools, key="multi_pool_display_pools")
    selected_pair = f3.selectbox("Pair", options=all_pairs, index=0, key="multi_pool_pair") if all_pairs else None

    st.caption(f"Current output dir: `{last_output_dir}`")

    summary_view = pool_summary[
        (pool_summary["horizon"] == selected_horizon)
        & (pool_summary["pool_name"].isin(selected_pool_filter))
    ].copy()

    if not summary_view.empty:
        m1, m2, m3, m4 = st.columns(4)
        best_excess = summary_view.sort_values("daily_mean_excess_return_pct", ascending=False).iloc[0] if "daily_mean_excess_return_pct" in summary_view.columns else None
        best_up = summary_view.sort_values("daily_mean_pool_up_ratio", ascending=False).iloc[0] if "daily_mean_pool_up_ratio" in summary_view.columns else None
        m1.metric("Pools", f"{summary_view['pool_name'].nunique():,}")
        m2.metric("Best excess pool", str(best_excess["pool_name"]) if best_excess is not None else "-")
        m3.metric("Best excess", fmt_num(best_excess["daily_mean_excess_return_pct"], 4) if best_excess is not None else "-")
        m4.metric("Best up ratio", fmt_num(best_up["daily_mean_pool_up_ratio"], 4) if best_up is not None else "-")

    tab_summary, tab_daily, tab_pair, tab_coverage, tab_raw = st.tabs(
        ["Pool summary", "Daily charts", "Pool1 vs Pool2", "Coverage", "Raw files"]
    )

    with tab_summary:
        summary_cols = [
            "pool_name",
            "horizon",
            "trading_days",
            "valid_days",
            "total_signal_rows",
            "daily_mean_signal_count",
            "daily_median_signal_count",
            "daily_mean_pool_return_pct",
            "daily_median_pool_return_pct",
            "daily_mean_pool_up_ratio",
            "daily_mean_market_return_pct",
            "daily_mean_market_up_ratio",
            "daily_mean_excess_return_pct",
            "daily_median_excess_return_pct",
            "daily_mean_excess_up_ratio",
            "positive_excess_return_day_ratio",
            "weighted_pool_avg_return_pct",
            "weighted_market_avg_return_pct",
            "weighted_excess_avg_return_pct",
        ]
        summary_cols = [c for c in summary_cols if c in summary_view.columns]
        sort_by = "daily_mean_excess_return_pct" if "daily_mean_excess_return_pct" in summary_view.columns else summary_cols[0]
        st.dataframe(
            clean_display_df(summary_view[summary_cols].sort_values(sort_by, ascending=False)),
            use_container_width=True,
            height=380,
        )
        show_download(summary_view, f"pool_summary_{selected_horizon}.csv", "Download pool summary")

    with tab_daily:
        daily_view = pool_daily[
            (pool_daily["horizon"] == selected_horizon)
            & (pool_daily["pool_name"].isin(selected_pool_filter))
        ].copy()

        if daily_view.empty:
            st.info("No daily metrics for current filter.")
        else:
            count_pivot = daily_view.pivot_table(index="date", columns="pool_name", values="signal_count", aggfunc="sum").sort_index()
            st.plotly_chart(
                build_line_chart_from_pivot(count_pivot, "出票数量", "signal_count", height=430),
                use_container_width=True,
            )

            c1, c2 = st.columns(2)
            ret_pivot = daily_view.pivot_table(index="date", columns="pool_name", values="pool_avg_return_pct", aggfunc="mean").sort_index()
            up_pivot = daily_view.pivot_table(index="date", columns="pool_name", values="pool_up_ratio", aggfunc="mean").sort_index()
            excess_pivot = daily_view.pivot_table(index="date", columns="pool_name", values="excess_avg_return_pct", aggfunc="mean").sort_index()
            excess_up_pivot = daily_view.pivot_table(index="date", columns="pool_name", values="excess_up_ratio", aggfunc="mean").sort_index() if "excess_up_ratio" in daily_view.columns else pd.DataFrame()

            with c1:
                st.plotly_chart(build_line_chart_from_pivot(ret_pivot, "平均收益", "return_pct", height=360), use_container_width=True)
            with c2:
                st.plotly_chart(build_line_chart_from_pivot(up_pivot, "上涨率", "up_ratio", height=360), use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                st.plotly_chart(build_line_chart_from_pivot(excess_pivot, "超额收益", "excess_return_pct", height=360), use_container_width=True)
            with c4:
                if not excess_up_pivot.empty:
                    st.plotly_chart(build_line_chart_from_pivot(excess_up_pivot, "超额上涨率", "excess_up_ratio", height=360), use_container_width=True)
                else:
                    st.info("当前结果没有 excess_up_ratio 列。")

    with tab_pair:
        pair_summary_view = pairwise_summary[pairwise_summary["horizon"] == selected_horizon].copy()
        pair_cols = [
            "pair",
            "horizon",
            "compare_days",
            "both_active_days",
            "pool_a_mean_count",
            "pool_b_mean_count",
            "mean_common_count",
            "mean_jaccard_ratio",
            "pool_a_mean_return_pct",
            "pool_b_mean_return_pct",
            "a_minus_b_mean_return_pct",
            "a_minus_b_median_return_pct",
            "pool_a_mean_up_ratio",
            "pool_b_mean_up_ratio",
            "a_minus_b_mean_up_ratio",
            "pool_a_mean_excess_pct",
            "pool_b_mean_excess_pct",
            "a_minus_b_mean_excess_pct",
            "a_win_days",
            "b_win_days",
            "a_win_ratio",
        ]
        pair_cols = [c for c in pair_cols if c in pair_summary_view.columns]
        pair_sort = "a_minus_b_mean_return_pct" if "a_minus_b_mean_return_pct" in pair_summary_view.columns else pair_cols[0]
        st.dataframe(
            clean_display_df(pair_summary_view[pair_cols].sort_values(pair_sort, ascending=False)),
            use_container_width=True,
            height=340,
        )
        show_download(pair_summary_view, f"pairwise_summary_{selected_horizon}.csv", "Download pairwise summary")

        if selected_pair:
            st.markdown(f"**Pair daily detail: {selected_pair}**")
            pair_daily_view = pairwise_daily[
                (pairwise_daily["horizon"] == selected_horizon)
                & (pairwise_daily["pair"] == selected_pair)
            ].copy()

            if not pair_daily_view.empty:
                chart_cols = [
                    "a_minus_b_avg_return_pct",
                    "a_minus_b_excess_avg_return_pct",
                    "a_minus_b_up_ratio",
                ]
                chart_cols = [c for c in chart_cols if c in pair_daily_view.columns]

                if chart_cols:
                    st.plotly_chart(
                        build_line_chart_from_pivot(
                            pair_daily_view.set_index("date")[chart_cols],
                            title="Pair daily difference",
                            yaxis_title="difference",
                            height=420,
                        ),
                        use_container_width=True,
                    )

                st.dataframe(
                    clean_display_df(pair_daily_view.sort_values("date", ascending=False)),
                    use_container_width=True,
                    height=360,
                )

    with tab_coverage:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**每日覆盖情况**")
            if not daily_coverage.empty:
                st.dataframe(clean_display_df(daily_coverage.sort_values("date", ascending=False)), use_container_width=True, height=420)
                show_download(daily_coverage, "daily_coverage.csv", "Download daily coverage")
        with c2:
            st.markdown("**Pairwise overlap coverage**")
            if selected_pair and not pairwise_coverage.empty:
                pair_cov_view = pairwise_coverage[pairwise_coverage["pair"] == selected_pair].copy()
                st.dataframe(clean_display_df(pair_cov_view.sort_values("date", ascending=False)), use_container_width=True, height=420)
                show_download(pair_cov_view, f"pairwise_coverage_{selected_pair.replace(' ', '_')}.csv", "Download pairwise coverage")

    with tab_raw:
        raw_map = {
            "1_pool_daily_metrics.csv": pool_daily,
            "2_pool_summary.csv": pool_summary,
            "3_pairwise_daily_compare.csv": pairwise_daily,
            "4_pairwise_summary.csv": pairwise_summary,
            "5_daily_coverage.csv": daily_coverage,
            "6_pairwise_coverage_daily.csv": pairwise_coverage,
        }
        raw_name = st.selectbox("Raw file", list(raw_map.keys()))
        raw_df = raw_map[raw_name]
        st.dataframe(clean_display_df(raw_df.head(1000)), use_container_width=True, height=520)
        show_download(raw_df, raw_name, f"Download {raw_name}")


# ======================================================================================
# App router
# ======================================================================================


with st.sidebar:
    st.header("Page")
    page = st.radio(
        "Select page",
        [
            "Single Pool Viewer",
            "Analyze Pool Indicator",
            "N Pool Compare Runner",
        ],
    )

    reload_btn = st.button("Reload data", use_container_width=True)
    if reload_btn:
        st.cache_data.clear()
        st.rerun()


if page == "Single Pool Viewer":
    render_pool_viewer()
elif page == "Analyze Pool Indicator":
    render_analyze_pool_indicator()
elif page == "N Pool Compare Runner":
    render_multi_pool_compare_runner()
