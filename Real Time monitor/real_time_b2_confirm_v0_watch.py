from __future__ import annotations

import importlib.util
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


# =============================================================================
# User config
# =============================================================================

PROJECT_ROOT = Path(r"C:\Users\zyf37\Desktop\Trade Backtest v1.0.0")

REALTIME_DIR = PROJECT_ROOT / "Real Time monitor"

B2_STRATEGY_PATH = PROJECT_ROOT / "selection_strategies" / "b2_confirm_select_strategy_v0.py"
RENKO_V4_STRATEGY_PATH = PROJECT_ROOT / "selection_strategies" / "renko_chart_select_strategy_v4.py"

MARKET_CACHE_DIR = Path(
    r"C:\Users\zyf37\Desktop\BackTest Data\market_cache\daily_bars_by_symbol"
)

OUTPUT_DIR = REALTIME_DIR / "output"

INTERVAL_SECONDS = 60
TOP_N = 50

# Startup local history tail length per symbol.
HISTORY_TAIL_N = 160

# Only trade Shanghai/Shenzhen main board.
ONLY_MAIN_BOARD = True

EXPORT_CSV = True

PRINT_WATCH_LIST = True
WATCH_TOP_N = 30

MIN_PRICE = 3.0
MAX_PRICE = 50.0

EXCLUDE_ST = True
EXCLUDE_NEW_STOCK = True

# Eastmoney f5 volume is usually in lots/shou.
# If local history volume is shares, use 100. If local history volume is lots, use 1.
REALTIME_VOLUME_MULTIPLIER = 100.0

# KDJ fallback params.
KDJ_N = 9
KDJ_M1 = 3
KDJ_M2 = 3

# If True, print first few strategy errors each round for debugging.
PRINT_ERROR_SAMPLE = True
ERROR_SAMPLE_N = 8


# =============================================================================
# Time helpers
# =============================================================================

def now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def date_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def today_ts() -> pd.Timestamp:
    return pd.Timestamp(datetime.now().date())


# =============================================================================
# Load external strategies
# =============================================================================

def load_strategy_module(strategy_path: Path, module_name: str) -> Any:
    if not strategy_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {strategy_path}")

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    spec = importlib.util.spec_from_file_location(module_name, strategy_path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load strategy module from: {strategy_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


# =============================================================================
# Eastmoney realtime fetch: main board only
# =============================================================================

def fetch_main_board_realtime() -> pd.DataFrame:
    """
    Fetch realtime quotes, then keep Shanghai/Shenzhen main board only:
    - 00xxxx: Shenzhen main board
    - 60xxxx: Shanghai main board
    """

    url = "https://push2.eastmoney.com/api/qt/clist/get"

    fields = ",".join([
        "f12",   # code
        "f14",   # name
        "f2",    # latest price
        "f3",    # return pct
        "f4",    # change
        "f5",    # volume, usually lots
        "f6",    # amount
        "f7",    # amplitude pct
        "f8",    # turnover pct
        "f10",   # volume ratio
        "f15",   # high
        "f16",   # low
        "f17",   # open
        "f18",   # prev close
        "f20",   # total market value
        "f21",   # float market value
        "f23",   # pb
    ])

    # Shanghai/Shenzhen A-share ranges; local filter keeps 00/60.
    fs = "m:1+t:2,m:1+t:23,m:0+t:6,m:0+t:80"

    page = 1
    page_size = 500
    rows: list[dict] = []

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://quote.eastmoney.com/",
    }

    while True:
        params = {
            "pn": page,
            "pz": page_size,
            "po": "1",
            "np": "1",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": "2",
            "invt": "2",
            "fid": "f3",
            "fs": fs,
            "fields": fields,
            "_": int(time.time() * 1000),
        }

        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()

        data = resp.json().get("data") or {}
        diff = data.get("diff") or []

        if not diff:
            break

        rows.extend(diff)

        if len(diff) < page_size:
            break

        page += 1

    if not rows:
        raise RuntimeError("Eastmoney returned no realtime quote rows")

    raw = pd.DataFrame(rows)

    rename_map = {
        "f12": "code",
        "f14": "name",
        "f2": "close",
        "f3": "realtime_return_pct",
        "f4": "realtime_change",
        "f5": "realtime_volume_raw",
        "f6": "amount",
        "f7": "amplitude_pct",
        "f8": "turnover_pct",
        "f10": "volume_ratio",
        "f15": "high",
        "f16": "low",
        "f17": "open",
        "f18": "prev_close",
        "f20": "total_market_value",
        "f21": "float_market_value",
        "f23": "pb",
    }

    df = raw.rename(columns=rename_map).copy()

    df["code"] = df["code"].astype(str).str.zfill(6)
    df["name"] = df["name"].astype(str)

    if ONLY_MAIN_BOARD:
        df = df[df["code"].str.startswith(("00", "60"))].copy()

    numeric_cols = [
        "close",
        "realtime_return_pct",
        "realtime_change",
        "realtime_volume_raw",
        "amount",
        "amplitude_pct",
        "turnover_pct",
        "volume_ratio",
        "high",
        "low",
        "open",
        "prev_close",
        "total_market_value",
        "float_market_value",
        "pb",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["close"].notna() & (df["close"] > 0)].copy()
    df = df[df["open"].notna() & (df["open"] > 0)].copy()
    df = df[df["high"].notna() & (df["high"] > 0)].copy()
    df = df[df["low"].notna() & (df["low"] > 0)].copy()

    if EXCLUDE_ST:
        df = df[~df["name"].str.contains("ST", case=False, na=False)].copy()

    if EXCLUDE_NEW_STOCK:
        df = df[~df["name"].str.startswith("N")].copy()

    df = df[df["close"].between(MIN_PRICE, MAX_PRICE)].copy()

    df["volume"] = df["realtime_volume_raw"] * REALTIME_VOLUME_MULTIPLIER
    df["date"] = today_ts()

    df["dist_high_pct"] = (
        (df["high"] - df["close"]) / df["high"].replace(0, np.nan) * 100.0
    )
    df["price_above_open"] = df["close"] >= df["open"]

    return df


# =============================================================================
# Historical data load
# =============================================================================

def normalize_code(x: object) -> str:
    s = str(x).strip()

    if "#" in s:
        s = s.split("#")[-1]

    s = s.replace(".SZ", "").replace(".SH", "")
    s = s.replace("SZ", "").replace("SH", "")
    s = s.strip()

    return s.zfill(6)[-6:]


def extract_code_from_path(path: Path) -> str | None:
    stem = path.stem

    parts = stem.replace(".", "#").split("#")
    for p in reversed(parts):
        p = p.strip()
        if p.isdigit() and len(p) == 6:
            return p

    digits = "".join(ch for ch in stem if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]

    return None


def standardize_history_df(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """
    Compatible local market cache standardization.
    Required output columns:
        date, open, high, low, close, volume
    """

    out = df.copy()

    rename_candidates = {
        "日期": "date",
        "时间": "date",
        "trade_date": "date",
        "datetime": "date",
        "交易日期": "date",

        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",

        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "VOLUME": "volume",
        "AMOUNT": "amount",
    }

    out = out.rename(columns={c: rename_candidates[c] for c in out.columns if c in rename_candidates})

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)

    if missing:
        raise KeyError(f"{code} historical data missing columns: {sorted(missing)}")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out[out["date"].notna()].copy()

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "amount" in out.columns:
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce")

    out = out.dropna(subset=["open", "high", "low", "close", "volume"]).copy()

    out = (
        out.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    out["code"] = code

    return out


def load_history_cache(market_cache_dir: Path) -> dict[str, pd.DataFrame]:
    if not market_cache_dir.exists():
        raise FileNotFoundError(f"Market cache dir not found: {market_cache_dir}")

    files = sorted(market_cache_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {market_cache_dir}")

    history: dict[str, pd.DataFrame] = {}

    print("=" * 120)
    print(f"[{now_str()}] Start loading local market cache")
    print(f"market_cache_dir: {market_cache_dir}")
    print(f"parquet files    : {len(files)}")
    print("=" * 120)

    loaded = 0
    skipped = 0
    failed = 0

    pbar = tqdm(files, desc="Loading local market cache", unit="file", ncols=120)

    for path in pbar:
        code = extract_code_from_path(path)

        if not code:
            skipped += 1
            pbar.set_postfix(loaded=loaded, skipped=skipped, failed=failed)
            continue

        if ONLY_MAIN_BOARD and not code.startswith(("00", "60")):
            skipped += 1
            pbar.set_postfix(loaded=loaded, skipped=skipped, failed=failed)
            continue

        try:
            df = pd.read_parquet(path)
            df = standardize_history_df(df, code)

            # Remove today rows to avoid conflict with realtime dynamic T0.
            df = df[df["date"] < today_ts()].copy()

            if len(df) < 30:
                skipped += 1
                pbar.set_postfix(loaded=loaded, skipped=skipped, failed=failed)
                continue

            history[code] = df.tail(HISTORY_TAIL_N).reset_index(drop=True)
            loaded += 1

        except Exception:
            failed += 1

        pbar.set_postfix(loaded=loaded, skipped=skipped, failed=failed)

    print("=" * 120)
    print(f"[{now_str()}] Local market cache loaded")
    print(f"loaded : {loaded}")
    print(f"skipped: {skipped}")
    print(f"failed : {failed}")
    print("=" * 120)

    if not history:
        raise RuntimeError("No valid historical data loaded.")

    return history


# =============================================================================
# Indicator helpers
# =============================================================================

def tdx_sma(series: pd.Series, n: int, m: int) -> pd.Series:
    """
    TongDaXin SMA(X,N,M):
        Y = (M*X + (N-M)*Y') / N
    pandas ewm alpha=m/n is a practical approximation.
    """
    return series.ewm(alpha=m / n, adjust=False).mean()


def ensure_j_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    If no J column exists, compute KDJ J on the fly.
    B2 strategy accepts j/J/kdj_j etc.
    """

    out = df.copy()

    j_candidates = ["j", "J", "kdj_j", "KDJ_J", "j_value", "J_VALUE"]
    if any(c in out.columns for c in j_candidates):
        return out

    low_n = out["low"].rolling(KDJ_N, min_periods=1).min()
    high_n = out["high"].rolling(KDJ_N, min_periods=1).max()

    rsv = (out["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100.0
    rsv = rsv.fillna(50.0)

    k = tdx_sma(rsv, KDJ_M1, 1)
    d = tdx_sma(k, KDJ_M2, 1)
    j = 3.0 * k - 2.0 * d

    out["j"] = j

    return out


def try_add_project_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try computing project-level indicators for Renko strategies.
    If it fails, return original df so B2 can still work.
    """
    out = df.copy()

    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        from indicators import add_all_indicators

        out = add_all_indicators(out)

    except Exception:
        return df

    return out


# =============================================================================
# Dynamic T0 + strategy execution
# =============================================================================

def build_dynamic_t0_row(r: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([{
        "date": today_ts(),
        "open": float(r["open"]),
        "high": float(r["high"]),
        "low": float(r["low"]),
        "close": float(r["close"]),
        "volume": float(r["volume"]),
        "amount": float(r["amount"]) if "amount" in r and pd.notna(r["amount"]) else np.nan,
        "code": str(r["code"]),
        "name": str(r["name"]),
        "realtime_return_pct": (
            float(r["realtime_return_pct"]) if pd.notna(r["realtime_return_pct"]) else np.nan
        ),
        "volume_ratio": float(r["volume_ratio"]) if pd.notna(r["volume_ratio"]) else np.nan,
        "turnover_pct": float(r["turnover_pct"]) if pd.notna(r["turnover_pct"]) else np.nan,
        "dist_high_pct": float(r["dist_high_pct"]) if pd.notna(r["dist_high_pct"]) else np.nan,
    }])


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_bool(x: Any) -> bool:
    try:
        if pd.isna(x):
            return False
        return bool(x)
    except Exception:
        return False


def run_one_symbol_realtime(
    code: str,
    realtime_row: pd.Series,
    history: dict[str, pd.DataFrame],
    b2_strategy_module: Any,
    renko_strategy_module: Any,
) -> dict[str, Any] | None:
    hist = history.get(code)

    if hist is None or hist.empty:
        return None

    t0 = build_dynamic_t0_row(realtime_row)

    df = pd.concat([hist, t0], ignore_index=True)
    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    df = ensure_j_column(df)
    df_with_indicators = try_add_project_indicators(df)

    # -------------------------------------------------------------------------
    # Run B2 Confirm V0
    # -------------------------------------------------------------------------
    b2_selected = False
    b2_error = ""
    b2_last = None

    try:
        b2_out = b2_strategy_module.select(df.copy())
        b2_last = b2_out.iloc[-1]
        b2_selected = bool(b2_last.get("selected", False))
    except Exception as e:
        b2_error = f"{type(e).__name__}: {e}"

    # -------------------------------------------------------------------------
    # Run Renko v4
    # -------------------------------------------------------------------------
    renko_selected = False
    renko_error = ""
    renko_last = None

    try:
        renko_out = renko_strategy_module.select(df_with_indicators.copy())
        renko_last = renko_out.iloc[-1]
        renko_selected = bool(renko_last.get("selected", False))
    except Exception as e:
        renko_error = f"{type(e).__name__}: {e}"

    if b2_last is None and renko_last is None:
        return {
            "date": today_ts().strftime("%Y-%m-%d"),
            "time": now_str(),
            "code": code,
            "name": realtime_row.get("name", ""),
            "selected": False,
            "scheme": "ERROR",
            "b2_selected": False,
            "renko_v4_selected": False,
            "watch_score": -999,
            "b2_error": b2_error,
            "renko_error": renko_error,
        }

    last = b2_last if b2_last is not None else renko_last

    if b2_selected and renko_selected:
        scheme = "BOTH"
    elif b2_selected:
        scheme = "B2_CONFIRM_V0"
    elif renko_selected:
        scheme = "RENKO_V4"
    else:
        scheme = "NONE"

    selected = b2_selected or renko_selected

    watch_score = 0
    if b2_last is not None:
        watch_score += int(safe_bool(b2_last.get("b1_within_b2_lookback", False))) * 30
        watch_score += int(safe_bool(b2_last.get("b2_return_ok", False))) * 20
        watch_score += int(safe_bool(b2_last.get("b2_bullish_candle", False))) * 10
        watch_score += int(safe_bool(b2_last.get("b2_volume_up", False))) * 15
        watch_score += int(safe_bool(b2_last.get("b2_j_ok", False))) * 15
        watch_score += int(safe_bool(b2_last.get("b2_upper_shadow_ok", False))) * 10

    if renko_selected:
        watch_score += 100

    def get_from_any(col: str, default: Any = np.nan) -> Any:
        if renko_last is not None and col in renko_last.index:
            return renko_last.get(col, default)
        if b2_last is not None and col in b2_last.index:
            return b2_last.get(col, default)
        if last is not None and col in last.index:
            return last.get(col, default)
        return default

    return {
        "date": today_ts().strftime("%Y-%m-%d"),
        "time": now_str(),
        "code": code,
        "name": realtime_row.get("name", ""),

        "selected": selected,
        "scheme": scheme,
        "b2_selected": b2_selected,
        "renko_v4_selected": renko_selected,
        "watch_score": watch_score,

        "price": safe_float(get_from_any("close", np.nan)),
        "open": safe_float(get_from_any("open", np.nan)),
        "high": safe_float(get_from_any("high", np.nan)),
        "low": safe_float(get_from_any("low", np.nan)),
        "volume": safe_float(get_from_any("volume", np.nan)),
        "prev_volume": safe_float(get_from_any("prev_volume", np.nan)),
        "volume_ratio_prev": safe_float(get_from_any("volume_ratio_prev", np.nan)),
        "daily_return_pct": safe_float(get_from_any("daily_return_pct", np.nan)),
        "j": safe_float(get_from_any("j", get_from_any("J", get_from_any("kdj_j", np.nan)))),
        "upper_shadow_ratio": safe_float(get_from_any("upper_shadow_ratio", np.nan)),
        "b1_days_ago_for_b2": safe_float(get_from_any("b1_days_ago_for_b2", np.nan)),
        "b2_quality_score": safe_float(get_from_any("b2_quality_score", np.nan)),

        # Renko diagnostics, if available.
        "renko_score": safe_float(get_from_any("score", np.nan)),
        "renko_score_pct": safe_float(get_from_any("score_pct", np.nan)),
        "brick_value": safe_float(get_from_any("brick_value", np.nan)),
        "brick_prev_1": safe_float(get_from_any("brick_prev_1", np.nan)),
        "brick_prev_2": safe_float(get_from_any("brick_prev_2", np.nan)),

        "realtime_return_pct": safe_float(realtime_row.get("realtime_return_pct", np.nan)),
        "volume_ratio": safe_float(realtime_row.get("volume_ratio", np.nan)),
        "turnover_pct": safe_float(realtime_row.get("turnover_pct", np.nan)),
        "dist_high_pct": safe_float(realtime_row.get("dist_high_pct", np.nan)),

        "b1_within_b2_lookback": safe_bool(get_from_any("b1_within_b2_lookback", False)),
        "b2_return_ok": safe_bool(get_from_any("b2_return_ok", False)),
        "b2_bullish_candle": safe_bool(get_from_any("b2_bullish_candle", False)),
        "b2_volume_up": safe_bool(get_from_any("b2_volume_up", False)),
        "b2_j_ok": safe_bool(get_from_any("b2_j_ok", False)),
        "b2_upper_shadow_ok": safe_bool(get_from_any("b2_upper_shadow_ok", False)),
        "b2_double_volume": safe_bool(get_from_any("b2_double_volume", False)),
        "b2_tiny_upper_shadow": safe_bool(get_from_any("b2_tiny_upper_shadow", False)),

        "b2_error": b2_error,
        "renko_error": renko_error,
    }


def run_realtime_selection(
    realtime_df: pd.DataFrame,
    history: dict[str, pd.DataFrame],
    b2_strategy_module: Any,
    renko_strategy_module: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    realtime_df = realtime_df.copy()
    realtime_df["code"] = realtime_df["code"].astype(str).str.zfill(6)

    realtime_df = realtime_df[realtime_df["code"].isin(history.keys())].copy()

    for _, r in realtime_df.iterrows():
        code = str(r["code"]).zfill(6)

        result = run_one_symbol_realtime(
            code=code,
            realtime_row=r,
            history=history,
            b2_strategy_module=b2_strategy_module,
            renko_strategy_module=renko_strategy_module,
        )

        if result is not None:
            rows.append(result)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    out = out.sort_values(
        by=["selected", "scheme", "watch_score", "daily_return_pct", "volume_ratio_prev"],
        ascending=[False, True, False, False, False],
    ).reset_index(drop=True)

    return out


# =============================================================================
# Print / export
# =============================================================================

def fmt_float(x: Any, width: int = 8, precision: int = 3) -> str:
    try:
        if pd.isna(x):
            return " " * (width - 3) + "nan"
        return f"{float(x):{width}.{precision}f}"
    except Exception:
        return " " * (width - 3) + "nan"


def print_error_samples(result: pd.DataFrame) -> None:
    if not PRINT_ERROR_SAMPLE or result.empty:
        return

    err = result[
        (result.get("b2_error", "") != "") | (result.get("renko_error", "") != "")
    ].copy()

    if err.empty:
        return

    print("\n>>> ERROR_SAMPLE")
    print("-" * 180)
    for i, (_, r) in enumerate(err.head(ERROR_SAMPLE_N).iterrows(), start=1):
        print(
            f"{i:>2d}. {r.get('code')} {str(r.get('name', '')):<8s} | "
            f"b2_error={r.get('b2_error', '')} | renko_error={r.get('renko_error', '')}"
        )


def print_selected(
    result: pd.DataFrame,
    realtime_count: int,
    fetch_cost: float,
    calc_cost: float,
) -> None:
    print("\n" + "=" * 180)

    selected_count = 0
    b2_count = 0
    renko_count = 0
    both_count = 0

    if not result.empty:
        selected_count = int(result["selected"].sum())
        b2_count = int(result["b2_selected"].sum())
        renko_count = int(result["renko_v4_selected"].sum())
        both_count = int((result["scheme"] == "BOTH").sum())

    print(
        f"[{now_str()}] Realtime strategy pool | "
        f"realtime_main_board={realtime_count} | calculated={len(result)} | "
        f"selected={selected_count} | B2={b2_count} | RENKO_V4={renko_count} | BOTH={both_count} | "
        f"fetch={fetch_cost:.2f}s | calc={calc_cost:.2f}s"
    )
    print("=" * 180)

    if result.empty:
        print("No calculated result.")
        return

    selected = result[result["selected"] == True].copy()

    if selected.empty:
        print("当前没有符合 B2 Confirm V0 或 Renko v4 的股票。")
    else:
        print("\n>>> SELECTED")
        print("-" * 180)

        show = selected.head(TOP_N)

        for i, (_, r) in enumerate(show.iterrows(), start=1):
            print(
                f"{i:>2d}. {r['code']} {str(r['name']):<8s} | "
                f"scheme={str(r['scheme']):<14s} | "
                f"price={fmt_float(r['price'], 8, 3)} | "
                f"ret={fmt_float(r['daily_return_pct'], 7, 2)}% | "
                f"J={fmt_float(r['j'], 7, 2)} | "
                f"vol/prev={fmt_float(r['volume_ratio_prev'], 6, 2)} | "
                f"upper={fmt_float(r['upper_shadow_ratio'], 6, 2)} | "
                f"brick={fmt_float(r['brick_value'], 7, 2)} | "
                f"renko_score={fmt_float(r['renko_score'], 7, 2)} | "
                f"renko_pct={fmt_float(r['renko_score_pct'], 7, 2)} | "
                f"B1_days={fmt_float(r['b1_days_ago_for_b2'], 4, 0)} | "
                f"dist_high={fmt_float(r['dist_high_pct'], 6, 2)}% | "
                f"turnover={fmt_float(r['turnover_pct'], 6, 2)}%"
            )

    if PRINT_WATCH_LIST:
        watch = result[result["selected"] == False].copy()
        watch = watch[watch["watch_score"] > 0].copy()
        watch = watch.sort_values(
            by=["watch_score", "daily_return_pct", "volume_ratio_prev"],
            ascending=[False, False, False],
        ).head(WATCH_TOP_N)

        if not watch.empty:
            print("\n>>> WATCH_LIST_NOT_SELECTED")
            print("-" * 180)

            for i, (_, r) in enumerate(watch.iterrows(), start=1):
                print(
                    f"{i:>2d}. {r['code']} {str(r['name']):<8s} | "
                    f"scheme={str(r['scheme']):<14s} | "
                    f"watch={r['watch_score']:3.0f} | "
                    f"price={fmt_float(r['price'], 8, 3)} | "
                    f"ret={fmt_float(r['daily_return_pct'], 7, 2)}% | "
                    f"J={fmt_float(r['j'], 7, 2)} | "
                    f"vol/prev={fmt_float(r['volume_ratio_prev'], 6, 2)} | "
                    f"upper={fmt_float(r['upper_shadow_ratio'], 6, 2)} | "
                    f"brick={fmt_float(r['brick_value'], 7, 2)} | "
                    f"B1_recent={r['b1_within_b2_lookback']} | "
                    f"return_ok={r['b2_return_ok']} | "
                    f"bull={r['b2_bullish_candle']} | "
                    f"vol_up={r['b2_volume_up']} | "
                    f"j_ok={r['b2_j_ok']} | "
                    f"upper_ok={r['b2_upper_shadow_ok']}"
                )

    print_error_samples(result)


def export_result(result: pd.DataFrame) -> None:
    if not EXPORT_CSV or result.empty:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_path = OUTPUT_DIR / f"b2_renko_v4_realtime_selected_{date_str()}.csv"
    all_path = OUTPUT_DIR / f"b2_renko_v4_realtime_all_{date_str()}.csv"

    result.to_csv(all_path, index=False, encoding="utf-8-sig")

    selected = result[result["selected"] == True].copy()
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")

    print(f"\n已导出全部结果: {all_path}")
    print(f"已导出入选结果: {selected_path}")


def print_config() -> None:
    print("=" * 180)
    print("B2 Confirm V0 + Renko v4 realtime stock pool")
    print("=" * 180)
    print(f"PROJECT_ROOT              : {PROJECT_ROOT}")
    print(f"REALTIME_DIR              : {REALTIME_DIR}")
    print(f"B2_STRATEGY_PATH          : {B2_STRATEGY_PATH}")
    print(f"RENKO_V4_STRATEGY_PATH    : {RENKO_V4_STRATEGY_PATH}")
    print(f"MARKET_CACHE_DIR          : {MARKET_CACHE_DIR}")
    print(f"OUTPUT_DIR                : {OUTPUT_DIR}")
    print(f"Refresh interval          : {INTERVAL_SECONDS} seconds")
    print(f"Only main board 00 / 60   : {ONLY_MAIN_BOARD}")
    print(f"History tail rows         : {HISTORY_TAIL_N}")
    print(f"Realtime volume multiplier: {REALTIME_VOLUME_MULTIPLIER}")
    print(f"Price range               : {MIN_PRICE} ~ {MAX_PRICE}")
    print(f"Exclude ST                : {EXCLUDE_ST}")
    print(f"Exclude N new stock       : {EXCLUDE_NEW_STOCK}")
    print(f"Print watch list          : {PRINT_WATCH_LIST}")
    print(f"CSV export                : {EXPORT_CSV}")
    print("Press Ctrl + C to stop")
    print("=" * 180)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print_config()

    b2_strategy_module = load_strategy_module(
        B2_STRATEGY_PATH,
        module_name="b2_confirm_select_strategy_v0",
    )

    renko_strategy_module = load_strategy_module(
        RENKO_V4_STRATEGY_PATH,
        module_name="renko_chart_select_strategy_v4",
    )

    history = load_history_cache(MARKET_CACHE_DIR)

    while True:
        try:
            round_start = time.time()

            print(f"\n[{now_str()}] Start fetching realtime main board quotes...")

            fetch_start = time.time()
            realtime_df = fetch_main_board_realtime()
            fetch_cost = time.time() - fetch_start

            print(
                f"[{now_str()}] Realtime quotes fetched: {len(realtime_df)} rows | "
                f"fetch_cost={fetch_cost:.2f}s"
            )

            calc_start = time.time()

            result = run_realtime_selection(
                realtime_df=realtime_df,
                history=history,
                b2_strategy_module=b2_strategy_module,
                renko_strategy_module=renko_strategy_module,
            )

            calc_cost = time.time() - calc_start

            print_selected(
                result=result,
                realtime_count=len(realtime_df),
                fetch_cost=fetch_cost,
                calc_cost=calc_cost,
            )

            export_result(result)

            round_cost = time.time() - round_start
            sleep_seconds = max(1, INTERVAL_SECONDS - round_cost)

            print(
                f"[{now_str()}] Round cost={round_cost:.2f}s | "
                f"next wait={sleep_seconds:.2f}s"
            )

            time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            print("\nManual stop.")
            break

        except Exception as e:
            print(f"[{now_str()}] ERROR: {type(e).__name__}: {e}")
            time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
