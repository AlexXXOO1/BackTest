from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================
# Pool Quality Analysis
#
# Main purpose:
# - Frequently compare v0 pool with any vn pool.
# - Cache market index so fixed market data does not need to be reloaded every run.
# - Cache old/v0 forward returns so v0 does not need to be recalculated every run.
#
# Default usage:
#   python .\tools\pool_quality_analysis.py --new v2
#
# After updating market TXT data:
#   python .\tools\pool_quality_analysis.py --new v2 --rebuild-market-cache --rebuild-old-cache
# ============================================================


DEFAULT_DATA_ROOT = Path(r"C:\Users\zyf37\Desktop\BackTest Data")
DEFAULT_POOLS_DIR = DEFAULT_DATA_ROOT / "pools"
DEFAULT_DATA_DIR = DEFAULT_DATA_ROOT / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "pools"
DEFAULT_CACHE_DIR = DEFAULT_POOLS_DIR / "cache"

DEFAULT_OLD_STRATEGY = "v2"
DEFAULT_NEW_STRATEGY = "v1"

DEFAULT_MAIN_RETURN_COL = "t3_close_from_t1_open_pct"
DEFAULT_WORKERS = 8

STRATEGY_PREFIX = "renko_chart_select_strategy_"


DATE_CANDIDATES = ["date", "datetime", "trade_date", "signal_date", "日期"]
CODE_CANDIDATES = ["code", "symbol", "ts_code", "stock_code", "证券代码", "股票代码"]

NUMERIC_CN_RENAME = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
    "成交额": "amount",
}

ASCII_RENAME = {
    "date": "date",
    "datetime": "date",
    "trade_date": "date",
    "signal_date": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
}


@dataclass
class MarketCacheItem:
    path: Path
    df: Optional[pd.DataFrame]
    error: Optional[str] = None


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    if obj is None:
        return None

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def extract_6digit_code(value: Any) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    m = re.search(r"(\d{6})", text)
    return m.group(1) if m else None


def normalize_strategy_name(name: str) -> str:
    text = str(name).strip()

    if text.endswith(".parquet"):
        return Path(text).stem.replace("_pool", "")

    if text.startswith(STRATEGY_PREFIX):
        return text

    return STRATEGY_PREFIX + text


def strategy_to_short_name(strategy: str) -> str:
    if strategy.startswith(STRATEGY_PREFIX):
        return strategy.replace(STRATEGY_PREFIX, "")
    return strategy


def resolve_pool_path(value: str, pools_dir: Path) -> Path:
    text = str(value).strip()

    if text.endswith(".parquet"):
        return Path(text)

    strategy_name = normalize_strategy_name(text)
    return pools_dir / f"{strategy_name}_pool.parquet"


def build_output_path(old_value: str, new_value: str, output_dir: Path) -> Path:
    old_strategy = normalize_strategy_name(old_value)
    new_strategy = normalize_strategy_name(new_value)

    old_short = strategy_to_short_name(old_strategy)
    new_short = strategy_to_short_name(new_strategy)

    filename = f"pool_quality_analysis_{old_short}_vs_{new_short}.json"
    return output_dir / filename


def detect_col(cols: Iterable[str], candidates: list[str]) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in cols}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for c in cols:
        low = str(c).lower()
        if any(cand.lower() in low for cand in candidates):
            return c

    return None


def parse_date_series(raw: pd.Series) -> pd.Series:
    raw = raw.astype(str).str.strip()

    out = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")

    mask = out.isna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(raw.loc[mask], format="%Y/%m/%d", errors="coerce")

    mask = out.isna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(raw.loc[mask], format="%Y%m%d", errors="coerce")

    mask = out.isna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(raw.loc[mask], format="%d/%m/%Y", errors="coerce")

    mask = out.isna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(raw.loc[mask], format="%d-%m-%Y", errors="coerce")

    return out


def parse_mixed_date(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()
    raw = raw.str.replace("\ufeff", "", regex=False)

    out = parse_date_series(raw)

    slash_mask = raw.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$", na=False)
    if slash_mask.any():
        out.loc[slash_mask] = pd.to_datetime(
            raw.loc[slash_mask],
            format="%d/%m/%Y",
            errors="coerce",
        )

    ymd_mask = raw.str.match(r"^\d{8}$", na=False)
    if ymd_mask.any():
        out.loc[ymd_mask] = pd.to_datetime(
            raw.loc[ymd_mask],
            format="%Y%m%d",
            errors="coerce",
        )

    return out.dt.normalize()


def read_tdx_txt(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    encodings = ["utf-8-sig", "gbk", "gb18030", "utf-8"]
    last_error = None
    lines = None

    for enc in encodings:
        try:
            lines = path.read_text(encoding=enc, errors="strict").splitlines()
            break
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

    if lines is None:
        try:
            lines = path.read_text(encoding="gb18030", errors="ignore").splitlines()
        except Exception as e:
            return None, f"unreadable_text: {type(e).__name__}: {e}; last={last_error}"

    data_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if re.match(
            r"^(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{1,2}-\d{1,2}|\d{8})\s*,",
            line,
        ):
            data_lines.append(line)

    if not data_lines:
        return None, "no_data_rows_found"

    from io import StringIO

    csv_text = "date,open,high,low,close,volume,amount\n" + "\n".join(data_lines)

    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception as e:
        return None, f"csv_parse_failed: {type(e).__name__}: {e}"

    df["date"] = parse_mixed_date(df["date"])

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["date", "open", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    if df.empty:
        return None, "empty_after_cleaning"

    return df, None


def read_market_file(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return read_tdx_txt(path)

    try:
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".csv":
            df = pd.read_csv(path, encoding="utf-8-sig")
        else:
            return None, f"unsupported_suffix:{suffix}"
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="gb18030")
        except Exception as e:
            return None, f"read_failed: {type(e).__name__}: {e}"
    except Exception as e:
        return None, f"read_failed: {type(e).__name__}: {e}"

    rename = {}

    for col in df.columns:
        c = str(col).strip()

        if c in NUMERIC_CN_RENAME:
            rename[col] = NUMERIC_CN_RENAME[c]
        elif c.lower() in ASCII_RENAME:
            rename[col] = ASCII_RENAME[c.lower()]

    df = df.rename(columns=rename)

    if not {"date", "open", "close"}.issubset(set(df.columns)):
        return None, f"missing_required_cols:{list(df.columns)}"

    df["date"] = parse_mixed_date(df["date"])

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["date", "open", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    if df.empty:
        return None, "empty_after_cleaning"

    return df, None


def _load_market_file_for_index(
    path: Path,
) -> tuple[Optional[str], Optional[MarketCacheItem], Optional[str]]:
    code = extract_6digit_code(path.name)

    if not code:
        return None, None, "no_code"

    df, err = read_market_file(path)

    if err:
        return code, None, err

    return code, MarketCacheItem(path=path, df=df, error=None), None


def build_market_index(
    data_dir: Path,
    workers: int = DEFAULT_WORKERS,
) -> tuple[dict[str, MarketCacheItem], dict[str, Any]]:
    files = []

    for suffix in ["*.txt", "*.csv", "*.parquet"]:
        files.extend(data_dir.rglob(suffix))

    code_to_path: dict[str, Path] = {}

    for path in files:
        code = extract_6digit_code(path.name)

        if not code:
            continue

        if code not in code_to_path:
            code_to_path[code] = path

    load_items = list(code_to_path.items())

    index: dict[str, MarketCacheItem] = {}
    errors = Counter()
    examples = []

    progress = (
        tqdm(total=len(load_items), desc="Loading market files", unit="file")
        if tqdm
        else None
    )

    if workers <= 1:
        for _, path in load_items:
            loaded_code, item, err = _load_market_file_for_index(path)

            if err:
                errors[err.split(":")[0]] += 1
            elif loaded_code and item and item.df is not None:
                index[loaded_code] = item

                if len(examples) < 5:
                    examples.append(
                        {
                            "code": loaded_code,
                            "path": str(item.path),
                            "rows": int(len(item.df)),
                            "first_date": item.df["date"].min().strftime("%Y-%m-%d"),
                            "last_date": item.df["date"].max().strftime("%Y-%m-%d"),
                        }
                    )

            if progress:
                progress.update(1)

    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_load_market_file_for_index, path): code
                for code, path in load_items
            }

            for future in as_completed(futures):
                try:
                    loaded_code, item, err = future.result()
                except Exception as e:
                    errors[f"worker_failed:{type(e).__name__}"] += 1

                    if progress:
                        progress.update(1)

                    continue

                if err:
                    errors[err.split(":")[0]] += 1
                elif loaded_code and item and item.df is not None:
                    index[loaded_code] = item

                    if len(examples) < 5:
                        examples.append(
                            {
                                "code": loaded_code,
                                "path": str(item.path),
                                "rows": int(len(item.df)),
                                "first_date": item.df["date"].min().strftime("%Y-%m-%d"),
                                "last_date": item.df["date"].max().strftime("%Y-%m-%d"),
                            }
                        )

                if progress:
                    progress.update(1)

    if progress:
        progress.close()

    diagnostics = {
        "data_dir": str(data_dir),
        "market_files_found_recursively": len(files),
        "market_unique_codes_detected": len(load_items),
        "market_files_loaded_by_code": len(index),
        "market_loaded_examples": examples,
        "market_file_error_counts": dict(errors.most_common(20)),
        "market_load_workers": workers,
        "loaded_from_market_cache": False,
    }

    return index, diagnostics


def get_market_index_with_cache(
    data_dir: Path,
    cache_path: Path,
    workers: int,
    use_cache: bool = True,
    rebuild_cache: bool = False,
) -> tuple[dict[str, MarketCacheItem], dict[str, Any]]:
    """
    Load market index from cache if possible.

    This avoids repeatedly reading all fixed TongDaXin TXT files.
    Rebuild this cache after updating market TXT data or changing the TXT parser.
    """
    if use_cache and cache_path.exists() and not rebuild_cache:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)

        index = payload.get("market_index", {})
        diagnostics = payload.get("diagnostics", {})
        diagnostics["loaded_from_market_cache"] = True
        diagnostics["market_cache_path"] = str(cache_path)
        diagnostics["market_cache_rows"] = int(len(index))

        return index, diagnostics

    index, diagnostics = build_market_index(data_dir=data_dir, workers=workers)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "market_index": index,
        "diagnostics": diagnostics,
    }

    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    diagnostics["loaded_from_market_cache"] = False
    diagnostics["market_cache_path"] = str(cache_path)
    diagnostics["market_cache_saved"] = True

    return index, diagnostics


def normalize_pool(
    df: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()

    date_col = detect_col(out.columns, DATE_CANDIDATES)
    code_col = detect_col(out.columns, CODE_CANDIDATES)

    if date_col is None:
        raise ValueError(f"{label}: cannot detect date column. columns={list(out.columns)}")

    if code_col is None:
        for col in out.columns:
            sample = out[col].dropna().astype(str).head(50).tolist()

            if any(extract_6digit_code(v) for v in sample):
                code_col = col
                break

    if code_col is None:
        raise ValueError(f"{label}: cannot detect code column. columns={list(out.columns)}")

    out["__signal_date"] = parse_mixed_date(out[date_col])
    out["__code6"] = out[code_col].map(extract_6digit_code)
    out["__key"] = (
        out["__signal_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + out["__code6"].astype(str)
    )

    before = len(out)

    out = out.dropna(subset=["__signal_date", "__code6"]).copy()
    out = out.drop_duplicates("__key", keep="first").reset_index(drop=True)

    diag = {
        "label": label,
        "original_rows": before,
        "valid_rows_after_date_code_clean": len(out),
        "detected_date_col": str(date_col),
        "detected_code_col": str(code_col),
        "columns": [str(c) for c in df.columns],
        "sample_keys": out["__key"].head(5).tolist(),
    }

    return out, diag


def build_date_position_map(mdf: pd.DataFrame) -> dict[np.datetime64, int]:
    dates = mdf["date"].values
    return {np.datetime64(d): i for i, d in enumerate(dates)}


def add_forward_returns(
    pool: pd.DataFrame,
    market_index: dict[str, MarketCacheItem],
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Add T+1/T+2/T+3 forward returns.

    Optimized groupby version:
    - Groups pool rows by stock code.
    - Loads each stock's market DataFrame only once.
    - Builds date-position map once per stock.
    - Avoids slow row-by-row full scanning.

    Main assumption:
    - T0 = signal date.
    - T+1/T+2/T+3 are the next 1/2/3 trading rows in that stock's market data.
    """
    out = pool.copy()

    return_cols = [
        "t1_open_from_t0_close_pct",
        "t1_close_from_t0_close_pct",
        "t2_open_from_t0_close_pct",
        "t2_close_from_t0_close_pct",
        "t3_open_from_t0_close_pct",
        "t3_close_from_t0_close_pct",
        "t1_close_from_t1_open_pct",
        "t2_close_from_t1_open_pct",
        "t3_close_from_t1_open_pct",
    ]

    for c in return_cols:
        out[c] = np.nan

    reasons = Counter()
    matched_examples = []

    if out.empty:
        diag = {
            "label": label,
            "loaded_from_cache": False,
            "forward_calc_mode": "groupby_fast",
            "forward_missing_reason_counts_all": {},
            "matched_examples": [],
        }
        return out, diag

    grouped = out.groupby("__code6", sort=False)
    group_iter = grouped

    if tqdm:
        group_iter = tqdm(
            grouped,
            total=out["__code6"].nunique(dropna=True),
            desc=f"Calculating returns by code: {label}",
            unit="code",
        )

    for code, group in group_iter:
        if code is None or pd.isna(code):
            reasons["missing_code"] += int(len(group))
            continue

        item = market_index.get(str(code))

        if item is None or item.df is None:
            reasons["market_txt_not_found_or_unreadable"] += int(len(group))
            continue

        mdf = item.df

        if len(mdf) < 4:
            reasons["market_file_too_short"] += int(len(group))
            continue

        # Make sure market data has contiguous integer positions.
        mdf = mdf.reset_index(drop=True)

        # Convert to NumPy arrays once per code. This is much faster than repeated iloc.
        dates = mdf["date"].values
        open_arr = pd.to_numeric(mdf["open"], errors="coerce").to_numpy(dtype=float)
        close_arr = pd.to_numeric(mdf["close"], errors="coerce").to_numpy(dtype=float)

        date_pos_map = {np.datetime64(d): i for i, d in enumerate(dates)}

        # Map every signal date in this stock group to its T0 position.
        signal_dates = group["__signal_date"].values
        positions = np.array(
            [date_pos_map.get(np.datetime64(d), -1) for d in signal_dates],
            dtype=np.int64,
        )

        group_index = group.index.to_numpy()

        missing_mask = positions < 0
        if missing_mask.any():
            reasons["signal_date_not_in_market_file"] += int(missing_mask.sum())

        enough_future_mask = (positions >= 0) & ((positions + 3) < len(mdf))
        not_enough_mask = (positions >= 0) & ((positions + 3) >= len(mdf))
        if not_enough_mask.any():
            reasons["not_enough_future_bars_t3"] += int(not_enough_mask.sum())

        if not enough_future_mask.any():
            continue

        valid_positions = positions[enough_future_mask]
        valid_out_index = group_index[enough_future_mask]

        t0_close = close_arr[valid_positions]
        t1_open = open_arr[valid_positions + 1]

        valid_price_mask = (
            np.isfinite(t0_close)
            & np.isfinite(t1_open)
            & (t0_close > 0)
            & (t1_open > 0)
        )

        if (~valid_price_mask).any():
            reasons["invalid_base_price"] += int((~valid_price_mask).sum())

        if not valid_price_mask.any():
            continue

        final_positions = valid_positions[valid_price_mask]
        final_out_index = valid_out_index[valid_price_mask]

        t0_close = close_arr[final_positions]
        t1_open = open_arr[final_positions + 1]

        t1_open_v = open_arr[final_positions + 1]
        t1_close_v = close_arr[final_positions + 1]
        t2_open_v = open_arr[final_positions + 2]
        t2_close_v = close_arr[final_positions + 2]
        t3_open_v = open_arr[final_positions + 3]
        t3_close_v = close_arr[final_positions + 3]

        out.loc[final_out_index, "t1_open_from_t0_close_pct"] = (t1_open_v / t0_close - 1) * 100
        out.loc[final_out_index, "t1_close_from_t0_close_pct"] = (t1_close_v / t0_close - 1) * 100
        out.loc[final_out_index, "t2_open_from_t0_close_pct"] = (t2_open_v / t0_close - 1) * 100
        out.loc[final_out_index, "t2_close_from_t0_close_pct"] = (t2_close_v / t0_close - 1) * 100
        out.loc[final_out_index, "t3_open_from_t0_close_pct"] = (t3_open_v / t0_close - 1) * 100
        out.loc[final_out_index, "t3_close_from_t0_close_pct"] = (t3_close_v / t0_close - 1) * 100

        out.loc[final_out_index, "t1_close_from_t1_open_pct"] = (t1_close_v / t1_open - 1) * 100
        out.loc[final_out_index, "t2_close_from_t1_open_pct"] = (t2_close_v / t1_open - 1) * 100
        out.loc[final_out_index, "t3_close_from_t1_open_pct"] = (t3_close_v / t1_open - 1) * 100

        reasons["ok"] += int(len(final_out_index))

        if len(matched_examples) < 5:
            need = 5 - len(matched_examples)
            for out_idx, pos in zip(final_out_index[:need], final_positions[:need]):
                matched_examples.append(
                    {
                        "key": out.at[out_idx, "__key"],
                        "market_file": str(item.path),
                        "t0_close": float(close_arr[pos]),
                        "t1_open": float(open_arr[pos + 1]),
                        "t3_close": float(close_arr[pos + 3]),
                        "t3_close_from_t1_open_pct": float(out.at[out_idx, "t3_close_from_t1_open_pct"]),
                    }
                )

    diag = {
        "label": label,
        "loaded_from_cache": False,
        "forward_calc_mode": "groupby_fast",
        "forward_missing_reason_counts_all": dict(reasons.most_common()),
        "matched_examples": matched_examples,
    }

    return out, diag



def get_forward_returns_with_cache(
    pool: pd.DataFrame,
    market_index: dict[str, MarketCacheItem],
    label: str,
    cache_path: Optional[Path] = None,
    use_cache: bool = True,
    rebuild_cache: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if cache_path and use_cache and cache_path.exists() and not rebuild_cache:
        cached = pd.read_parquet(cache_path)

        diag = {
            "label": label,
            "loaded_from_cache": True,
            "cache_path": str(cache_path),
            "cached_rows": int(len(cached)),
        }

        return cached, diag

    ret, diag = add_forward_returns(pool, market_index, label)

    diag["loaded_from_cache"] = False

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        ret.to_parquet(cache_path, index=False)
        diag["cache_path"] = str(cache_path)
        diag["cache_saved"] = True

    return ret, diag


def max_drawdown_from_returns_pct(returns: pd.Series) -> Optional[float]:
    s = pd.to_numeric(returns, errors="coerce").dropna() / 100.0

    if s.empty:
        return None

    equity = (1.0 + s).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0

    return float(dd.min() * 100.0)


def summarize_quality(df: pd.DataFrame, main_return_col: str) -> dict[str, Any]:
    res: dict[str, Any] = {"count": int(len(df))}

    if main_return_col not in df.columns:
        res.update(
            {
                "main_return_col": main_return_col,
                "valid_main_return_count": 0,
                "win_rate_pct": None,
                "avg_trade_return_pct": None,
                "median_trade_return_pct": None,
                "total_return_pct_compounded_estimated": None,
                "max_drawdown_pct_estimated": None,
                "max_single_loss_pct": None,
                "max_single_profit_pct": None,
            }
        )
        return res

    r = pd.to_numeric(df[main_return_col], errors="coerce").dropna()

    res["main_return_col"] = main_return_col
    res["valid_main_return_count"] = int(len(r))

    if len(r) == 0:
        res.update(
            {
                "win_rate_pct": None,
                "avg_trade_return_pct": None,
                "median_trade_return_pct": None,
                "total_return_pct_compounded_estimated": None,
                "max_drawdown_pct_estimated": None,
                "max_single_loss_pct": None,
                "max_single_profit_pct": None,
            }
        )
    else:
        res.update(
            {
                "win_rate_pct": float((r > 0).mean() * 100),
                "avg_trade_return_pct": float(r.mean()),
                "median_trade_return_pct": float(r.median()),
                "total_return_pct_compounded_estimated": float(((1 + r / 100.0).prod() - 1) * 100),
                "max_drawdown_pct_estimated": max_drawdown_from_returns_pct(r),
                "max_single_loss_pct": float(r.min()),
                "max_single_profit_pct": float(r.max()),
            }
        )

    extra_cols = [
        "t1_open_from_t0_close_pct",
        "t1_close_from_t0_close_pct",
        "t2_close_from_t0_close_pct",
        "t3_close_from_t0_close_pct",
        "t1_close_from_t1_open_pct",
        "t2_close_from_t1_open_pct",
        "t3_close_from_t1_open_pct",
    ]

    for c in extra_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()

            res[c + "_avg"] = float(s.mean()) if len(s) else None
            res[c + "_median"] = float(s.median()) if len(s) else None
            res[c + "_win_rate_pct"] = float((s > 0).mean() * 100) if len(s) else None
            res[c + "_valid_count"] = int(len(s))

    return res


def subset_by_keys(df: pd.DataFrame, keys: set[str]) -> pd.DataFrame:
    return df[df["__key"].isin(keys)].copy()


def build_delta_summary(
    old_summary: dict[str, Any],
    new_summary: dict[str, Any],
) -> dict[str, Any]:
    fields = [
        "count",
        "valid_main_return_count",
        "win_rate_pct",
        "avg_trade_return_pct",
        "median_trade_return_pct",
        "total_return_pct_compounded_estimated",
        "max_drawdown_pct_estimated",
        "max_single_loss_pct",
        "max_single_profit_pct",
    ]

    delta = {}

    for field in fields:
        old_value = old_summary.get(field)
        new_value = new_summary.get(field)

        if old_value is None or new_value is None:
            delta[field + "_change"] = None
            continue

        try:
            delta[field + "_change"] = float(new_value) - float(old_value)
        except Exception:
            delta[field + "_change"] = None

    return delta


def print_compact_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print("POOL QUALITY ANALYSIS")
    print("=" * 88)

    cfg = report["config"]

    print(f"Old strategy : {cfg['old_strategy']}")
    print(f"New strategy : {cfg['new_strategy']}")
    print(f"Main return  : {cfg['main_return_col']}")
    print(f"Output       : {cfg['output']}")

    print("\n" + "-" * 88)
    print("Cache status")
    print("-" * 88)

    market_diag = report["diagnostics"]["market"]
    old_forward = report["diagnostics"]["old_forward"]

    print(f"Market loaded from cache : {market_diag.get('loaded_from_market_cache')}")
    print(f"Market cache path        : {market_diag.get('market_cache_path')}")
    print(f"Old loaded from cache    : {old_forward.get('loaded_from_cache')}")
    print(f"Old cache path           : {old_forward.get('cache_path')}")

    print("\n" + "-" * 88)
    print("Pool size")
    print("-" * 88)

    size = report["pool_size_summary"]

    print(f"Old count       : {size['old_count']}")
    print(f"New count       : {size['new_count']}")
    print(f"Common count    : {size['common_count']}")
    print(f"Only old count  : {size['only_old_count']}")
    print(f"Only new count  : {size['only_new_count']}")
    print(f"Count change    : {size['new_vs_old_count_change']}")
    print(f"Count change %  : {size['new_vs_old_count_change_pct']}")

    old_all = report["quality_summary"]["old_all"]
    new_all = report["quality_summary"]["new_all"]
    delta = report["delta_summary"]["new_all_minus_old_all"]

    print("\n" + "-" * 88)
    print("Main quality comparison")
    print("-" * 88)
    print(f"{'Metric':38} {'Old':>14} {'New':>14} {'Change':>14}")
    print("-" * 88)

    rows = [
        ("valid_main_return_count", "Valid trades"),
        ("win_rate_pct", "Win rate %"),
        ("avg_trade_return_pct", "Avg return %"),
        ("median_trade_return_pct", "Median return %"),
        ("total_return_pct_compounded_estimated", "Compounded return %"),
        ("max_drawdown_pct_estimated", "Max drawdown %"),
        ("max_single_loss_pct", "Max single loss %"),
        ("max_single_profit_pct", "Max single profit %"),
    ]

    for key, label in rows:
        old_v = old_all.get(key)
        new_v = new_all.get(key)
        chg_v = delta.get(key + "_change")

        print(f"{label:38} {str(old_v):>14} {str(new_v):>14} {str(chg_v):>14}")

    print("\n" + "-" * 88)
    print("Forward diagnostics")
    print("-" * 88)

    print("Old forward:")
    print(
        json.dumps(
            to_jsonable(old_forward.get("forward_missing_reason_counts_all", {})),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("New forward:")
    print(
        json.dumps(
            to_jsonable(report["diagnostics"]["new_forward"].get("forward_missing_reason_counts_all", {})),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n" + "=" * 88)
    print("DONE")
    print("=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pool Quality Analysis: compare v2 pool with any vn pool."
    )

    parser.add_argument("--old", default=DEFAULT_OLD_STRATEGY, help="Old strategy. Default: v2.")
    parser.add_argument("--new", default=DEFAULT_NEW_STRATEGY, help="New strategy. Example: v1, v2, v1_j_range.")

    parser.add_argument("--pools-dir", type=Path, default=DEFAULT_POOLS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)

    parser.add_argument("--old-pool", type=Path, default=None, help="Optional full old pool parquet path.")
    parser.add_argument("--new-pool", type=Path, default=None, help="Optional full new pool parquet path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional full output json path.")

    parser.add_argument("--main-return-col", default=DEFAULT_MAIN_RETURN_COL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    parser.add_argument("--no-cache", action="store_true", help="Disable all cache usage.")
    parser.add_argument("--rebuild-market-cache", action="store_true", help="Force rebuild market index cache.")
    parser.add_argument("--rebuild-old-cache", action="store_true", help="Force rebuild old pool forward-return cache.")

    parser.add_argument("--cache-new", action="store_true", help="Also cache new pool forward returns. Default is disabled.")
    parser.add_argument(
        "--rebuild-new-cache",
        action="store_true",
        help="Force rebuild new pool forward-return cache. Only works with --cache-new.",
    )

    args = parser.parse_args()

    old_pool_path = args.old_pool if args.old_pool else resolve_pool_path(args.old, args.pools_dir)
    new_pool_path = args.new_pool if args.new_pool else resolve_pool_path(args.new, args.pools_dir)

    output_path = args.output if args.output else build_output_path(args.old, args.new, args.output_dir)

    old_strategy = normalize_strategy_name(args.old)
    new_strategy = normalize_strategy_name(args.new)

    old_short = strategy_to_short_name(old_strategy)
    new_short = strategy_to_short_name(new_strategy)

    market_cache_path = args.cache_dir / "market_index_cache.pkl"
    old_cache_path = args.cache_dir / f"{old_strategy}_forward_returns.parquet"
    new_cache_path = args.cache_dir / f"{new_strategy}_forward_returns.parquet"

    print("=" * 88)
    print("Pool Quality Analysis")
    print("=" * 88)
    print(f"Old strategy : {old_strategy}")
    print(f"New strategy : {new_strategy}")
    print(f"Old pool     : {old_pool_path}")
    print(f"New pool     : {new_pool_path}")
    print(f"Data dir     : {args.data_dir}")
    print(f"Output       : {output_path}")
    print(f"Cache dir    : {args.cache_dir}")
    print(f"Market cache : {market_cache_path}")
    print(f"Workers      : {args.workers}")
    print(f"Return col   : {args.main_return_col}")
    print(f"Use cache    : {not args.no_cache}")
    print(f"Rebuild market cache : {args.rebuild_market_cache}")
    print(f"Rebuild old cache    : {args.rebuild_old_cache}")
    print(f"Cache new            : {args.cache_new}")

    if tqdm is None:
        print("\nWARNING: tqdm is not installed. Progress bar disabled.")
        print("Install it with: pip install tqdm\n")

    if not old_pool_path.exists():
        raise FileNotFoundError(f"Old pool not found: {old_pool_path}")

    if not new_pool_path.exists():
        raise FileNotFoundError(f"New pool not found: {new_pool_path}")

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {args.data_dir}")

    print("\nReading pool files...")
    old_raw = pd.read_parquet(old_pool_path)
    new_raw = pd.read_parquet(new_pool_path)

    print("Normalizing pool files...")
    old_pool, old_pool_diag = normalize_pool(old_raw, old_short)
    new_pool, new_pool_diag = normalize_pool(new_raw, new_short)

    print("Loading or building market index...")
    market_index, market_diag = get_market_index_with_cache(
        data_dir=args.data_dir,
        cache_path=market_cache_path,
        workers=args.workers,
        use_cache=not args.no_cache,
        rebuild_cache=args.rebuild_market_cache,
    )

    print("Loading or calculating old pool forward returns...")
    old_ret, old_ret_diag = get_forward_returns_with_cache(
        pool=old_pool,
        market_index=market_index,
        label=old_short,
        cache_path=old_cache_path,
        use_cache=not args.no_cache,
        rebuild_cache=args.rebuild_old_cache,
    )

    if args.cache_new:
        print("Loading or calculating new pool forward returns...")
        new_ret, new_ret_diag = get_forward_returns_with_cache(
            pool=new_pool,
            market_index=market_index,
            label=new_short,
            cache_path=new_cache_path,
            use_cache=not args.no_cache,
            rebuild_cache=args.rebuild_new_cache,
        )
    else:
        print("Calculating new pool forward returns...")
        new_ret, new_ret_diag = add_forward_returns(new_pool, market_index, new_short)

    old_keys = set(old_ret["__key"])
    new_keys = set(new_ret["__key"])

    common_keys = old_keys & new_keys
    only_old_keys = old_keys - new_keys
    only_new_keys = new_keys - old_keys

    old_all_summary = summarize_quality(old_ret, args.main_return_col)
    new_all_summary = summarize_quality(new_ret, args.main_return_col)

    common_old_summary = summarize_quality(subset_by_keys(old_ret, common_keys), args.main_return_col)
    common_new_summary = summarize_quality(subset_by_keys(new_ret, common_keys), args.main_return_col)

    only_old_summary = summarize_quality(subset_by_keys(old_ret, only_old_keys), args.main_return_col)
    only_new_summary = summarize_quality(subset_by_keys(new_ret, only_new_keys), args.main_return_col)

    report = {
        "config": {
            "old_strategy": old_strategy,
            "new_strategy": new_strategy,
            "old_pool": str(old_pool_path),
            "new_pool": str(new_pool_path),
            "data_dir": str(args.data_dir),
            "output": str(output_path),
            "cache_dir": str(args.cache_dir),
            "market_cache_path": str(market_cache_path),
            "old_cache_path": str(old_cache_path),
            "new_cache_path": str(new_cache_path),
            "main_return_col": args.main_return_col,
            "buy_sell_assumption_for_main_return": "Buy at T+1 open, sell at T+3 close by default.",
            "workers": args.workers,
            "use_cache": not args.no_cache,
            "rebuild_market_cache": args.rebuild_market_cache,
            "rebuild_old_cache": args.rebuild_old_cache,
            "cache_new": args.cache_new,
            "rebuild_new_cache": args.rebuild_new_cache,
        },
        "diagnostics": {
            "old_pool": old_pool_diag,
            "new_pool": new_pool_diag,
            "market": market_diag,
            "old_forward": old_ret_diag,
            "new_forward": new_ret_diag,
        },
        "pool_size_summary": {
            "old_count": int(len(old_ret)),
            "new_count": int(len(new_ret)),
            "common_count": int(len(common_keys)),
            "only_old_count": int(len(only_old_keys)),
            "only_new_count": int(len(only_new_keys)),
            "new_vs_old_count_change": int(len(new_ret) - len(old_ret)),
            "new_vs_old_count_change_pct": float((len(new_ret) / len(old_ret) - 1) * 100)
            if len(old_ret)
            else None,
        },
        "quality_summary": {
            "old_all": old_all_summary,
            "new_all": new_all_summary,
            "common_old_view": common_old_summary,
            "common_new_view": common_new_summary,
            "only_old_removed_by_new": only_old_summary,
            "only_new_added_by_new": only_new_summary,
        },
        "delta_summary": {
            "new_all_minus_old_all": build_delta_summary(old_all_summary, new_all_summary),
            "common_new_minus_common_old": build_delta_summary(common_old_summary, common_new_summary),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(to_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_compact_summary(report)
    print(f"\nSaved JSON report: {output_path}")


if __name__ == "__main__":
    main()
