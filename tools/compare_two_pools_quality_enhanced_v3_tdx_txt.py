from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_OLD_POOL = Path(r"C:\Users\zyf37\Desktop\BackTest Data\pools\renko_chart_select_strategy_v1_pool.parquet")
DEFAULT_NEW_POOL = Path(r"C:\Users\zyf37\Desktop\BackTest Data\pools\renko_chart_select_strategy_v1_j_range_pool.parquet")
DEFAULT_DATA_DIR = Path(r"C:\Users\zyf37\Desktop\BackTest Data\data")
DEFAULT_OUTPUT = Path(r"C:\Users\zyf37\Desktop\BackTest Data\pools\pool_quality_compare_v1_vs_v1_j_range_enhanced_v3.json")

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
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if math.isnan(float(obj)) or math.isinf(float(obj)):
            return None
        return float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if pd.isna(obj) if not isinstance(obj, (list, tuple, dict, Path)) else False:
        return None
    return obj


def extract_6digit_code(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    m = re.search(r"(\d{6})", text)
    return m.group(1) if m else None


def detect_col(cols: Iterable[str], candidates: list[str]) -> Optional[str]:
    lower_map = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    # fuzzy
    for c in cols:
        low = str(c).lower()
        if any(cand.lower() in low for cand in candidates):
            return c
    return None


def parse_mixed_date(s: pd.Series) -> pd.Series:
    """Parse dates from formats such as 2025-04-01, 20250401, and TDX DD/MM/YYYY."""
    raw = s.astype(str).str.strip()
    raw = raw.str.replace("\ufeff", "", regex=False)

    out = pd.to_datetime(raw, errors="coerce")

    # TDX export sample uses DD/MM/YYYY, e.g. 02/08/2021.
    slash_mask = raw.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$", na=False)
    if slash_mask.any():
        out.loc[slash_mask] = pd.to_datetime(raw.loc[slash_mask], format="%d/%m/%Y", errors="coerce")

    # YYYYMMDD
    ymd_mask = raw.str.match(r"^\d{8}$", na=False)
    if ymd_mask.any():
        out.loc[ymd_mask] = pd.to_datetime(raw.loc[ymd_mask], format="%Y%m%d", errors="coerce")

    return out.dt.normalize()


def read_tdx_txt(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Read TongDaXin TXT export like:
        920000 安徽凤凰 日线 前复权
              日期    开盘    最高    最低    收盘    成交量    成交额
        02/08/2021,5.51,5.60,...
    """
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
        # Keep only real csv rows beginning with date-like token.
        if re.match(r"^(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{1,2}-\d{1,2}|\d{8})\s*,", line):
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
    df = df.dropna(subset=["date", "open", "close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
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
    df = df.dropna(subset=["date", "open", "close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if df.empty:
        return None, "empty_after_cleaning"
    return df, None


def build_market_index(data_dir: Path) -> tuple[dict[str, MarketCacheItem], dict[str, Any]]:
    files = []
    for suffix in ["*.txt", "*.csv", "*.parquet"]:
        files.extend(data_dir.rglob(suffix))

    index: dict[str, MarketCacheItem] = {}
    errors = Counter()
    examples = []
    for path in files:
        code = extract_6digit_code(path.name)
        if not code:
            continue
        if code in index:
            continue
        df, err = read_market_file(path)
        if err:
            errors[err.split(":")[0]] += 1
        else:
            index[code] = MarketCacheItem(path=path, df=df, error=None)
            if len(examples) < 5:
                examples.append({
                    "code": code,
                    "path": str(path),
                    "rows": int(len(df)),
                    "first_date": df["date"].min().strftime("%Y-%m-%d"),
                    "last_date": df["date"].max().strftime("%Y-%m-%d"),
                })

    diagnostics = {
        "data_dir": str(data_dir),
        "market_files_found_recursively": len(files),
        "market_files_loaded_by_code": len(index),
        "market_loaded_examples": examples,
        "market_file_error_counts": dict(errors.most_common(20)),
    }
    return index, diagnostics


def normalize_pool(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    date_col = detect_col(out.columns, DATE_CANDIDATES)
    code_col = detect_col(out.columns, CODE_CANDIDATES)

    if date_col is None:
        raise ValueError(f"{label}: cannot detect date column. columns={list(out.columns)}")
    if code_col is None:
        # fallback: maybe file/symbol col contains code
        for col in out.columns:
            sample = out[col].dropna().astype(str).head(50).tolist()
            if any(extract_6digit_code(v) for v in sample):
                code_col = col
                break
    if code_col is None:
        raise ValueError(f"{label}: cannot detect code column. columns={list(out.columns)}")

    out["__signal_date"] = parse_mixed_date(out[date_col])
    out["__code6"] = out[code_col].map(extract_6digit_code)
    out["__key"] = out["__signal_date"].dt.strftime("%Y-%m-%d") + "|" + out["__code6"].astype(str)

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


def add_forward_returns(pool: pd.DataFrame, market_index: dict[str, MarketCacheItem], label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = pool.copy()
    return_cols = [
        "t1_open_from_t0_close_pct", "t1_close_from_t0_close_pct",
        "t2_open_from_t0_close_pct", "t2_close_from_t0_close_pct",
        "t3_open_from_t0_close_pct", "t3_close_from_t0_close_pct",
        "t1_close_from_t1_open_pct", "t2_close_from_t1_open_pct", "t3_close_from_t1_open_pct",
    ]
    for c in return_cols:
        out[c] = np.nan

    reasons = Counter()
    matched_examples = []

    for idx, row in out.iterrows():
        code = row["__code6"]
        signal_date = row["__signal_date"]
        item = market_index.get(code)
        if item is None or item.df is None:
            reasons["market_txt_not_found_or_unreadable"] += 1
            continue
        mdf = item.df
        pos_arr = np.flatnonzero(mdf["date"].values == np.datetime64(signal_date))
        if len(pos_arr) == 0:
            reasons["signal_date_not_in_market_file"] += 1
            continue
        pos = int(pos_arr[0])
        if pos + 3 >= len(mdf):
            reasons["not_enough_future_bars_t3"] += 1
            continue

        t0 = mdf.iloc[pos]
        t1 = mdf.iloc[pos + 1]
        t2 = mdf.iloc[pos + 2]
        t3 = mdf.iloc[pos + 3]
        t0_close = float(t0["close"])
        t1_open = float(t1["open"])
        if t0_close <= 0 or t1_open <= 0:
            reasons["invalid_base_price"] += 1
            continue

        out.at[idx, "t1_open_from_t0_close_pct"] = (float(t1["open"]) / t0_close - 1) * 100
        out.at[idx, "t1_close_from_t0_close_pct"] = (float(t1["close"]) / t0_close - 1) * 100
        out.at[idx, "t2_open_from_t0_close_pct"] = (float(t2["open"]) / t0_close - 1) * 100
        out.at[idx, "t2_close_from_t0_close_pct"] = (float(t2["close"]) / t0_close - 1) * 100
        out.at[idx, "t3_open_from_t0_close_pct"] = (float(t3["open"]) / t0_close - 1) * 100
        out.at[idx, "t3_close_from_t0_close_pct"] = (float(t3["close"]) / t0_close - 1) * 100
        out.at[idx, "t1_close_from_t1_open_pct"] = (float(t1["close"]) / t1_open - 1) * 100
        out.at[idx, "t2_close_from_t1_open_pct"] = (float(t2["close"]) / t1_open - 1) * 100
        out.at[idx, "t3_close_from_t1_open_pct"] = (float(t3["close"]) / t1_open - 1) * 100
        reasons["ok"] += 1

        if len(matched_examples) < 5:
            matched_examples.append({
                "key": row["__key"],
                "market_file": str(item.path),
                "t0_close": t0_close,
                "t1_open": t1_open,
                "t3_close": float(t3["close"]),
                "t3_close_from_t1_open_pct": out.at[idx, "t3_close_from_t1_open_pct"],
            })

    diag = {
        "label": label,
        "forward_missing_reason_counts_all": dict(reasons.most_common()),
        "matched_examples": matched_examples,
    }
    return out, diag


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
        res.update({
            "main_return_col": main_return_col,
            "valid_main_return_count": 0,
            "win_rate_pct": None,
            "avg_trade_return_pct": None,
            "median_trade_return_pct": None,
            "total_return_pct_compounded_estimated": None,
            "max_drawdown_pct_estimated": None,
            "max_single_loss_pct": None,
        })
        return res

    r = pd.to_numeric(df[main_return_col], errors="coerce").dropna()
    res["main_return_col"] = main_return_col
    res["valid_main_return_count"] = int(len(r))
    if len(r) == 0:
        res.update({
            "win_rate_pct": None,
            "avg_trade_return_pct": None,
            "median_trade_return_pct": None,
            "total_return_pct_compounded_estimated": None,
            "max_drawdown_pct_estimated": None,
            "max_single_loss_pct": None,
        })
    else:
        res.update({
            "win_rate_pct": float((r > 0).mean() * 100),
            "avg_trade_return_pct": float(r.mean()),
            "median_trade_return_pct": float(r.median()),
            "total_return_pct_compounded_estimated": float(((1 + r / 100.0).prod() - 1) * 100),
            "max_drawdown_pct_estimated": max_drawdown_from_returns_pct(r),
            "max_single_loss_pct": float(r.min()),
            "max_single_profit_pct": float(r.max()),
        })

    for c in [
        "t1_open_from_t0_close_pct", "t1_close_from_t0_close_pct",
        "t2_close_from_t0_close_pct", "t3_close_from_t0_close_pct",
        "t1_close_from_t1_open_pct", "t2_close_from_t1_open_pct", "t3_close_from_t1_open_pct",
    ]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            res[c + "_avg"] = float(s.mean()) if len(s) else None
            res[c + "_valid_count"] = int(len(s))
    return res


def subset_by_keys(df: pd.DataFrame, keys: set[str]) -> pd.DataFrame:
    return df[df["__key"].isin(keys)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two selection pool parquet files and calculate forward returns from TDX TXT exports.")
    parser.add_argument("--old-pool", type=Path, default=DEFAULT_OLD_POOL)
    parser.add_argument("--new-pool", type=Path, default=DEFAULT_NEW_POOL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--main-return-col", default="t3_close_from_t1_open_pct")
    args = parser.parse_args()

    old_raw = pd.read_parquet(args.old_pool)
    new_raw = pd.read_parquet(args.new_pool)

    old_pool, old_pool_diag = normalize_pool(old_raw, "old_v1")
    new_pool, new_pool_diag = normalize_pool(new_raw, "new_v1_j_range")

    market_index, market_diag = build_market_index(args.data_dir)

    old_ret, old_ret_diag = add_forward_returns(old_pool, market_index, "old_v1")
    new_ret, new_ret_diag = add_forward_returns(new_pool, market_index, "new_v1_j_range")

    old_keys = set(old_ret["__key"])
    new_keys = set(new_ret["__key"])
    common_keys = old_keys & new_keys
    only_old_keys = old_keys - new_keys
    only_new_keys = new_keys - old_keys

    report = {
        "config": {
            "old_pool": str(args.old_pool),
            "new_pool": str(args.new_pool),
            "data_dir": str(args.data_dir),
            "output": str(args.output),
            "main_return_col": args.main_return_col,
            "buy_sell_assumption_for_main_return": "Buy at T+1 open, sell at T+3 close by default.",
        },
        "diagnostics": {
            "old_pool": old_pool_diag,
            "new_pool": new_pool_diag,
            "market": market_diag,
            "old_forward": old_ret_diag,
            "new_forward": new_ret_diag,
        },
        "pool_size_summary": {
            "old_v1_count": int(len(old_ret)),
            "new_v1_j_range_count": int(len(new_ret)),
            "common_count": int(len(common_keys)),
            "only_old_count": int(len(only_old_keys)),
            "only_new_count": int(len(only_new_keys)),
            "new_vs_old_count_change": int(len(new_ret) - len(old_ret)),
            "new_vs_old_count_change_pct": float((len(new_ret) / len(old_ret) - 1) * 100) if len(old_ret) else None,
        },
        "quality_summary": {
            "old_v1_all": summarize_quality(old_ret, args.main_return_col),
            "new_v1_j_range_all": summarize_quality(new_ret, args.main_return_col),
            "common_old_view": summarize_quality(subset_by_keys(old_ret, common_keys), args.main_return_col),
            "common_new_view": summarize_quality(subset_by_keys(new_ret, common_keys), args.main_return_col),
            "only_old_removed_by_new_condition": summarize_quality(subset_by_keys(old_ret, only_old_keys), args.main_return_col),
            "only_new_added_by_new_condition": summarize_quality(subset_by_keys(new_ret, only_new_keys), args.main_return_col),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved JSON report: {args.output}")
    print(json.dumps(to_jsonable(report["pool_size_summary"]), ensure_ascii=False, indent=2))
    print("Market diagnostics:")
    print(json.dumps(to_jsonable(market_diag), ensure_ascii=False, indent=2))
    print("Old forward diagnostics:")
    print(json.dumps(to_jsonable(old_ret_diag), ensure_ascii=False, indent=2))
    print("New forward diagnostics:")
    print(json.dumps(to_jsonable(new_ret_diag), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
