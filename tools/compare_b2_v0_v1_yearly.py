from __future__ import annotations

r"""
Compare B2 v0 vs new v1 pools by year.

Buy: T+1 open
Sell modes: T+2 open / T+2 close / T+3 open / T+3 close

Run:
python .\tools\compare_b2_v0_v1_yearly.py

Save detail:
python .\tools\compare_b2_v0_v1_yearly.py --save-detail
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

DEFAULT_POOL_V0 = Path(r"C:\Users\zyf37\Desktop\BackTest Data\pools\b2_confirm_select_strategy_v0_pool.parquet")
DEFAULT_POOL_V1 = Path(r"C:\Users\zyf37\Desktop\BackTest Data\pools\b2_confirm_select_strategy_v1_pool.parquet")
DEFAULT_TXT_DIR = Path(r"C:\Users\zyf37\Desktop\BackTest Data\data")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\zyf37\Desktop\BackTest Data\output\b2_v0_v1_yearly_compare")

ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936", "mbcs")

POOL_FEATURE_ALIAS_MAP: dict[str, list[str]] = {
    "b2_j": ["b2_j", "b2_j_value", "J", "j"],
    "daily_return_pct": ["daily_return_pct", "pct_change_close"],
    "b2_volume_ratio": ["b2_volume_ratio", "volume_ratio_prev"],
    "b1_days_ago": ["b1_days_ago", "b1_days_ago_for_b2"],
    "b1_j_value": ["b1_j_value"],
    "b1_volume_value": ["b1_volume_value"],
    "b2_quality_score": ["b2_quality_score", "score"],
    "score": ["score", "b2_quality_score"],
    "score_pct": ["score_pct"],
    "ma20": ["ma20"],
    "ma50": ["ma50"],
    "ma20_gt_ma50": ["ma20_gt_ma50"],
    "prior_20d_double_volume_count": ["prior_20d_double_volume_count"],
    "prior_20d_has_double_volume": ["prior_20d_has_double_volume"],
    "double_volume_bar": ["double_volume_bar"],
    "b1_ma20_gt_ma50": ["b1_ma20_gt_ma50"],
    "b1_prior_20d_has_double_volume": ["b1_prior_20d_has_double_volume"],
    "b1_stage_low_position": ["b1_stage_low_position"],
    "b1_low_or_extreme_volume": ["b1_low_or_extreme_volume"],
    "b2_after_b1_within_3d": ["b2_after_b1_within_3d"],
    "b2_no_or_tiny_upper_shadow": ["b2_no_or_tiny_upper_shadow"],
    "b2_long_upper_shadow_reject": ["b2_long_upper_shadow_reject"],
    "b2_strong_volume": ["b2_strong_volume"],
    "upper_shadow_ratio": ["upper_shadow_ratio"],
    "position_in_range_20": ["position_in_range_20"],
    "range_width_20": ["range_width_20"],
    "distance_to_previous_n_low": ["distance_to_previous_n_low"],
    "distance_to_ma20": ["distance_to_ma20"],
    "distance_to_bbi": ["distance_to_bbi"],
    "distance_to_yellow": ["distance_to_yellow"],
    "volume_ratio_ma5": ["volume_ratio_ma5"],
    "volume_q20_20": ["volume_q20_20"],
    "volume_min_20": ["volume_min_20"],
    "b1_low_range_position": ["b1_low_range_position"],
    "b1_near_previous_n_low": ["b1_near_previous_n_low"],
    "b1_in_range_bottom": ["b1_in_range_bottom"],
    "b1_position_ok": ["b1_position_ok"],
    "b1_j_ok": ["b1_j_ok"],
    "b1_low_volume": ["b1_low_volume"],
    "b1_extreme_low_volume": ["b1_extreme_low_volume"],
    "b1_not_break_prev_low": ["b1_not_break_prev_low"],
    "b1_valid": ["b1_valid"],
    "b1_within_b2_lookback": ["b1_within_b2_lookback"],
    "b2_return_ok": ["b2_return_ok"],
    "b2_bullish_candle": ["b2_bullish_candle"],
    "b2_volume_up": ["b2_volume_up"],
    "b2_double_volume": ["b2_double_volume"],
    "b2_j_ok": ["b2_j_ok"],
    "b2_upper_shadow_ok": ["b2_upper_shadow_ok"],
    "b2_tiny_upper_shadow": ["b2_tiny_upper_shadow"],
    "b2_upper_shadow_warning": ["b2_upper_shadow_warning"],
}


def iter_progress(items: Iterable, total: int | None = None, desc: str = ""):
    return tqdm(items, total=total, desc=desc) if tqdm is not None else items


def read_text(path: Path) -> str:
    last = None
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except Exception as exc:
            last = exc
    try:
        return path.read_text(encoding="gb18030", errors="replace")
    except Exception as exc:
        raise RuntimeError(f"{last}; fallback error: {exc}") from exc


def parse_date_token(token: str):
    token = token.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%d/%m/%Y", "%Y.%m.%d"):
        try:
            return pd.to_datetime(token, format=fmt, errors="raise")
        except Exception:
            pass
    dt = pd.to_datetime(token, errors="coerce")
    return None if pd.isna(dt) else dt


def to_float(x: str):
    x = x.strip().replace(",", "")
    if x in {"", "-", "--", "nan", "NaN", "None"}:
        return None
    try:
        return float(x)
    except Exception:
        return None


def parse_market_line(line: str):
    line = line.strip().replace("\ufeff", "")
    line = line.replace(",", " ").replace("，", " ").replace(";", " ")
    line = re.sub(r"\s+", " ", line)
    parts = [x for x in line.split(" ") if x]
    if len(parts) < 6:
        return None
    dt = parse_date_token(parts[0])
    if dt is None:
        return None
    nums = [v for v in (to_float(p) for p in parts[1:]) if v is not None]
    if len(nums) < 5:
        return None
    o, h, l, c, vol = nums[:5]
    amount = nums[5] if len(nums) >= 6 else None
    if o <= 0 or h <= 0 or l <= 0 or c <= 0 or h < l:
        return None
    return {
        "date": pd.Timestamp(dt).normalize(),
        "open": float(o),
        "high": float(h),
        "low": float(l),
        "close": float(c),
        "volume": float(vol),
        "amount": float(amount) if amount is not None else None,
    }


def read_tdx_txt(path: Path) -> pd.DataFrame:
    rows = []
    for line in read_text(path).splitlines():
        row = parse_market_line(line)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError("No valid market rows parsed")
    df = pd.DataFrame(rows)
    return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def normalize_symbol(raw: str) -> str:
    s = str(raw).strip().upper().replace(".", "#")
    if "#" in s:
        left, right = s.split("#", 1)
        right = re.sub(r"\D", "", right)
        if left in {"SH", "SZ"} and len(right) == 6:
            return f"{left}#{right}"
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 6:
        code = digits[-6:]
        return f"{'SH' if code.startswith('6') else 'SZ'}#{code}"
    return s


def infer_pool_columns(df: pd.DataFrame) -> tuple[str, str]:
    lower = {str(c).lower(): str(c) for c in df.columns}
    date_col = next((lower[x] for x in ["date", "signal_date", "trade_date", "datetime"] if x in lower), "")
    symbol_col = next((lower[x] for x in ["symbol", "code", "ts_code", "stock_code", "security_code"] if x in lower), "")
    if not date_col:
        date_col = next((str(c) for c in df.columns if "date" in str(c).lower()), "")
    if not symbol_col:
        symbol_col = next((str(c) for c in df.columns if "symbol" in str(c).lower() or "code" in str(c).lower()), "")
    if not date_col or not symbol_col:
        raise ValueError(f"Cannot infer date/symbol columns from: {list(df.columns)}")
    return date_col, symbol_col


def load_pool(path: Path, strategy: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Pool file not found: {path}")
    df = pd.read_parquet(path)
    date_col, symbol_col = infer_pool_columns(df)
    out = df.copy()
    out["_signal_date"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out["_symbol"] = out[symbol_col].map(normalize_symbol)
    out["_strategy"] = strategy
    out = out.dropna(subset=["_signal_date"])
    out = out[out["_symbol"].astype(str).str.len() >= 8].copy()
    out["_year"] = out["_signal_date"].dt.year
    print(f"[POOL] {strategy}: {path}")
    print(f"       rows={len(out):,} | date_col={date_col} | symbol_col={symbol_col}")
    if len(out):
        print(f"       min_date={out['_signal_date'].min()} | max_date={out['_signal_date'].max()}")
    return out.reset_index(drop=True)


def plain_value(v):
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        low = v.strip().lower()
        if low == "true":
            return 1
        if low == "false":
            return 0
    return v


def add_pool_features(row_out: dict, pool_row: pd.Series):
    for canonical, candidates in POOL_FEATURE_ALIAS_MAP.items():
        for src in candidates:
            if src in pool_row.index:
                row_out[canonical] = plain_value(pool_row[src])
                break


def build_market_cache(txt_dir: Path, symbols: list[str]):
    cache, warnings = {}, []
    for symbol in iter_progress(symbols, total=len(symbols), desc="Loading market TXT"):
        path = txt_dir / f"{symbol}.txt"
        if not path.exists():
            warnings.append({"symbol": symbol, "path": str(path), "error": "file not found"})
            continue
        try:
            cache[symbol] = read_tdx_txt(path)
        except Exception as exc:
            warnings.append({"symbol": symbol, "path": str(path), "error": str(exc)})
            print(f"[WARN] Skip {path}: {exc}")
    return cache, warnings


def pct(buy: float, sell: float) -> float:
    return 0.0 if buy == 0 else (sell / buy - 1.0) * 100.0


def analyze_pool(pool: pd.DataFrame, market_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    strategy = pool["_strategy"].iloc[0] if len(pool) else ""
    for _, r in iter_progress(pool.iterrows(), total=len(pool), desc=f"Analyzing {strategy}"):
        symbol = r["_symbol"]
        signal_date = pd.Timestamp(r["_signal_date"]).normalize()
        market = market_cache.get(symbol)
        if market is None or market.empty:
            continue
        future = market[market["date"] > signal_date].sort_values("date").head(3)
        if len(future) < 3:
            continue
        t1, t2, t3 = future.iloc[0], future.iloc[1], future.iloc[2]
        buy = float(t1["open"])
        t0 = market[market["date"] == signal_date]
        t0_close = float(t0.iloc[-1]["close"]) if len(t0) else None
        ret_t2_open = pct(buy, float(t2["open"]))
        ret_t2_close = pct(buy, float(t2["close"]))
        ret_t3_open = pct(buy, float(t3["open"]))
        ret_t3_close = pct(buy, float(t3["close"]))
        max_high_t1_t2 = max(float(t1["high"]), float(t2["high"]))
        min_low_t1_t2 = min(float(t1["low"]), float(t2["low"]))
        max_high_t1_t3 = max(float(t1["high"]), float(t2["high"]), float(t3["high"]))
        min_low_t1_t3 = min(float(t1["low"]), float(t2["low"]), float(t3["low"]))
        out = {
            "strategy": r["_strategy"], "symbol": symbol, "signal_date": signal_date.date().isoformat(), "year": int(signal_date.year),
            "t1_date": pd.Timestamp(t1["date"]).date().isoformat(), "t2_date": pd.Timestamp(t2["date"]).date().isoformat(), "t3_date": pd.Timestamp(t3["date"]).date().isoformat(),
            "t0_close": t0_close, "buy_t1_open": buy,
            "t1_open": buy, "t1_close": float(t1["close"]), "t1_high": float(t1["high"]), "t1_low": float(t1["low"]),
            "t2_open": float(t2["open"]), "t2_close": float(t2["close"]), "t2_high": float(t2["high"]), "t2_low": float(t2["low"]),
            "t3_open": float(t3["open"]), "t3_close": float(t3["close"]), "t3_high": float(t3["high"]), "t3_low": float(t3["low"]),
            "ret_t1_open_to_t2_open_pct": ret_t2_open,
            "ret_t1_open_to_t2_close_pct": ret_t2_close,
            "ret_t1_open_to_t3_open_pct": ret_t3_open,
            "ret_t1_open_to_t3_close_pct": ret_t3_close,
            "win_t2_open": ret_t2_open > 0, "win_t2_close": ret_t2_close > 0,
            "win_t3_open": ret_t3_open > 0, "win_t3_close": ret_t3_close > 0,
            "max_opportunity_t1_t2_pct": pct(buy, max_high_t1_t2),
            "max_drawdown_t1_t2_pct": pct(buy, min_low_t1_t2),
            "max_opportunity_t1_t3_pct": pct(buy, max_high_t1_t3),
            "max_drawdown_t1_t3_pct": pct(buy, min_low_t1_t3),
            "t1_open_gap_pct": ((buy / t0_close - 1.0) * 100.0) if t0_close else None,
        }
        for x in [2, 3, 5]:
            out[f"hit_plus_{x}pct_t1_t2"] = out["max_opportunity_t1_t2_pct"] >= x
            out[f"hit_minus_{x}pct_t1_t2"] = out["max_drawdown_t1_t2_pct"] <= -x
            out[f"hit_plus_{x}pct_t1_t3"] = out["max_opportunity_t1_t3_pct"] >= x
            out[f"hit_minus_{x}pct_t1_t3"] = out["max_drawdown_t1_t3_pct"] <= -x
        add_pool_features(out, r)
        rows.append(out)
    return pd.DataFrame(rows)


def mean(df, col): return 0.0 if col not in df or len(df) == 0 else round(float(pd.to_numeric(df[col], errors="coerce").mean()), 6)
def median(df, col): return 0.0 if col not in df or len(df) == 0 else round(float(pd.to_numeric(df[col], errors="coerce").median()), 6)
def maxv(df, col): return 0.0 if col not in df or len(df) == 0 else round(float(pd.to_numeric(df[col], errors="coerce").max()), 6)
def minv(df, col): return 0.0 if col not in df or len(df) == 0 else round(float(pd.to_numeric(df[col], errors="coerce").min()), 6)
def rate(df, col): return 0.0 if col not in df or len(df) == 0 else round(float(pd.Series(df[col]).astype(bool).mean()) * 100.0, 6)


def summarize(detail: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    rows = []
    modes = [
        ("t2_open", "win_t2_open", "ret_t1_open_to_t2_open_pct"),
        ("t2_close", "win_t2_close", "ret_t1_open_to_t2_close_pct"),
        ("t3_open", "win_t3_open", "ret_t1_open_to_t3_open_pct"),
        ("t3_close", "win_t3_close", "ret_t1_open_to_t3_close_pct"),
    ]
    for strategy in sorted(detail["strategy"].dropna().unique()):
        s_df = detail[detail["strategy"] == strategy]
        for period, y_df in [(str(y), s_df[s_df["year"] == y]) for y in years] + [("ALL", s_df)]:
            year_val = 0 if period == "ALL" else int(period)
            for mode, win_col, ret_col in modes:
                rows.append({
                    "strategy": strategy, "period": period, "year": year_val, "sell_mode": mode,
                    "trade_count": int(len(y_df)),
                    "win_rate_pct": rate(y_df, win_col),
                    "avg_ret_pct": mean(y_df, ret_col),
                    "median_ret_pct": median(y_df, ret_col),
                    "best_ret_pct": maxv(y_df, ret_col),
                    "worst_ret_pct": minv(y_df, ret_col),
                    "avg_max_opportunity_t1_t2_pct": mean(y_df, "max_opportunity_t1_t2_pct"),
                    "avg_max_drawdown_t1_t2_pct": mean(y_df, "max_drawdown_t1_t2_pct"),
                    "avg_max_opportunity_t1_t3_pct": mean(y_df, "max_opportunity_t1_t3_pct"),
                    "avg_max_drawdown_t1_t3_pct": mean(y_df, "max_drawdown_t1_t3_pct"),
                    "hit_plus_2pct_t1_t2_rate_pct": rate(y_df, "hit_plus_2pct_t1_t2"),
                    "hit_plus_5pct_t1_t2_rate_pct": rate(y_df, "hit_plus_5pct_t1_t2"),
                    "hit_minus_2pct_t1_t2_rate_pct": rate(y_df, "hit_minus_2pct_t1_t2"),
                    "hit_minus_5pct_t1_t2_rate_pct": rate(y_df, "hit_minus_5pct_t1_t2"),
                    "hit_plus_2pct_t1_t3_rate_pct": rate(y_df, "hit_plus_2pct_t1_t3"),
                    "hit_plus_5pct_t1_t3_rate_pct": rate(y_df, "hit_plus_5pct_t1_t3"),
                    "hit_minus_2pct_t1_t3_rate_pct": rate(y_df, "hit_minus_2pct_t1_t3"),
                    "hit_minus_5pct_t1_t3_rate_pct": rate(y_df, "hit_minus_5pct_t1_t3"),
                })
    return pd.DataFrame(rows)


def build_delta(summary: pd.DataFrame) -> pd.DataFrame:
    base, comp = "v0", "v1"
    a = summary[summary["strategy"] == base]
    b = summary[summary["strategy"] == comp]
    keys = ["period", "year", "sell_mode"]
    m = a.merge(b, on=keys, suffixes=("_v0", "_v1"), how="outer")
    out = m[keys].copy()
    out["base_strategy"] = base
    out["compare_strategy"] = comp
    out["delta_name"] = "v1_minus_v0"
    out["trade_count_v0"] = m["trade_count_v0"].fillna(0).astype(int)
    out["trade_count_v1"] = m["trade_count_v1"].fillna(0).astype(int)
    out["trade_count_delta"] = out["trade_count_v1"] - out["trade_count_v0"]
    out["trade_count_keep_pct"] = (out["trade_count_v1"] / out["trade_count_v0"].replace(0, pd.NA) * 100).round(6)
    metrics = [c for c in summary.columns if c not in {"strategy", "period", "year", "sell_mode", "trade_count"}]
    for col in metrics:
        out[f"{col}_v0"] = pd.to_numeric(m[f"{col}_v0"], errors="coerce")
        out[f"{col}_v1"] = pd.to_numeric(m[f"{col}_v1"], errors="coerce")
        out[f"{col}_delta"] = (out[f"{col}_v1"] - out[f"{col}_v0"]).round(6)
    out["better_by_avg_ret"] = out["avg_ret_pct_delta"] > 0
    out["better_by_win_rate"] = out["win_rate_pct_delta"] > 0
    out["better_both"] = out["better_by_avg_ret"] & out["better_by_win_rate"]
    return out.sort_values(["year", "sell_mode"]).reset_index(drop=True)


def best_modes(summary: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    rows = []
    for (strategy, period), g in summary.groupby(["strategy", "period"], dropna=False):
        avg_row = g.sort_values("avg_ret_pct", ascending=False).iloc[0]
        win_row = g.sort_values("win_rate_pct", ascending=False).iloc[0]
        rows.append({
            "strategy": strategy, "period": period,
            "best_avg_sell_mode": avg_row["sell_mode"], "best_avg_ret_pct": avg_row["avg_ret_pct"], "best_avg_win_rate_pct": avg_row["win_rate_pct"], "best_avg_trade_count": avg_row["trade_count"],
            "best_win_sell_mode": win_row["sell_mode"], "best_win_rate_pct": win_row["win_rate_pct"], "best_win_avg_ret_pct": win_row["avg_ret_pct"], "best_win_trade_count": win_row["trade_count"],
        })
    out = pd.DataFrame(rows)
    order = {str(y): y for y in years} | {"ALL": 9999}
    out["_order"] = out["period"].map(order).fillna(9998)
    return out.sort_values(["_order", "strategy"]).drop(columns="_order").reset_index(drop=True)


def feature_coverage(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in POOL_FEATURE_ALIAS_MAP if c in detail.columns]:
        n = int(detail[col].notna().sum())
        rows.append({"feature": col, "non_null": n, "non_null_pct": round(n / max(len(detail), 1) * 100, 4)})
    cov = pd.DataFrame(rows).sort_values(["non_null", "feature"], ascending=[False, True])
    print("-" * 120)
    print("Mapped feature coverage in trade detail")
    print("-" * 120)
    print(cov.to_string(index=False) if len(cov) else "[WARN] No mapped feature columns in detail.")
    return cov


def parse_years(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(description="Compare B2 v0 vs new v1 yearly quality.")
    p.add_argument("--pool-v0", type=Path, default=DEFAULT_POOL_V0)
    p.add_argument("--pool-v1", type=Path, default=DEFAULT_POOL_V1)
    p.add_argument("--txt-dir", type=Path, default=DEFAULT_TXT_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--years", default="2022,2023,2024,2025,2026")
    p.add_argument("--save-detail", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    years = parse_years(args.years)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 120)
    print("Compare B2 v0 vs new v1 by year")
    print("=" * 120)
    print(f"Pool v0    : {args.pool_v0}")
    print(f"Pool v1    : {args.pool_v1}")
    print(f"TXT dir    : {args.txt_dir}")
    print(f"Output dir : {args.output_dir}")
    print(f"Years      : {years}")
    print("-" * 120)
    pool_v0 = load_pool(args.pool_v0, "v0")
    pool_v1 = load_pool(args.pool_v1, "v1")
    if args.limit > 0:
        pool_v0 = pool_v0.head(args.limit).copy()
        pool_v1 = pool_v1.head(args.limit).copy()
    print(f"v0 pool rows: {len(pool_v0):,}")
    print(f"v1 pool rows: {len(pool_v1):,}")
    symbols = sorted(set(pool_v0["_symbol"].dropna().unique()) | set(pool_v1["_symbol"].dropna().unique()))
    print(f"Unique symbols to load: {len(symbols):,}")
    market_cache, warnings = build_market_cache(args.txt_dir, symbols)
    print(f"Market files loaded: {len(market_cache):,}")
    print(f"Market warnings    : {len(warnings):,}")
    detail = pd.concat([analyze_pool(pool_v0, market_cache), analyze_pool(pool_v1, market_cache)], ignore_index=True)
    cov = feature_coverage(detail)
    summary = summarize(detail, years)
    delta = build_delta(summary)
    best = best_modes(summary, years)
    summary_path = args.output_dir / "b2_v0_v1_yearly_summary.csv"
    delta_path = args.output_dir / "b2_v0_v1_yearly_delta.csv"
    best_path = args.output_dir / "b2_v0_v1_best_sell_modes.csv"
    cov_path = args.output_dir / "b2_v0_v1_detail_feature_coverage.csv"
    detail_path = args.output_dir / "b2_v0_v1_trade_detail.csv"
    warn_path = args.output_dir / "b2_v0_v1_market_read_warnings.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    delta.to_csv(delta_path, index=False, encoding="utf-8-sig")
    best.to_csv(best_path, index=False, encoding="utf-8-sig")
    cov.to_csv(cov_path, index=False, encoding="utf-8-sig")
    if warnings:
        pd.DataFrame(warnings).to_csv(warn_path, index=False, encoding="utf-8-sig")
    if args.save_detail:
        detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print("-" * 120)
    print("Best sell modes")
    print("-" * 120)
    print(best.to_string(index=False))
    print("-" * 120)
    print("Delta preview: v1 - v0, T+2 close / T+3 close")
    print("-" * 120)
    preview = delta[delta["sell_mode"].isin(["t2_close", "t3_close"])]
    cols = ["period", "sell_mode", "trade_count_v0", "trade_count_v1", "trade_count_keep_pct", "win_rate_pct_delta", "avg_ret_pct_delta", "median_ret_pct_delta", "worst_ret_pct_delta", "hit_minus_5pct_t1_t3_rate_pct_delta"]
    print(preview[[c for c in cols if c in preview.columns]].to_string(index=False))
    print("-" * 120)
    print("Output files")
    print("-" * 120)
    print(f"Summary         : {summary_path}")
    print(f"Delta           : {delta_path}")
    print(f"Best modes      : {best_path}")
    print(f"Feature coverage: {cov_path}")
    if warnings:
        print(f"Warnings        : {warn_path}")
    print(f"Detail          : {detail_path if args.save_detail else 'not saved. Add --save-detail to save trade detail.'}")
    print("=" * 120)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
