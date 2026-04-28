from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.common import extract_code, save_json
from core.data_store import MarketDataStore
from core.indicator_store import IndicatorStore
from core.pool_store import PoolStore, build_pool_from_indicators
from selection_strategies import SELECTION_STRATEGY_REGISTRY
from trade_strategies import TRADE_STRATEGY_REGISTRY, get_candidate_selector, get_trade_strategy, trade_record_to_dict


def build_or_update_indicator_cache(config, incremental: bool = False) -> pd.DataFrame:
    market_store = MarketDataStore(config.txt_dir, config.market_cache_dir)
    if not market_store.list_cached_symbols():
        print("Market cache is empty. Importing TDX TXT files first...")
        market_store.import_txt_files(end_date=config.end_date)
    indicator_store = IndicatorStore(config.indicator_cache_path)
    if incremental or not indicator_store.exists():
        return indicator_store.build(
            market_store=market_store,
            n1=config.n1,
            n2=config.n2,
            end_date=config.end_date,
            incremental=incremental,
        )
    return indicator_store.read()


def build_pool_for_range(config, overwrite: bool = True) -> Path:
    indicator_df = build_or_update_indicator_cache(config, incremental=False)
    pool_df = build_pool_from_indicators(
        indicator_df=indicator_df,
        selection_strategy=config.selection_strategy,
        start_date=config.start_date,
        end_date=config.end_date,
        n1=config.n1,
        n2=config.n2,
    )
    pool_store = PoolStore(config.pools_dir)
    return pool_store.write_replace_range(config.selection_strategy, pool_df, config.start_date, config.end_date)


def ensure_pool_exists(config, signal_date: pd.Timestamp) -> Path:
    pool_store = PoolStore(config.pools_dir)
    pool_path = pool_store.pool_path(config.selection_strategy)
    if pool_path.exists():
        pool_df = pool_store.read_date(config.selection_strategy, signal_date)
        if not pool_df.empty:
            return pool_path
    print(f"Pool data not found for {signal_date.strftime('%Y-%m-%d')}, generating unified pool file now: {pool_path.name}")
    one_day_config = replace(config, start_date=signal_date, end_date=signal_date)
    return build_pool_for_range(one_day_config, overwrite=True)


def load_pool_df(config, signal_date: pd.Timestamp) -> pd.DataFrame:
    ensure_pool_exists(config, signal_date)
    pool_df = PoolStore(config.pools_dir).read_date(config.selection_strategy, signal_date)
    return pool_df


def select_candidates(signal_df: pd.DataFrame, signal_date: pd.Timestamp, config) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df
    selector = get_candidate_selector(config.trade_strategy)
    if selector is None:
        return signal_df.iloc[[0]].reset_index(drop=True)
    return selector(signal_df=signal_df, signal_date=signal_date, config=config)


def inject_pool_fields_into_df(df: pd.DataFrame, signal_date: pd.Timestamp, pool_row: dict) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    idx_list = df.index[df["date"].dt.normalize() == pd.Timestamp(signal_date).normalize()].tolist()
    if not idx_list:
        return df
    signal_idx = idx_list[0]
    for col, value in pool_row.items():
        if col in {"date"}:
            continue
        if col not in df.columns:
            df[col] = pd.Series([None] * len(df), dtype="object")
        elif isinstance(value, (str, bool, np.bool_)):
            df[col] = df[col].astype("object")
        df.at[signal_idx, col] = value
    return df


def execute_trade_from_store(market_store: MarketDataStore, signal_date: pd.Timestamp, capital_alloc: float, config, pool_row: dict, code: str = ""):
    file_name = str(pool_row.get("file") or "")
    symbol = str(pool_row.get("symbol") or Path(file_name).stem)
    df = market_store.get_symbol_data(symbol)
    if df.empty:
        df = market_store.get_symbol_data(file_name)
    if df.empty:
        return None

    df = inject_pool_fields_into_df(df, signal_date, pool_row)

    # Avoid passing duplicated keyword arguments into trade strategy functions.
    # Pool rows may contain metadata columns with names such as trade_strategy,
    # file_name, or code. These are already passed explicitly or handled by the
    # selected trade strategy, so they must not be forwarded again through
    # **pool_row.
    reserved_kwargs = {
        "df",
        "signal_date",
        "capital_alloc",
        "lot_size",
        "commission_rate",
        "stamp_tax_rate",
        "slippage_rate",
        "code",
        "file_name",
        "trade_strategy",
        "pool_row",
    }
    extra_pool_kwargs = {
        k: v for k, v in pool_row.items() if k not in reserved_kwargs
    }

    return get_trade_strategy(config.trade_strategy)(
        df=df,
        signal_date=signal_date,
        capital_alloc=capital_alloc,
        lot_size=config.lot_size,
        commission_rate=config.commission_rate,
        stamp_tax_rate=config.stamp_tax_rate,
        slippage_rate=config.slippage_rate,
        code=code,
        file_name=file_name,
        pool_row=pool_row,
        **extra_pool_kwargs,
    )


def print_trade_detail(trade_dict: dict) -> None:
    print("  - Trade completed")
    print(f"    File: {trade_dict['file']} | Code: {trade_dict['code']}")
    print(f"    Signal date: {trade_dict['signal_date']} | Buy date: {trade_dict['buy_date']} | Sell date: {trade_dict['sell_date']}")
    print(f"    Buy price: {trade_dict['buy_price']} | Sell price: {trade_dict['sell_price']} | Shares: {trade_dict['shares']}")
    print(f"    Gross PnL: {trade_dict['gross_pnl']} | Net PnL: {trade_dict['net_pnl']} | Return: {trade_dict['ret_pct']}%")
    print(f"    Exit rule: {trade_dict['exit_rule']}")


def build_daily_detail(signal_date, capital_before, capital_after, signal_df, selected_df, trade_dicts) -> dict:
    day_net_pnl = float(sum(x["net_pnl"] for x in trade_dicts)) if trade_dicts else 0.0
    day_avg_ret_pct = float(np.mean([x["ret_pct"] for x in trade_dicts])) if trade_dicts else None
    return {
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "signal_count": int(len(signal_df)),
        "selected_count": int(len(selected_df)),
        "executed_count": int(len(trade_dicts)),
        "capital_before": round(capital_before, 2),
        "capital_after": round(capital_after, 2),
        "day_net_pnl": round(day_net_pnl, 2),
        "day_avg_ret_pct": round(day_avg_ret_pct, 4) if day_avg_ret_pct is not None else None,
        "signals": signal_df.to_dict(orient="records") if not signal_df.empty else [],
        "selected_signals": selected_df.to_dict(orient="records") if not selected_df.empty else [],
        "trades": trade_dicts,
    }


def build_summary(initial_capital: float, final_capital: float, trade_df: pd.DataFrame) -> dict:
    valid = trade_df.dropna(subset=["net_pnl"]).copy() if not trade_df.empty else pd.DataFrame()
    return {
        "initial_capital": round(initial_capital, 2),
        "final_capital": round(final_capital, 2),
        "net_profit": round(final_capital - initial_capital, 2),
        "total_return_pct": round((final_capital / initial_capital - 1) * 100, 4),
        "trade_count": int(len(trade_df)),
        "win_count": int((valid["net_pnl"] > 0).sum()) if not valid.empty else 0,
        "loss_count": int((valid["net_pnl"] < 0).sum()) if not valid.empty else 0,
        "win_rate_pct": round((valid["net_pnl"] > 0).mean() * 100, 4) if not valid.empty else None,
        "avg_trade_ret_pct": round(float(valid["ret_pct"].mean()), 4) if not valid.empty else None,
        "avg_trade_net_pnl": round(float(valid["net_pnl"].mean()), 2) if not valid.empty else None,
        "max_profit_trade": round(float(valid["net_pnl"].max()), 2) if not valid.empty else None,
        "max_loss_trade": round(float(valid["net_pnl"].min()), 2) if not valid.empty else None,
    }


def run_selector(config, overwrite: bool = False) -> None:
    if config.selection_strategy not in SELECTION_STRATEGY_REGISTRY:
        raise ValueError(f"Unknown selection strategy: {config.selection_strategy}")
    pool_store = PoolStore(config.pools_dir)
    pool_path = pool_store.pool_path(config.selection_strategy)
    if pool_path.exists() and not overwrite:
        existing = pool_store.read(config.selection_strategy)
        if not existing.empty:
            existing_dates = pd.to_datetime(existing["date"]).dt.normalize()
            start_ts = pd.Timestamp(config.start_date).normalize()
            end_ts = pd.Timestamp(config.end_date).normalize()
            if ((existing_dates >= start_ts) & (existing_dates <= end_ts)).any():
                print(f"Unified pool already exists for this range, skipped: {pool_path}")
                return
    print("========== Start unified pool build ==========")
    print(f"Selection strategy: {config.selection_strategy}")
    print(f"TXT directory: {config.txt_dir}")
    print(f"Market cache directory: {config.market_cache_dir}")
    print(f"Indicator cache path: {config.indicator_cache_path}")
    print(f"Pool path: {pool_path}")
    output_path = build_pool_for_range(config, overwrite=True)
    pool_df = pool_store.read_date(config.selection_strategy, config.start_date)
    full_pool = pool_store.read(config.selection_strategy)
    print(f"Unified pool saved: {output_path}")
    print(f"Total rows in pool file: {len(full_pool)}")


def run_backtest(config) -> Path | None:
    if config.selection_strategy not in SELECTION_STRATEGY_REGISTRY:
        raise ValueError(f"Unknown selection strategy: {config.selection_strategy}")
    if config.trade_strategy not in TRADE_STRATEGY_REGISTRY:
        raise ValueError(f"Unknown trade strategy: {config.trade_strategy}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.pools_dir.mkdir(parents=True, exist_ok=True)
    market_store = MarketDataStore(config.txt_dir, config.market_cache_dir)
    if not market_store.list_cached_symbols():
        print("Market cache is empty. Importing TDX TXT files first...")
        market_store.import_txt_files(end_date=config.end_date)

    trade_dates = market_store.get_trade_dates(config.start_date, config.end_date)
    if not trade_dates:
        print("No available trade dates in the configured range.")
        return None

    equity = config.initial_capital
    all_trade_records: list[dict] = []
    daily_details: list[dict] = []
    occupied_until: pd.Timestamp | None = None
    skipped_by_position = 0

    for idx, signal_date in enumerate(trade_dates, start=1):
        print(f"\n########## [{idx}/{len(trade_dates)}] Backtest date: {signal_date.strftime('%Y-%m-%d')} ##########")
        capital_before = equity
        trade_dicts: list[dict] = []

        if occupied_until is not None and signal_date < occupied_until:
            skipped_by_position += 1
            print(f"{signal_date.strftime('%Y-%m-%d')} | Position capital occupied, skip new signal | Previous sell date: {occupied_until.strftime('%Y-%m-%d')} | Equity {equity:.2f}")
            daily_details.append({"signal_date": signal_date.strftime("%Y-%m-%d"), "signal_count": 0, "selected_count": 0, "executed_count": 0, "capital_before": round(capital_before, 2), "capital_after": round(equity, 2), "day_net_pnl": 0.0, "day_avg_ret_pct": None, "skip_reason": "position_occupied", "occupied_until": occupied_until.strftime("%Y-%m-%d"), "signals": [], "selected_signals": [], "trades": []})
            continue

        signal_df = load_pool_df(config, signal_date)
        selected_df = select_candidates(signal_df, signal_date, config) if not signal_df.empty else pd.DataFrame(columns=signal_df.columns)

        if not selected_df.empty:
            capital_alloc = equity
            print(f"Selected today {len(selected_df)} stock(s), single-position full-equity allocation: {capital_alloc:.2f}")
            for _, selected_row in selected_df.iterrows():
                row_dict = selected_row.to_dict()
                file_name = str(row_dict.get("file") or "")
                code = extract_code(file_name)
                trade_record = execute_trade_from_store(market_store, signal_date, capital_alloc, config, row_dict, code=code)
                if trade_record is None:
                    print(f"  - Skipped {file_name}, reason: incomplete trade record.")
                    continue
                trade_dict = trade_record_to_dict(trade_record)
                trade_dicts.append(trade_dict)
                all_trade_records.append(trade_dict)
                print_trade_detail(trade_dict)
                sell_date = pd.to_datetime(trade_dict.get("sell_date"), errors="coerce")
                if pd.notna(sell_date):
                    occupied_until = pd.Timestamp(sell_date).normalize()

        day_net_pnl = float(sum(x["net_pnl"] for x in trade_dicts)) if trade_dicts else 0.0
        equity += day_net_pnl
        daily_details.append(build_daily_detail(signal_date, capital_before, equity, signal_df, selected_df, trade_dicts))
        print(f"{signal_date.strftime('%Y-%m-%d')} | Signals {len(signal_df)} | Selected {len(selected_df)} | Executed {len(trade_dicts)} | Daily net PnL {day_net_pnl:.2f} | Equity {equity:.2f}")

    trade_df = pd.DataFrame(all_trade_records)
    summary = build_summary(config.initial_capital, equity, trade_df)
    summary["skipped_by_position"] = int(skipped_by_position)
    result = {
        "config": {
            "txt_dir": str(config.txt_dir),
            "market_cache_dir": str(config.market_cache_dir),
            "indicator_cache_path": str(config.indicator_cache_path),
            "output_dir": str(config.output_dir),
            "pools_dir": str(config.pools_dir),
            "start_date": config.start_date.strftime("%Y-%m-%d"),
            "end_date": config.end_date.strftime("%Y-%m-%d"),
            "selection_strategy": config.selection_strategy,
            "available_selection_strategies": sorted(SELECTION_STRATEGY_REGISTRY),
            "trade_strategy": config.trade_strategy,
            "available_trade_strategies": sorted(TRADE_STRATEGY_REGISTRY),
            "initial_capital": config.initial_capital,
            "lot_size": config.lot_size,
            "commission_rate": config.commission_rate,
            "stamp_tax_rate": config.stamp_tax_rate,
            "slippage_rate": config.slippage_rate,
            "n1": config.n1,
            "n2": config.n2,
        },
        "summary": summary,
        "daily_details": daily_details,
        "trades": all_trade_records,
    }
    json_name = f"{config.start_date.strftime('%Y-%m-%d')}_{config.end_date.strftime('%Y-%m-%d')}_{config.selection_strategy}_{config.trade_strategy}_backtest.json"
    result_json = config.output_dir / json_name
    save_json(result, result_json)
    print("\n========== Backtest completed ==========")
    print(f"Selection strategy: {config.selection_strategy}")
    print(f"Trade strategy: {config.trade_strategy}")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"Result JSON: {result_json}")
    return result_json
