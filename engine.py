from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import time

import numpy as np
import pandas as pd

from core.common import extract_code, save_json
from core.data_store import MarketDataStore
from core.indicator_store import IndicatorStore
from core.pool_store import PoolStore, build_pool_from_indicators
from selection_strategies import SELECTION_STRATEGY_REGISTRY
from trade_strategies import (
    TRADE_STRATEGY_REGISTRY,
    get_candidate_selector,
    get_trade_strategy,
    trade_record_to_dict,
)






def color_text(text: str, color_code: str) -> str:
    return f"\033[{color_code}m{text}\033[0m"


def yellow_text(text: str) -> str:
    return color_text(text, "93")


def green_text(text: str) -> str:
    return color_text(text, "92")


def red_text(text: str) -> str:
    return color_text(text, "91")


def cyan_text(text: str) -> str:
    return color_text(text, "96")


def gray_text(text: str) -> str:
    return color_text(text, "90")


def print_line(char: str = "-", width: int = 88) -> None:
    print(char * width)


def print_section(title: str, char: str = "=") -> None:
    print()
    print_line(char)
    print(title)
    print_line(char)


def print_day_header(idx: int, total: int, signal_date: pd.Timestamp) -> None:
    print()
    print_line("=", 96)
    print(
        cyan_text(
            f"[{idx}/{total}] Backtest date: {signal_date.strftime('%Y-%m-%d')}"
        )
    )
    print_line("=", 96)


def print_key_value(label: str, value: Any, indent: int = 2) -> None:
    print(f"{' ' * indent}{label:<26}: {value}")


def print_skip_reason(file_name: str, reason: str, index: int | None = None) -> None:
    prefix = f"Candidate #{index}" if index is not None else "Candidate"
    print(
        gray_text(
            f"    - {prefix} skipped | File: {file_name or 'N/A'} | Reason: {reason}"
        )
    )


def print_trade_detail(trade_dict: dict) -> None:
    print(green_text("    + Trade executed"))

    print_key_value("File", trade_dict.get("file", ""), indent=6)
    print_key_value("Code", trade_dict.get("code", ""), indent=6)

    print_key_value(
        "Dates",
        (
            f"Signal {trade_dict.get('signal_date')} | "
            f"Buy {trade_dict.get('buy_date')} | "
            f"Sell {trade_dict.get('sell_date')}"
        ),
        indent=6,
    )

    print_key_value(
        "Price / Shares",
        (
            f"Buy {trade_dict.get('buy_price')} | "
            f"Sell {trade_dict.get('sell_price')} | "
            f"Shares {trade_dict.get('shares')}"
        ),
        indent=6,
    )

    net_pnl = float(trade_dict.get("net_pnl", 0.0))
    ret_pct = trade_dict.get("ret_pct", None)

    pnl_text = (
        green_text(f"{net_pnl:.2f}") if net_pnl >= 0 else red_text(f"{net_pnl:.2f}")
    )

    print_key_value(
        "PnL",
        (
            f"Gross {trade_dict.get('gross_pnl')} | "
            f"Net {pnl_text} | "
            f"Return {ret_pct}%"
        ),
        indent=6,
    )

    print_key_value("Exit rule", trade_dict.get("exit_rule", ""), indent=6)


def print_settlement_capital(
    signal_date: pd.Timestamp,
    capital_before: float,
    day_net_pnl: float,
    capital_after: float,
) -> None:
    pnl_text = (
        f"+{day_net_pnl:.2f}" if day_net_pnl >= 0 else f"{day_net_pnl:.2f}"
    )

    print(
        yellow_text(
            f"  Settlement | Date {signal_date.strftime('%Y-%m-%d')} | "
            f"Before {capital_before:.2f} | "
            f"Daily PnL {pnl_text} | "
            f"After {capital_after:.2f}"
        )
    )


def print_daily_summary(
    signal_date: pd.Timestamp,
    signal_count: int,
    candidate_count: int,
    executed_count: int,
    day_net_pnl: float,
    equity: float,
) -> None:
    pnl_text = (
        green_text(f"{day_net_pnl:.2f}")
        if day_net_pnl >= 0
        else red_text(f"{day_net_pnl:.2f}")
    )

    print_line("-", 96)
    print(
        f"  Daily summary | "
        f"Date {signal_date.strftime('%Y-%m-%d')} | "
        f"Signals {signal_count} | "
        f"Candidates {candidate_count} | "
        f"Executed {executed_count} | "
        f"Daily PnL {pnl_text} | "
        f"Equity {equity:.2f}"
    )


def print_final_summary(summary: dict, result_json: Path, config) -> None:
    print_section("Backtest completed", "=")

    print_key_value("Selection strategy", config.selection_strategy)
    print_key_value("Trade strategy", config.trade_strategy)
    print_key_value("Start date", config.start_date.strftime("%Y-%m-%d"))
    print_key_value("End date", config.end_date.strftime("%Y-%m-%d"))

    print_line("-", 88)

    for key, value in summary.items():
        if key in {"net_profit", "total_return_pct", "avg_trade_net_pnl"}:
            if isinstance(value, (int, float)) and value < 0:
                value = red_text(str(value))
            else:
                value = green_text(str(value))
        print_key_value(key, value)

    print_line("-", 88)
    print_key_value("Result JSON", result_json)






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

    return pool_store.write_replace_range(
        config.selection_strategy,
        pool_df,
        config.start_date,
        config.end_date,
    )


def ensure_pool_exists(config, signal_date: pd.Timestamp) -> Path:
    """
    Ensure the unified pool file exists.

    Important:
    - A unified pool file may exist even when a specific signal_date has no candidates.
    - An empty daily pool should not trigger rebuilding.
    - Only rebuild when the pool parquet file itself does not exist.
    """
    pool_store = PoolStore(config.pools_dir)
    pool_path = pool_store.pool_path(config.selection_strategy)

    if pool_path.exists():
        return pool_path

    print(
        yellow_text(
            f"Pool file not found. Generating unified pool file now: {pool_path.name}"
        )
    )

    return build_pool_for_range(config, overwrite=True)


def load_pool_df(config, signal_date: pd.Timestamp) -> pd.DataFrame:
    ensure_pool_exists(config, signal_date)
    pool_df = PoolStore(config.pools_dir).read_date(
        config.selection_strategy,
        signal_date,
    )
    return pool_df


def select_candidates(
    signal_df: pd.DataFrame,
    signal_date: pd.Timestamp,
    config,
) -> pd.DataFrame:
    if signal_df.empty:
        return signal_df

    selector = get_candidate_selector(config.trade_strategy)

    if selector is None:
        return signal_df.iloc[[0]].reset_index(drop=True)

    return selector(
        signal_df=signal_df,
        signal_date=signal_date,
        config=config,
    )






def inject_pool_fields_into_df(
    df: pd.DataFrame,
    signal_date: pd.Timestamp,
    pool_row: dict,
) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    idx_list = df.index[
        df["date"].dt.normalize() == pd.Timestamp(signal_date).normalize()
    ].tolist()

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


def execute_trade_from_store(
    market_store: MarketDataStore,
    signal_date: pd.Timestamp,
    capital_alloc: float,
    config,
    pool_row: dict,
    code: str = "",
):
    file_name = str(pool_row.get("file") or "")
    symbol = str(pool_row.get("symbol") or Path(file_name).stem)

    df = market_store.get_symbol_data(symbol)

    if df.empty:
        df = market_store.get_symbol_data(file_name)

    if df.empty:
        return None

    df = inject_pool_fields_into_df(df, signal_date, pool_row)

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






def build_daily_detail(
    signal_date,
    capital_before,
    capital_after,
    signal_df,
    selected_df,
    trade_dicts,
) -> dict:
    day_net_pnl = float(sum(x["net_pnl"] for x in trade_dicts)) if trade_dicts else 0.0

    day_avg_ret_pct = (
        float(np.mean([x["ret_pct"] for x in trade_dicts]))
        if trade_dicts
        else None
    )

    return {
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "signal_count": int(len(signal_df)),
        "selected_count": int(len(selected_df)),
        "executed_count": int(len(trade_dicts)),
        "capital_before": round(capital_before, 2),
        "capital_after": round(capital_after, 2),
        "day_net_pnl": round(day_net_pnl, 2),
        "day_avg_ret_pct": (
            round(day_avg_ret_pct, 4) if day_avg_ret_pct is not None else None
        ),
        "signals": signal_df.to_dict(orient="records") if not signal_df.empty else [],
        "selected_signals": (
            selected_df.to_dict(orient="records") if not selected_df.empty else []
        ),
        "trades": trade_dicts,
    }


def build_summary(
    initial_capital: float,
    final_capital: float,
    trade_df: pd.DataFrame,
) -> dict:
    valid = (
        trade_df.dropna(subset=["net_pnl"]).copy()
        if not trade_df.empty
        else pd.DataFrame()
    )

    return {
        "initial_capital": round(initial_capital, 2),
        "final_capital": round(final_capital, 2),
        "net_profit": round(final_capital - initial_capital, 2),
        "total_return_pct": round((final_capital / initial_capital - 1) * 100, 4),
        "trade_count": int(len(trade_df)),
        "win_count": int((valid["net_pnl"] > 0).sum()) if not valid.empty else 0,
        "loss_count": int((valid["net_pnl"] < 0).sum()) if not valid.empty else 0,
        "win_rate_pct": (
            round((valid["net_pnl"] > 0).mean() * 100, 4)
            if not valid.empty
            else None
        ),
        "avg_trade_ret_pct": (
            round(float(valid["ret_pct"].mean()), 4) if not valid.empty else None
        ),
        "avg_trade_net_pnl": (
            round(float(valid["net_pnl"].mean()), 2) if not valid.empty else None
        ),
        "max_profit_trade": (
            round(float(valid["net_pnl"].max()), 2) if not valid.empty else None
        ),
        "max_loss_trade": (
            round(float(valid["net_pnl"].min()), 2) if not valid.empty else None
        ),
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

    print_section("Start unified pool build", "=")
    print_key_value("Selection strategy", config.selection_strategy)
    print_key_value("TXT directory", config.txt_dir)
    print_key_value("Market cache directory", config.market_cache_dir)
    print_key_value("Indicator cache path", config.indicator_cache_path)
    print_key_value("Pool path", pool_path)

    output_path = build_pool_for_range(config, overwrite=True)
    full_pool = pool_store.read(config.selection_strategy)

    print_line("-", 88)
    print_key_value("Unified pool saved", output_path)
    print_key_value("Total rows in pool file", len(full_pool))






def run_backtest(config) -> Path | None:
    if config.selection_strategy not in SELECTION_STRATEGY_REGISTRY:
        raise ValueError(f"Unknown selection strategy: {config.selection_strategy}")

    if config.trade_strategy not in TRADE_STRATEGY_REGISTRY:
        raise ValueError(f"Unknown trade strategy: {config.trade_strategy}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.pools_dir.mkdir(parents=True, exist_ok=True)

    market_store = MarketDataStore(config.txt_dir, config.market_cache_dir)

    if not market_store.list_cached_symbols():
        print(yellow_text("Market cache is empty. Importing TDX TXT files first..."))
        market_store.import_txt_files(end_date=config.end_date)

    trade_dates = market_store.get_trade_dates(config.start_date, config.end_date)

    if not trade_dates:
        print(red_text("No available trade dates in the configured range."))
        return None

    print_section("Start backtest", "=")
    print_key_value("Selection strategy", config.selection_strategy)
    print_key_value("Trade strategy", config.trade_strategy)
    print_key_value("Start date", config.start_date.strftime("%Y-%m-%d"))
    print_key_value("End date", config.end_date.strftime("%Y-%m-%d"))
    print_key_value("Initial capital", f"{config.initial_capital:.2f}")
    print_key_value("Total trade dates", len(trade_dates))

    equity = config.initial_capital
    all_trade_records: list[dict] = []
    daily_details: list[dict] = []
    occupied_until: pd.Timestamp | None = None
    skipped_by_position = 0

    for idx, signal_date in enumerate(trade_dates, start=1):
        print_day_header(idx, len(trade_dates), signal_date)

        capital_before = equity
        trade_dicts: list[dict] = []

        if occupied_until is not None and signal_date < occupied_until:
            skipped_by_position += 1

            print(
                yellow_text(
                    f"  Position occupied. Skip new signal. "
                    f"Previous sell date: {occupied_until.strftime('%Y-%m-%d')} | "
                    f"Current equity: {equity:.2f}"
                )
            )

            daily_details.append(
                {
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    "signal_count": 0,
                    "selected_count": 0,
                    "executed_count": 0,
                    "capital_before": round(capital_before, 2),
                    "capital_after": round(equity, 2),
                    "day_net_pnl": 0.0,
                    "day_avg_ret_pct": None,
                    "skip_reason": "position_occupied",
                    "occupied_until": occupied_until.strftime("%Y-%m-%d"),
                    "signals": [],
                    "selected_signals": [],
                    "trades": [],
                }
            )

            print_daily_summary(
                signal_date=signal_date,
                signal_count=0,
                candidate_count=0,
                executed_count=0,
                day_net_pnl=0.0,
                equity=equity,
            )

            print_settlement_capital(
                signal_date=signal_date,
                capital_before=capital_before,
                day_net_pnl=0.0,
                capital_after=equity,
            )

            continue

        signal_df = load_pool_df(config, signal_date)

        print_key_value("Raw signals", len(signal_df), indent=2)

        selected_df = (
            select_candidates(signal_df, signal_date, config)
            if not signal_df.empty
            else pd.DataFrame(columns=signal_df.columns)
        )

        print_key_value("Ranked candidates", len(selected_df), indent=2)

        if selected_df.empty:
            print(gray_text("  No candidate for this date."))
        else:
            capital_alloc = equity

            print_key_value(
                "Capital allocation",
                f"{capital_alloc:.2f} / single-position full-equity",
                indent=2,
            )

            executed_today = False

            print_line("-", 96)
            print("  Candidate execution:")

            for candidate_idx, (_, selected_row) in enumerate(
                selected_df.iterrows(),
                start=1,
            ):
                row_dict = selected_row.to_dict()
                file_name = str(row_dict.get("file") or "")
                code = extract_code(file_name)

                score_pct = row_dict.get("score_pct", None)

                print(
                    f"    > Try candidate #{candidate_idx} | "
                    f"File: {file_name or 'N/A'} | "
                    f"Code: {code or 'N/A'} | "
                    f"score_pct: {score_pct}"
                )

                trade_record = execute_trade_from_store(
                    market_store=market_store,
                    signal_date=signal_date,
                    capital_alloc=capital_alloc,
                    config=config,
                    pool_row=row_dict,
                    code=code,
                )

                if trade_record is None:
                    print_skip_reason(
                        file_name=file_name,
                        reason="trade filter not passed or not enough future bars",
                        index=candidate_idx,
                    )
                    continue

                trade_dict = trade_record_to_dict(trade_record)

                trade_dicts.append(trade_dict)
                all_trade_records.append(trade_dict)

                print_trade_detail(trade_dict)

                time.sleep(0.3)

                sell_date = pd.to_datetime(
                    trade_dict.get("sell_date"),
                    errors="coerce",
                )

                if pd.notna(sell_date):
                    occupied_until = pd.Timestamp(sell_date).normalize()

                executed_today = True

                print(
                    green_text(
                        f"    Stop trying more candidates because one trade "
                        f"has been executed for {signal_date.strftime('%Y-%m-%d')}."
                    )
                )

                break

            if not executed_today:
                print(
                    yellow_text(
                        f"  No candidate passed execution filters on "
                        f"{signal_date.strftime('%Y-%m-%d')}."
                    )
                )

        day_net_pnl = (
            float(sum(x["net_pnl"] for x in trade_dicts))
            if trade_dicts
            else 0.0
        )

        equity += day_net_pnl

        daily_details.append(
            build_daily_detail(
                signal_date=signal_date,
                capital_before=capital_before,
                capital_after=equity,
                signal_df=signal_df,
                selected_df=selected_df,
                trade_dicts=trade_dicts,
            )
        )

        print_daily_summary(
            signal_date=signal_date,
            signal_count=len(signal_df),
            candidate_count=len(selected_df),
            executed_count=len(trade_dicts),
            day_net_pnl=day_net_pnl,
            equity=equity,
        )

        print_settlement_capital(
            signal_date=signal_date,
            capital_before=capital_before,
            day_net_pnl=day_net_pnl,
            capital_after=equity,
        )

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

    json_name = (
        f"{config.start_date.strftime('%Y-%m-%d')}_"
        f"{config.end_date.strftime('%Y-%m-%d')}_"
        f"{config.selection_strategy}_"
        f"{config.trade_strategy}_backtest.json"
    )

    result_json = config.output_dir / json_name
    save_json(result, result_json)

    print_final_summary(summary, result_json, config)

    return result_json
