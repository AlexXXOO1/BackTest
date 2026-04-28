from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TradeRecord:
    code: str
    file: str
    signal_date: str
    buy_date: str
    buy_price: float
    shares: int
    buy_amount: float
    buy_cost: float
    exit_rule: str
    sell_date: str
    sell_price: float
    sell_amount: float
    sell_cost: float
    gross_pnl: float
    net_pnl: float
    ret_pct: float
    hold_days: int
    t2_close_ret_pct: float
    trade_strategy: str



def trade_record_to_dict(record: TradeRecord) -> dict:
    return asdict(record)
