from __future__ import annotations

"""
Renko trade strategy v4:
Score filter + no ST + stop loss + T+3 close exit.

Compatible with current engine.py call style:
    EXECUTE_FUNC(
        df=df,
        signal_date=signal_date,
        cash=cash,
        code=code,
        **extra_pool_kwargs,
    )

Core rules:
1. Do not buy stocks whose name contains ST, *ST, S*ST, 退, 退市.
2. Do not buy if score_pct < MIN_SCORE_PCT.
3. Buy at T+1 open.
4. Stop loss:
   - stop_price = T0 close * 0.98
   - if T+1 close < stop_price, sell at T+2 open
   - else if T+2 close < stop_price, sell at T+3 open
   - else sell at T+3 close.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd


STRATEGY_NAME = "renko_trade_strategy_v4_score_no_st_t3_close"


# =============================================================================
# Parameters
# =============================================================================

# Score filter
MIN_SCORE_PCT = 40.0

# If no score_pct >= 60, whether to fallback to score_pct >= 40.
ALLOW_SCORE_FALLBACK = False
FALLBACK_MIN_SCORE_PCT = 40.0

# Optional T+1 open gap filter
USE_T1_OPEN_GAP_FILTER = False
MIN_T1_OPEN_GAP_PCT = -2.0
MAX_T1_OPEN_GAP_PCT = 2.0

# Stop loss rule
USE_STOP_LOSS = True
STOP_LOSS_RATIO = 0.98

# Trading cost / capital defaults
DEFAULT_CASH = 20000.0
DEFAULT_LOT_SIZE = 100
DEFAULT_COMMISSION_RATE = 0.0003
DEFAULT_STAMP_TAX_RATE = 0.001


# =============================================================================
# Result object
# =============================================================================

@dataclass
class TradeResult:
    code: str
    stock_name: str
    signal_date: pd.Timestamp
    buy_date: pd.Timestamp
    sell_date: pd.Timestamp
    buy_price: float
    sell_price: float
    shares: int
    gross_pnl: float
    net_pnl: float
    ret_pct: float
    score_pct: float
    score: float
    score_rank_key: float
    stop_price: float
    exit_rule: str
    strategy: str


# =============================================================================
# Basic helpers
# =============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def normalize_stock_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_stock_name_from_data(data: dict[str, Any] | pd.Series) -> str:
    """
    Supported stock-name columns:
    - 股票名
    - stock_name
    - name
    - security_name
    - stockName
    """
    for col in ["股票名", "stock_name", "name", "security_name", "stockName"]:
        try:
            value = data.get(col, "")
        except Exception:
            value = ""

        name = normalize_stock_name(value)

        if name:
            return name

    return ""


def is_st_stock_name(stock_name: str) -> bool:
    """
    Return True if stock name contains ST-like keywords.
    """
    if not stock_name:
        return False

    name = stock_name.upper().replace(" ", "")

    keywords = [
        "ST",
        "*ST",
        "S*ST",
        "退",
        "退市",
    ]

    return any(k.upper() in name for k in keywords)


def build_pool_row_from_kwargs(
    *,
    candidate: Any = None,
    code: str = "",
    kwargs: dict[str, Any],
) -> pd.Series:
    """
    Build one candidate row from:
    1. candidate if engine passes it.
    2. otherwise use **extra_pool_kwargs.
    """
    if isinstance(candidate, pd.Series):
        row = candidate.copy()
    elif isinstance(candidate, dict):
        row = pd.Series(candidate)
    else:
        row = pd.Series(kwargs)

    if "code" not in row.index or not str(row.get("code", "")).strip():
        row["code"] = code

    return row


def should_skip_by_score(row: pd.Series) -> bool:
    """
    Return True if score filter rejects this candidate.
    """
    if "score_pct" not in row.index:
        # If pool has no score_pct, do not block by score.
        return False

    score_pct = safe_float(row.get("score_pct"), default=-999.0)

    if score_pct >= MIN_SCORE_PCT:
        return False

    if ALLOW_SCORE_FALLBACK and score_pct >= FALLBACK_MIN_SCORE_PCT:
        return False

    return True


# =============================================================================
# Market data helpers
# =============================================================================

def sort_market_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "date" not in out.columns:
        raise KeyError("Market df does not contain 'date' column.")

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def get_future_bar(
    df: pd.DataFrame,
    signal_date: pd.Timestamp,
    offset: int,
) -> pd.Series | None:
    """
    Get T+offset bar by trading-day order.
    """
    if df is None or df.empty:
        return None

    data = sort_market_df(df)
    signal_date = pd.Timestamp(signal_date).normalize()

    idx_list = data.index[data["date"] == signal_date].tolist()

    if not idx_list:
        return None

    idx = idx_list[0]
    target_idx = idx + offset

    if target_idx >= len(data):
        return None

    return data.iloc[target_idx]


def resolve_cash(
    *,
    cash: Any = None,
    config: Any = None,
    kwargs: dict[str, Any],
) -> float:
    """
    Resolve available cash from engine parameters or config.
    """
    for value in [
        cash,
        kwargs.get("cash"),
        kwargs.get("current_cash"),
        kwargs.get("available_cash"),
        kwargs.get("capital"),
        kwargs.get("initial_cash"),
    ]:
        resolved = safe_float(value, default=-1.0)
        if resolved > 0:
            return resolved

    if config is not None:
        for attr in ["cash", "current_cash", "initial_cash", "initial_capital", "capital"]:
            if hasattr(config, attr):
                resolved = safe_float(getattr(config, attr), default=-1.0)
                if resolved > 0:
                    return resolved

    return DEFAULT_CASH


def calculate_shares(
    cash: float,
    buy_price: float,
    lot_size: int = DEFAULT_LOT_SIZE,
) -> int:
    """
    A-share lot calculation.
    """
    if buy_price <= 0 or cash <= 0:
        return 0

    raw_shares = int(cash // buy_price)
    shares = raw_shares // lot_size * lot_size

    return int(shares)


def apply_costs(
    *,
    buy_price: float,
    sell_price: float,
    shares: int,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    stamp_tax_rate: float = DEFAULT_STAMP_TAX_RATE,
) -> tuple[float, float, float]:
    """
    Return:
    - gross_pnl
    - net_pnl
    - ret_pct
    """
    buy_amount = buy_price * shares
    sell_amount = sell_price * shares

    gross_pnl = sell_amount - buy_amount

    buy_commission = buy_amount * commission_rate
    sell_commission = sell_amount * commission_rate
    stamp_tax = sell_amount * stamp_tax_rate

    net_pnl = gross_pnl - buy_commission - sell_commission - stamp_tax

    if buy_amount > 0:
        ret_pct = net_pnl / buy_amount * 100.0
    else:
        ret_pct = 0.0

    return gross_pnl, net_pnl, ret_pct


# =============================================================================
# Optional candidate selector
# =============================================================================

def choose_candidate(candidates: pd.DataFrame) -> pd.Series | None:
    """
    Optional selector.

    Some engine versions use this function.
    Your current engine appears to iterate candidates itself,
    so execute_trade also repeats ST and score filtering.
    """
    if candidates is None or candidates.empty:
        return None

    df = candidates.copy()

    if "selected" in df.columns:
        selected = pd.to_numeric(df["selected"], errors="coerce").fillna(0).astype(int)
        df = df[selected == 1].copy()

    if df.empty:
        return None

    # Filter ST by stock name if stock-name column exists.
    if any(col in df.columns for col in ["股票名", "stock_name", "name", "security_name", "stockName"]):
        names = df.apply(get_stock_name_from_data, axis=1)
        df = df[~names.map(is_st_stock_name)].copy()

    if df.empty:
        return None

    # Score filter
    if "score_pct" in df.columns:
        score_pct = pd.to_numeric(df["score_pct"], errors="coerce")
        df_high = df[score_pct >= MIN_SCORE_PCT].copy()

        if not df_high.empty:
            df = df_high
        elif ALLOW_SCORE_FALLBACK:
            df = df[score_pct >= FALLBACK_MIN_SCORE_PCT].copy()
        else:
            return None

    # Ensure ranking columns
    for col in [
        "score_pct",
        "score_rank_key",
        "close_to_short_trend",
        "brick_reversal_ratio",
        "daily_return_pct",
        "score",
    ]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "code" not in df.columns:
        df["code"] = ""

    df = df.sort_values(
        by=[
            "score_pct",
            "score_rank_key",
            "close_to_short_trend",
            "brick_reversal_ratio",
            "daily_return_pct",
            "code",
        ],
        ascending=[
            False,
            False,
            True,
            True,
            False,
            True,
        ],
        na_position="last",
    ).reset_index(drop=True)

    if df.empty:
        return None

    return df.iloc[0]


# =============================================================================
# Execute function compatible with engine.py
# =============================================================================

def execute_trade(
    *,
    df: pd.DataFrame,
    signal_date: Any,
    cash: Any = None,
    config: Any = None,
    candidate: Any = None,
    code: str = "",
    lot_size: int = DEFAULT_LOT_SIZE,
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    stamp_tax_rate: float = DEFAULT_STAMP_TAX_RATE,
    **kwargs: Any,
) -> TradeResult | None:
    """
    Compatible with current engine.py.

    Trade rules:
    1. Skip ST stock by stock name.
    2. Skip score_pct < MIN_SCORE_PCT.
    3. Buy at T+1 open.
    4. Stop loss:
       - stop_price = T0 close * STOP_LOSS_RATIO
       - if T+1 close < stop_price, sell at T+2 open
       - else if T+2 close < stop_price, sell at T+3 open
       - else sell at T+3 close.
    """
    row = build_pool_row_from_kwargs(
        candidate=candidate,
        code=code,
        kwargs=kwargs,
    )

    code_value = str(row.get("code", code)).strip()
    stock_name = get_stock_name_from_data(row)

    # -------------------------------------------------------------------------
    # ST filter
    # -------------------------------------------------------------------------
    if is_st_stock_name(stock_name):
        print(f"{STRATEGY_NAME}: skip ST stock: {code_value} {stock_name}")
        return None

    # -------------------------------------------------------------------------
    # Score filter
    # -------------------------------------------------------------------------
    if should_skip_by_score(row):
        score_pct = safe_float(row.get("score_pct"), default=-999.0)
        print(
            f"{STRATEGY_NAME}: skip by score_pct: "
            f"{code_value} {stock_name} score_pct={score_pct:.4f}, "
            f"min_score_pct={MIN_SCORE_PCT}"
        )
        return None

    signal_date = pd.Timestamp(signal_date).normalize()

    t0_bar = get_future_bar(df, signal_date, 0)
    t1_bar = get_future_bar(df, signal_date, 1)
    t2_bar = get_future_bar(df, signal_date, 2)
    t3_bar = get_future_bar(df, signal_date, 3)

    if t0_bar is None or t1_bar is None or t2_bar is None or t3_bar is None:
        return None

    # Required price columns
    required_t0_cols = ["close"]
    required_t1_cols = ["open", "close"]
    required_t2_cols = ["open", "close"]
    required_t3_cols = ["open", "close"]

    for col in required_t0_cols:
        if col not in t0_bar.index:
            return None

    for col in required_t1_cols:
        if col not in t1_bar.index:
            return None

    for col in required_t2_cols:
        if col not in t2_bar.index:
            return None

    for col in required_t3_cols:
        if col not in t3_bar.index:
            return None

    t0_close = safe_float(t0_bar["close"])
    t1_open = safe_float(t1_bar["open"])
    t1_close = safe_float(t1_bar["close"])
    t2_open = safe_float(t2_bar["open"])
    t2_close = safe_float(t2_bar["close"])
    t3_open = safe_float(t3_bar["open"])
    t3_close = safe_float(t3_bar["close"])

    if t0_close <= 0 or t1_open <= 0:
        return None

    buy_price = t1_open
    buy_date = pd.Timestamp(t1_bar["date"]).normalize()

    # -------------------------------------------------------------------------
    # Optional T+1 open gap filter
    # -------------------------------------------------------------------------
    if USE_T1_OPEN_GAP_FILTER:
        t1_open_gap_pct = (buy_price / t0_close - 1.0) * 100.0

        if not (MIN_T1_OPEN_GAP_PCT <= t1_open_gap_pct <= MAX_T1_OPEN_GAP_PCT):
            print(
                f"{STRATEGY_NAME}: skip by T+1 open gap: "
                f"{code_value} {stock_name} gap={t1_open_gap_pct:.2f}%"
            )
            return None

    # -------------------------------------------------------------------------
    # Stop loss / exit rule
    # -------------------------------------------------------------------------
    stop_price = buy_price  * STOP_LOSS_RATIO

    if USE_STOP_LOSS and t1_close > 0 and t1_close < stop_price:
        if t2_open <= 0:
            return None

        sell_price = t2_open
        sell_date = pd.Timestamp(t2_bar["date"]).normalize()
        exit_rule = "stop_loss_t1_close_below_t0_close_98pct_sell_t2_open"

    elif USE_STOP_LOSS and t2_close > 0 and t2_close < stop_price:
        if t3_open <= 0:
            return None

        sell_price = t3_open
        sell_date = pd.Timestamp(t3_bar["date"]).normalize()
        exit_rule = "stop_loss_t2_close_below_t0_close_98pct_sell_t3_open"

    else:
        if t3_close <= 0:
            return None

        sell_price = t3_close
        sell_date = pd.Timestamp(t3_bar["date"]).normalize()
        exit_rule = "t3_close"

    # -------------------------------------------------------------------------
    # Position sizing
    # -------------------------------------------------------------------------
    resolved_cash = resolve_cash(
        cash=cash,
        config=config,
        kwargs=kwargs,
    )

    shares = calculate_shares(
        cash=resolved_cash,
        buy_price=buy_price,
        lot_size=lot_size,
    )

    if shares <= 0:
        return None

    gross_pnl, net_pnl, ret_pct = apply_costs(
        buy_price=buy_price,
        sell_price=sell_price,
        shares=shares,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
    )

    result = TradeResult(
        code=code_value,
        stock_name=stock_name,
        signal_date=signal_date,
        buy_date=buy_date,
        sell_date=sell_date,
        buy_price=buy_price,
        sell_price=sell_price,
        shares=shares,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        ret_pct=ret_pct,
        score_pct=safe_float(row.get("score_pct"), default=0.0),
        score=safe_float(row.get("score"), default=0.0),
        score_rank_key=safe_float(row.get("score_rank_key"), default=0.0),
        stop_price=stop_price,
        exit_rule=exit_rule,
        strategy=STRATEGY_NAME,
    )

    print(
        f"{STRATEGY_NAME} executed: "
        f"{code_value} {stock_name} | "
        f"score_pct={result.score_pct:.4f} | "
        f"stop_price={stop_price:.4f} | "
        f"exit_rule={exit_rule} | "
        f"buy={buy_price:.4f} | "
        f"sell={sell_price:.4f} | "
        f"ret_pct={ret_pct:.4f}%"
    )

    return result


def trade_record_to_dict(record: TradeResult | dict[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {}

    if isinstance(record, dict):
        return record

    return {
        "code": record.code,
        "stock_name": record.stock_name,
        "signal_date": record.signal_date,
        "buy_date": record.buy_date,
        "sell_date": record.sell_date,
        "buy_price": record.buy_price,
        "sell_price": record.sell_price,
        "shares": record.shares,
        "gross_pnl": record.gross_pnl,
        "net_pnl": record.net_pnl,
        "ret_pct": record.ret_pct,
        "score_pct": record.score_pct,
        "score": record.score,
        "score_rank_key": record.score_rank_key,
        "stop_price": record.stop_price,
        "exit_rule": record.exit_rule,
        "strategy": record.strategy,
    }


# =============================================================================
# Registry aliases
# =============================================================================

# Your trade_strategies/registry.py requires this name.
EXECUTE_FUNC = execute_trade

# Compatibility aliases.
TRADE_FUNC = execute_trade

SELECT_CANDIDATE_FUNC = choose_candidate
CANDIDATE_SELECTOR_FUNC = choose_candidate
GET_CANDIDATE_FUNC = choose_candidate