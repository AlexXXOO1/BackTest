renko_trade_strategy_v0
=======================

Install:
1. Copy trade_strategies/renko_trade_strategy_v0.py into your project's trade_strategies folder.
2. Open trade_strategies/registry.py.
3. Add "renko_trade_strategy_v0" to ACTIVE_TRADE_MODULES.

Trading logic:
- T0 is the pool signal date.
- Only selected=True/1 rows are eligible.
- If multiple symbols are selected on the same T0, the strategy chooses one candidate only.
- Buy at T+1 open.
- Sell at T+3 close.
- The engine's occupied_until logic should skip new entries while position capital is occupied.

Run example:
python main.py --start-date 2024-01-01 --end-date 2026-04-24 --selection-strategy renko_chart_select_strategy_v1 --trade-strategy renko_trade_strategy_v0
