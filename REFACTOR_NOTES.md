# Refactor Notes

## Selection chain

The maintained selection strategy is `renko_chart_select_strategy_v0`. The legacy `v8` alias is no longer used.

The selected flag is named `selected`.

## Indicator chain

Reusable indicators are stored in the `indicators/` package. The indicator layer should only calculate market facts, not strategy decisions.

- `core.py`: TDX-compatible helper functions such as REF, HHV, LLV, MA, and SMA.
- `brick.py`: brick momentum variables and brick reversal indicators.
- `trend.py`: trend-line and price-position indicators.
- `momentum.py`: KDJ and MACD indicators.
- `volume.py`: volume confirmation and surge-shrink pullback indicators.
- `candle_patterns.py`: reusable candle-pattern and price-action facts.
- `quality.py`: reusable renko chart quality facts.

## Strategy layer

Strategy-specific decisions are kept in `selection_strategies/renko_chart_select_strategy_v0.py`:

- scoring weights
- score threshold
- hard risk filter combination
- final `selected` decision

A new strategy should call indicators from `indicators/` and keep filtering, thresholds, scoring, and ranking logic inside the strategy layer.
