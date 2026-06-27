# Investment Research Report: AAPL

Educational research tool only. This is not financial advice, not a guarantee of returns, and not a real-money trading system.

## Assumptions
- Educational research tool only. This is not financial advice, not a guarantee of returns, and not a real-money trading system.
- Price policy: raw_close - Use the raw Close column from the data provider.
- Label policy: three_class - BUY, HOLD, and SELL are explicit classes; HOLD means no sufficient edge.
- Cost assumptions: 10.0 bps transaction cost + 5.0 bps slippage per position change.

## Data Summary
- rows: 1254
- date_start: 2021-06-28T00:00:00
- date_end: 2026-06-25T00:00:00

## Rule-Based Result
- signal: SELL
- score: -2
- breakdown: {'moving_avg': 'HOLD', 'momentum': 'SELL', 'trend_strength': 'SELL', 'volatility': 'BUY', 'sharpe': 'BUY'}
- current_price: 275.1499938964844
- short_ma: 299.7250015258789
- long_ma: 291.15080017089844
- reason: Multi-factor rule-based trading model (trend, momentum, risk-adjusted returns)

## ML Prediction
- label: -1
- signal: SELL
- probabilities: {'SELL': 0.45097424828050753, 'HOLD': 0.285711861323122, 'BUY': 0.2633138903963705}

## Model Metrics
- mean_accuracy: 0.3651282051282051
- std_accuracy: 0.10665680427779008
- mean_f1: 0.28073993461136004
- classes: [-1, 0, 1]
- class_names: ['SELL', 'HOLD', 'BUY']

## Backtest Metrics
- cumulative_return: 0.1491283618710988
- buy_hold_return: 1.1140507365136836
- max_drawdown: -0.37656238008151866
- annualized_volatility: 0.21011803092247
- sharpe_ratio: 0.27552018028875913
- hit_rate: 0.46722454672245467
- turnover: 85.0
- total_cost_rate: 0.0015

## Risk Summary
- annualized_volatility: 0.27665227125805925
- sharpe_ratio: 0.584291668002933
- sortino_ratio: 0.8464703486133708
- max_drawdown: -0.3343370956600401
- value_at_risk_95: -0.027193814015144667
- conditional_value_at_risk_95: -0.0391533469020241

## Monte Carlo Metrics
- expected_return: 0.04722014718026169
- median_return: 0.03402242745578177
- probability_of_loss: 0.39
- value_at_risk_95: -0.1553501057074358
- conditional_value_at_risk_95: -0.20498653476176348
- expected_max_drawdown: -0.1186223824295716
- probability_below_start: 0.39
- probability_stop_loss_hit: 0.288

## Warnings
- No guardrail warnings were generated for this run.

## Limitations
- Educational research only; not financial advice.
- Market data may be delayed, revised, incomplete, or provider-dependent.
- Backtests and Monte Carlo scenarios are assumption-driven and do not predict guaranteed outcomes.
- No brokerage integration or real-money trading readiness is provided.
