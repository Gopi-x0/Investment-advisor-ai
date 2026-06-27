# Project Roadmap

Created: 2026-06-27

Goal: turn this from a learning CLI into a stronger, more realistic investment-analysis project while staying honest about uncertainty and avoiding fake "financial advisor" confidence.

## Implementation Status

Status date: 2026-06-27

All six roadmap phases have been implemented in the local codebase:

- Phase 1: tests, explicit price policy, explicit HOLD label policy, leakage-free ML pipeline, transaction costs/slippage, educational disclaimer.
- Phase 2: richer feature set, sklearn pipelines, walk-forward validation, model metrics, artifact metadata, baseline model choices, permutation importance.
- Phase 3: risk metrics and Monte Carlo scenario modules.
- Phase 4: Streamlit dashboard.
- Phase 5: watchlist and portfolio analytics.
- Phase 6: saved reports, warnings, assumptions, limitations, and usability guardrails.

Remaining work is hardening: pinned dependencies, CI, deterministic fixture data, richer dashboard controls, production-grade data handling, and human-approved changes to business-critical policies such as default price policy and neutral-score behavior.

## Recommended Direction

Build this as an educational decision-support tool, not an automated trading bot. The strongest version of the project should explain risk, show scenarios, validate models out of sample, and make it clear when the model is uncertain.

Best near-term product shape:

1. Keep the Python CLI working.
2. Add tests and reliable data handling.
3. Add offline training/backtesting.
4. Add Monte Carlo and risk analytics.
5. Add a Streamlit GUI.
6. Later, consider a FastAPI backend plus a richer web frontend.

## Phase 1: Make The Core Trustworthy

Before adding flashier features, make the current calculations defensible.

- Add unit tests for indicators, labels, recommendations, and data loading.
- Decide the price policy: raw `Close`, `Adj Close`, or yfinance auto-adjusted prices.
- Decide whether HOLD is a real ML class or just a fallback output.
- Fix feature-selection leakage in walk-forward validation.
- Add transaction cost and slippage assumptions to any backtest.
- Add a clear educational-use disclaimer in the CLI and future GUI.

Why this matters: a GUI or Monte Carlo layer on top of shaky labels and leaky validation makes the project look stronger while making the conclusions less trustworthy.

## Phase 2: Level Up The ML Process

### Leakage-Free Training Pipeline

- Use sklearn `Pipeline` objects for scaling, feature selection, and model training.
- Fit feature selection inside each train fold, not before splitting.
- Keep a final training step separate from validation.
- Store selected features, scaler, model parameters, training date, and label policy.

### Better Validation

- Keep walk-forward validation.
- Add rolling backtests with transaction costs.
- Track accuracy, F1, precision/recall by class, confusion matrix, max drawdown, cumulative return, hit rate, and turnover.
- Compare against baselines:
  - buy-and-hold
  - moving-average rule model
  - random/always-hold baseline if HOLD becomes a class

### Better Labels

- Decide between:
  - Binary labels: BUY/SELL only.
  - Three-class labels: BUY/HOLD/SELL.
- For real-life usability, three-class is usually more honest because "do nothing" is often the best action.
- Consider labels based on future return after costs, not just raw future price movement.

### Better Features

Add features gradually and test each group:

- Lagged returns: 1-day, 5-day, 20-day, 60-day.
- Rolling volatility: 10-day, 20-day, 60-day.
- Trend features: price vs moving averages, moving-average slope.
- Market regime: SPY return, SPY volatility, VIX-like proxy if available.
- Relative strength: ticker return minus market return.
- Drawdown features: distance from recent high.
- Volume features: volume spikes and rolling volume z-score.

Avoid adding many features at once. Feature bloat is an easy way to overfit.

### Model Upgrades

Good next models:

- Logistic Regression as an interpretable baseline.
- Random Forest as current tree baseline.
- Gradient Boosting for stronger nonlinear patterns.
- Calibrated classifiers for probability estimates.

Later experiments:

- XGBoost/LightGBM if you want a serious tabular ML benchmark.
- Simple neural nets only after the tabular baseline is strong.
- LSTM/transformer experiments only as an advanced optional branch; they are easy to overfit on small stock datasets.

### Explainability

- Start with permutation importance.
- Show top features for each prediction.
- Later add SHAP if the model stack supports it cleanly.
- In the GUI, show "why this signal happened" rather than only BUY/SELL.

## Phase 3: Monte Carlo And Risk Analytics

Monte Carlo is a strong feature if framed as scenario analysis, not prediction.

### First Monte Carlo Version

Use historical bootstrapping:

- Pull daily returns for the selected ticker.
- Sample returns with replacement.
- Simulate many future paths, such as 1,000 paths over 20, 60, or 252 trading days.
- Show percentile bands: 5th, 25th, 50th, 75th, 95th.
- Report probability of loss, expected return, Value at Risk, Conditional Value at Risk, and expected max drawdown.

Why bootstrap first: it makes fewer assumptions than geometric Brownian motion and is easier to explain.

### Second Monte Carlo Version

Add parametric scenarios:

- Geometric Brownian motion using estimated drift and volatility.
- Volatility-stressed scenario.
- Bear-market scenario.
- Bull-market scenario.

### Risk Metrics To Add

- Annualized volatility.
- Sharpe ratio.
- Sortino ratio.
- Maximum drawdown.
- Value at Risk.
- Conditional Value at Risk.
- Probability of ending below current price.
- Probability of hitting a stop-loss threshold.

## Phase 4: GUI

Best first GUI: Streamlit.

Why Streamlit fits this project:

- Python-first.
- Fast to build.
- Easy charts and controls.
- Works well for ML demos and dashboards.
- No need for a separate backend at first.

Recommended pages:

- Ticker overview.
- Rule-based signal.
- ML signal and confidence.
- Feature importance.
- Backtest results.
- Monte Carlo simulator.
- Portfolio/watchlist view.
- Model diagnostics.

Useful controls:

- Ticker input.
- Time period selector.
- Label strategy selector.
- Model selector.
- Lookahead window.
- Risk-free rate.
- Transaction cost assumption.
- Monte Carlo horizon and simulation count.

Later GUI path:

- FastAPI backend for analysis endpoints.
- React or Next.js frontend for a polished app.
- SQLite/Postgres for saved runs, watchlists, and model results.

## Phase 5: Portfolio-Level Analysis

Single-stock signals are useful for learning, but real-life investing is portfolio-based.

Add:

- Watchlists.
- Multi-ticker analysis.
- Correlation matrix.
- Portfolio volatility.
- Diversification score.
- Efficient frontier.
- Risk parity allocation.
- Max drawdown by portfolio.
- Monte Carlo portfolio paths.

This is where the project starts feeling much more like a real investment-analysis tool.

## Phase 6: Real-Life Usability Guardrails

To be credible, the app should avoid implying certainty.

Add:

- Educational-use disclaimer.
- Data source and last updated timestamp.
- Backtest assumptions display.
- Transaction cost assumptions.
- Confidence and uncertainty display.
- "No trade / insufficient edge" outcome.
- Saved analysis reports.
- Clear warning when data is too short, labels are imbalanced, or validation variance is high.

Do not add automated real-money trading until the project has strong tests, backtests, audit logs, and explicit risk controls.

## Suggested Build Order

1. Restore Python environment and run tests.
2. Fix any failing tests.
3. Add backtesting with transaction costs.
4. Fix feature-selection leakage.
5. Decide HOLD label policy.
6. Add model artifact persistence.
7. Add Monte Carlo scenario module.
8. Add Streamlit dashboard.
9. Add portfolio/watchlist support.
10. Add explainability and saved reports.

## Best Portfolio Story For This Project

Position the project as:

"A Python investment research dashboard that compares rule-based signals, ML signals, backtests, and Monte Carlo risk scenarios with transparent assumptions."

That is stronger, more honest, and more impressive than "AI stock picker."
