# Investment Advisor AI

Educational investment research and decision-support tooling for comparing rule-based signals, leakage-free ML validation, financial-statement fundamentals, backtests, risk metrics, Monte Carlo scenarios, decision translation, and portfolio/watchlist analytics.

This project is not financial advice, does not guarantee returns, and is not designed for automated real-money trading.

## What It Does

- Downloads historical market data with `yfinance`.
- Makes the price policy explicit: default is raw `Close`; adjusted close is supported.
- Makes the label policy explicit: default is three-class `BUY` / `HOLD` / `SELL`.
- Runs a multi-factor rule-based signal.
- Builds a leakage-free sklearn ML pipeline with fold-local preprocessing and feature selection.
- Reports validation metrics, class probabilities, feature importance, and warning guardrails.
- Adds a decision translation layer that turns model/risk outputs into action status, conviction, suggested allocation, and dollars-at-risk.
- Pulls yfinance financial statement data where available and scores fundamentals across growth, margins, free cash flow, leverage, and valuation.
- Backtests ML signals with transaction costs and slippage.
- Calculates risk metrics including volatility, Sharpe, Sortino, max drawdown, VaR, and CVaR.
- Runs Monte Carlo bootstrap scenarios.
- Strengthens portfolio diagnostics with concentration, effective positions, pairwise correlation, risk contribution, and portfolio Monte Carlo paths.
- Saves model artifacts plus JSON and Markdown analysis reports.
- Provides a Streamlit dashboard organized around Decision, Fundamentals, Model, Risk, Portfolio, and Report tabs.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m pip install -r requirements-dev.txt
```

If your shell exposes `python` instead of `py`, use `python` in the same commands.

## Run The CLI

```powershell
py src\main.py
```

Successful CLI runs save:

- model artifacts under `artifacts/models/`
- timestamped JSON and Markdown reports under `reports/YYYY-MM-DD/`

The dashboard also saves a timestamped Markdown report for each completed run, so reports can be compared over time instead of replacing a single `latest` file.

## Run The Streamlit Dashboard

```powershell
py -m streamlit run streamlit_app.py
```

## Verify

```powershell
py -m pytest
py -m ruff check src tests streamlit_app.py
py -m compileall src tests streamlit_app.py
```

Verification should be rerun after recreating the local Python environment. The previous verified baseline before the decision/fundamentals UI update was `52 passed`, ruff passed, compileall passed, and a live `AAPL` CLI smoke run completed.

## Documentation

- `PROJECT_ROADMAP.md`: phased product and technical roadmap.
- `PROJECT_MAP.md`: current architecture and entry points.
- `CODEBASE_AUDIT.md`: audit notes, completed cleanup, roadmap implementation summary, and remaining risks.
- `NEXT_CLEANUP_STEPS.md`: safest next hardening steps.
