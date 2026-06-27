# Investment Advisor AI

Educational investment research and decision-support tooling for comparing rule-based signals, leakage-free ML validation, backtests, risk metrics, Monte Carlo scenarios, and portfolio/watchlist analytics.

This project is not financial advice, does not guarantee returns, and is not designed for automated real-money trading.

## What It Does

- Downloads historical market data with `yfinance`.
- Makes the price policy explicit: default is raw `Close`; adjusted close is supported.
- Makes the label policy explicit: default is three-class `BUY` / `HOLD` / `SELL`.
- Runs a multi-factor rule-based signal.
- Builds a leakage-free sklearn ML pipeline with fold-local preprocessing and feature selection.
- Reports validation metrics, class probabilities, feature importance, and warning guardrails.
- Backtests ML signals with transaction costs and slippage.
- Calculates risk metrics including volatility, Sharpe, Sortino, max drawdown, VaR, and CVaR.
- Runs Monte Carlo bootstrap scenarios.
- Saves model artifacts plus JSON and Markdown analysis reports.
- Provides a Streamlit dashboard with ticker and portfolio/watchlist views.

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
- JSON and Markdown reports under `reports/`

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

Latest local verification: `52 passed`, ruff passed, compileall passed, and a live `AAPL` CLI smoke run completed.

## Documentation

- `PROJECT_ROADMAP.md`: phased product and technical roadmap.
- `PROJECT_MAP.md`: current architecture and entry points.
- `CODEBASE_AUDIT.md`: audit notes, completed cleanup, roadmap implementation summary, and remaining risks.
- `NEXT_CLEANUP_STEPS.md`: safest next hardening steps.
