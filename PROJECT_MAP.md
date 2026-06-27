# Project Map

Updated: 2026-06-27

Scope: current repository map after the roadmap implementation. The app is positioned as an educational investment research and decision-support tool, not a financial advisor, stock picker, trading bot, or real-money trading system.

## Current App Shape

`Investment-advisor-ai` now has two user-facing entry points:

- A reliable CLI that downloads data, prints assumptions/disclaimers, runs the rule model, trains a leakage-free ML pipeline, backtests predictions with costs, runs risk and Monte Carlo analysis, saves a model artifact, and writes JSON/Markdown reports.
- A Streamlit dashboard that exposes ticker analysis, model diagnostics, Monte Carlo results, saved-report downloads, and portfolio/watchlist analytics.

The project remains a local Python app. There is no backend API, database, migration system, auth layer, brokerage integration, or automated trading workflow.

## Entry Points

- `src/main.py`: CLI entry point. Run with `py src\main.py`.
- `streamlit_app.py`: Streamlit dashboard. Run with `py -m streamlit run streamlit_app.py`.
- `tests/`: pytest suite for indicators, labels, recommendations, data loading, backtesting, risk metrics, Monte Carlo, portfolio logic, reports, and research orchestration.
- `src/test_data_loader.py`: manual live-data smoke script retained for now.
- `example/*.py`: unrelated educational/course examples retained for owner review.

## Runtime Flow

1. Load config and print educational-use assumptions from `src/config.py`.
2. Download and clean market data with `src/data_loader.py`.
3. Apply explicit price policy: default `raw_close`; optional `adjusted_close`.
4. Run the multi-factor rule model through `src/model.py` and `src/advisor.py`.
5. Build technical and return/volatility feature rows in `src/ml_dataset.py`.
6. Create explicit BUY/HOLD/SELL labels in `src/ml_labels.py`.
7. Train and validate a leakage-free sklearn pipeline in `src/ml_pipeline.py`.
8. Predict the latest ML signal and calculate permutation importance.
9. Run signal backtesting with transaction costs/slippage in `src/backtesting.py`.
10. Calculate risk analytics in `src/risk_metrics.py`.
11. Run Monte Carlo scenarios in `src/monte_carlo.py`.
12. Persist model artifacts and reports through `src/ml_pipeline.py` and `src/reports.py`.
13. Streamlit uses `src/research.py` for structured analysis and `src/portfolio.py` for watchlist/portfolio analytics.

## Major Directories

### Root

- `README.md`: setup, run, verification, and positioning.
- `requirements.txt`: runtime dependencies.
- `requirements-dev.txt`: pytest and ruff.
- `pyproject.toml`: project metadata plus pytest/ruff config.
- `PROJECT_ROADMAP.md`: staged roadmap.
- `CODEBASE_AUDIT.md`: audit and implementation history.
- `NEXT_CLEANUP_STEPS.md`: next safe hardening steps.

### `src/`

- `config.py`: explicit price policy, HOLD label policy, costs/slippage assumptions, risk settings, and educational disclaimer.
- `data_loader.py`: `yfinance` download, MultiIndex column normalization, numeric cleanup, and price-policy application.
- `indicators.py`: technical indicator calculations.
- `advisor.py`: rule-based recommendation scoring.
- `model.py`: existing rule-model wrapper and legacy ML helpers kept for compatibility.
- `ml_dataset.py`: feature construction with a shared `FEATURE_COLUMNS` schema.
- `ml_labels.py`: leakage-aware label generation with date-aligned indexes.
- `ml_pipeline.py`: sklearn pipelines, walk-forward validation, prediction, feature importance, artifact persistence.
- `backtesting.py`: signal backtesting with explicit transaction costs and slippage.
- `risk_metrics.py`: volatility, Sharpe, Sortino, drawdown, VaR, and CVaR.
- `monte_carlo.py`: bootstrap and parametric scenario simulations.
- `portfolio.py`: watchlist returns, correlations, risk-parity weights, efficient frontier samples, and portfolio Monte Carlo.
- `research.py`: structured end-to-end analysis orchestration for the dashboard and future APIs.
- `reports.py`: JSON/Markdown report payload generation and saving.

### `tests/`

Active automated pytest coverage includes:

- recommendation scoring
- indicator basics
- data loading and price policy
- label generation and date alignment
- ML dataset schema
- ML validation guardrails
- leakage-free training pipeline behavior
- backtesting with costs
- risk metrics
- Monte Carlo simulation summaries
- portfolio/watchlist logic
- report serialization
- research orchestration validation

Latest local result: `52 passed`, with three expected sklearn warnings from tiny synthetic feature-selection fixtures.

### `example/`

Educational TensorFlow/penguin/bakechef examples are unrelated to the investment-research runtime. They remain in place because earlier audit risk was Medium/unclear due possible coursework/reference value.

### Generated Local Artifacts

- `artifacts/models/`: saved model artifacts from CLI runs.
- `reports/`: saved JSON and Markdown analysis reports from CLI runs.
- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `venv/`: generated/local environment artifacts.

## Remaining Risk Areas

- Live `yfinance` data can be delayed, revised, unavailable, or shaped differently across package versions.
- The rule-based neutral-score policy still returns SELL through existing recommendation behavior; changing that would alter outputs and needs explicit approval.
- Validation metrics are educational and can vary materially by ticker, date range, and market regime.
- The project does not yet have pinned dependency versions, CI, cached fixture data, or a backend API.
- The Streamlit dashboard is a local research interface, not a production web app.
- No database schema, auth, audit log, brokerage integration, or real-money order routing exists.

## Cleanup Decision Principles

- Keep the CLI working while adding features.
- Treat label logic, price policy, backtesting, risk metrics, and recommendation behavior as business-critical educational logic.
- Avoid implying guaranteed returns or investment advice.
- Do not remove Medium/High risk files without owner approval.
- Prefer small, tested changes over broad rewrites.
