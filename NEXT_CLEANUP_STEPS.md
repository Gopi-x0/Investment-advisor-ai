# Next Cleanup Steps

Updated: 2026-06-27

The roadmap implementation is now in place and verified locally. The next work should be production-hardening and polish, not broad rewrites.

## Completed Since The Original Cleanup Plan

- Added pytest and ruff tooling.
- Added tests for indicators, labels, recommendations, data loading, validation, backtesting, ML pipeline, risk metrics, Monte Carlo, portfolio logic, reports, and research orchestration.
- Made price policy explicit.
- Made HOLD label policy explicit.
- Fixed leakage by moving preprocessing and feature selection into sklearn pipelines.
- Added transaction costs and slippage to backtesting.
- Added model artifact persistence.
- Added baseline model choices and validation metrics.
- Added Monte Carlo scenario analysis.
- Added risk analytics.
- Added Streamlit dashboard.
- Added watchlist/portfolio support.
- Added explainability via permutation importance.
- Added saved JSON/Markdown reports.
- Added educational-use disclaimers and guardrails.
- Fixed live CLI label/date alignment so reports and artifacts are produced end to end.

## Recommended Stage 3 Refactors

1. Split CLI printing from orchestration.
   - Move the current CLI analysis flow fully onto `src/research.py`.
   - Keep `src/main.py` as input/output glue only.
   - Risk: Medium, because it touches the main user workflow.

2. Add deterministic fixture data.
   - Store a small static OHLCV fixture under `tests/fixtures/`.
   - Use it for CLI/research smoke tests without network access.
   - Risk: Low.

3. Add a real baseline-comparison module.
   - Compare ML against buy-and-hold, always-hold, and the rule model in one report section.
   - Risk: Medium, because definitions must be consistent with cost assumptions.

4. Add dashboard controls for config values.
   - Expose label policy, lookahead window, cost bps, slippage bps, Monte Carlo horizon, and simulation count.
   - Risk: Medium, because invalid combinations need guardrails.

5. Introduce lightweight package structure.
   - Consider installing the project in editable mode and using package imports.
   - Risk: Medium, because import paths and entry-point commands will change.

## Business-Critical Issues Needing Approval

- Rule-based neutral-score behavior in `src/advisor.py`.
  - Current behavior: score `0` returns SELL.
  - Likely educationally clearer behavior: score `0` returns HOLD.
  - Needs approval because recommendation outputs will change.

- Default price policy.
  - Current default: raw `Close`.
  - Alternative: adjusted close for split/dividend-aware historical research.
  - Needs approval because all historical returns, labels, and signals can change.

- Label-policy threshold design.
  - Current default: volatility-adjusted three-class labels.
  - Alternative: after-cost return thresholds or triple-barrier first-touch labels with HOLD.
  - Needs approval because training targets and metrics will change.

- Production data provider.
  - Current source: free `yfinance`.
  - Production-like alternatives need licensing, rate-limit handling, retries, and data-quality checks.

## Tests To Add Next

- CLI smoke test using a static fixture and monkeypatched data loader.
- Streamlit/report integration smoke test that exercises `build_research_report_from_data`.
- Baseline comparison tests for buy-and-hold, always-hold, rule-model, and ML strategies.
- Portfolio edge-case tests for missing dates, single ticker, zero variance, and all-cash/zero-weight inputs.
- Monte Carlo reproducibility tests with fixed seeds across all scenario types.
- Artifact compatibility test that loads a saved model and validates metadata.

## Production-Hardening Steps

- Pin dependency versions or add a lockfile.
- Add CI to run install, lint, tests, and compile checks.
- Add source metadata to downloaded datasets: ticker, provider, download timestamp, period, price policy.
- Add caching for market data with explicit freshness controls.
- Add structured logging instead of print-only diagnostics.
- Add model and report version fields.
- Add better error messages in Streamlit for too-short data, single-class labels, and failed downloads.
- Add dependency/security scanning.
- Add a small user guide explaining every metric and limitation in plain language.

## Questions For Human Decision

- Should `example/` remain in this repository as coursework/reference material?
- Should `src/test_data_loader.py` remain as a manual smoke script now that pytest coverage exists?
- Should a neutral rule score become HOLD?
- Should adjusted close become the default price policy?
- Should reports be stored only locally, or should a future SQLite database track saved runs?
- Should the next GUI iteration stay in Streamlit or move toward FastAPI plus a frontend?
