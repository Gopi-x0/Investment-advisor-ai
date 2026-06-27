# Codebase Audit

Audit date: 2026-06-27

Scope: full repository audit based on tracked files plus visible generated/local directories. The initial audit created `PROJECT_MAP.md` and `CODEBASE_AUDIT.md`. A follow-up safe-cleanup pass has now made small Stage 1 and Stage 2 changes documented below.

## Roadmap Implementation Update

Update date: 2026-06-27

### Completed Roadmap Items

- Reliable CLI kept working and now completes the full analysis path on live data.
- Automated tests added or expanded for indicators, labels, recommendations, data loading, backtesting, risk metrics, portfolio logic, reports, and research validation.
- Explicit price policy added in `src/config.py`; default is `raw_close`, with `adjusted_close` supported.
- Explicit HOLD label policy added; default is three-class BUY/HOLD/SELL.
- Leakage-free sklearn training pipeline added in `src/ml_pipeline.py`; scaling and feature selection happen inside the pipeline.
- Backtesting added with transaction costs and slippage in `src/backtesting.py`.
- Model artifact persistence added under `artifacts/models/`.
- Baseline model choices added: logistic regression, random forest, and gradient boosting.
- Monte Carlo module added in `src/monte_carlo.py`.
- Risk analytics added in `src/risk_metrics.py`.
- Streamlit dashboard added in `streamlit_app.py`.
- Portfolio/watchlist support added in `src/portfolio.py` and surfaced in the dashboard.
- Explainability added through permutation importance summaries.
- Saved JSON/Markdown analysis reports added through `src/reports.py`.
- Educational-use disclaimers, assumptions, limitations, and guardrail warnings added across CLI, dashboard, and reports.

### Files Changed Or Added

- `README.md`: updated setup, CLI/dashboard usage, verification commands, outputs, and educational positioning.
- `PROJECT_MAP.md`: refreshed current architecture, entry points, runtime flow, tests, and remaining risks.
- `PROJECT_ROADMAP.md`: marked all roadmap phases implemented and clarified remaining hardening work.
- `NEXT_CLEANUP_STEPS.md`: updated next safe refactors, approval-required decisions, tests, and hardening steps.
- `pyproject.toml`: project metadata plus pytest/ruff configuration.
- `requirements.txt`: added Streamlit to runtime dependencies.
- `requirements-dev.txt`: added developer test/lint dependencies.
- `streamlit_app.py`: new Streamlit dashboard.
- `src/config.py`: new assumptions, policy, and cost configuration module.
- `src/data_loader.py`: explicit price policy, yfinance MultiIndex normalization, safer cleanup behavior.
- `src/ml_dataset.py`: shared feature schema and richer feature construction.
- `src/ml_labels.py`: explicit label policy, date-aligned labels, after-cost return label helper, safer empty outputs.
- `src/ml_pipeline.py`: leakage-free training pipeline, walk-forward validation, prediction, feature importance, artifact persistence.
- `src/backtesting.py`: signal backtesting with cost/slippage assumptions.
- `src/risk_metrics.py`: risk metric calculations.
- `src/monte_carlo.py`: bootstrap and parametric scenarios.
- `src/portfolio.py`: portfolio/watchlist analytics.
- `src/research.py`: structured end-to-end research report orchestration.
- `src/reports.py`: report payload generation and JSON/Markdown persistence.
- `src/main.py`: CLI upgraded to assumptions, ML pipeline, backtesting, risk, Monte Carlo, artifact saving, and report saving.
- `src/model.py`: kept existing rule workflow and compatibility helpers, with safer metric handling.
- `tests/`: added/expanded focused pytest coverage across the meaningful modules.

### Issues Fixed

- Fixed the original ML leakage risk by using sklearn pipelines with fold-local preprocessing and feature selection.
- Fixed live CLI label/date misalignment where feature rows used dates but labels used row numbers, causing zero joined ML rows.
- Added explicit pricing, label, cost, and slippage assumptions instead of implicit behavior.
- Added guardrails for too-small datasets, missing columns, empty labels, and single-class labels.
- Added report and artifact persistence for reproducibility.
- Added Streamlit import-path setup so the dashboard runs from the repo root.
- Added yfinance MultiIndex-column normalization for newer provider output shapes.

### Issues Intentionally Skipped

- Rule-based score `0` still maps through existing recommendation behavior; changing it to HOLD would alter outputs and needs approval.
- Default price policy remains `raw_close`; switching to adjusted close would change historical returns/signals and needs approval.
- `example/` remains untouched because earlier audit risk was Medium/unclear due possible coursework/reference value.
- `src/test_data_loader.py` remains as a manual smoke script.
- No database, auth, backend API, broker integration, or automated real-money trading features were added.
- Dependencies are not pinned yet.

### New Or Remaining Risks

- Live `yfinance` data can fail, change shape, be delayed, or be revised.
- ML metrics are educational and can vary heavily across ticker, date range, and market regime.
- The Streamlit dashboard is useful for local analysis but is not production-deployed software.
- Tiny synthetic ML tests trigger expected sklearn feature-selection runtime warnings.
- Generated artifacts and reports are local files; there is no versioned run database.

### Verification Results

- `py -m pytest`: passed, `52 passed`, `3 warnings`.
- `py -m ruff check src tests streamlit_app.py`: passed.
- `py -m compileall src tests streamlit_app.py`: passed.
- `py -c "import streamlit_app; print('streamlit import ok')"`: passed; Streamlit printed expected bare-mode warnings because it was not launched with `streamlit run`.
- `echo AAPL | py src\main.py`: passed end to end against live data, producing ML validation, backtest metrics, risk metrics, Monte Carlo metrics, `artifacts\models\AAPL_latest.joblib`, `reports\AAPL_latest.json`, and `reports\AAPL_latest.md`.

## Cleanup Update

Cleanup date: 2026-06-27

### Cleanup Completed

- Stage 1 safe cleanup removed clearly unused imports from `src/main.py` and `src/ml_labels.py`.
- Stage 2 safe quality improvements added small validation and consistency improvements without changing recommendation scoring, label semantics, model choice, or data source policy.
- No files or directories were deleted.
- No dependencies were removed.
- No database/schema work was performed because the project has no database layer.

### Files Changed

- `src/main.py`
  - Removed unused imports: `from pyexpat import features`, unused label strategy imports, and unused pandas import.
  - Imports `FEATURE_COLUMNS` from `src/ml_dataset.py` instead of duplicating the feature column list.
  - Added `ML_N_SPLITS` constant for the existing walk-forward split count.
  - Added `validate_ml_inputs` to fail early with clear messages when labeled ML data is empty, missing columns, too small for walk-forward validation, or has fewer than two label classes.
- `src/ml_dataset.py`
  - Added shared `FEATURE_COLUMNS` constant.
  - Returns an empty DataFrame with the expected feature schema when no feature rows can be built, instead of failing later with a missing `date` index.
- `src/ml_labels.py`
  - Removed unused numpy import.
  - Added empty-label guards to triple-barrier label functions to avoid divide-by-zero errors in summary printing.
- `src/model.py`
  - Added `zero_division=0` to walk-forward weighted F1 calculation to avoid warnings/unstable metric behavior on edge-case class distributions.
- `src/data_loader.py`
  - Replaced an informal invalid-ticker message with production-appropriate wording.

### Issues Fixed

- Removed low-risk unused imports.
- Reduced duplicated feature schema between `main.py` and `ml_dataset.py`.
- Added clearer ML setup validation before sklearn training.
- Prevented no-label TBL summaries from dividing by zero.
- Made walk-forward F1 metric handling consistent with `evaluate_model`.
- Improved invalid ticker UX without changing data-fetch behavior.

### Issues Skipped

- `example/` was not deleted because the audit marked it Medium/unclear due possible coursework/reference value.
- `src/test_data_loader.py` was not removed because it should be replaced by real tests first.
- `src/model.py::train_model` and `evaluate_model` were not removed because they support the commented holdout-evaluation workflow.
- Rule-based neutral-score behavior was not changed because it changes recommendation output.
- Feature-selection leakage in walk-forward validation was not changed because it affects reported ML metrics and needs tests/approval.
- Label semantics were not changed because they define training truth data.
- Adjusted-vs-raw price policy was not changed because it affects all financial calculations.

### New Risks Discovered

- No new production risks were introduced by the cleanup pass.
- The Python execution environment remains a verification blocker in this sandbox: `python` and `py` are unavailable, and `venv/pyvenv.cfg` points at a missing base interpreter.

### Verification Results

- `git -c safe.directory='C:/Users/dhawa/OneDrive/Desktop/UNI/CS Projects/Investment-advisor-ai' status --short` ran successfully.
- `git -c safe.directory='C:/Users/dhawa/OneDrive/Desktop/UNI/CS Projects/Investment-advisor-ai' diff --check` passed with exit code 0. Git reported line-ending normalization warnings for touched source files.
- Source-only `rg` checks confirmed removed import strings and informal ticker wording are no longer present in active source.
- `python --version` failed because `python` is not available on PATH in this sandbox.
- `.\venv\Scripts\python.exe --version` failed because the checked-in local virtual environment points at a missing base interpreter.
- Runtime Python validation, tests, and CLI smoke checks could not be run in this sandbox because no working Python interpreter is available.

## Testing And Roadmap Update

Update date: 2026-06-27

### Files Changed

- `pyproject.toml`
  - Added initial project metadata and pytest/ruff configuration.
- `requirements-dev.txt`
  - Added pytest and ruff as development dependencies.
- `tests/test_advisor.py`
  - Added tests for current BUY, SELL, and zero-score behavior.
- `tests/test_data_loader.py`
  - Added mocked yfinance tests for data cleanup, empty downloads, blank tickers, and download failures.
- `tests/test_indicators.py`
  - Added baseline tests for moving averages, momentum, volume ratio, and price position.
- `tests/test_main_validation.py`
  - Added tests for ML dataset validation guardrails.
- `tests/test_ml_dataset.py`
  - Added tests for feature schema and insufficient-data behavior.
- `tests/test_ml_labels.py`
  - Added tests for normal labels and empty triple-barrier outputs.
- `README.md`
  - Added setup, run, and verification commands.
- `PROJECT_MAP.md`
  - Updated to reflect test/tooling additions.
- `NEXT_CLEANUP_STEPS.md`
  - Updated progress against the previous test/tooling checklist.
- `PROJECT_ROADMAP.md`
  - Added a staged roadmap for Monte Carlo, GUI, stronger ML, portfolio analysis, and production hardening.

### Issues Fixed

- Added a real automated-test foundation where the repo previously only had a manual smoke script.
- Added dev tooling config without forcing a broad style rewrite.
- Removed remaining unused active imports from `src/main.py` for `train_model` and `evaluate_model`.
- Documented a realistic product direction that emphasizes risk analytics, honest uncertainty, and backtesting over "AI stock picker" claims.

### Issues Skipped

- No rule-based recommendation behavior was changed.
- No ML label policy was changed.
- No feature-selection leakage fix was implemented yet.
- No Monte Carlo or GUI code was added yet; those are planned after tests can run.
- No paid data provider, brokerage integration, or real-money trading workflow was added.

### Verification Status

- Tests were written but could not be executed in this sandbox until the Python environment is restored.
- The next required verification commands are `python -m pytest` and `python -m ruff check src tests`.

## 1. Executive Summary

### What The App Appears To Do

`Investment-advisor-ai` is a beginner Python CLI project that downloads historical stock data, calculates technical indicators, produces a rule-based BUY/SELL recommendation, builds an ML feature dataset, labels historical rows, trains a scikit-learn model during the CLI run, and predicts the latest trading signal.

It should be treated as educational software, not production financial advice.

### Main Architecture

- CLI orchestration: `src/main.py`
- Market data access: `src/data_loader.py`
- Technical indicators: `src/indicators.py`
- Rule-based recommendation scoring: `src/advisor.py` and `src/model.py::run_ruleBasedModel`
- ML feature generation: `src/ml_dataset.py`
- ML label generation: `src/ml_labels.py`
- ML training/evaluation/prediction helpers: `src/model.py`
- Design notes: `src/plan/*.md`
- Unrelated examples: `example/*.py`, `example/penguins.csv`

There is no frontend, backend API, database, migration system, persistent model artifact, package metadata, CI/CD, lint config, or formal automated test suite.

### Main Technologies

- Python
- pandas
- numpy
- scikit-learn
- yfinance
- TensorFlow and matplotlib appear only in unrelated example scripts and are not declared in `requirements.txt`.

### Main Risks

- Rule-based scoring cannot return HOLD for a neutral score; `src/advisor.py:66-67` forces neutral and negative scores to SELL.
- ML validation has data leakage: `src/model.py:149-151` selects top features on the full dataset before time-series validation.
- Label generation strategies are inconsistent and some rows may be mislabeled, especially `src/ml_labels.py:82-89`.
- Current main flow trains a model on every run, with no persistence, no reproducibility metadata, no tests, and no safeguards for too-small or single-class datasets.
- Market data comes from live `yfinance` calls without retries, caching, source snapshots, or explicit adjusted/raw price policy.
- No automated tests cover business-critical indicator, label, or recommendation behavior.

### Biggest Cleanup Opportunities

- Separate active investment-advisor code from unrelated `example/` course/demo code.
- Move `src/test_data_loader.py` into real tests or replace it with pytest coverage.
- Remove unused imports and stale commented training code after test coverage exists.
- Recreate local virtual environment outside source control assumptions; the current `venv` is ignored and broken in this sandbox.
- Update or archive stale docs in `src/plan/`.

### Biggest Code Quality Opportunities

- Split `src/main.py` into CLI, orchestration, configuration, and reporting layers.
- Make label generation deterministic, tested, and aligned with documented behavior.
- Remove ML evaluation leakage by performing feature selection inside each training fold.
- Centralize constants and assumptions such as lookahead days, thresholds, risk-free rate, model choice, and feature columns.
- Add validation before model training and prediction.
- Convert print-heavy helpers into pure functions returning structured results where practical.

### Ratings

- Overall code quality: 4/10
- Production readiness: 2/10

The core educational idea is coherent and modular enough to improve, but the current project lacks tests, packaging, reproducibility, production error handling, and safe ML validation.

## 2. Repository Map

### Root Directory

- Purpose: project metadata, license, dependency list, ignore rules.
- Status: active but minimal.
- Key files:
  - `README.md`: one-line project description.
  - `requirements.txt`: runtime dependencies.
  - `.gitignore`: broad Python ignore template.
  - `LICENSE`: MIT-style license.
- What depends on it: installation and developer setup depend on `requirements.txt`; repo hygiene depends on `.gitignore`.
- Recommendation: keep, then expand README/setup docs and add package/test/lint metadata.

### `src/`

- Purpose: active investment advisor implementation.
- Status: active.
- Key files:
  - `main.py`: CLI entry point and app orchestration.
  - `data_loader.py`: downloads and cleans stock data.
  - `indicators.py`: calculates moving averages, momentum, volatility, Sharpe ratio, RSI, MACD, Bollinger position, ATR, volume ratio, and price position.
  - `advisor.py`: rule-based recommendation scoring.
  - `ml_dataset.py`: builds one row of features per historical date.
  - `ml_labels.py`: creates supervised learning labels.
  - `model.py`: builds/trains/evaluates classifiers and predicts latest signal.
  - `test_data_loader.py`: manual live-data smoke script.
- What depends on it: `src/main.py` imports nearly all active modules.
- Recommendation: keep. Refactor only after tests exist around current behavior.

### `src/plan/`

- Purpose: ML design notes.
- Status: partially active, partially stale.
- Key files:
  - `ml_plan.md`
  - `ml_labels.md`
- What depends on it: no runtime dependency.
- Evidence of staleness: docs describe Logistic Regression, `train_model.py`, `ml_model.py`, saved `model.pkl`, 5-day/2-percent labels; current code uses RandomForest/GradientBoosting in `model.py`, current `main.py` uses 20-day volatility-adjusted triple-barrier labels, and no model is persisted.
- Recommendation: keep for now; update once intended architecture is approved.

### `example/`

- Purpose: educational TensorFlow examples unrelated to the current app runtime.
- Status: possibly dead relative to the investment-advisor app; may be classwork/reference.
- Key files:
  - `penguins.py`
  - `Penguins_Overfitting.py`
  - `bakechef_net_starter.py`
  - `penguins.csv`
- What depends on it: no active `src` code references it.
- Recommendation: review with owner. If this repo should be production-focused, archive or remove later. If examples are intentional, document them and fix paths/dependencies.

### `tests/`

- Purpose: expected test directory.
- Status: empty and untracked by Git.
- Key files: none.
- What depends on it: no current tooling.
- Recommendation: keep as test root and add pytest tests.

### `.vscode/`

- Purpose: local/editor settings.
- Status: active editor config.
- Key file: `settings.json`.
- What depends on it: VS Code behavior only.
- Recommendation: keep if repository-level editor config is desired.

### `venv/`

- Purpose: local virtual environment.
- Status: generated local artifact, ignored by `.gitignore`, not tracked.
- Key evidence: `git ls-files` does not include it; `venv/pyvenv.cfg` points to a missing base interpreter in this sandbox.
- What depends on it: local development only.
- Recommendation: do not treat as source. Recreate locally as needed.

### `src/__pycache__/`

- Purpose: generated Python bytecode.
- Status: generated local artifact, ignored, not tracked.
- Recommendation: safe to clean later; not part of source.

## 3. Dead Code / Dead Directory Analysis

### `example/bakechef_net_starter.py`

- Why it appears unused: no imports or runtime references from `src`; it is a tutorial template, not investment-advisor logic.
- Evidence: `rg` found no references from active modules; file contains `quit()` in `getDataFromSample`; expects missing `bakechefData.csv`; imports TensorFlow/matplotlib not declared in requirements.
- Risk if removed: Low for current app, Medium if the repo intentionally stores coursework.
- Recommended action: Keep for now pending owner approval; likely archive/delete later.
- Verification before removal: confirm no assignment/reference requirement; run `rg "bakechef|bakechef_net_starter"` after removal.

### `example/penguins.py`

- Why it appears unused: no active app references; it hardcodes an external path instead of using repo-local `example/penguins.csv`.
- Evidence: path points to `C:/Users/dhawa/OneDrive/Desktop/UNI/CPSC 383/penguins/penguins.csv`; TensorFlow/matplotlib are absent from `requirements.txt`.
- Risk if removed: Low for current app, Medium if retained for coursework.
- Recommended action: Keep for now; archive/delete later if repo focus is investment advisor only.
- Verification before removal: confirm with owner; run `rg "penguins.py|penguins.csv|CPSC 383"`.

### `example/Penguins_Overfitting.py`

- Why it appears unused: no active app references and overlaps with `example/penguins.py`.
- Evidence: same TensorFlow/penguin domain and hardcoded external dataset path; not imported by `src`.
- Risk if removed: Low for current app, Medium if retained as experiment history.
- Recommended action: Merge/archive/delete after owner approval.
- Verification before removal: confirm whether it documents overfitting experiment results.

### `example/penguins.csv`

- Why it appears unused: local CSV exists but penguin scripts currently use an absolute external path.
- Evidence: scripts reference `C:/Users/dhawa/.../penguins.csv`, not `example/penguins.csv`.
- Risk if removed: Medium because it may be the intended local data copy.
- Recommended action: Keep for now. If examples remain, update scripts to use local relative path.
- Verification before removal: search all branches/docs if available; confirm examples are not needed.

### `src/test_data_loader.py`

- Why it appears unused: not under `tests`, not a pytest test, no assertions, not imported by app.
- Evidence: standalone `main()` with live `get_stock_data("AAPL", "1y")`.
- Risk if removed: Low after replacing with pytest coverage.
- Recommended action: Refactor into `tests/test_data_loader.py` with mocked yfinance response, or keep as documented manual smoke script.
- Verification before removal: add automated tests for ticker validation, empty data, yfinance exception handling, and successful cleanup.

### `src/ml_labels.py::create_labels`, `create_normal_labels`, `create_TBL`

- Why it appears unused: imported in `src/main.py:4` but current main calls only `create_TBL_volatility_adjustment` at `src/main.py:57`.
- Evidence: no active call sites except imports and comments.
- Risk if removed: Medium because they are plausible experimental label strategies and documented in `src/plan`.
- Recommended action: Keep for now; either expose as configured strategies or move unused variants to tests/docs.
- Verification before removal: decide intended label strategy; add tests for remaining strategy.

### `src/model.py::train_model` and `evaluate_model`

- Why it appears unused: imported in `src/main.py` but active calls are commented at `src/main.py:75-79`; current flow uses `train_model_walk_forward`.
- Evidence: no active runtime call.
- Risk if removed: Medium because they support simpler train/test experimentation.
- Recommended action: Keep for now; either wire behind a mode/config or move to experimentation module.
- Verification before removal: confirm workflow no longer needs simple holdout evaluation.

### `venv/`

- Why it appears unused as source: generated environment, ignored, not tracked.
- Evidence: `.gitignore` ignores `venv/`; `git ls-files` excludes it.
- Risk if removed: Low for repository, Medium for local convenience.
- Recommended action: Do not commit; recreate when needed.
- Verification before removal: ensure dependencies can be installed from `requirements.txt`.

### `src/__pycache__/`

- Why it appears unused as source: generated Python bytecode cache.
- Evidence: ignored by `.gitignore`; not tracked.
- Risk if removed: Low.
- Recommended action: Delete during safe cleanup if desired.
- Verification before removal: none beyond confirming it is ignored/untracked.

## 4. Code Quality / Strength Audit

### Maintainability

#### `src/main.py`

- Problem: the CLI entry point handles user input, data download, rule-based display, feature engineering, label generation, model training, evaluation printing, prediction, and feature-importance printing.
- Why it makes code weaker: every behavior change risks the whole flow; tests must drive input/print/network/model training together.
- Better approach: keep `main.py` thin; move orchestration into a service function such as `run_analysis(ticker, config)` that returns structured results; keep display formatting separate.
- Risk of changing it: Medium because this is the primary app entry point.

#### `src/model.py`

- Problem: model factory, feature selection, scaling, validation, final training, evaluation, rule-model wrapper, and print logging live together.
- Why it makes code weaker: ML evaluation behavior and CLI output are coupled; feature-selection leakage is easier to miss.
- Better approach: separate `model_factory`, `training`, `evaluation`, and `prediction` concerns; make functions return data and let CLI print.
- Risk of changing it: Medium.

#### `src/ml_dataset.py` and `src/main.py`

- Problem: feature column schema is duplicated. `build_feature_dataset` defines emitted columns, while `main.py:60-66` hardcodes the expected list.
- Why it makes code weaker: adding/removing a feature can produce runtime `KeyError` or silently omit intended features.
- Better approach: centralize feature column names in one constant or return `(features_df, feature_columns)`.
- Risk of changing it: Low to Medium.

#### `src/indicators.py` and `src/ml_labels.py`

- Problem: `normalize_series` is duplicated in both files.
- Why it makes code weaker: fixes for yfinance/pandas column shape issues must be made twice.
- Better approach: move data shape normalization to a shared utility once tests exist.
- Risk of changing it: Low.

#### Naming and style

- Problem: names mix camelCase, snake_case, misspellings, and inconsistent capitalization: `movingAvg_Recommandation`, `trend_strength_Recommendation`, `create_TBL`.
- Why it makes code weaker: harder searchability and onboarding; typo becomes API surface.
- Better approach: standardize to snake_case while keeping compatibility wrappers temporarily if needed.
- Risk of changing it: Medium because current imports/calls depend on names.

### Architecture Quality

#### UI and business logic are mixed

- Current structure: `src/main.py` prints decisions and also chooses label strategy, feature columns, train/validation strategy, and model parameters.
- Why it is fragile: user-facing presentation changes are coupled to ML and recommendation behavior.
- Recommended structure: CLI layer calls an application service that returns rule result, ML result, diagnostics, and warnings.
- Fix now or later: Later, after adding tests.

#### Data access and data policy are thin

- Current structure: `src/data_loader.py` downloads raw yfinance data and drops missing rows.
- Why it is fragile: financial results depend on raw vs adjusted prices, missing-data rules, and source freshness, but those assumptions are not explicit.
- Recommended structure: create a data service that validates ticker/period, records source metadata, normalizes columns, and exposes a documented price policy.
- Fix now or later: Start soon with tests; deeper policy work later.

#### Validation is spread out and incomplete

- Current structure: some functions check empty data and Close column; training flow assumes non-empty joined data, enough samples, enough labels, and enough classes.
- Why it is fragile: realistic ticker histories can fail deep inside sklearn with unclear errors.
- Recommended structure: central preflight validation before feature generation and model training.
- Fix now or later: Fix soon.

#### ML experiment logic is in runtime path

- Current structure: main trains and evaluates every time the user asks for a ticker.
- Why it is fragile: slow, non-reproducible, and not production-like.
- Recommended structure: separate offline training/backtesting from online inference; persist model artifacts with metadata.
- Fix now or later: Later for production hardening; tests first.

### Business Logic Quality

#### P1: Neutral rule-based scores become SELL

- Path: `src/advisor.py:66-67`
- Business risk: neutral evidence is reported as SELL, which can create misleading recommendations.
- Example failure scenario: moving average says BUY and momentum says SELL with equal weighted score; user sees SELL instead of HOLD/neutral.
- Suggested safer design: return HOLD when `total_score == 0`, BUY when positive, SELL when negative; add tests for positive, negative, and zero scores.
- Priority: P1.

#### P1: Feature selection leaks future/test information into validation

- Path: `src/model.py:149-151` calling `select_top_features(X, y)` before `TimeSeriesSplit`.
- Business risk: reported validation metrics can be overly optimistic, making the model look safer than it is.
- Example failure scenario: a feature selected because it works on future folds is used to evaluate earlier folds, inflating mean accuracy/F1.
- Suggested safer design: perform feature selection inside each fold using only `X_train, y_train`; for final model, fit feature selection on all training data after validation.
- Priority: P1.

#### P1: `create_labels` keeps rows that do not have future returns

- Path: `src/ml_labels.py:79-89`
- Business risk: last `lookahead_days` rows can be labeled HOLD even though no future outcome exists; early rows with missing volatility threshold also default to HOLD.
- Example failure scenario: model learns HOLD from rows that were not actually evaluated, weakening or biasing training.
- Suggested safer design: initialize labels as missing, assign only rows with valid future return and valid threshold, then drop missing rows.
- Priority: P1.

#### P1: Current main label strategy is binary despite prediction code supporting HOLD

- Path: `src/main.py:57`, `src/ml_labels.py:174-184`, `src/model.py:218-249`
- Business risk: current `create_TBL_volatility_adjustment` drops no-touch outcomes, so the trained model usually sees only BUY and SELL, while `predict_latest_signal` advertises HOLD support.
- Example failure scenario: model rarely or never predicts HOLD even when uncertainty is high.
- Suggested safer design: decide whether HOLD is a real class; if yes, encode no-touch rows as 0 and include them intentionally; if no, remove HOLD from current ML messaging.
- Priority: P1.

#### P2: Raw price policy is not explicit

- Path: `src/data_loader.py:21`, `src/ml_labels.py`, `src/indicators.py`
- Business risk: calculations based on raw `Close` can be distorted by splits/dividends compared with adjusted prices.
- Example failure scenario: a historical split creates a false huge move in labels/indicators.
- Suggested safer design: define whether indicators and returns use `Adj Close` or auto-adjusted data; add tests with split-like synthetic data.
- Priority: P2.

#### P2: Thresholds and assumptions are hardcoded

- Path: `src/advisor.py`, `src/ml_labels.py`, `src/model.py`, `src/main.py`
- Business risk: changing strategy assumptions requires code edits and can create inconsistent behavior.
- Example failure scenario: plan says 5-day 2-percent labels, main uses 20-day volatility-adjusted labels.
- Suggested safer design: central config object for periods, thresholds, model type, lookahead, and feature count.
- Priority: P2.

#### P2: No transaction costs, slippage, or backtesting guardrails

- Path: current repo-wide design gap.
- Business risk: recommendation quality is judged by classification metrics but not practical trading outcomes.
- Example failure scenario: model predicts frequent profitable-looking flips that would fail after costs.
- Suggested safer design: add offline backtest metrics before presenting performance as useful.
- Priority: P2.

### Reliability and Error Handling

- `src/data_loader.py:21-23`: wraps all yfinance errors as connection errors; no retry, timeout policy, or rate-limit handling.
- `src/data_loader.py:26`: user-facing error text is unprofessional for production.
- `src/data_loader.py:28-36`: drops all rows with any missing values without reporting how many rows were lost.
- `src/main.py:26-33`: broad catch prints only `Error:` and returns; no structured logging or recovery.
- `src/main.py:59-68`: no validation that join produced non-empty dataset or all feature columns exist.
- `src/main.py:82`: no guard for too few samples for `TimeSeriesSplit(n_splits=5)` or single-class labels.
- `src/ml_dataset.py:26-89`: catches all feature errors, prints warning, and skips rows; this can hide data quality problems.
- `src/ml_labels.py:131` and `src/ml_labels.py:188`: divide by `len(labels_series)` without checking for zero labels.
- `src/model.py:169`: `f1_score` does not specify `zero_division`, so unusual class distributions can emit warnings or unstable behavior.

### Data Integrity

There is no database or persistent application state. Data integrity here means correctness, traceability, and reproducibility of market data, features, labels, and model outputs.

- Market data is fetched live each run and not snapshotted, so the same ticker can produce different results over time.
- There is no source timestamp, yfinance version record, dependency version pinning, or model metadata.
- Generated labels are not audited or saved for inspection.
- Feature rows use `data.index[i]` in `src/ml_dataset.py:60`; after `data_loader` resets the index, this is a row number, not necessarily the actual market date.
- Deletes/updates are not relevant because there is no database.
- Financial auditability is currently low: there is no record of why a recommendation was made beyond console output.

### Security / Safety

- No secrets were found in source.
- No auth/role model is present because this is a local CLI, not a server.
- Ticker input is not used in shell/database contexts, so command injection risk is low.
- Dependency supply-chain risk is meaningful because `requirements.txt` is unpinned.
- Live network dependency on yfinance means availability and upstream behavior affect outputs.
- The CLI does not prominently display an educational-use/not-financial-advice warning.
- Example files include absolute local paths that reveal user machine structure; not secret, but not production-clean.

### Performance

- `src/ml_dataset.py` loops through all rows and repeatedly slices historical data. This is acceptable for 5 years of daily data but could become slow for longer periods or many tickers.
- `src/main.py` trains a model on every run, which is slow and wasteful for interactive use.
- `src/model.py` uses `n_jobs=-1` for RandomForest, which can consume all CPU on a developer machine.
- No caching exists for yfinance downloads or generated features.
- No large-payload concerns exist because there is no API/server.

### UI/UX Code Quality

- UI is a CLI, not a graphical frontend.
- Output is verbose and includes debug `dataset.head()`/`dataset.tail()` at `src/main.py:70-73`.
- Errors are plain text and not consistently actionable.
- The CLI asks for a ticker but does not show disclaimers, confidence, data date, or whether recommendation is educational only.
- Progress output from model/helper functions is mixed with business logic, making clean UI changes harder.

### Testability

- Important logic is hard to test because `main.py` mixes input, network, model training, and printing.
- Best unit-test candidates:
  - `src/advisor.py` recommendation scoring.
  - `src/indicators.py` indicator functions on synthetic data.
  - `src/ml_labels.py` edge cases: no future data, no barrier touched, zero labels, volatility NaNs.
  - `src/ml_dataset.py` feature column contract and date alignment.
  - `src/data_loader.py` with mocked `yf.download`.
- Integration-test candidates:
  - `run_analysis` flow on a fixed local fixture after refactor.
  - rule-based vs ML output shape on deterministic data.
- Regression tests needed:
  - neutral score returns HOLD if that behavior is approved.
  - no label leakage for future-unavailable rows.
  - feature selection in validation uses training data only.

## 5. Inconsistency Analysis

### Naming

- Where: `movingAvg_Recommandation`, `momentum_Recommendation`, `create_TBL`, `run_ruleBasedModel`, snake_case functions elsewhere.
- Problem: inconsistent API style and misspellings make code harder to search and maintain.
- Recommended standard: snake_case function names and correctly spelled `recommendation`.
- Safe to fix now: not until tests exist; use wrappers if renamed.

### Documentation vs implementation

- Where: `src/plan/ml_plan.md`, `src/plan/ml_labels.md`, `src/main.py`, `src/model.py`, `src/ml_labels.py`.
- Problem: docs describe Logistic Regression, saved models, `train_model.py`, `ml_model.py`, and 5-day labels; implementation uses RandomForest/GradientBoosting, no saved model, and 20-day TBL labels.
- Recommended standard: docs must state current architecture or be marked as historical plan.
- Safe to fix now: yes, documentation-only update is low risk after approval.

### Error handling

- Where: `data_loader.py`, `main.py`, `ml_dataset.py`, `ml_labels.py`.
- Problem: broad catches, print-only warnings, and deep sklearn failures are mixed.
- Recommended standard: validate early, raise domain-specific errors, print user-facing messages only in CLI.
- Safe to fix now: partially; add tests first for behavior-sensitive paths.

### Validation

- Where: indicator functions check only basic data shape; model training assumes dataset health.
- Problem: insufficient checks for minimum rows, valid columns, nonzero denominators, non-empty labels, and class count.
- Recommended standard: central validation before each pipeline stage.
- Safe to fix now: yes for adding non-invasive validations, but output-changing behavior should be tested.

### API/return shapes

- Where: `advisor.overall_Recommendation` returns tuple; `model.run_ruleBasedModel` returns dict; training functions return tuples; evaluation returns dict.
- Problem: callers must know positional tuple structure.
- Recommended standard: use dataclasses or named dictionaries for cross-module contracts.
- Safe to fix now: later; medium risk.

### Business logic placement

- Where: `main.py` chooses label strategy and hardcoded feature list.
- Problem: strategy and feature policy are in presentation/orchestration code.
- Recommended standard: central config and application service.
- Safe to fix now: later.

### Styling/UI patterns

- Where: CLI print statements across `main.py`, `model.py`, `ml_dataset.py`, `ml_labels.py`.
- Problem: helper functions have presentation side effects.
- Recommended standard: helpers return results; CLI prints.
- Safe to fix now: later.

## 6. Dependency Audit

### Declared Dependencies

`requirements.txt` contains:

- `pandas`: used in active source.
- `numpy`: used in active source.
- `scikit-learn`: used in active source.
- `yfinance`: used in active source.

### Issues

- Versions are unpinned, so installs are not reproducible.
- No Python version is declared.
- No dev dependencies are declared for tests/lint/type checking.
- TensorFlow and matplotlib are imported by `example/*.py` but not declared. This is fine if examples are out of scope, but broken if examples are expected to run.
- The local `venv` is present but not usable in this sandbox because it points to a missing base Python executable. `python` and `py` were also unavailable on PATH.
- Could not run `pip freeze` or `pip check` due the broken interpreter.

### Possible Missing Dev Dependencies

- `pytest`
- `ruff`
- `mypy` or `pyright` if type checking is desired
- `pre-commit` if formatting/lint checks should be standardized

### Unused Or Unclear Dependencies

- No clearly unused active runtime dependency among the four declared packages.
- TensorFlow/matplotlib are undeclared example-only dependencies, not active app dependencies.

## 7. Test / Build / Lint Audit

### What Exists

- `requirements.txt`
- Manual smoke script: `src/test_data_loader.py`
- Empty `tests/` directory
- `.gitignore`
- VS Code setting file

### What Is Missing

- Automated tests
- pytest config
- linting config
- formatting config
- type checking config
- CI/CD
- package metadata (`pyproject.toml`, `setup.py`, or similar)
- model training/inference scripts as separate commands
- seed/fixture data for deterministic tests
- build/deploy files
- error monitoring/logging
- database migrations and seed files are not applicable because there is no database.

### What Should Be Added First

1. Minimal `pyproject.toml` with Python version, pytest config, and ruff config.
2. Unit tests for `advisor.py`, `ml_labels.py`, and `indicators.py`.
3. Mocked tests for `data_loader.py`.
4. A deterministic fixture dataset for integration tests.
5. CI that runs install, lint, and tests.

### Verification Commands To Run

Current exact commands are listed in section 11. In this sandbox, runtime commands could not be executed because no working Python interpreter was available.

## 8. Quality Ratings

- Maintainability: 5/10. Small files and simple modules help, but orchestration, printing, ML behavior, and configuration are too coupled.
- Architecture: 4/10. The app has rough module boundaries, but no application service, package structure, offline/online ML split, or config layer.
- Business logic safety: 3/10. HOLD handling, label consistency, feature-selection leakage, and undocumented financial assumptions are significant.
- Data integrity: 4/10. No database risk, but no reproducible market data snapshots, label audit trail, or model metadata.
- Error handling: 3/10. Broad catches, print-only warnings, and missing preflight checks can produce confusing failures.
- Security posture: 6/10. Local CLI with no secrets or auth surface, but unpinned dependencies and live network reliance remain risks.
- Testability: 4/10. Pure functions exist, but main flow and ML helpers are print/network/training heavy and there are no tests.
- Performance: 6/10. Fine for one ticker and 5 years of daily data, but retraining every run and repeated feature slicing will not scale well.
- UI code quality: 4/10. CLI works as a learning interface, but output is noisy and not separated from logic.
- Overall code strength: 4/10. Good educational foundation, not yet strong enough for production-like finance use.

## 9. Cleanup and Improvement Plan

### Stage 1: Safe Cleanup

- Files likely affected:
  - `src/main.py`
  - `src/ml_labels.py`
  - `example/*.py`
  - local generated `src/__pycache__/`
  - local generated `venv/`
- Risk level: Low, except example removal is Medium until owner confirms.
- Intended outcome:
  - Remove unused imports such as `from pyexpat import features`, unused pandas import in `main.py`, unused label imports, unused `np` in `ml_labels.py`.
  - Remove generated caches from local workspace if desired.
  - Decide whether `example/` belongs in this repo.
  - Keep behavior unchanged.
- Verification:
  - `git status --short`
  - `rg -n "from pyexpat|import pandas as pd|create_normal_labels|create_TBL\\b" src/main.py src/ml_labels.py`
  - Run existing CLI after a working Python environment is restored.

### Stage 2: Safe Quality Improvements

- Files likely affected:
  - `src/advisor.py`
  - `src/indicators.py`
  - `src/ml_labels.py`
  - `src/ml_dataset.py`
  - `src/data_loader.py`
  - `tests/`
- Risk level: Low to Medium.
- Intended outcome:
  - Add pytest tests for current behavior before changing outputs.
  - Add validation for empty joined dataset, too few samples, and single-class labels.
  - Standardize user-facing errors in CLI.
  - Add safer zero-length checks in TBL label functions.
  - Centralize feature column list.
- Verification:
  - `python -m pytest`
  - `python src/main.py` with a known ticker
  - test fixture confirms generated feature columns and labels.

### Stage 3: Medium-Risk Refactors

- Files likely affected:
  - `src/main.py`
  - `src/model.py`
  - `src/ml_dataset.py`
  - `src/ml_labels.py`
  - `src/advisor.py`
- Risk level: Medium.
- Intended outcome:
  - Split CLI from pipeline orchestration.
  - Move strategy/config constants into one configuration module.
  - Fix neutral rule score behavior if approved.
  - Fix feature-selection leakage inside walk-forward validation.
  - Decide whether ML uses binary BUY/SELL or three-class BUY/HOLD/SELL labels.
  - Make training/evaluation functions return structured results without printing.
- Verification:
  - Regression tests for old/new output expectations.
  - Walk-forward tests prove feature selector sees only train folds.
  - Snapshot tests for CLI output if keeping CLI.

### Stage 4: Production Hardening

- Files likely affected:
  - New package/config files
  - `src/data_loader.py`
  - `src/model.py`
  - `src/main.py`
  - `tests/`
  - CI workflow files
- Risk level: Medium to High.
- Intended outcome:
  - Add `pyproject.toml`, CI, lint, formatting, and dependency pinning.
  - Separate offline training/backtesting from online inference.
  - Persist model artifacts with feature schema, label policy, training date, package versions, and evaluation metrics.
  - Add data source metadata and adjusted/raw price policy.
  - Add clear educational/not-financial-advice messaging.
  - Add logging and structured errors.
- Verification:
  - CI passes install/lint/test.
  - Model artifact can be trained and loaded reproducibly.
  - Backtest report uses fixed fixture or snapshotted market data.

## 10. Do-Not-Touch List

Do not edit these areas until tests and intended behavior are approved:

- `src/advisor.py`: recommendation scoring is business-critical and currently has output-changing ambiguity around HOLD.
- `src/indicators.py`: technical indicator math feeds both rule and ML models.
- `src/ml_labels.py`: label logic determines model truth data and has major business/ML impact.
- `src/model.py`: validation/training logic affects reported model quality.
- `src/ml_dataset.py`: feature schema affects trained model inputs.
- `src/data_loader.py`: data source and price adjustment policy affect all downstream results.
- `src/main.py`: primary app entry point; refactors can break the only user workflow.
- `requirements.txt`: changing versions can change yfinance/pandas/sklearn behavior.
- `example/`: do not delete until user confirms it is not needed for coursework/reference.

## 11. Verification Commands

Use these after cleanup or refactor work. Commands assume a working Python installation is available.

### Environment

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

### Current App Smoke Checks

```powershell
python src\main.py
python src\test_data_loader.py
```

`src\test_data_loader.py` requires network access to yfinance and is a manual smoke check, not an automated test.

### Recommended After Adding Tooling

```powershell
python -m pytest
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src
```

### Git/Repository Checks

```powershell
git status --short
git ls-files
rg --files -g '!venv/**' -g '!**/__pycache__/**'
```

If Git reports dubious ownership under the sandbox user, use the non-writing form:

```powershell
git -c safe.directory='C:/Users/dhawa/OneDrive/Desktop/UNI/CS Projects/Investment-advisor-ai' status --short
```

### Database/Migration Checks

Not applicable. The repository currently has no database, schema, migrations, seed files, or persistence layer.
