from dataclasses import dataclass

from backtesting import BacktestResult, run_signal_backtest
from config import AnalysisConfig, DEFAULT_ANALYSIS_CONFIG
from data_loader import get_stock_data
from decision import InvestorProfile, build_decision_summary
from fundamentals import FundamentalAnalysis, fetch_yfinance_fundamentals
from ml_dataset import FEATURE_COLUMNS, build_feature_dataset
from ml_labels import create_labels
from ml_pipeline import WalkForwardResult, permutation_importance_summary, predict_latest_signal, walk_forward_train
from model import run_ruleBasedModel
from monte_carlo import MonteCarloResult, historical_bootstrap_simulation
from risk_metrics import summarize_return_risk


@dataclass
class ResearchReport:
    ticker: str
    data_rows: int
    date_range: tuple
    rule_result: dict
    training_result: WalkForwardResult
    ml_prediction: dict
    feature_importance: object
    backtest: BacktestResult
    risk_summary: dict
    monte_carlo: MonteCarloResult
    fundamentals: FundamentalAnalysis | None
    decision_summary: dict
    warnings: list[str]


def build_research_report(
    ticker,
    period="5y",
    config: AnalysisConfig = DEFAULT_ANALYSIS_CONFIG,
    model_name="random_forest",
    include_fundamentals=True,
    investor_profile: InvestorProfile | None = None,
):
    data = get_stock_data(ticker, period=period, price_policy=config.price_policy)
    fundamentals = None
    fundamental_warning = None
    if include_fundamentals:
        try:
            fundamentals = fetch_yfinance_fundamentals(ticker)
        except Exception as exc:
            fundamental_warning = f"Fundamentals unavailable: {exc}"
    return build_research_report_from_data(
        ticker,
        data,
        config=config,
        model_name=model_name,
        fundamentals=fundamentals,
        fundamental_warning=fundamental_warning,
        investor_profile=investor_profile,
    )


def build_research_report_from_data(
    ticker,
    data,
    config: AnalysisConfig = DEFAULT_ANALYSIS_CONFIG,
    model_name="random_forest",
    fundamentals: FundamentalAnalysis | None = None,
    fundamental_warning: str | None = None,
    investor_profile: InvestorProfile | None = None,
):
    rule_result = run_ruleBasedModel(data)
    feature_frame = build_feature_dataset(data)
    labels = create_labels(
        data,
        lookahead_days=config.lookahead_days,
        volatility_window=config.volatility_window,
        k=config.volatility_barrier_multiplier,
        label_policy=config.label_policy,
    )
    dataset = feature_frame.join(labels.rename("label"), how="inner")
    validate_research_dataset(dataset, config.walk_forward_splits)

    features = dataset[FEATURE_COLUMNS]
    y = dataset["label"]
    training_result = walk_forward_train(
        features,
        y,
        model_name=model_name,
        n_splits=config.walk_forward_splits,
        top_n_features=config.top_n_features,
        config=config,
    )
    ml_prediction = predict_latest_signal(training_result.model, features.iloc[[-1]])
    feature_importance = permutation_importance_summary(training_result.model, features, y)
    backtest = run_signal_backtest(dataset["current_price"], training_result.oos_predictions, config.costs)
    risk_summary = summarize_return_risk(data["Close"], risk_free_rate=config.risk_free_rate)
    returns = data["Close"].pct_change().dropna()
    monte_carlo = historical_bootstrap_simulation(
        returns,
        start_price=float(data["Close"].iloc[-1]),
        horizon_days=config.monte_carlo_horizon_days,
        n_simulations=config.monte_carlo_simulations,
        stop_loss=config.monte_carlo_stop_loss,
    )

    warnings = build_report_warnings(training_result, dataset)
    if fundamental_warning:
        warnings.append(fundamental_warning)
    decision_summary = build_decision_summary(
        rule_result=rule_result,
        ml_prediction=ml_prediction,
        model_metrics=training_result.metrics,
        backtest_metrics=backtest.metrics,
        risk_summary=risk_summary,
        monte_carlo_metrics=monte_carlo.metrics,
        fundamentals=fundamentals,
        investor_profile=investor_profile,
    )
    date_range = (data["Date"].min(), data["Date"].max()) if "Date" in data.columns else (data.index.min(), data.index.max())
    return ResearchReport(
        ticker=ticker,
        data_rows=len(data),
        date_range=date_range,
        rule_result=rule_result,
        training_result=training_result,
        ml_prediction=ml_prediction,
        feature_importance=feature_importance,
        backtest=backtest,
        risk_summary=risk_summary,
        monte_carlo=monte_carlo,
        fundamentals=fundamentals,
        decision_summary=decision_summary,
        warnings=warnings,
    )


def validate_research_dataset(dataset, n_splits):
    if dataset.empty:
        raise ValueError("No labeled feature rows were created for research analysis.")
    missing = [column for column in FEATURE_COLUMNS if column not in dataset.columns]
    if missing:
        raise ValueError(f"Research dataset is missing feature columns: {missing}")
    if "label" not in dataset.columns:
        raise ValueError("Research dataset is missing the label column.")
    if len(dataset) <= n_splits:
        raise ValueError(f"Need more than {n_splits} labeled rows for walk-forward validation.")
    if dataset["label"].nunique() < 2:
        raise ValueError("Need at least two label classes for ML research analysis.")


def build_report_warnings(training_result, dataset):
    warnings = []
    if training_result.metrics["std_accuracy"] > 0.10:
        warnings.append("High validation variance: model performance is unstable across time splits.")
    if abs(training_result.metrics["mean_accuracy"] - training_result.metrics["mean_f1"]) > 0.05:
        warnings.append("Accuracy/F1 gap detected: class imbalance may be affecting model quality.")
    if dataset["label"].value_counts(normalize=True).max() > 0.75:
        warnings.append("Label imbalance detected: one class dominates the training set.")
    return warnings
