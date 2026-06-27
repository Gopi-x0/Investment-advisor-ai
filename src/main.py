from backtesting import run_signal_backtest
from config import DEFAULT_ANALYSIS_CONFIG, describe_assumptions
from data_loader import get_stock_data
from ml_dataset import FEATURE_COLUMNS, build_feature_dataset
from ml_labels import create_labels
from ml_pipeline import (
    permutation_importance_summary,
    predict_latest_signal,
    save_model_artifact,
    walk_forward_train,
)
from model import run_ruleBasedModel
from monte_carlo import historical_bootstrap_simulation
from risk_metrics import summarize_return_risk
from reports import build_report_payload, save_report_payload

CONFIG = DEFAULT_ANALYSIS_CONFIG
ML_N_SPLITS = CONFIG.walk_forward_splits


def validate_ml_inputs(dataset, feature_columns, n_splits):
    if dataset.empty:
        raise ValueError("No labeled feature rows were created for ML training.")

    if "label" not in dataset.columns:
        raise ValueError("ML dataset is missing the 'label' column.")

    missing_columns = [column for column in feature_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"ML dataset is missing feature columns: {missing_columns}")

    if len(dataset) <= n_splits:
        raise ValueError(
            f"Need more than {n_splits} labeled samples for walk-forward validation; "
            f"got {len(dataset)}."
        )

    if dataset["label"].nunique() < 2:
        raise ValueError("Need at least two label classes for ML training.")


def main():
    print("Investment Advisor AI setup complete.")
    print("")
    for assumption in describe_assumptions(CONFIG):
        print(f"- {assumption}")

    ticker = input("Enter a stock ticker (e.g. AAPL): ").strip().upper()

    print(f"\nDownloading data for {ticker}...")

    if not ticker:
        print("Ticker cannot be empty.")
        return

    try:
        data = get_stock_data(ticker, price_policy=CONFIG.price_policy)
        print(f"\nDownloaded {len(data)} days of data")
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}\n")
        result = run_ruleBasedModel(data)
    except Exception as e:
        print("Error:", e)
        return

    print("=" * 60)
    print(" MULTI-FACTOR RULE-BASED TRADING MODEL ".center(60, "="))
    print("=" * 60)

    print(f"\nTicker: {ticker}")
    print(f"Signal: {result['signal']}")
    print(f"Overall Score: {result['score']}\n")

    print(f"Current Price: {result['current_price']:.2f}")
    print(f"Short MA (20): {result['short_ma']:.2f}")
    print(f"Long MA (50): {result['long_ma']:.2f}\n")

    print("Indicator Breakdown:")
    for indicator, decision in result["breakdown"].items():
        print(f"  - {indicator.replace('_', ' ').title()}: {decision}")

    print("")
    print("=" * 60)
    print(" MACHINE LEARNING MODEL ".center(60, "="))
    print("=" * 60)

    try:
        feature_frame = build_feature_dataset(data)
        labels = create_labels(
            data,
            lookahead_days=CONFIG.lookahead_days,
            volatility_window=CONFIG.volatility_window,
            k=CONFIG.volatility_barrier_multiplier,
            label_policy=CONFIG.label_policy,
        )
        dataset = feature_frame.join(labels.rename("label"), how="inner")
        validate_ml_inputs(dataset, FEATURE_COLUMNS, ML_N_SPLITS)
    except ValueError as e:
        print("ML setup error:", e)
        return

    features = dataset[FEATURE_COLUMNS]
    labels = dataset["label"]

    print("")
    print(dataset.head())
    print("")
    print(dataset.tail())

    training_result = walk_forward_train(
        features,
        labels,
        model_name="random_forest",
        n_splits=ML_N_SPLITS,
        top_n_features=CONFIG.top_n_features,
        config=CONFIG,
    )

    print("\n" + "=" * 60)
    print(" MODEL ANALYSIS ".center(60, "="))
    print("=" * 60)

    print(
        f"Mean Accuracy:  {training_result.metrics['mean_accuracy']:.4f} "
        f"+/- {training_result.metrics['std_accuracy']:.4f}"
    )
    print(f"Mean F1 Score:  {training_result.metrics['mean_f1']:.4f}")
    print(f"Classes: {training_result.metrics['class_names']}")

    if training_result.metrics["std_accuracy"] > 0.10:
        print("\nHIGH VARIANCE - Model is unstable across time periods")
    else:
        print("\nModel shows consistent performance across time periods")

    if abs(training_result.metrics["mean_accuracy"] - training_result.metrics["mean_f1"]) > 0.05:
        print("Accuracy-F1 gap detected - possible class imbalance")

    latest_features = features.iloc[[-1]]
    ml_prediction = predict_latest_signal(training_result.model, latest_features)

    print("\nML Recommendation:")
    print("ML Signal:", ml_prediction["signal"])
    if ml_prediction["probabilities"]:
        print("Class Probabilities:", ml_prediction["probabilities"])

    feature_importance = permutation_importance_summary(training_result.model, features, labels)

    print("\nFeature Importance:")
    print(feature_importance.head(10))

    backtest = run_signal_backtest(dataset["current_price"], training_result.oos_predictions, CONFIG.costs)

    print("\nBacktest Summary:")
    for key, value in backtest.metrics.items():
        print(f"{key}: {value:.4f}")

    returns = data["Close"].pct_change().dropna()
    risk_summary = summarize_return_risk(data["Close"], risk_free_rate=CONFIG.risk_free_rate)
    monte_carlo = historical_bootstrap_simulation(
        returns,
        start_price=float(data["Close"].iloc[-1]),
        horizon_days=CONFIG.monte_carlo_horizon_days,
        n_simulations=CONFIG.monte_carlo_simulations,
        stop_loss=CONFIG.monte_carlo_stop_loss,
    )

    print("\nRisk Summary:")
    for key, value in risk_summary.items():
        print(f"{key}: {value:.4f}")

    print("\nMonte Carlo Bootstrap Scenario:")
    for key, value in monte_carlo.metrics.items():
        print(f"{key}: {value:.4f}")

    artifact_path = save_model_artifact(training_result, f"artifacts/models/{ticker}_latest.joblib")
    print(f"\nSaved model artifact: {artifact_path}")

    report_payload = build_report_payload(
        ticker=ticker,
        config=CONFIG,
        rule_result=result,
        ml_prediction=ml_prediction,
        model_metrics=training_result.metrics,
        model_metadata=training_result.metadata,
        feature_importance=feature_importance,
        backtest_metrics=backtest.metrics,
        risk_summary=risk_summary,
        monte_carlo_metrics=monte_carlo.metrics,
        warnings=[],
        data_summary={
            "rows": len(data),
            "date_start": data["Date"].min(),
            "date_end": data["Date"].max(),
        },
    )
    report_paths = save_report_payload(report_payload, output_dir="reports", stem=f"{ticker}_latest")
    print(f"Saved analysis reports: {report_paths['json']} and {report_paths['markdown']}")


if __name__ == "__main__":
    main()
