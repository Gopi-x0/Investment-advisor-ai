from pyexpat import features
from data_loader import get_stock_data
from ml_dataset import build_feature_dataset
from ml_labels import create_labels
from model import (
    run_ruleBasedModel,
    train_random_forest,
    evaluate_model,
    predict_latest_signal,
    get_feature_importances
)
import pandas as pd

def main():
    print("Investment Advisor AI setup complete.")

    ticker = input("Enter a stock ticker (e.g. AAPL): ").strip().upper()

    print(f"\nDownloading data for {ticker}...")

    if not ticker:
        print("Ticker cannot be empty.")
        return
    try:
        data = get_stock_data(ticker)
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
    print(" MACHINE LEARNING (RANDOM FOREST) MODEL ".center(60, "="))
    print("=" * 60) 

    features = build_feature_dataset(data)
    labels = create_labels(data)

    dataset = features.join(labels.rename("label"), how="inner")   
    feature_cols = [
    "current_price", "short_ma", "long_ma", "momentum", "volatility", 
    "trend_strength", "sharpe_ratio",
    "price_above_short_ma", "price_above_long_ma", "short_above_long_ma", "ma_spread",
    "rsi", "macd_histogram", "bb_position", "atr", "volume_ratio", 
    "price_position_52w", "momentum_5d", "momentum_20d"
    ]
    features = dataset[feature_cols]
    labels = dataset[labels]

    rf_model, X_test, y_test = train_random_forest(features, labels)

    results = evaluate_model(rf_model, X_test, y_test)

    print("\nRandom Forest Model Performance:")
    print("Accuracy:", results["accuracy"])
    print(results["report"])

    latest_features = features.iloc[[-1]]
    ml_signal = predict_latest_signal(rf_model, latest_features)

    print("\nML Recommendation (Random Forest):")
    print("ML Signal:", ml_signal)

    feature_importance = get_feature_importances(rf_model,features.columns)

    print("\nFeature Importance:")
    print(feature_importance)

    print("\nLabel distribution:")
    print(dataset["label"].value_counts(normalize=True))


if __name__ == "__main__":
    main()