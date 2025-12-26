from pyexpat import features
from data_loader import get_stock_data
from ml_dataset import build_feature_dataset
from ml_labels import create_labels
from model import (
    run_model,
    train_random_forest,
    evaluate_model,
    predict_latest_signal,
    get_feature_importances
)
import pandas as pd

def main():
    print("Investment Advisor AI setup complete.")

    ticker = input("Enter a stock ticker (e.g. AAPL): ").strip().upper()

    if not ticker:
        print("Ticker cannot be empty.")
        return
    try:
        data = get_stock_data(ticker)
        result = run_model(data)
    except Exception as e:
        print("Error:", e)
        return

    print(f"Ticker: {ticker}")
    print(f"Signal: {result['signal']}")
    print(f"Current Price: {result['current_price']:.2f}")
    print(f"Short MA: {result['short_ma']:.2f}")
    print(f"Long MA: {result['long_ma']:.2f}")
    print(f"Reason: {result['reason']}")

    features = build_feature_dataset(data)
    labels = create_labels(data)

    dataset = features.join(labels.rename("label"), how="inner")

    features = dataset[
    [
        "current_price",
        "short_ma",
        "long_ma",
        "momentum",
        "volatility",
        "trend_strength",
    ]
]
    labels = dataset["label"]


    rf_model, X_test, y_test = train_random_forest(features, labels)

    results = evaluate_model(rf_model, X_test, y_test)

    print("\nRandom Forest Model Performance:")
    print("Accuracy:", results["accuracy"])
    print(results["report"])

    latest_features = features.iloc[[-1]]
    ml_signal = predict_latest_signal(rf_model, latest_features)

    print("\nML Recommendation (Random Forest):")
    print("ML Signal:", ml_signal)

    feature_importance = get_feature_importances(
        rf_model,
        features.columns
    )

    print("\nFeature Importance:")
    print(feature_importance)

    print("\nLabel distribution:")
    print(dataset["label"].value_counts(normalize=True))


if __name__ == "__main__":
    main()