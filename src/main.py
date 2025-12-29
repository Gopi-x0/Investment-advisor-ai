from pyexpat import features
from data_loader import get_stock_data
from ml_dataset import build_feature_dataset
from ml_labels import create_labels, create_normal_labels, create_TBL, create_TBL_volatility_adjustment
from model import (
    run_ruleBasedModel,
    train_model,
    train_model_walk_forward,
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
    print(" MACHINE LEARNING MODEL ".center(60, "="))
    print("=" * 60) 

    features = build_feature_dataset(data)
    labels = create_TBL_volatility_adjustment(data, lookahead_days=20,volatility_window=20,k= 1.3) 

    dataset = features.join(labels.rename("label"), how="inner")   
    feature_cols = [
    "current_price", "short_ma", "long_ma", "momentum", "volatility", 
    "trend_strength", "sharpe_ratio",
    "price_above_short_ma", "price_above_long_ma", "short_above_long_ma", "ma_spread",
    "rsi", "macd_histogram", "bb_position", "atr", "volume_ratio", 
    "price_position_52w", "momentum_5d", "momentum_20d"
    ]
    features = dataset[feature_cols]
    labels = dataset["label"]

    print("")
    print(dataset.head())
    print("")
    print(dataset.tail())

    #rf_model, X_test, y_test, scaler, selected_features  = train_model(features, labels, use_feature_selection= True, top_n_features=10, use_gradient_boost = False)
    #results = evaluate_model(rf_model, X_test, y_test)
    #print("\nRandom Forest Model Performance:")
    #print("Accuracy:", results["accuracy"])
    #print(results["report"])

    #OR walk forward validation
    rf_model, cv_results, selected_features, scaler = train_model_walk_forward(features, labels, n_splits=5, use_feature_selection=True, top_n_features=10, use_gradient_boost= False)

    print("\n" + "="*60)
    print(" MODEL ANALYSIS ".center(60, "="))
    print("="*60)

    print(f"Mean Accuracy:  {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
    print(f"Mean F1 Score:  {cv_results['mean_f1']:.4f}")
    print(f"Accuracy Range: {min(cv_results['all_accuracies']):.4f} to {max(cv_results['all_accuracies']):.4f}")

    # Stability check
    if cv_results['std_accuracy'] > 0.10:
        print("\nHIGH VARIANCE - Model is unstable across time periods")
    else:
        print("\nModel shows consistent performance across time periods")

    # Class balance check
    if abs(cv_results['mean_accuracy'] - cv_results['mean_f1']) > 0.05:
        print("Accuracy-F1 gap detected - possible class imbalance")
    ################################### 

    latest_features = features.iloc[[-1]]
    ml_signal = predict_latest_signal(rf_model, latest_features, scaler, selected_features)

    print("\nML Recommendation:")
    print("ML Signal:", ml_signal) 

    feature_importance = get_feature_importances(rf_model, selected_features)

    print("\nFeature Importance:")
    print(feature_importance)

    #("\nLabel distribution:")
    #print(dataset["label"].value_counts(normalize=True))

if __name__ == "__main__":
    main()