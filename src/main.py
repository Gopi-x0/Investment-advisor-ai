from pyexpat import features
from data_loader import get_stock_data
from ml_dataset import build_feature_dataset
from model import run_model
from ml_labels import create_labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
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
    
    label_map = {"SELL": -1, "HOLD": 0, "BUY": 1}
    dataset["label_num"] = dataset["label"].map(label_map)
    
    features = dataset[[ "current_price", "short_ma", "long_ma", "momentum", "volatility", "trend_strength"]]
    labels = dataset["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, shuffle=False
    )   

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        class_weight="balanced"
    )

    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    print("\nRandom Forest Model Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred)) 
    
    latest_features = features.iloc[[-1]]
    ml_signal_num = rf_model.predict(latest_features)[0]

    inv_label_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
    ml_signal = inv_label_map[ml_signal_num]

    print("\nML Recommendation (Random Forest):")
    print("ML Signal:", ml_signal)
   
    feature_importance = pd.Series(
        rf_model.feature_importances_,
        index= features.columns
    ).sort_values(ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

if __name__ == "__main__":
    main()