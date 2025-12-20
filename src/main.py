from pyexpat import features
from data_loader import get_stock_data
from ml_dataset import build_feature_dataset
from model import run_model
from ml_labels import create_labels

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
    print(features.head())
    print(features.tail())

    labels = create_labels(data)
    print(labels.value_counts())
    print(labels.head())


if __name__ == "__main__":
    main()