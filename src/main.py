from data_loader import get_stock_data
from model import run_model

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

if __name__ == "__main__":
    main()