from data_loader import get_stock_data

def main():
    ticker = "AAPL"
    period = "1y"
    try:
        data = get_stock_data(ticker, period)
        print(f"Successfully retrieved data for {ticker}:")
        print(data.head())
    except Exception as e:
        print("Error occurred:")
        print(e)

if __name__ == "__main__":
    main()