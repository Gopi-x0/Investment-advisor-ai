from data_loader import get_stock_data
from indicators import getMovingAverages
from advisor import movingAvg_Recommandation

def main():
    print("Investment Advisor AI setup complete.")
   
    ticker = "AAPL"
    data = get_stock_data(ticker)
    currentPrice, shortMA, longMA = getMovingAverages(data, 20, 50)
    recommendation = movingAvg_Recommandation(currentPrice, shortMA, longMA)
    print(f"Ticker: {ticker}")
    print(f"Current Price: {currentPrice:.2f}")
    print(f"Short MA: {shortMA:.2f}")
    print(f"Long MA: {longMA:.2f}")
    print(f"Recommendation: {recommendation}")

if __name__ == "__main__":
    main()