import yfinance as yf
import pandas as pd

def get_stock_data(ticker, period="5y"): #Change years for ML
    """
    Downloads and cleans historical stock price data.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        period (str): Time period (default '1y')

    Returns:
        pd.DataFrame: Cleaned stock data
    """

    # Validate ticker input
    if not isinstance(ticker, str) or ticker.strip() == "":
        raise ValueError("Ticker must be a non-empty string.")
    # Download stock data
    try:
        data = yf.download(ticker, period=period, auto_adjust=False)
    except Exception as e:
        raise ConnectionError(f"Failed to download data. Check your internet connection.") from e
    # handle empty data
    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Please check if the ticker even exists you goof.")
    # clean data
    data = data.reset_index()
    
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    existing_numeric_columns = [col for col in numeric_columns if col in data.columns]
    data[existing_numeric_columns] = data[existing_numeric_columns].apply(pd.to_numeric, errors='coerce')
            
    #handle missing values
    data = data.dropna()
    return data
