import pandas as pd
import yfinance as yf

from config import PricePolicy


PRICE_COLUMNS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}


def normalize_yfinance_columns(data):
    if not isinstance(data.columns, pd.MultiIndex):
        return data

    for level in range(data.columns.nlevels):
        values = {str(value) for value in data.columns.get_level_values(level)}
        if PRICE_COLUMNS.intersection(values):
            normalized = data.copy()
            normalized.columns = normalized.columns.get_level_values(level)
            return normalized

    raise ValueError("Downloaded data has an unsupported column format.")


def apply_price_policy(data, price_policy: PricePolicy = "raw_close"):
    if price_policy == "raw_close":
        data.attrs["price_policy"] = "raw_close"
        data.attrs["price_column"] = "Close"
        return data

    if price_policy == "adjusted_close":
        if "Adj Close" not in data.columns:
            raise ValueError("Adjusted close price policy requires an 'Adj Close' column.")
        data = data.copy()
        data["Raw Close"] = data["Close"]
        data["Close"] = data["Adj Close"]
        data.attrs["price_policy"] = "adjusted_close"
        data.attrs["price_column"] = "Adj Close"
        return data

    raise ValueError(f"Unsupported price policy: {price_policy}")


def get_stock_data(ticker, period="5y", price_policy: PricePolicy = "raw_close"):
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
        raise ConnectionError("Failed to download data. Check your internet connection.") from e
    # handle empty data
    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Please check the ticker symbol.")
    data = normalize_yfinance_columns(data)
    # clean data
    data = data.reset_index()

    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    existing_numeric_columns = [col for col in numeric_columns if col in data.columns]
    data[existing_numeric_columns] = data[existing_numeric_columns].apply(pd.to_numeric, errors="coerce")

    # handle missing values
    data = data.dropna()
    return apply_price_policy(data, price_policy)
