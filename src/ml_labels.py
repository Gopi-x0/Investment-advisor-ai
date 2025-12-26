import numpy as np
import pandas as pd

def create_labels(data, lookahead_days=20, volatility_window=20, k =1.0):
    """
    Create BUY / HOLD / SELL labels based on future returns.

    Parameters:
    - data: pandas DataFrame with a 'Close' column
    - lookahead_days: how many days ahead to look
    - volatility_window: window size for calculating volatility
    - k: multiplier for volatility threshold

    Returns:
    - pandas Series of labels aligned with data index
    """
    if data is None or data.empty:
        raise ValueError("Data is empty or None")

    if "Close" not in data.columns:
        raise ValueError("Data must contain a 'Close' column")
    

    df = pd.DataFrame(index=data.index)

    # Calculate volatility
    volatility = data["Close"].rolling(window=volatility_window).std()

    # Dynamic threshold
    threshold = k * volatility

    # future return aligned to index
    close = data["Close"].squeeze()

    future_return = (close.shift(-lookahead_days) - close) / close


    df["label"] = 0  # HOLD

    df.loc[future_return >= threshold, "label"] = 1    # BUY
    df.loc[future_return <= -threshold, "label"] = -1  # SELL

    # drop rows where future return is NaN (end of series)
    df = df.dropna()

    return df["label"]


