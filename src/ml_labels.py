import pandas as pd

def create_labels(data, lookahead_days=5, threshold=0.02):
    """
    Create BUY / HOLD / SELL labels based on future returns.

    Parameters:
    - data: pandas DataFrame with a 'Close' column
    - lookahead_days: how many days ahead to look
    - threshold: return threshold for BUY/SELL

    Returns:
    - pandas Series of labels aligned with data index
    """
    if data is None or data.empty:
        raise ValueError("Data is empty or None")

    if "Close" not in data.columns:
        raise ValueError("Data must contain a 'Close' column")
    
    close_prices = data["Close"].values
    labels = [None] * len(close_prices)

    for i in range(len(close_prices) - lookahead_days):
        future_price = close_prices[i + lookahead_days]
        current_price = close_prices[i]
        future_return = (future_price - current_price) / current_price

        if future_return > threshold:
            labels[i] = "BUY"
        elif future_return < -threshold:
            labels[i] = "SELL"
        else:
            labels[i] = "HOLD"

    # For the last 'lookahead_days' entries, we cannot compute future returns

    labels_series = pd.Series(labels, index=data.index)

    # Drop rows where label is None (not enough future data)
    labels_series = labels_series.dropna()

    return labels_series

