import pandas as pd
import numpy as np

def normalize_series(data, column):
    """Helper to ensure we're working with a Series"""
    if isinstance(data[column], pd.DataFrame):
        return data[column].iloc[:, 0]
    return data[column]

def create_normal_labels(data, lookahead_days=20, threshold =0.05):
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
            labels[i] = 1
        elif future_return < -threshold:
            labels[i] = -1
        #else: #Optional
        #    labels[i] = 0

    labels_series = pd.Series(labels, index=data.index)
    labels_series = labels_series.dropna()

    return labels_series

def create_labels(data, lookahead_days=20, volatility_window=20, k=1.0):
    """
    Create BUY / HOLD / SELL labels based on future returns with dynamic volatility threshold.

    Parameters:
    - data: pandas DataFrame with a 'Close' column
    - lookahead_days: how many days ahead to look (default: 20)
    - volatility_window: window size for calculating volatility (default: 20)
    - k: multiplier for volatility threshold (default: 1.0)

    Returns:
    - pandas Series of labels aligned with data index
      1 = BUY, 0 = HOLD, -1 = SELL
    """
    if data is None or data.empty:
        raise ValueError("Data is empty or None")

    if "Close" not in data.columns:
        raise ValueError("Data must contain a 'Close' column")
    
    close = normalize_series(data, "Close")
    
    # Calculate returns (percentage change)
    returns = close.pct_change()
    
    # Calculate rolling volatility (std of returns)
    volatility = returns.rolling(window=volatility_window).std()
    
    # Dynamic threshold based on volatility
    threshold = k * volatility
    
    # Calculate future return aligned to current index
    future_return = (close.shift(-lookahead_days) - close) / close
    
    # Initialize labels as HOLD (0)
    labels = pd.Series(0, index=data.index, name="label")
    
    labels[future_return >= threshold] = 1 #buy
    labels[future_return <= -threshold] = -1 #sell
    
    labels = labels.dropna()
    return labels


def create_TBL(data, lookahead_days=20, upper_barrier=0.05, lower_barrier=-0.05, volatility_window = 20):
    """
    Triple Barrier Labeling: Labels based on which barrier is hit first.
    """
    if data is None or data.empty:
        raise ValueError("Data is empty or None")
    if "Close" not in data.columns:
        raise ValueError("Data must contain a 'Close' column")
    
    close = normalize_series(data, "Close")

    labels = []

    for i in range(len(close) - lookahead_days):
        current_price = close.iloc[i]
        future_prices = close.iloc[i+1:i+1+lookahead_days]

        returns = (future_prices - current_price) / current_price
        
        upper_touch = returns[returns >= upper_barrier]
        lower_touch = returns[returns <= lower_barrier]
        
        upper_day = upper_touch.index[0] if len(upper_touch) > 0 else None
        lower_day = lower_touch.index[0] if len(lower_touch) > 0 else None

        if upper_day is not None and (lower_day is None or upper_day < lower_day):
            labels.append(1)
        elif lower_day is not None:
            labels.append(-1)
        else:
            labels.append(None)

    labels.extend([None] * lookahead_days)

    labels_series = pd.Series(labels, index=data.index)
    labels_series = labels_series.dropna()
    buy = (labels_series == 1).sum()
    sell = (labels_series == -1).sum()

    print(f"✓ TBL: {buy} BUY ({buy/len(labels_series)*100:.1f}%), "
          f"{sell} SELL ({sell/len(labels_series)*100:.1f}%), ")
    
    return labels_series


def create_TBL_volatility_adjustment(data, lookahead_days=20, k=1.3, volatility_window = 20):
    """
    Triple Barrier Labeling: Labels based on which barrier is hit first.
    """
    if data is None or data.empty:
        raise ValueError("Data is empty or None")
    if "Close" not in data.columns:
        raise ValueError("Data must contain a 'Close' column")
    
    close = normalize_series(data, "Close")

    returns = close.pct_change()
    rolling_vol = returns.rolling(window=volatility_window).std()    

    labels = []

    for i in range(len(close) - lookahead_days):
        current_price = close.iloc[i]
        future_prices = close.iloc[i+1:i+1+lookahead_days]

        current_vol = rolling_vol.iloc[i]
        
        if pd.isna(current_vol):
            labels.append(None)
            continue

        upper_barrier = k * current_vol
        lower_barrier = -k* current_vol

        returns = (future_prices - current_price) / current_price
        
        upper_touch = returns[returns >= upper_barrier]
        lower_touch = returns[returns <= lower_barrier]
        
        upper_day = upper_touch.index[0] if len(upper_touch) > 0 else None
        lower_day = lower_touch.index[0] if len(lower_touch) > 0 else None

        if upper_day is not None and (lower_day is None or upper_day < lower_day):
            labels.append(1)
        elif lower_day is not None:
            labels.append(-1)
        else:
            labels.append(None)

    labels.extend([None] * lookahead_days)

    labels_series = pd.Series(labels, index=data.index)
    labels_series = labels_series.dropna()
    buy = (labels_series == 1).sum()
    sell = (labels_series == -1).sum()

    print(f"✓ TBL: {buy} BUY ({buy/len(labels_series)*100:.1f}%), "
          f"{sell} SELL ({sell/len(labels_series)*100:.1f}%), ")
    
    return labels_series
