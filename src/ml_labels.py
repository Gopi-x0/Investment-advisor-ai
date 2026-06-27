import pandas as pd
from config import LabelPolicy


def label_index(data):
    if "Date" in data.columns:
        return data["Date"]
    return data.index


def normalize_series(data, column):
    """Helper to ensure we're working with a Series"""
    if isinstance(data[column], pd.DataFrame):
        return data[column].iloc[:, 0]
    return data[column]


def close_series(data):
    close = normalize_series(data, "Close").copy()
    close.index = label_index(data)
    return close


def create_normal_labels(data, lookahead_days=20, threshold=0.05):
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
    
    close = close_series(data)
    close_prices = close.values
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

    labels_series = pd.Series(labels, index=close.index)
    labels_series = labels_series.dropna()

    return labels_series

def create_labels(data, lookahead_days=20, volatility_window=20, k=1.0, label_policy: LabelPolicy = "three_class"):
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

    close = close_series(data)
    returns = close.pct_change()
    threshold = k * returns.rolling(window=volatility_window).std()
    future_return = (close.shift(-lookahead_days) - close) / close

    labels = pd.Series(None, index=close.index, name="label", dtype="object")
    valid_rows = future_return.notna() & threshold.notna()

    if label_policy == "three_class":
        labels[valid_rows] = 0

    labels[valid_rows & (future_return > threshold)] = 1
    labels[valid_rows & (future_return < -threshold)] = -1

    if label_policy == "binary":
        labels = labels[labels != 0]

    labels = labels.dropna()
    return labels.astype(int)


def create_return_labels(
    data,
    lookahead_days=20,
    buy_threshold=0.03,
    sell_threshold=-0.03,
    cost_rate=0.0,
    label_policy: LabelPolicy = "three_class",
):
    if data is None or data.empty:
        raise ValueError("Data is empty or None")
    if "Close" not in data.columns:
        raise ValueError("Data must contain a 'Close' column")

    close = close_series(data)
    future_return = (close.shift(-lookahead_days) - close) / close
    labels = pd.Series(None, index=close.index, name="label", dtype="object")
    valid_rows = future_return.notna()

    if label_policy == "three_class":
        labels[valid_rows] = 0

    labels[valid_rows & (future_return > buy_threshold + cost_rate)] = 1
    labels[valid_rows & (future_return < sell_threshold - cost_rate)] = -1

    if label_policy == "binary":
        labels = labels[labels != 0]

    labels = labels.dropna()
    return labels.astype(int)


def create_TBL(data, lookahead_days=20, upper_barrier=0.05, lower_barrier=-0.05, volatility_window = 20):
    """
    Triple Barrier Labeling: Labels based on which barrier is hit first.
    """
    if data is None or data.empty:
        raise ValueError("Data is empty or None")
    if "Close" not in data.columns:
        raise ValueError("Data must contain a 'Close' column")
    
    close = close_series(data)

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

    labels.extend([None] * (len(close) - len(labels)))

    labels_series = pd.Series(labels, index=close.index)
    labels_series = labels_series.dropna()

    if labels_series.empty:
        print("TBL: 0 BUY (0.0%), 0 SELL (0.0%), ")
        return labels_series

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
    
    close = close_series(data)

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

    labels.extend([None] * (len(close) - len(labels)))

    labels_series = pd.Series(labels, index=close.index)
    labels_series = labels_series.dropna()

    if labels_series.empty:
        print("TBL: 0 BUY (0.0%), 0 SELL (0.0%), ")
        return labels_series

    buy = (labels_series == 1).sum()
    sell = (labels_series == -1).sum()

    print(f"✓ TBL: {buy} BUY ({buy/len(labels_series)*100:.1f}%), "
          f"{sell} SELL ({sell/len(labels_series)*100:.1f}%), ")
    
    return labels_series
