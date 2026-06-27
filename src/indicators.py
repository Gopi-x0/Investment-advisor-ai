import pandas as pd
import numpy as np

def checkData(data):
    """Validate input data"""
    if data is None or data.empty:
        raise ValueError("Data is empty or None")
    if "Close" not in data.columns:
        raise ValueError("Data must contain 'Close' column")

def normalize_series(data, column):
    """Helper to ensure we're working with a Series"""
    if isinstance(data[column], pd.DataFrame):
        return data[column].iloc[:, 0]
    return data[column]

def getMovingAverages(data, short_range=20, long_range=50):
    """
    Calculate short and long-term moving averages.
    Returns current price and both MAs.
    """
    checkData(data)
    close = normalize_series(data, "Close")
    
    short_ma = close.tail(short_range).mean()
    long_ma = close.tail(long_range).mean()
    current_price = close.iloc[-1]
    
    return float(current_price), float(short_ma), float(long_ma)


def getMomentum(data, n_days=10):
    """
    Simple momentum: percentage change over n days
    """
    checkData(data)
    close = normalize_series(data, "Close")
    
    if len(close) < n_days + 1:
        return 0.0
    
    current_price = close.iloc[-1]
    past_price = close.iloc[-n_days - 1]
    
    momentum = (current_price - past_price) / past_price
    return float(momentum)


def getTrendStrength(data, long_range=50):
    """
    Measure how far current price is from long-term MA
    """
    checkData(data)
    close = normalize_series(data, "Close")
    
    current_price = close.iloc[-1]
    long_ma = close.tail(long_range).mean()
    
    trend_strength = (current_price - long_ma) / long_ma
    return float(trend_strength)


def getReturns(data):
    """
    Calculate daily returns using pct_change (more efficient)
    """
    checkData(data)
    close = normalize_series(data, "Close")
    returns = close.pct_change().dropna()
    return returns.values


def getVolatility(data, window=None):
    """
    Historical volatility: standard deviation of returns
    """
    checkData(data)
    close = normalize_series(data, "Close")
    
    returns = close.pct_change().dropna()
    
    if window and len(returns) > window:
        volatility = returns.tail(window).std()
    else:
        volatility = returns.std()
    
    return float(volatility)


def getSharpeRatio(data, window=None, risk_free_rate=0.02):
    """
    Sharpe Ratio: risk-adjusted return
    """
    checkData(data)
    returns = getReturns(data)
    
    if len(returns) < 2:
        return 0.0
    
    # Use recent window if specified
    if window and len(returns) > window:
        recent_returns = returns[-window:]
    else:
        recent_returns = returns
    
    avg_return = np.mean(recent_returns)
    volatility = np.std(recent_returns)
    
    if volatility == 0 or np.isnan(volatility):
        return 0.0
    
    # Annualize assuming 252 trading days
    annual_return = avg_return * 252
    annual_volatility = volatility * np.sqrt(252)
    
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    return float(sharpe_ratio)

def getRSI(data, period=14):
    """
    Relative Strength Index using Wilder's smoothing
    Range: 0-100 (>70 overbought, <30 oversold)
    """
    checkData(data)
    close = normalize_series(data, "Close")
    
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    
    # Wilder's smoothing
    avg_gain = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return float(rsi.iloc[-1])


def getMACD(data, fast=12, slow=26, signal=9):
    """
    Moving Average Convergence Divergence
    Returns histogram value (MACD - Signal)
    """
    checkData(data)
    close = normalize_series(data, "Close")
    
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return float(histogram.iloc[-1])


def getBollingerPosition(data, period=20, num_std=2):
    """
    Where price sits relative to Bollinger Bands
    Returns: -1 to 1 (negative = below middle, positive = above)
    """
    checkData(data)
    close = normalize_series(data, "Close")
    
    middle_band = close.tail(period).mean()
    std = close.tail(period).std()
    
    upper_band = middle_band + (num_std * std)
    lower_band = middle_band - (num_std * std)
    
    current_price = close.iloc[-1]
    
    # Normalize position within bands
    if upper_band == lower_band:
        return 0.0
    
    bb_position = (current_price - middle_band) / (upper_band - middle_band)
    return float(bb_position)


def getATR(data, period=14):
    """
    Average True Range: measures volatility
    """
    checkData(data)
    
    # Check if we have High/Low columns
    if "High" not in data.columns or "Low" not in data.columns:
        # Fallback to using Close volatility
        return getVolatility(data, window=period) * 100
    
    high = normalize_series(data, "High")
    low = normalize_series(data, "Low")
    close = normalize_series(data, "Close")
    
    close_prev = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    return float(atr.iloc[-1])


def getVolumeRatio(data, period=20):
    """
    Current volume vs average volume
    Returns 1.0 if volume data not available
    """
    if "Volume" not in data.columns:
        return 1.0
    
    volume = normalize_series(data, "Volume")
    
    current_vol = volume.iloc[-1]
    avg_vol = volume.tail(period).mean()
    
    if avg_vol == 0:
        return 1.0
    
    return float(current_vol / avg_vol)


def getPricePosition(data, period=252):
    """
    Where current price sits in recent range (0 to 1)
    0 = at low, 1 = at high, 0.5 = middle
    """
    checkData(data)
    close = normalize_series(data, "Close")
    
    window_data = close.tail(period)
    
    if len(window_data) < 2:
        return 0.5
    
    low = window_data.min()
    high = window_data.max()
    current = close.iloc[-1]
    
    if high == low:
        return 0.5
    
    position = (current - low) / (high - low)
    return float(position)