import pandas as pd
import numpy as np

def getMovingAverages(data, shortRange, longRange):
    checkData(data)

    shortMA = data["Close"].tail(shortRange).mean().iloc[0]
    longMA = data["Close"].tail(longRange).mean().iloc[0]
    currentPrice = data["Close"].iloc[-1].iloc[0]
    
    return currentPrice, shortMA, longMA

def getMomentum(data, nDays):
    checkData(data)
    
    currentPrice = data["Close"].iloc[-1].iloc[0]
    pastPrice = data["Close"].iloc[-nDays - 1].iloc[0]

    momentum = (currentPrice - pastPrice) / pastPrice
    return momentum

def getTrendStrength(data, longRange):
    checkData(data)

    currentPrice = data["Close"].iloc[-1].iloc[0]
    longMA = data["Close"].tail(longRange).mean().iloc[0]

    return (currentPrice - longMA) / longMA


##could use pct_change as that is faster 
def getReturns(data): 
    checkData(data)
    prices = data["Close"].iloc[:,0].values
    returns = []

    for i in range(1, len(prices)):
        dailyReturn = (prices[i] - prices[i - 1]) / prices[i - 1]
        returns.append(dailyReturn)
    
    return returns

def getVolatility(data):
    checkData(data)
    returns = getReturns(data)
    volatility = np.std(returns)
    return volatility

def getSharpeRatio(data): 
    checkData(data)
    
    returns = getReturns(data)

    avgReturn = np.mean(returns)
    volatility = np.std(returns)
    
    if volatility == 0 or volatility is None:
        return None
    
    sharpeRatio = avgReturn / volatility
    return sharpeRatio

def getATR(data, period=14):
    """
    Average True Range (Wilder)
    """
    checkData(data)

    high = data["High"]
    low = data["Low"]
    close_prev = data["Close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return float(atr.iloc[-1])

def getRSI(data, period=14):
    """
    RSI using Wilder's smoothing (industry standard)
    """
    checkData(data)

    delta = data["Close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return float(rsi.iloc[-1])


def checkData(data):
    if data is None or data.empty:
        raise ValueError("Data is empty or None")
    
