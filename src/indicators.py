import pandas as pd
import numpy as np

def getMovingAverages(data, shortRange, longRange):
    checkData(data)

    shortMA = float(data["Close"].tail(shortRange).mean())
    longMA = float(data["Close"].tail(longRange).mean())
    currentPrice = float(data["Close"].iloc[-1])

    return currentPrice, shortMA, longMA

def getMomentum(data, nDays): 
    checkData(data)
    
    currentPrice = data["Close"].iloc[-1]
    pastPrice = data["Close"].iloc[-nDays-1]

    momentum = (currentPrice - pastPrice) / pastPrice
    return momentum

def getTrendStrength(data, longRange): 
    checkData(data)
    
    currentPrice = data["Close"].iloc[-1]
    longMA = float(data["Close"].tail(longRange).mean())

    trendStrength = (currentPrice - longMA) / longMA
    return trendStrength

##could use pct_change as that is faster 
def getReturns(data): 
    prices = data["Close"].values
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


def checkData(data):
    if data is None or data.empty:
        raise ValueError("Data is empty or None")
    
