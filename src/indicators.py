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


def checkData(data):
    if data is None or data.empty:
        raise ValueError("Data is empty or None")
    
