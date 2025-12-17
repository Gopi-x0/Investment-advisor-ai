import pandas as pd

def getMovingAverages(data, shortRange, longRange):
    if data is None or data.empty:
        return None, None, None

    shortMA = float(data["Close"].tail(shortRange).mean())
    longMA = float(data["Close"].tail(longRange).mean())
    currentPrice = float(data["Close"].iloc[-1])

    return currentPrice, shortMA, longMA
