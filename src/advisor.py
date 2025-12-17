def movingAvg_Recommandation(currentPrice, shortMA, longMA): 
    if shortMA > longMA and currentPrice > longMA:
        return "BUY"
    if shortMA < longMA and currentPrice < longMA:
        return "SELL"
    return "HOLD"