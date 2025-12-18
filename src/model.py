from indicators import getMovingAverages
from advisor import movingAvg_Recommandation

def run_model(data):
    """
    Runs the investment advisor model.
    Returns a structured recommendation.
    """

    currentPrice, shortMA, longMA = getMovingAverages(data, 20, 50)
    signal = movingAvg_Recommandation(currentPrice, shortMA, longMA)

    return {
        "signal": signal,
        "current_price": currentPrice,
        "short_ma": shortMA,
        "long_ma": longMA,
        "reason": "Based on 20-day and 50-day moving averages"
    }
