def movingAvg_Recommandation(currentPrice, shortMA, longMA): 
    if shortMA > longMA and currentPrice > longMA:
        return "BUY"
    if shortMA < longMA and currentPrice < longMA:
        return "SELL"
    return "HOLD"

def momentum_Recommendation(momentum, threshold=0.0):
    if momentum > threshold:
        return "BUY"
    if momentum < -threshold:
        return "SELL"
    return "HOLD"


def trend_strength_Recommendation(trendStrength, threshold=0.02):
    if trendStrength > threshold:
        return "BUY"
    if trendStrength < -threshold:
        return "SELL"
    return "HOLD"


def volatility_Recommendation(volatility, maxVolatility=0.03):
    if volatility <= maxVolatility:
        return "BUY"   # acceptable risk
    return "HOLD"      # too risky to buy


def sharpe_Recommendation(sharpeRatio, minSharpe=0.5):
    if sharpeRatio >= minSharpe:
        return "BUY"
    if sharpeRatio < 0:
        return "SELL"
    return "HOLD"

def overall_Recommendation(currentPrice, shortMA, longMA, momentum, trendStrength, volatility, sharpeRatio):
    #map/dictionary of key(recommandation topic) and whether to buy/sell/hold based on that rule
    recommendations = {
        "moving_avg": movingAvg_Recommandation(currentPrice, shortMA, longMA),
        "momentum": momentum_Recommendation(momentum),
        "trend_strength": trend_strength_Recommendation(trendStrength),
        "volatility": volatility_Recommendation(volatility),
        "sharpe": sharpe_Recommendation(sharpeRatio),
    }

    # Scoring map
    score_map = {"BUY": 1, "HOLD": 0, "SELL": -1}
    total_score = sum(score_map[r] for r in recommendations.values())

    # Final decision thresholds
    if total_score >= 2:
        final_decision = "BUY"
    elif total_score <= -2:
        final_decision = "SELL"
    else:
        final_decision = "HOLD"

    return final_decision, total_score, recommendations