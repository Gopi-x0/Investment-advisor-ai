import pandas as pd
from indicators import (
    getMovingAverages,
    getMomentum,
    getVolatility,
    getTrendStrength
)

def build_feature_dataset(data, short_range=20, long_range=50, momentum_days=10):
    rows = []
    min_days = max(short_range, long_range, momentum_days) + 1

    for i in range(min_days, len(data)):
        # Slice data up to day i (NO FUTURE)
        historical_data = data.iloc[:i]

        current_price, short_ma, long_ma = getMovingAverages(
            historical_data, short_range, long_range
        )

        momentum = getMomentum(historical_data, momentum_days)
        volatility = getVolatility(historical_data)
        trend_strength = getTrendStrength(historical_data, long_range)

        rows.append({
            "date": data.index[i],
            "current_price": current_price,
            "short_ma": short_ma,
            "long_ma": long_ma,
            "momentum": momentum,
            "volatility": volatility,
            "trend_strength": trend_strength
        })

    return pd.DataFrame(rows).set_index("date")