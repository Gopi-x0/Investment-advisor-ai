import pandas as pd
from indicators import (
    getMovingAverages,
    getMomentum,
    getVolatility,
    getTrendStrength,
    getSharpeRatio,
    getRSI,
    getMACD,
    getBollingerPosition,
    getATR,
    getVolumeRatio,
    getPricePosition
)

def build_feature_dataset(data, short_range=20, long_range=50, momentum_days=10):
    rows = []
    min_days = max(short_range, long_range, momentum_days, 60) + 1
    
    print(f"Building features from day {min_days} to {len(data)}...")
    
    for i in range(min_days, len(data)):
        # Slice data up to day i (NO FUTURE LEAKAGE!)
        historical_data = data.iloc[:i+1]
        
        try:
            # === CORE FEATURES ===
            current_price, short_ma, long_ma = getMovingAverages(
                historical_data, short_range, long_range
            )
            
            momentum = getMomentum(historical_data, momentum_days)
            volatility = getVolatility(historical_data, window=20)
            trend_strength = getTrendStrength(historical_data, long_range)
            sharpe_ratio = getSharpeRatio(historical_data, window=60)
            
            # === MOMENTUM FEATURES (Multiple Timeframes) ===
            momentum_5d = getMomentum(historical_data, 5)
            momentum_20d = getMomentum(historical_data, 20)
            
            # === TECHNICAL INDICATORS ===
            rsi = getRSI(historical_data, period=14)
            macd_histogram = getMACD(historical_data)
            bb_position = getBollingerPosition(historical_data)
            atr = getATR(historical_data)
            
            # === VOLUME ===
            volume_ratio = getVolumeRatio(historical_data)
            
            # === PRICE POSITION ===
            price_position_52w = getPricePosition(historical_data, period=252)
            
            # === DERIVED FEATURES ===
            price_above_short_ma = 1.0 if current_price > short_ma else 0.0
            price_above_long_ma = 1.0 if current_price > long_ma else 0.0
            short_above_long_ma = 1.0 if short_ma > long_ma else 0.0
            ma_spread = (short_ma - long_ma) / long_ma
            
            rows.append({
                "date": data.index[i],
                "current_price": current_price,
                "short_ma": short_ma,
                "long_ma": long_ma,
                "momentum": momentum,
                "volatility": volatility,
                "trend_strength": trend_strength,
                "sharpe_ratio": sharpe_ratio,
                # Momentum (multiple timeframes)
                "momentum_5d": momentum_5d,
                "momentum_20d": momentum_20d,
                # Technical indicators
                "rsi": rsi,
                "macd_histogram": macd_histogram,
                "bb_position": bb_position,
                "atr": atr,
                # Volume
                "volume_ratio": volume_ratio,
                # Price position
                "price_position_52w": price_position_52w,
                # Derived
                "price_above_short_ma": price_above_short_ma,
                "price_above_long_ma": price_above_long_ma,
                "short_above_long_ma": short_above_long_ma,
                "ma_spread": ma_spread,
            })
            
        except Exception as e:
            print(f"Warning: Could not calculate features at index {i}: {e}")
            continue
    
    df = pd.DataFrame(rows).set_index("date")
    
    print(f"Feature dataset built: {len(df)} samples, {len(df.columns)} features")
    print(f"✓ Features: {list(df.columns)}")
    
    return df