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

FEATURE_COLUMNS = [
    "current_price", "short_ma", "long_ma", "momentum", "volatility",
    "trend_strength", "sharpe_ratio",
    "price_above_short_ma", "price_above_long_ma", "short_above_long_ma", "ma_spread",
    "rsi", "macd_histogram", "bb_position", "atr", "volume_ratio",
    "price_position_52w", "momentum_5d", "momentum_20d",
    "return_1d", "return_5d", "return_20d", "return_60d",
    "volatility_10d", "volatility_60d", "ma_slope_20d",
    "drawdown_from_52w_high", "volume_zscore_20d"
]


def _series(data, column):
    value = data[column]
    if isinstance(value, pd.DataFrame):
        return value.iloc[:, 0]
    return value


def _safe_pct_change(close, days):
    if len(close) < days + 1:
        return 0.0
    past_price = close.iloc[-days - 1]
    if past_price == 0:
        return 0.0
    return float((close.iloc[-1] - past_price) / past_price)


def _volume_zscore(data, period=20):
    if "Volume" not in data.columns:
        return 0.0
    volume = _series(data, "Volume").tail(period)
    std = volume.std()
    if std == 0 or pd.isna(std):
        return 0.0
    return float((volume.iloc[-1] - volume.mean()) / std)

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

            close = _series(historical_data, "Close")
            return_1d = _safe_pct_change(close, 1)
            return_5d = _safe_pct_change(close, 5)
            return_20d = _safe_pct_change(close, 20)
            return_60d = _safe_pct_change(close, 60)
            volatility_10d = getVolatility(historical_data, window=10)
            volatility_60d = getVolatility(historical_data, window=60)

            ma_20 = close.rolling(20).mean()
            ma_slope_20d = 0.0
            if len(ma_20.dropna()) > 20 and ma_20.iloc[-21] != 0:
                ma_slope_20d = float((ma_20.iloc[-1] - ma_20.iloc[-21]) / ma_20.iloc[-21])

            high_52w = close.tail(252).max()
            drawdown_from_52w_high = 0.0 if high_52w == 0 else float((current_price / high_52w) - 1)
            volume_zscore_20d = _volume_zscore(historical_data, period=20)
            
            # === DERIVED FEATURES ===
            price_above_short_ma = 1.0 if current_price > short_ma else 0.0
            price_above_long_ma = 1.0 if current_price > long_ma else 0.0
            short_above_long_ma = 1.0 if short_ma > long_ma else 0.0
            ma_spread = (short_ma - long_ma) / long_ma
            
            rows.append({
                "date": data["Date"].iloc[i] if "Date" in data.columns else data.index[i],
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
                "return_1d": return_1d,
                "return_5d": return_5d,
                "return_20d": return_20d,
                "return_60d": return_60d,
                "volatility_10d": volatility_10d,
                "volatility_60d": volatility_60d,
                "ma_slope_20d": ma_slope_20d,
                "drawdown_from_52w_high": drawdown_from_52w_high,
                "volume_zscore_20d": volume_zscore_20d,
                # Derived
                "price_above_short_ma": price_above_short_ma,
                "price_above_long_ma": price_above_long_ma,
                "short_above_long_ma": short_above_long_ma,
                "ma_spread": ma_spread,
            })
            
        except Exception as e:
            print(f"Warning: Could not calculate features at index {i}: {e}")
            continue
    
    if rows:
        df = pd.DataFrame(rows).set_index("date")
    else:
        df = pd.DataFrame(columns=FEATURE_COLUMNS)
        df.index.name = "date"
    
    print(f"Feature dataset built: {len(df)} samples, {len(df.columns)} features")
    print(f"Features: {list(df.columns)}")
    
    return df
