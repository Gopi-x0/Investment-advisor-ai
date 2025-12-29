from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from indicators import getMovingAverages, getMomentum, getTrendStrength, getVolatility, getSharpeRatio
from advisor import overall_Recommendation

def run_ruleBasedModel(data):
    current_price, short_ma, long_ma = getMovingAverages(data, 20, 50)

    momentum = getMomentum(data, 14)
    trend_strength = getTrendStrength(data, 50)
    volatility = getVolatility(data)
    sharpe_ratio = getSharpeRatio(data)

    final_decision, score, breakdown = overall_Recommendation(
        current_price,
        short_ma,
        long_ma,
        momentum,
        trend_strength,
        volatility,
        sharpe_ratio
    )

    return {
        "signal": final_decision,
        "score": score,
        "breakdown": breakdown,
        "current_price": current_price,
        "short_ma": short_ma,
        "long_ma": long_ma,
        "reason": "Multi-factor rule-based trading model (trend, momentum, risk-adjusted returns)"
    }

def train_random_forest(X, y):
    """
    Train a Random Forest Classifier on the provided features and labels.

    Parameters:
    - X: pandas DataFrame of features
    - y: pandas Series of labels

    Returns:
    - model: trained RandomForestClassifier
    - X_test: test features
    - y_test: test labels
    - y_pred: predicted labels for test set
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def predict_latest_signal(model, latest_features):
    """
    Predict the investment signal for the latest features using the trained model.

    Parameters:
    - model: trained RandomForestClassifier
    - latest_features: pandas DataFrame with the latest feature set

    Returns:
    - signal: predicted signal ('BUY', 'SELL')
    """
    prediction = model.predict(latest_features)[0]
    
    if prediction == 1:
        return "BUY"
    elif prediction == -1:
        return "SELL"
    else:
        return "HOLD"

def get_feature_importances(model, feature_names):
    """
    Get feature importances from the trained model.

    Parameters:
    - model: trained RandomForestClassifier
    - feature_names: list of feature names

    Returns:
    - importances: pandas Series of feature importances
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    """
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0)
    }

