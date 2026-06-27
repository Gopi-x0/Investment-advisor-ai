from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
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


def select_top_features(X, y, top_n=10):
    """
    Select top N most important features using a quick Random Forest.
    
    Returns:
    - List of top feature names
    """
    print(f"\nSelecting top {top_n} features...")
    
    # Quick RF to get feature importances
    rf_temp = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5,
        random_state=42, 
        n_jobs=-1
    )
    rf_temp.fit(X, y)
    
    # Get importances
    importances = pd.Series(rf_temp.feature_importances_, index=X.columns)
    top_features = importances.nlargest(top_n).index.tolist()
    
    print(f"Top {top_n} features selected:")
    for i, feat in enumerate(top_features, 1):
        print(f"  {i:2d}. {feat:25s} ({importances[feat]:.4f})")
    
    removed = set(X.columns) - set(top_features)
    if removed:
        print(f"\nRemoved {len(removed)} weak features:")
        for feat in removed:
            print(f"  - {feat:25s} ({importances[feat]:.4f})")
    print("")
    return top_features

def build_model(use_gradient_boost=False, final=False):
    if not use_gradient_boost: 
        return RandomForestClassifier(
            n_estimators=200,          
            max_depth=10,                
            min_samples_split=50,       
            min_samples_leaf=40,        
            max_features='sqrt',        
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,                  
            oob_score=final             
        )
    else:
        return GradientBoostingClassifier(
            n_estimators=200,           
            learning_rate=0.05,
            max_depth=3,                
            min_samples_split=50,       
            min_samples_leaf=40,        
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )

def scale_split(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler


def train_model(X, y, use_feature_selection=True, top_n_features=10, use_gradient_boost = False): 
    # Feature selection
    if use_feature_selection and len(X.columns) > top_n_features:
        selected_features = select_top_features(X, y, top_n=top_n_features)
        X = X[selected_features]
    else:
        selected_features = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print("\nTraining...")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples:  {len(X_test)}")
    print(f"  Features:         {len(selected_features)}")
    
    X_train_scaled, X_test_scaled, scaler = scale_split(X_train, X_test)
    model = build_model(use_gradient_boost, final = True)

    model.fit(X_train_scaled, y_train)
    
    if use_gradient_boost:
        print("Training complete!")
    else:
        print(f"Training complete! (OOB score: {model.oob_score_:.4f})")

    return model, X_test_scaled, y_test, scaler, selected_features

def train_model_walk_forward(X, y, n_splits=5, use_feature_selection=True, top_n_features=10, use_gradient_boost=False):
    
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION ({n_splits} splits)".center(60, "="))
    print('='*60)
    
    if use_feature_selection and len(X.columns) > top_n_features:
        selected_features = select_top_features(X, y, top_n=top_n_features)
        X = X[selected_features]
    else:
        selected_features = list(X.columns)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accuracies, f1_scores, all_predictions = [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train_scaled, X_test_scaled, scaler = scale_split(X_train, X_test)
        model = build_model(use_gradient_boost)

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
        accuracies.append(accuracy)
        f1_scores.append(f1)

        print(f"Fold {fold}/{n_splits}:")
        print(f"  Train: samples {X_train.index[0]} to {X_train.index[-1]} ({len(X_train)} samples)")
        print(f"  Test:  samples {X_test.index[0]} to {X_test.index[-1]} ({len(X_test)} samples)")
        print(f"F1 Score: {f1:.4f}\n")

        all_predictions.append({
            'fold': fold,
            'y_test': y_test,
            'y_pred': y_pred,
            'accuracy': accuracy,
            'f1': f1
        })

    #combine everything trained and test
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_f1 = np.mean(f1_scores)
    
    scaler_final = StandardScaler()
    X_scaled_final = pd.DataFrame(
        scaler_final.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    final_model = build_model(use_gradient_boost, final = True)
    final_model.fit(X_scaled_final, y)

    if use_gradient_boost:
        print("Training complete!")
    else:
        print(f"Training complete! (OOB score: {final_model.oob_score_:.4f})")

    cv_results = {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_f1': mean_f1,
        'all_accuracies': accuracies,
        'all_f1_scores': f1_scores,
        'fold_predictions': all_predictions
    }

    return final_model, cv_results, selected_features, scaler_final

def predict_latest_signal(model, latest_features, scaler=None, selected_features=None):
    """
    Predict the investment signal for the latest features using the trained model.

    Parameters:
    - model: trained classifier
    - latest_features: pandas DataFrame with the latest feature set
    - scaler: fitted StandardScaler (optional)
    - selected_features: list of feature names to use (optional)

    Returns:
    - signal: predicted signal ('BUY', 'SELL', 'HOLD')
    """
    # Select features if specified
    if selected_features:
        latest_features = latest_features[selected_features]
    
    # Scale if scaler provided
    if scaler:
        latest_features = pd.DataFrame(
            scaler.transform(latest_features),
            columns=latest_features.columns,
            index=latest_features.index
        )
    
    prediction = model.predict(latest_features)[0]
    
    if prediction == 1:
        return "BUY"
    elif prediction == -1:
        return "SELL"
    else:
        return "HOLD"

def get_feature_importances(model, feature_names):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": cm
    }
