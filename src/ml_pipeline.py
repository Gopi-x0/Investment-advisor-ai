from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import AnalysisConfig, DEFAULT_ANALYSIS_CONFIG

SIGNAL_NAMES = {
    -1: "SELL",
    0: "HOLD",
    1: "BUY",
}


@dataclass
class WalkForwardResult:
    model: Pipeline
    selected_features: list[str]
    metrics: dict
    fold_results: list[dict]
    oos_predictions: pd.Series
    metadata: dict


def build_training_pipeline(model_name="random_forest", top_n_features=15, n_features=None):
    k = "all" if n_features is None else min(top_n_features, n_features)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("feature_selection", SelectKBest(score_func=f_classif, k=k)),
            ("model", _build_classifier(model_name)),
        ]
    )


def _build_classifier(model_name):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42)
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=40,
            max_features="sqrt",
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def walk_forward_train(
    X,
    y,
    model_name="random_forest",
    n_splits=5,
    top_n_features=15,
    config: AnalysisConfig = DEFAULT_ANALYSIS_CONFIG,
):
    if len(X) <= n_splits:
        raise ValueError(f"Need more than {n_splits} samples for walk-forward validation.")
    if y.nunique() < 2:
        raise ValueError("Need at least two label classes for ML training.")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    oos_predictions = pd.Series(index=X.index, dtype=float, name="prediction")
    class_labels = sorted(int(value) for value in y.unique())

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline = build_training_pipeline(model_name, top_n_features, n_features=X_train.shape[1])
        pipeline.fit(X_train, y_train)
        y_pred = pd.Series(pipeline.predict(X_test), index=X_test.index)
        oos_predictions.loc[X_test.index] = y_pred

        precision, recall, f1_by_class, support = precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=class_labels,
            zero_division=0,
        )

        fold_results.append(
            {
                "fold": fold,
                "train_start": str(X_train.index[0]),
                "train_end": str(X_train.index[-1]),
                "test_start": str(X_test.index[0]),
                "test_end": str(X_test.index[-1]),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, y_pred, labels=class_labels).tolist(),
                "precision_by_class": precision.tolist(),
                "recall_by_class": recall.tolist(),
                "f1_by_class": f1_by_class.tolist(),
                "support_by_class": support.tolist(),
            }
        )

    final_model = build_training_pipeline(model_name, top_n_features, n_features=X.shape[1])
    final_model.fit(X, y)
    selected_features = get_selected_features(final_model, list(X.columns))

    metrics = {
        "mean_accuracy": float(np.mean([fold["accuracy"] for fold in fold_results])),
        "std_accuracy": float(np.std([fold["accuracy"] for fold in fold_results])),
        "mean_f1": float(np.mean([fold["f1_weighted"] for fold in fold_results])),
        "classes": class_labels,
        "class_names": [SIGNAL_NAMES.get(value, str(value)) for value in class_labels],
    }

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "feature_columns": list(X.columns),
        "selected_features": selected_features,
        "price_policy": config.price_policy,
        "label_policy": config.label_policy,
        "lookahead_days": config.lookahead_days,
        "transaction_cost_bps": config.costs.transaction_cost_bps,
        "slippage_bps": config.costs.slippage_bps,
    }
    return WalkForwardResult(final_model, selected_features, metrics, fold_results, oos_predictions.dropna(), metadata)


def get_selected_features(pipeline, feature_names):
    selector = pipeline.named_steps.get("feature_selection")
    if selector is None or selector.k == "all":
        return list(feature_names)
    mask = selector.get_support()
    return [feature for feature, keep in zip(feature_names, mask) if keep]


def predict_latest_signal(pipeline, latest_features):
    raw_prediction = int(pipeline.predict(latest_features)[0])
    probabilities = {}
    model = pipeline.named_steps.get("model")
    if hasattr(model, "predict_proba"):
        proba = pipeline.predict_proba(latest_features)[0]
        probabilities = {
            SIGNAL_NAMES.get(int(label), str(label)): float(value)
            for label, value in zip(pipeline.classes_, proba)
        }
    return {
        "label": raw_prediction,
        "signal": SIGNAL_NAMES.get(raw_prediction, str(raw_prediction)),
        "probabilities": probabilities,
    }


def permutation_importance_summary(pipeline, X, y, n_repeats=5):
    result = permutation_importance(pipeline, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def save_model_artifact(result: WalkForwardResult, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result.model,
            "metadata": result.metadata,
            "metrics": result.metrics,
        },
        path,
    )
    return path
