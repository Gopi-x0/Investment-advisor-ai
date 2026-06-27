import pandas as pd

from config import DEFAULT_ANALYSIS_CONFIG
from ml_pipeline import predict_latest_signal, save_model_artifact, walk_forward_train


def training_frame(rows=36):
    X = pd.DataFrame(
        {
            "a": range(rows),
            "b": [value % 3 for value in range(rows)],
            "c": [1 if value % 2 else 0 for value in range(rows)],
        },
        index=pd.RangeIndex(rows),
    )
    y = pd.Series([1, 0, -1] * (rows // 3), index=X.index, name="label")
    return X, y


def test_walk_forward_train_returns_oos_predictions_and_final_model():
    X, y = training_frame()

    result = walk_forward_train(
        X,
        y,
        model_name="logistic_regression",
        n_splits=3,
        top_n_features=2,
        config=DEFAULT_ANALYSIS_CONFIG,
    )

    assert result.metrics["mean_accuracy"] >= 0
    assert result.metrics["mean_f1"] >= 0
    assert len(result.fold_results) == 3
    assert len(result.oos_predictions) > 0
    assert len(result.selected_features) == 2


def test_predict_latest_signal_returns_signal_and_probabilities():
    X, y = training_frame()
    result = walk_forward_train(X, y, model_name="logistic_regression", n_splits=3, top_n_features=2)

    prediction = predict_latest_signal(result.model, X.iloc[[-1]])

    assert prediction["signal"] in {"BUY", "HOLD", "SELL"}
    assert isinstance(prediction["probabilities"], dict)


def test_save_model_artifact_writes_metadata(tmp_path):
    X, y = training_frame()
    result = walk_forward_train(X, y, model_name="logistic_regression", n_splits=3, top_n_features=2)

    path = save_model_artifact(result, tmp_path / "model.joblib")

    assert path.exists()
