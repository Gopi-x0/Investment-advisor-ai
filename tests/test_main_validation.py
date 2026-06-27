import pandas as pd
import pytest

from main import validate_ml_inputs
from ml_dataset import FEATURE_COLUMNS


def valid_dataset(rows=6):
    data = pd.DataFrame({column: range(rows) for column in FEATURE_COLUMNS})
    data["label"] = [1, -1] * (rows // 2) + ([1] if rows % 2 else [])
    return data


def test_validate_ml_inputs_accepts_valid_dataset():
    validate_ml_inputs(valid_dataset(rows=6), FEATURE_COLUMNS, n_splits=5)


def test_validate_ml_inputs_rejects_empty_dataset():
    dataset = pd.DataFrame(columns=FEATURE_COLUMNS + ["label"])

    with pytest.raises(ValueError, match="No labeled feature rows"):
        validate_ml_inputs(dataset, FEATURE_COLUMNS, n_splits=5)


def test_validate_ml_inputs_rejects_missing_label_column():
    dataset = pd.DataFrame({column: range(6) for column in FEATURE_COLUMNS})

    with pytest.raises(ValueError, match="missing the 'label' column"):
        validate_ml_inputs(dataset, FEATURE_COLUMNS, n_splits=5)


def test_validate_ml_inputs_rejects_missing_feature_column():
    dataset = valid_dataset(rows=6).drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="missing feature columns"):
        validate_ml_inputs(dataset, FEATURE_COLUMNS, n_splits=5)


def test_validate_ml_inputs_rejects_too_few_samples():
    with pytest.raises(ValueError, match="Need more than 5 labeled samples"):
        validate_ml_inputs(valid_dataset(rows=5), FEATURE_COLUMNS, n_splits=5)


def test_validate_ml_inputs_rejects_one_class_labels():
    dataset = valid_dataset(rows=6)
    dataset["label"] = 1

    with pytest.raises(ValueError, match="at least two label classes"):
        validate_ml_inputs(dataset, FEATURE_COLUMNS, n_splits=5)
