import pandas as pd

from ml_dataset import FEATURE_COLUMNS, build_feature_dataset


def test_feature_columns_are_unique():
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


def test_build_feature_dataset_returns_empty_schema_when_not_enough_rows():
    data = pd.DataFrame(
        {
            "Close": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Volume": [1000, 1100, 1200],
        }
    )

    result = build_feature_dataset(data)

    assert result.empty
    assert list(result.columns) == FEATURE_COLUMNS
    assert result.index.name == "date"


def test_feature_columns_include_phase_two_features():
    assert "return_60d" in FEATURE_COLUMNS
    assert "volatility_60d" in FEATURE_COLUMNS
    assert "drawdown_from_52w_high" in FEATURE_COLUMNS
