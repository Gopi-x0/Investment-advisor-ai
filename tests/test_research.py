import pandas as pd
import pytest

from config import AnalysisConfig
from research import build_report_warnings, validate_research_dataset
from ml_dataset import FEATURE_COLUMNS


def valid_research_dataset(rows=8):
    data = pd.DataFrame({column: range(rows) for column in FEATURE_COLUMNS})
    data["label"] = [1, 0, -1, 0] * (rows // 4)
    return data


def test_validate_research_dataset_accepts_valid_data():
    validate_research_dataset(valid_research_dataset(), n_splits=3)


def test_validate_research_dataset_rejects_missing_columns():
    dataset = valid_research_dataset().drop(columns=[FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="missing feature columns"):
        validate_research_dataset(dataset, n_splits=3)


def test_build_report_warnings_flags_imbalanced_labels():
    class Result:
        metrics = {"std_accuracy": 0.02, "mean_accuracy": 0.50, "mean_f1": 0.49}

    dataset = valid_research_dataset(rows=8)
    dataset["label"] = [1, 1, 1, 1, 1, 1, 1, 0]

    warnings = build_report_warnings(Result(), dataset)

    assert any("Label imbalance" in warning for warning in warnings)


def test_analysis_config_can_be_tuned_for_dashboard_controls():
    config = AnalysisConfig(monte_carlo_horizon_days=20, monte_carlo_simulations=50)

    assert config.monte_carlo_horizon_days == 20
    assert config.monte_carlo_simulations == 50
