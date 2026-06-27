import pandas as pd

from ml_labels import create_TBL, create_TBL_volatility_adjustment, create_labels, create_normal_labels


def test_create_normal_labels_marks_buy_and_sell_moves():
    data = pd.DataFrame({"Close": [100.0, 110.0, 90.0]})

    labels = create_normal_labels(data, lookahead_days=1, threshold=0.05)

    assert labels.tolist() == [1, -1]


def test_create_tbl_returns_empty_series_when_no_future_window_exists():
    data = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})

    labels = create_TBL(data, lookahead_days=5)

    assert labels.empty


def test_create_tbl_volatility_adjustment_returns_empty_series_when_no_future_window_exists():
    data = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})

    labels = create_TBL_volatility_adjustment(data, lookahead_days=5)

    assert labels.empty


def test_create_labels_includes_hold_only_for_valid_future_rows():
    data = pd.DataFrame({"Close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]})

    labels = create_labels(data, lookahead_days=1, volatility_window=2, k=10.0, label_policy="three_class")

    assert set(labels.unique()) == {0}
    assert labels.index.max() < data.index.max()


def test_create_labels_aligns_to_date_column_when_present():
    dates = pd.date_range("2024-01-01", periods=6)
    data = pd.DataFrame(
        {
            "Date": dates,
            "Close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        }
    )

    labels = create_labels(data, lookahead_days=1, volatility_window=2, k=10.0, label_policy="three_class")

    assert labels.index.equals(pd.Index(dates[2:-1]))
