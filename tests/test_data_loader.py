import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

import data_loader


def sample_download_frame():
    data = pd.DataFrame(
        {
            "Open": ["100.0", "101.0"],
            "High": ["102.0", "103.0"],
            "Low": ["99.0", "100.0"],
            "Close": ["101.0", "102.0"],
            "Adj Close": ["101.0", "102.0"],
            "Volume": ["1000000", "1100000"],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    data.index.name = "Date"
    return data


def test_get_stock_data_rejects_blank_ticker():
    with pytest.raises(ValueError, match="Ticker must be a non-empty string"):
        data_loader.get_stock_data(" ")


def test_get_stock_data_cleans_downloaded_data(monkeypatch):
    def fake_download(ticker, period, auto_adjust):
        assert ticker == "AAPL"
        assert period == "1y"
        assert auto_adjust is False
        return sample_download_frame()

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    result = data_loader.get_stock_data("AAPL", period="1y")

    assert len(result) == 2
    assert "Date" in result.columns
    assert is_numeric_dtype(result["Close"])
    assert is_numeric_dtype(result["Volume"])
    assert result.attrs["price_policy"] == "raw_close"


def test_get_stock_data_raises_for_empty_download(monkeypatch):
    monkeypatch.setattr(data_loader.yf, "download", lambda *args, **kwargs: pd.DataFrame())

    with pytest.raises(ValueError, match="No data found"):
        data_loader.get_stock_data("MISSING")


def test_get_stock_data_wraps_download_errors(monkeypatch):
    def fake_download(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(data_loader.yf, "download", fake_download)

    with pytest.raises(ConnectionError, match="Failed to download data"):
        data_loader.get_stock_data("AAPL")


def test_get_stock_data_can_apply_adjusted_close_policy(monkeypatch):
    frame = sample_download_frame()
    frame.loc[frame.index[0], "Adj Close"] = "88.0"

    monkeypatch.setattr(data_loader.yf, "download", lambda *args, **kwargs: frame)

    result = data_loader.get_stock_data("AAPL", price_policy="adjusted_close")

    assert "Raw Close" in result.columns
    assert result.loc[0, "Close"] == 88.0
    assert result.attrs["price_policy"] == "adjusted_close"


def test_get_stock_data_flattens_yfinance_multiindex_columns(monkeypatch):
    frame = sample_download_frame()
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["AAPL"]])

    monkeypatch.setattr(data_loader.yf, "download", lambda *args, **kwargs: frame)

    result = data_loader.get_stock_data("AAPL")

    assert "Close" in result.columns
    assert is_numeric_dtype(result["Close"])
