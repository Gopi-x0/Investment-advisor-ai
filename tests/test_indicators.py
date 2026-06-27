import pandas as pd
import pytest

from indicators import getMomentum, getMovingAverages, getPricePosition, getVolumeRatio


def price_data(count=60):
    close = pd.Series(range(1, count + 1), dtype=float)
    return pd.DataFrame(
        {
            "Close": close,
            "High": close + 1,
            "Low": close - 1,
            "Volume": 1000,
        }
    )


def test_get_moving_averages_uses_recent_windows():
    current_price, short_ma, long_ma = getMovingAverages(price_data(), short_range=20, long_range=50)

    assert current_price == 60.0
    assert short_ma == pytest.approx(50.5)
    assert long_ma == pytest.approx(35.5)


def test_get_momentum_returns_percent_change_over_window():
    momentum = getMomentum(price_data(), n_days=10)

    assert momentum == pytest.approx(0.2)


def test_get_volume_ratio_and_price_position_have_stable_baselines():
    data = price_data()

    assert getVolumeRatio(data, period=20) == pytest.approx(1.0)
    assert getPricePosition(data, period=60) == pytest.approx(1.0)
