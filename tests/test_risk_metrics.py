import pandas as pd
import pytest

from risk_metrics import (
    annualized_volatility,
    conditional_value_at_risk,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarize_return_risk,
    value_at_risk,
)


def test_core_risk_metrics_return_floats():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])

    assert annualized_volatility(returns) > 0
    assert isinstance(sharpe_ratio(returns), float)
    assert isinstance(sortino_ratio(returns), float)
    assert value_at_risk(returns) < 0
    assert conditional_value_at_risk(returns) <= value_at_risk(returns)


def test_max_drawdown_from_price_series():
    prices = pd.Series([100.0, 120.0, 90.0, 110.0])

    assert max_drawdown(prices) == pytest.approx(-0.25)


def test_summarize_return_risk_includes_expected_keys():
    prices = pd.Series([100.0, 101.0, 99.0, 102.0, 103.0])

    summary = summarize_return_risk(prices)

    assert "annualized_volatility" in summary
    assert "conditional_value_at_risk_95" in summary
