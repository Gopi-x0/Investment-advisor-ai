import pandas as pd
import pytest

from portfolio import (
    correlation_matrix,
    diversification_score,
    efficient_frontier,
    normalize_weights,
    portfolio_monte_carlo,
    portfolio_returns,
    portfolio_summary,
    portfolio_volatility,
    risk_parity_weights,
)


def price_frame():
    return pd.DataFrame(
        {
            "AAA": [100.0, 101.0, 102.0, 103.0, 104.0],
            "BBB": [50.0, 49.0, 50.0, 51.0, 52.0],
            "CCC": [200.0, 202.0, 201.0, 203.0, 206.0],
        }
    )


def test_normalize_weights_defaults_to_equal_weights():
    weights = normalize_weights(None, ["AAA", "BBB"])

    assert weights.to_dict() == {"AAA": 0.5, "BBB": 0.5}


def test_portfolio_returns_and_volatility():
    returns = portfolio_returns(price_frame())

    assert len(returns) == 4
    assert portfolio_volatility(price_frame()) >= 0


def test_correlation_and_diversification_score():
    corr = correlation_matrix(price_frame())

    assert corr.shape == (3, 3)
    assert 0 <= diversification_score(price_frame()) <= 1


def test_risk_parity_weights_sum_to_one():
    weights = risk_parity_weights(price_frame())

    assert weights.sum() == pytest.approx(1.0)


def test_portfolio_summary_contains_core_metrics():
    summary = portfolio_summary(price_frame())

    assert "cumulative_return" in summary
    assert "diversification_score" in summary


def test_efficient_frontier_and_portfolio_monte_carlo_shapes():
    frontier = efficient_frontier(price_frame(), n_portfolios=20)
    paths = portfolio_monte_carlo(price_frame(), horizon_days=10, n_simulations=25)

    assert len(frontier) == 20
    assert paths.shape == (10, 25)
