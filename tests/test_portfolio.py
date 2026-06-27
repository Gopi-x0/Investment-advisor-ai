import pandas as pd
import pytest

from portfolio import (
    average_pairwise_correlation,
    correlation_matrix,
    diversification_score,
    efficient_frontier,
    highest_correlation_pair,
    normalize_weights,
    portfolio_diagnostics,
    portfolio_monte_carlo,
    portfolio_returns,
    portfolio_summary,
    portfolio_volatility,
    risk_contribution,
    risk_parity_weights,
    weight_concentration,
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


def test_portfolio_diagnostics_include_concentration_and_risk_contribution():
    weights = pd.Series({"AAA": 0.70, "BBB": 0.20, "CCC": 0.10})

    diagnostics = portfolio_diagnostics(price_frame(), weights)
    contribution = risk_contribution(price_frame(), weights)
    concentration = weight_concentration(weights)

    assert diagnostics["max_weight_symbol"] == "AAA"
    assert diagnostics["effective_positions"] == pytest.approx(concentration["effective_positions"])
    assert contribution.sum() == pytest.approx(1.0)
    assert diagnostics["warnings"]


def test_pairwise_correlation_helpers_return_summary_values():
    pair = highest_correlation_pair(price_frame())

    assert -1 <= average_pairwise_correlation(price_frame()) <= 1
    assert pair["pair"] is not None
    assert -1 <= pair["correlation"] <= 1


def test_efficient_frontier_and_portfolio_monte_carlo_shapes():
    frontier = efficient_frontier(price_frame(), n_portfolios=20)
    paths = portfolio_monte_carlo(price_frame(), horizon_days=10, n_simulations=25)

    assert len(frontier) == 20
    assert paths.shape == (10, 25)
