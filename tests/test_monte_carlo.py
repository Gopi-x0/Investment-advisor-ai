import pandas as pd

from monte_carlo import (
    geometric_brownian_motion_simulation,
    historical_bootstrap_simulation,
    scenario_simulations,
)


def sample_returns():
    return pd.Series([0.01, -0.02, 0.015, 0.005, -0.01, 0.02])


def test_historical_bootstrap_simulation_shapes_and_metrics():
    result = historical_bootstrap_simulation(sample_returns(), start_price=100.0, horizon_days=10, n_simulations=50)

    assert result.paths.shape == (10, 50)
    assert set(result.percentiles.columns) == {"p05", "p25", "p50", "p75", "p95"}
    assert "probability_of_loss" in result.metrics
    assert 0 <= result.metrics["probability_stop_loss_hit"] <= 1


def test_geometric_brownian_motion_simulation_shapes_and_metrics():
    result = geometric_brownian_motion_simulation(sample_returns(), start_price=100.0, horizon_days=8, n_simulations=25)

    assert result.paths.shape == (8, 25)
    assert "expected_max_drawdown" in result.metrics


def test_scenario_simulations_include_named_scenarios():
    scenarios = scenario_simulations(sample_returns(), start_price=100.0, horizon_days=5, n_simulations=10)

    assert {"bootstrap", "gbm_base", "volatility_stress", "bear_market", "bull_market"} == set(scenarios)
