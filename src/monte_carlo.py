from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_metrics import conditional_value_at_risk, max_drawdown, value_at_risk


@dataclass(frozen=True)
class MonteCarloResult:
    paths: pd.DataFrame
    percentiles: pd.DataFrame
    metrics: dict


def historical_bootstrap_simulation(
    returns,
    start_price,
    horizon_days=60,
    n_simulations=1000,
    random_state=42,
    stop_loss=0.10,
):
    returns = pd.Series(returns).dropna().astype(float)
    if returns.empty:
        raise ValueError("Monte Carlo simulation requires at least one historical return.")

    rng = np.random.default_rng(random_state)
    sampled_returns = rng.choice(returns.to_numpy(), size=(horizon_days, n_simulations), replace=True)
    paths = _returns_to_price_paths(sampled_returns, start_price)
    return summarize_paths(paths, start_price=start_price, stop_loss=stop_loss)


def geometric_brownian_motion_simulation(
    returns,
    start_price,
    horizon_days=60,
    n_simulations=1000,
    random_state=42,
    drift_multiplier=1.0,
    volatility_multiplier=1.0,
    stop_loss=0.10,
):
    returns = pd.Series(returns).dropna().astype(float)
    if len(returns) < 2:
        raise ValueError("GBM simulation requires at least two historical returns.")

    rng = np.random.default_rng(random_state)
    mu = returns.mean() * drift_multiplier
    sigma = returns.std() * volatility_multiplier
    simulated_returns = rng.normal(mu, sigma, size=(horizon_days, n_simulations))
    paths = _returns_to_price_paths(simulated_returns, start_price)
    return summarize_paths(paths, start_price=start_price, stop_loss=stop_loss)


def scenario_simulations(returns, start_price, horizon_days=60, n_simulations=1000, random_state=42, stop_loss=0.10):
    return {
        "bootstrap": historical_bootstrap_simulation(
            returns, start_price, horizon_days, n_simulations, random_state, stop_loss
        ),
        "gbm_base": geometric_brownian_motion_simulation(
            returns, start_price, horizon_days, n_simulations, random_state, 1.0, 1.0, stop_loss
        ),
        "volatility_stress": geometric_brownian_motion_simulation(
            returns, start_price, horizon_days, n_simulations, random_state, 1.0, 1.5, stop_loss
        ),
        "bear_market": geometric_brownian_motion_simulation(
            returns, start_price, horizon_days, n_simulations, random_state, -1.0, 1.25, stop_loss
        ),
        "bull_market": geometric_brownian_motion_simulation(
            returns, start_price, horizon_days, n_simulations, random_state, 1.5, 1.0, stop_loss
        ),
    }


def summarize_paths(paths, start_price, stop_loss=0.10):
    percentiles = pd.DataFrame(
        {
            "p05": paths.quantile(0.05, axis=1),
            "p25": paths.quantile(0.25, axis=1),
            "p50": paths.quantile(0.50, axis=1),
            "p75": paths.quantile(0.75, axis=1),
            "p95": paths.quantile(0.95, axis=1),
        }
    )
    terminal_returns = paths.iloc[-1] / start_price - 1
    path_drawdowns = paths.apply(max_drawdown, axis=0)
    stop_price = start_price * (1 - stop_loss)

    metrics = {
        "expected_return": float(terminal_returns.mean()),
        "median_return": float(terminal_returns.median()),
        "probability_of_loss": float((terminal_returns < 0).mean()),
        "value_at_risk_95": value_at_risk(terminal_returns, confidence=0.95),
        "conditional_value_at_risk_95": conditional_value_at_risk(terminal_returns, confidence=0.95),
        "expected_max_drawdown": float(path_drawdowns.mean()),
        "probability_below_start": float((paths.iloc[-1] < start_price).mean()),
        "probability_stop_loss_hit": float((paths <= stop_price).any(axis=0).mean()),
    }
    return MonteCarloResult(paths=paths, percentiles=percentiles, metrics=metrics)


def _returns_to_price_paths(simulated_returns, start_price):
    cumulative = np.cumprod(1 + simulated_returns, axis=0)
    paths = pd.DataFrame(cumulative * start_price)
    paths.index = pd.RangeIndex(1, len(paths) + 1, name="day")
    return paths
