import numpy as np
import pandas as pd

from risk_metrics import annualized_volatility, max_drawdown


def normalize_weights(weights, columns=None):
    if weights is None:
        if columns is None:
            raise ValueError("columns are required when weights are not provided.")
        return pd.Series(1 / len(columns), index=columns, dtype=float)

    weights = pd.Series(weights, index=columns, dtype=float) if not isinstance(weights, pd.Series) else weights.astype(float)
    total = weights.sum()
    if total == 0:
        raise ValueError("Portfolio weights must not sum to zero.")
    return weights / total


def price_returns(price_frame):
    return pd.DataFrame(price_frame).astype(float).pct_change().dropna(how="all")


def portfolio_returns(price_frame, weights=None):
    prices = pd.DataFrame(price_frame).dropna(how="all")
    weights = normalize_weights(weights, prices.columns)
    returns = price_returns(prices).fillna(0.0)
    return returns.dot(weights)


def correlation_matrix(price_frame):
    return price_returns(price_frame).corr()


def portfolio_volatility(price_frame, weights=None, periods_per_year=252):
    returns = portfolio_returns(price_frame, weights)
    return annualized_volatility(returns, periods_per_year=periods_per_year)


def diversification_score(price_frame):
    corr = correlation_matrix(price_frame).dropna(how="all", axis=0).dropna(how="all", axis=1)
    if corr.shape[0] <= 1:
        return 0.0
    mask = ~np.eye(corr.shape[0], dtype=bool)
    avg_corr = float(corr.to_numpy()[mask].mean())
    return float(np.clip(1 - avg_corr, 0, 2) / 2)


def risk_parity_weights(price_frame):
    returns = price_returns(price_frame)
    vols = returns.std().replace(0, np.nan)
    inverse_vol = 1 / vols
    inverse_vol = inverse_vol.fillna(0.0)
    if inverse_vol.sum() == 0:
        return normalize_weights(None, returns.columns)
    return inverse_vol / inverse_vol.sum()


def portfolio_equity_curve(price_frame, weights=None):
    returns = portfolio_returns(price_frame, weights)
    return (1 + returns).cumprod()


def portfolio_summary(price_frame, weights=None):
    weights = normalize_weights(weights, pd.DataFrame(price_frame).columns)
    equity = portfolio_equity_curve(price_frame, weights)
    returns = portfolio_returns(price_frame, weights)
    return {
        "weights": weights.to_dict(),
        "cumulative_return": float(equity.iloc[-1] - 1) if not equity.empty else 0.0,
        "annualized_volatility": annualized_volatility(returns),
        "max_drawdown": max_drawdown(equity),
        "diversification_score": diversification_score(price_frame),
    }


def efficient_frontier(price_frame, n_portfolios=500, random_state=42):
    returns = price_returns(price_frame).dropna()
    if returns.empty:
        raise ValueError("Efficient frontier requires at least one return row.")

    rng = np.random.default_rng(random_state)
    rows = []
    columns = list(returns.columns)
    for _ in range(n_portfolios):
        raw_weights = rng.random(len(columns))
        weights = raw_weights / raw_weights.sum()
        portfolio_return = returns.dot(weights)
        rows.append(
            {
                "expected_return": float(portfolio_return.mean() * 252),
                "volatility": float(portfolio_return.std() * np.sqrt(252)),
                "sharpe": 0.0 if portfolio_return.std() == 0 else float(portfolio_return.mean() / portfolio_return.std() * np.sqrt(252)),
                **{f"weight_{column}": float(weight) for column, weight in zip(columns, weights)},
            }
        )
    return pd.DataFrame(rows)


def portfolio_monte_carlo(price_frame, weights=None, horizon_days=60, n_simulations=1000, random_state=42):
    prices = pd.DataFrame(price_frame).dropna(how="all")
    weights = normalize_weights(weights, prices.columns)
    returns = price_returns(prices).fillna(0.0)
    if returns.empty:
        raise ValueError("Portfolio Monte Carlo requires at least one return row.")

    rng = np.random.default_rng(random_state)
    sampled_indices = rng.integers(0, len(returns), size=(horizon_days, n_simulations))
    sampled_returns = returns.to_numpy()[sampled_indices]
    weighted_returns = np.tensordot(sampled_returns, weights.to_numpy(), axes=([2], [0]))
    paths = pd.DataFrame(np.cumprod(1 + weighted_returns, axis=0))
    paths.index = pd.RangeIndex(1, horizon_days + 1, name="day")
    return paths
