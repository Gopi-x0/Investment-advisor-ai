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


def average_pairwise_correlation(price_frame):
    corr = correlation_matrix(price_frame).dropna(how="all", axis=0).dropna(how="all", axis=1)
    if corr.shape[0] <= 1:
        return 0.0
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.nanmean(corr.to_numpy()[mask]))


def highest_correlation_pair(price_frame):
    corr = correlation_matrix(price_frame)
    if corr.shape[0] <= 1:
        return {"pair": None, "correlation": 0.0}

    best_pair = None
    best_value = -np.inf
    columns = list(corr.columns)
    for i, first in enumerate(columns):
        for second in columns[i + 1 :]:
            value = corr.loc[first, second]
            if pd.notna(value) and value > best_value:
                best_value = float(value)
                best_pair = [first, second]

    if best_pair is None:
        return {"pair": None, "correlation": 0.0}
    return {"pair": best_pair, "correlation": best_value}


def weight_concentration(weights, columns=None):
    weights = normalize_weights(weights, columns)
    hhi = float((weights**2).sum())
    effective_positions = 0.0 if hhi == 0 else float(1 / hhi)
    max_symbol = str(weights.idxmax()) if not weights.empty else None
    max_weight = float(weights.max()) if not weights.empty else 0.0
    return {
        "hhi": hhi,
        "effective_positions": effective_positions,
        "max_weight_symbol": max_symbol,
        "max_weight": max_weight,
    }


def risk_contribution(price_frame, weights=None):
    prices = pd.DataFrame(price_frame).dropna(how="all")
    returns = price_returns(prices).fillna(0.0)
    weights = normalize_weights(weights, prices.columns)
    if returns.empty:
        return pd.Series(0.0, index=weights.index)

    covariance = returns.cov() * 252
    portfolio_variance = float(weights.T @ covariance @ weights)
    if portfolio_variance <= 0 or pd.isna(portfolio_variance):
        return pd.Series(0.0, index=weights.index)

    contribution = weights * covariance.dot(weights) / portfolio_variance
    return contribution.reindex(weights.index).fillna(0.0)


def portfolio_diagnostics(price_frame, weights=None):
    prices = pd.DataFrame(price_frame).dropna(how="all")
    weights = normalize_weights(weights, prices.columns)
    summary = portfolio_summary(prices, weights)
    concentration = weight_concentration(weights)
    risk_contrib = risk_contribution(prices, weights)
    highest_pair = highest_correlation_pair(prices)
    annual_volatility = summary["annualized_volatility"]
    top_risk_symbol = str(risk_contrib.idxmax()) if not risk_contrib.empty else None
    top_risk_pct = float(risk_contrib.max()) if not risk_contrib.empty else 0.0

    warnings = []
    if concentration["max_weight"] > 0.35:
        warnings.append(
            f"{concentration['max_weight_symbol']} is {concentration['max_weight']:.0%} of the portfolio."
        )
    if concentration["effective_positions"] < 3 and len(weights) >= 3:
        warnings.append("Portfolio is concentrated; effective position count is below 3.")
    if summary["diversification_score"] < 0.25 and len(weights) >= 2:
        warnings.append("Diversification score is low because holdings move similarly.")
    if annual_volatility > 0.30:
        warnings.append(f"Portfolio annualized volatility is high at {annual_volatility:.0%}.")
    if highest_pair["pair"] and highest_pair["correlation"] > 0.85:
        warnings.append(
            f"{highest_pair['pair'][0]} and {highest_pair['pair'][1]} are highly correlated."
        )
    if top_risk_pct > 0.50 and len(weights) >= 2:
        warnings.append(f"{top_risk_symbol} contributes more than half of portfolio variance.")

    return {
        "weights": weights.to_dict(),
        "cumulative_return": summary["cumulative_return"],
        "annualized_volatility": annual_volatility,
        "max_drawdown": summary["max_drawdown"],
        "diversification_score": summary["diversification_score"],
        "average_correlation": average_pairwise_correlation(prices),
        "highest_correlation_pair": highest_pair,
        "concentration_hhi": concentration["hhi"],
        "effective_positions": concentration["effective_positions"],
        "max_weight_symbol": concentration["max_weight_symbol"],
        "max_weight": concentration["max_weight"],
        "risk_contribution": risk_contrib.to_dict(),
        "top_risk_symbol": top_risk_symbol,
        "top_risk_contribution": top_risk_pct,
        "warnings": warnings,
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
