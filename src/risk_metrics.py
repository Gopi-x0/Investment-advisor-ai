import numpy as np
import pandas as pd


def to_returns(prices_or_returns, input_type="prices"):
    series = pd.Series(prices_or_returns).dropna().astype(float)
    if input_type == "returns":
        return series
    return series.pct_change().dropna()


def annualized_volatility(returns, periods_per_year=252):
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    returns = pd.Series(returns).dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess_daily_return = returns.mean() - (risk_free_rate / periods_per_year)
    return float(excess_daily_return / returns.std() * np.sqrt(periods_per_year))


def sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    downside_std = downside.std()
    if len(downside) < 2 or downside_std == 0 or pd.isna(downside_std):
        return 0.0
    excess_daily_return = returns.mean() - (risk_free_rate / periods_per_year)
    return float(excess_daily_return / downside_std * np.sqrt(periods_per_year))


def max_drawdown(prices_or_equity):
    series = pd.Series(prices_or_equity).dropna().astype(float)
    if series.empty:
        return 0.0
    running_max = series.cummax()
    drawdown = series / running_max - 1
    return float(drawdown.min())


def value_at_risk(returns, confidence=0.95):
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0
    return float(np.quantile(returns, 1 - confidence))


def conditional_value_at_risk(returns, confidence=0.95):
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0
    var = value_at_risk(returns, confidence)
    tail = returns[returns <= var]
    if tail.empty:
        return var
    return float(tail.mean())


def summarize_return_risk(prices_or_returns, input_type="prices", risk_free_rate=0.02):
    returns = to_returns(prices_or_returns, input_type=input_type)
    series = pd.Series(prices_or_returns).dropna().astype(float)
    return {
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate=risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate=risk_free_rate),
        "max_drawdown": max_drawdown((1 + returns).cumprod() if input_type == "returns" else series),
        "value_at_risk_95": value_at_risk(returns, confidence=0.95),
        "conditional_value_at_risk_95": conditional_value_at_risk(returns, confidence=0.95),
    }
