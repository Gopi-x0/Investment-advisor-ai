from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CostAssumptions


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    strategy_returns: pd.Series
    metrics: dict


def calculate_max_drawdown(equity_curve):
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return float(drawdown.min())


def run_signal_backtest(prices, signals, costs: CostAssumptions | None = None):
    costs = costs or CostAssumptions()
    aligned = pd.concat(
        [
            pd.Series(prices, name="price").astype(float),
            pd.Series(signals, name="signal").astype(float),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        empty = pd.Series(dtype=float)
        return BacktestResult(empty, empty, {"cumulative_return": 0.0, "buy_hold_return": 0.0})

    returns = aligned["price"].pct_change().fillna(0.0)
    position = aligned["signal"].clip(-1, 1)
    trade_size = position.diff().abs().fillna(position.abs())
    cost_drag = trade_size * costs.total_cost_rate
    strategy_returns = position.shift(1).fillna(0.0) * returns - cost_drag
    equity_curve = (1 + strategy_returns).cumprod()

    buy_hold_curve = (1 + returns).cumprod()
    annualized_volatility = float(strategy_returns.std() * np.sqrt(252)) if len(strategy_returns) > 1 else 0.0
    sharpe_ratio = 0.0
    if strategy_returns.std() and not pd.isna(strategy_returns.std()):
        sharpe_ratio = float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(252))

    active_returns = strategy_returns[position.shift(1).fillna(0.0) != 0]
    hit_rate = float((active_returns > 0).mean()) if len(active_returns) else 0.0

    metrics = {
        "cumulative_return": float(equity_curve.iloc[-1] - 1),
        "buy_hold_return": float(buy_hold_curve.iloc[-1] - 1),
        "max_drawdown": calculate_max_drawdown(equity_curve),
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "hit_rate": hit_rate,
        "turnover": float(trade_size.sum()),
        "total_cost_rate": costs.total_cost_rate,
    }
    return BacktestResult(equity_curve, strategy_returns, metrics)
