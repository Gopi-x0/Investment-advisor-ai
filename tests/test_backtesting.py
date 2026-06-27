import pandas as pd
import pytest

from backtesting import calculate_max_drawdown, run_signal_backtest
from config import CostAssumptions


def test_calculate_max_drawdown():
    equity = pd.Series([1.0, 1.2, 0.9, 1.1])

    assert calculate_max_drawdown(equity) == pytest.approx(-0.25)


def test_run_signal_backtest_includes_cost_drag_and_metrics():
    prices = pd.Series([100.0, 110.0, 105.0, 115.0])
    signals = pd.Series([1, 1, 0, -1])

    result = run_signal_backtest(prices, signals, CostAssumptions(transaction_cost_bps=10, slippage_bps=5))

    assert len(result.equity_curve) == 4
    assert result.metrics["turnover"] > 0
    assert result.metrics["total_cost_rate"] == pytest.approx(0.0015)
    assert "max_drawdown" in result.metrics
