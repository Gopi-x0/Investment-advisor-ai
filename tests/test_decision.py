import pytest

from decision import InvestorProfile, build_decision_summary


def test_decision_summary_blocks_weak_sell_signal():
    decision = build_decision_summary(
        rule_result={"signal": "SELL"},
        ml_prediction={"signal": "SELL", "probabilities": {"SELL": 0.45, "HOLD": 0.30, "BUY": 0.25}},
        model_metrics={"mean_f1": 0.28},
        backtest_metrics={
            "cumulative_return": 0.10,
            "buy_hold_return": 0.80,
            "sharpe_ratio": 0.20,
            "max_drawdown": -0.35,
        },
        monte_carlo_metrics={
            "expected_return": 0.02,
            "probability_of_loss": 0.50,
            "probability_stop_loss_hit": 0.30,
            "conditional_value_at_risk_95": -0.20,
        },
        investor_profile=InvestorProfile(account_value=20_000, max_single_position_pct=0.10),
    )

    assert decision["action"] == "Avoid Or Reduce"
    assert decision["suggested_position_value"] == 0
    assert decision["order_readiness"]["status"] == "Not order-ready"
    assert any("backtest" in item.lower() for item in decision["cautions"])


def test_decision_summary_sizes_strong_candidate_by_risk_budget():
    decision = build_decision_summary(
        rule_result={"signal": "BUY"},
        ml_prediction={"signal": "BUY", "probabilities": {"SELL": 0.10, "HOLD": 0.20, "BUY": 0.70}},
        model_metrics={"mean_f1": 0.60},
        backtest_metrics={
            "cumulative_return": 0.50,
            "buy_hold_return": 0.20,
            "sharpe_ratio": 1.20,
            "max_drawdown": -0.12,
        },
        monte_carlo_metrics={
            "expected_return": 0.08,
            "probability_of_loss": 0.30,
            "probability_stop_loss_hit": 0.10,
            "value_at_risk_95": -0.12,
            "conditional_value_at_risk_95": -0.25,
        },
        fundamentals={"score": 45, "rating": "Strong", "statement_dates": ["2024-12-31"]},
        investor_profile=InvestorProfile(account_value=10_000, max_single_position_pct=0.20, risk_budget_pct=0.02),
    )

    assert decision["action"] == "Research Buy Candidate"
    assert decision["suggested_position_value"] == pytest.approx(800.0)
    assert decision["translations"]["cvar_95_value"] == pytest.approx(500.0)
    assert decision["order_readiness"]["status"] == "Paper-trade candidate"
