from config import CostAssumptions, DEFAULT_ANALYSIS_CONFIG, describe_assumptions


def test_cost_assumptions_total_cost_rate_uses_basis_points():
    costs = CostAssumptions(transaction_cost_bps=10, slippage_bps=5)

    assert costs.total_cost_rate == 0.0015


def test_default_assumptions_are_visible_to_users():
    assumptions = describe_assumptions(DEFAULT_ANALYSIS_CONFIG)

    assert any("not financial advice" in item for item in assumptions)
    assert any("Price policy" in item for item in assumptions)
    assert any("Label policy" in item for item in assumptions)
    assert any("Cost assumptions" in item for item in assumptions)
