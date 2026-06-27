from dataclasses import dataclass
from typing import Literal

PricePolicy = Literal["raw_close", "adjusted_close"]
LabelPolicy = Literal["three_class", "binary"]

EDUCATIONAL_DISCLAIMER = (
    "Educational research tool only. This is not financial advice, not a guarantee of returns, "
    "and not a real-money trading system."
)

PRICE_POLICY_DESCRIPTIONS = {
    "raw_close": "Use the raw Close column from the data provider.",
    "adjusted_close": "Use Adj Close when available so historical prices reflect splits/dividends.",
}

LABEL_POLICY_DESCRIPTIONS = {
    "three_class": "BUY, HOLD, and SELL are explicit classes; HOLD means no sufficient edge.",
    "binary": "Only BUY and SELL classes are modeled; HOLD is not treated as a training target.",
}


@dataclass(frozen=True)
class CostAssumptions:
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0

    @property
    def total_cost_rate(self) -> float:
        return (self.transaction_cost_bps + self.slippage_bps) / 10_000


@dataclass(frozen=True)
class AnalysisConfig:
    price_policy: PricePolicy = "raw_close"
    label_policy: LabelPolicy = "three_class"
    lookahead_days: int = 20
    volatility_window: int = 20
    volatility_barrier_multiplier: float = 1.3
    walk_forward_splits: int = 5
    top_n_features: int = 15
    risk_free_rate: float = 0.02
    monte_carlo_horizon_days: int = 60
    monte_carlo_simulations: int = 1000
    monte_carlo_stop_loss: float = 0.10
    costs: CostAssumptions = CostAssumptions()


DEFAULT_ANALYSIS_CONFIG = AnalysisConfig()


def describe_assumptions(config: AnalysisConfig = DEFAULT_ANALYSIS_CONFIG) -> list[str]:
    return [
        EDUCATIONAL_DISCLAIMER,
        f"Price policy: {config.price_policy} - {PRICE_POLICY_DESCRIPTIONS[config.price_policy]}",
        f"Label policy: {config.label_policy} - {LABEL_POLICY_DESCRIPTIONS[config.label_policy]}",
        (
            "Cost assumptions: "
            f"{config.costs.transaction_cost_bps:.1f} bps transaction cost + "
            f"{config.costs.slippage_bps:.1f} bps slippage per position change."
        ),
    ]
