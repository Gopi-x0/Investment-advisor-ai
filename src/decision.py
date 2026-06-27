from dataclasses import dataclass


@dataclass(frozen=True)
class InvestorProfile:
    account_value: float = 10_000.0
    max_single_position_pct: float = 0.10
    risk_budget_pct: float = 0.02

    @property
    def max_single_position_value(self) -> float:
        return max(0.0, self.account_value * self.max_single_position_pct)

    @property
    def risk_budget_value(self) -> float:
        return max(0.0, self.account_value * self.risk_budget_pct)


def build_decision_summary(
    rule_result=None,
    ml_prediction=None,
    model_metrics=None,
    backtest_metrics=None,
    risk_summary=None,
    monte_carlo_metrics=None,
    fundamentals=None,
    portfolio_context=None,
    investor_profile: InvestorProfile | None = None,
):
    profile = investor_profile or InvestorProfile()
    rule_result = rule_result or {}
    ml_prediction = ml_prediction or {}
    model_metrics = model_metrics or {}
    backtest_metrics = backtest_metrics or {}
    risk_summary = risk_summary or {}
    monte_carlo_metrics = monte_carlo_metrics or {}
    fundamentals = _fundamentals_dict(fundamentals)
    portfolio_context = portfolio_context or {}

    score, drivers, cautions = _score_evidence(
        rule_result,
        ml_prediction,
        model_metrics,
        backtest_metrics,
        monte_carlo_metrics,
        fundamentals,
        portfolio_context,
    )
    action = _action_from_score(score)
    conviction = _conviction_from_prediction(ml_prediction, model_metrics)
    translations = _risk_translations(profile, risk_summary, monte_carlo_metrics)
    sizing = _position_sizing(profile, monte_carlo_metrics, action)

    if fundamentals and _fundamentals_are_insufficient(fundamentals):
        cautions.append("Financial-statement data was too incomplete to support the decision.")
    elif fundamentals and fundamentals.get("rating"):
        drivers.append(f"Fundamental rating: {fundamentals['rating']}.")
    elif not fundamentals:
        cautions.append("No financial-statement layer was available, so this is price/action driven only.")

    order_readiness = _order_readiness(action, conviction, backtest_metrics, fundamentals, sizing)

    return {
        "action": action,
        "score": round(score, 1),
        "conviction": conviction,
        "suggested_allocation_pct": sizing["suggested_allocation_pct"],
        "suggested_position_value": sizing["suggested_position_value"],
        "max_single_position_value": profile.max_single_position_value,
        "risk_budget_value": profile.risk_budget_value,
        "translations": translations,
        "drivers": drivers,
        "cautions": cautions,
        "order_readiness": order_readiness,
    }


def _fundamentals_dict(fundamentals):
    if fundamentals is None:
        return {}
    if isinstance(fundamentals, dict):
        return fundamentals
    if hasattr(fundamentals, "to_dict"):
        return fundamentals.to_dict()
    return getattr(fundamentals, "__dict__", {})


def _score_evidence(
    rule_result,
    ml_prediction,
    model_metrics,
    backtest_metrics,
    monte_carlo_metrics,
    fundamentals,
    portfolio_context,
):
    score = 0.0
    drivers = []
    cautions = []

    ml_signal = ml_prediction.get("signal")
    if ml_signal == "BUY":
        score += 15
        drivers.append("ML signal is BUY.")
    elif ml_signal == "SELL":
        score -= 15
        cautions.append("ML signal is SELL.")
    elif ml_signal == "HOLD":
        drivers.append("ML signal is HOLD, which supports waiting for a stronger edge.")

    confidence = _max_probability(ml_prediction)
    if confidence is not None:
        if confidence >= 0.65:
            score += 8
            drivers.append(f"Prediction confidence is relatively high at {confidence:.0%}.")
        elif confidence < 0.50:
            score -= 8
            cautions.append(f"Prediction confidence is weak at {confidence:.0%}.")
        else:
            cautions.append(f"Prediction confidence is moderate at {confidence:.0%}.")

    rule_signal = rule_result.get("signal")
    if rule_signal == "BUY":
        score += 8
        drivers.append("Rule-based trend/risk signal is BUY.")
    elif rule_signal == "SELL":
        score -= 8
        cautions.append("Rule-based trend/risk signal is SELL.")

    mean_f1 = _float_or_none(model_metrics.get("mean_f1"))
    if mean_f1 is not None:
        if mean_f1 >= 0.55:
            score += 8
            drivers.append(f"Walk-forward F1 is usable at {mean_f1:.2f}.")
        elif mean_f1 < 0.35:
            score -= 10
            cautions.append(f"Walk-forward F1 is weak at {mean_f1:.2f}.")

    cumulative_return = _float_or_none(backtest_metrics.get("cumulative_return"))
    buy_hold_return = _float_or_none(backtest_metrics.get("buy_hold_return"))
    if cumulative_return is not None and buy_hold_return is not None:
        if cumulative_return > buy_hold_return + 0.05:
            score += 12
            drivers.append("Backtest beat buy-and-hold after costs.")
        elif cumulative_return < buy_hold_return - 0.05:
            score -= 12
            cautions.append("Backtest underperformed buy-and-hold after costs.")

    sharpe = _float_or_none(backtest_metrics.get("sharpe_ratio"))
    if sharpe is not None:
        if sharpe >= 1.0:
            score += 8
            drivers.append(f"Backtest Sharpe is strong at {sharpe:.2f}.")
        elif sharpe < 0.5:
            score -= 5
            cautions.append(f"Backtest Sharpe is low at {sharpe:.2f}.")

    max_drawdown = _float_or_none(backtest_metrics.get("max_drawdown"))
    if max_drawdown is not None and max_drawdown <= -0.25:
        score -= 6
        cautions.append(f"Backtest max drawdown is deep at {max_drawdown:.0%}.")

    expected_return = _float_or_none(monte_carlo_metrics.get("expected_return"))
    probability_loss = _float_or_none(monte_carlo_metrics.get("probability_of_loss"))
    stop_loss_probability = _float_or_none(monte_carlo_metrics.get("probability_stop_loss_hit"))
    if expected_return is not None:
        if expected_return >= 0.05:
            score += 6
            drivers.append(f"Monte Carlo expected return is positive at {expected_return:.1%}.")
        elif expected_return < 0:
            score -= 6
            cautions.append(f"Monte Carlo expected return is negative at {expected_return:.1%}.")
    if probability_loss is not None and probability_loss >= 0.45:
        score -= 5
        cautions.append(f"Monte Carlo probability of loss is elevated at {probability_loss:.0%}.")
    if stop_loss_probability is not None and stop_loss_probability >= 0.25:
        score -= 5
        cautions.append(f"Stop-loss hit probability is elevated at {stop_loss_probability:.0%}.")

    fundamental_score = _float_or_none(fundamentals.get("score"))
    if fundamental_score is not None:
        if fundamental_score >= 30:
            score += 15
            drivers.append("Financial statements support a high-quality business profile.")
        elif fundamental_score <= -20:
            score -= 15
            cautions.append("Financial statements show material business or valuation concerns.")

    concentration_warning = portfolio_context.get("concentration_warning")
    if concentration_warning:
        score -= 5
        cautions.append(concentration_warning)

    return score, drivers, cautions


def _risk_translations(profile, risk_summary, monte_carlo_metrics):
    position_value = profile.max_single_position_value
    var_95 = abs(_float_or_zero(monte_carlo_metrics.get("value_at_risk_95")))
    cvar_95 = abs(_float_or_zero(monte_carlo_metrics.get("conditional_value_at_risk_95")))
    expected_return = _float_or_zero(monte_carlo_metrics.get("expected_return"))
    max_drawdown = abs(_float_or_zero(risk_summary.get("max_drawdown")))

    return {
        "position_value_used": round(position_value, 2),
        "expected_return_value": round(position_value * expected_return, 2),
        "var_95_value": round(position_value * var_95, 2),
        "cvar_95_value": round(position_value * cvar_95, 2),
        "historical_max_drawdown_value": round(position_value * max_drawdown, 2),
        "probability_of_loss": _float_or_none(monte_carlo_metrics.get("probability_of_loss")),
        "probability_stop_loss_hit": _float_or_none(monte_carlo_metrics.get("probability_stop_loss_hit")),
    }


def _position_sizing(profile, monte_carlo_metrics, action):
    max_position = profile.max_single_position_value
    cvar_95 = abs(_float_or_zero(monte_carlo_metrics.get("conditional_value_at_risk_95")))
    if cvar_95 > 0:
        risk_capped_value = profile.risk_budget_value / cvar_95
    else:
        risk_capped_value = max_position

    suggested_value = min(max_position, risk_capped_value)
    action_multiplier = {
        "Research Buy Candidate": 1.00,
        "Watch / Small Position Only": 0.50,
        "Watch / No New Buy": 0.00,
        "Avoid Or Reduce": 0.00,
    }[action]
    suggested_value *= action_multiplier

    return {
        "suggested_position_value": round(max(0.0, suggested_value), 2),
        "suggested_allocation_pct": round(max(0.0, suggested_value) / profile.account_value, 4)
        if profile.account_value
        else 0.0,
    }


def _order_readiness(action, conviction, backtest_metrics, fundamentals, sizing):
    blockers = []
    if action not in {"Research Buy Candidate", "Watch / Small Position Only"}:
        blockers.append("Decision layer does not currently support a new buy.")
    if conviction == "Low":
        blockers.append("Model conviction is low.")
    if sizing["suggested_position_value"] <= 0:
        blockers.append("Risk budget produces no suggested position size.")
    if not fundamentals:
        blockers.append("Financial statements have not been reviewed.")
    elif _fundamentals_are_insufficient(fundamentals):
        blockers.append("Financial statements were loaded but are insufficient for investment use.")

    cumulative_return = _float_or_none(backtest_metrics.get("cumulative_return"))
    buy_hold_return = _float_or_none(backtest_metrics.get("buy_hold_return"))
    if cumulative_return is not None and buy_hold_return is not None and cumulative_return < buy_hold_return:
        blockers.append("Strategy backtest trails buy-and-hold.")

    if blockers:
        return {
            "status": "Not order-ready",
            "blockers": blockers,
        }
    return {
        "status": "Paper-trade candidate",
        "blockers": [],
    }


def _action_from_score(score):
    if score >= 35:
        return "Research Buy Candidate"
    if score >= 15:
        return "Watch / Small Position Only"
    if score >= -15:
        return "Watch / No New Buy"
    return "Avoid Or Reduce"


def _fundamentals_are_insufficient(fundamentals):
    concerns = fundamentals.get("concerns") or []
    if not fundamentals.get("statement_dates"):
        return True
    return any("insufficient" in str(concern).lower() for concern in concerns)


def _conviction_from_prediction(ml_prediction, model_metrics):
    probability = _max_probability(ml_prediction)
    mean_f1 = _float_or_none(model_metrics.get("mean_f1"))
    if probability is None:
        return "Low"
    if probability >= 0.65 and (mean_f1 is None or mean_f1 >= 0.45):
        return "High"
    if probability >= 0.50:
        return "Medium"
    return "Low"


def _max_probability(ml_prediction):
    probabilities = ml_prediction.get("probabilities") or {}
    if not probabilities:
        return None
    values = [_float_or_none(value) for value in probabilities.values()]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _float_or_zero(value):
    converted = _float_or_none(value)
    return converted if converted is not None else 0.0


def _float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
