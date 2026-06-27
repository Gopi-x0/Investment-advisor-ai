import json
import sys
from pathlib import Path

import pandas as pd

SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    import streamlit as st
except ModuleNotFoundError as exc:
    raise SystemExit("Streamlit is not installed. Run `python -m pip install -r requirements.txt`.") from exc

from config import AnalysisConfig, CostAssumptions, DEFAULT_ANALYSIS_CONFIG, describe_assumptions
from data_loader import get_stock_data
from decision import InvestorProfile, build_decision_summary
from portfolio import (
    correlation_matrix,
    efficient_frontier,
    normalize_weights,
    portfolio_diagnostics,
    portfolio_monte_carlo,
    risk_parity_weights,
)
from reports import report_payload_to_markdown, research_report_to_payload, save_markdown_report
from research import build_research_report


st.set_page_config(page_title="Investment Research Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {border: 1px solid rgba(49, 51, 63, 0.16); border-radius: 8px; padding: 0.75rem;}
    div[data-testid="stDataFrame"] {border-radius: 8px;}
    section[data-testid="stSidebar"] .block-container {padding-top: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def pct(value):
    if value is None:
        return "n/a"
    return f"{float(value):.1%}"


def money(value):
    if value is None:
        return "n/a"
    return f"${float(value):,.2f}"


def compact_metric_frame(values):
    return pd.DataFrame(
        [{"metric": key.replace("_", " ").title(), "value": value} for key, value in values.items()]
    )


def parse_custom_weights(text, columns):
    if not text.strip():
        raise ValueError("Enter custom weights as AAPL:40, MSFT:35, SPY:25.")

    parsed = {}
    for item in text.split(","):
        if not item.strip():
            continue
        if ":" in item:
            symbol, raw_value = item.split(":", 1)
        elif "=" in item:
            symbol, raw_value = item.split("=", 1)
        else:
            raise ValueError(f"Could not parse weight item: {item}")
        symbol = symbol.strip().upper()
        raw_value = raw_value.strip().replace("%", "")
        parsed[symbol] = float(raw_value)

    missing = [column for column in columns if column not in parsed]
    if missing:
        for column in missing:
            parsed[column] = 0.0
    weights = pd.Series(parsed).reindex(columns).fillna(0.0)
    return normalize_weights(weights)


@st.cache_data(show_spinner=False, ttl=900)
def load_price_series(symbol, period, price_policy):
    return get_stock_data(symbol, period=period, price_policy=price_policy).set_index("Date")["Close"]


def build_portfolio_bundle(tickers, period, config, weight_mode, custom_weight_text):
    errors = []
    prices = {}
    for symbol in tickers:
        try:
            prices[symbol] = load_price_series(symbol, period, config.price_policy)
        except Exception as exc:
            errors.append(f"Could not load {symbol}: {exc}")

    if len(prices) < 2:
        return None, errors

    price_frame = pd.DataFrame(prices).dropna()
    if price_frame.empty:
        return None, errors + ["Portfolio price history has no overlapping dates."]

    if weight_mode == "Equal":
        weights = normalize_weights(None, price_frame.columns)
    elif weight_mode == "Custom":
        try:
            weights = parse_custom_weights(custom_weight_text, price_frame.columns)
        except ValueError as exc:
            errors.append(str(exc))
            weights = risk_parity_weights(price_frame)
    else:
        weights = risk_parity_weights(price_frame)

    diagnostics = portfolio_diagnostics(price_frame, weights)
    return {"prices": price_frame, "weights": weights, "diagnostics": diagnostics}, errors


st.title("Investment Research Dashboard")
st.caption("Educational decision-support tool. Not financial advice or a real-money trading system.")

with st.sidebar.form("analysis_controls"):
    st.header("Research Setup")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    watchlist = st.text_input("Portfolio / watchlist", value="AAPL,MSFT,SPY")
    period = st.selectbox("History period", ["1y", "2y", "5y", "10y"], index=2)
    model_name = st.selectbox("Model", ["random_forest", "logistic_regression", "gradient_boosting"], index=0)
    include_fundamentals = st.checkbox("Include fundamentals", value=True)

    st.header("Decision Inputs")
    account_value = st.number_input("Account value", min_value=100.0, value=10_000.0, step=500.0)
    max_position_pct = st.slider("Max single position", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
    risk_budget_pct = st.slider("Risk budget per idea", min_value=0.25, max_value=10.0, value=2.0, step=0.25)

    st.header("Model Assumptions")
    price_policy = st.selectbox("Price policy", ["raw_close", "adjusted_close"], index=0)
    label_policy = st.selectbox("Label policy", ["three_class", "binary"], index=0)
    lookahead_days = st.number_input("Lookahead days", min_value=5, max_value=120, value=20, step=5)
    cost_bps = st.number_input("Transaction cost bps", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    slippage_bps = st.number_input("Slippage bps", min_value=0.0, max_value=100.0, value=5.0, step=1.0)

    st.header("Scenario Assumptions")
    monte_carlo_horizon = st.number_input("Monte Carlo horizon", min_value=10, max_value=252, value=60, step=10)
    monte_carlo_sims = st.number_input("Monte Carlo simulations", min_value=100, max_value=5000, value=1000, step=100)
    stop_loss = st.slider("Stop-loss threshold", min_value=2.5, max_value=50.0, value=10.0, step=2.5)
    risk_free_rate = st.number_input("Risk-free rate", min_value=0.0, max_value=0.20, value=0.02, step=0.005)

    st.header("Portfolio")
    weight_mode = st.selectbox("Weighting", ["Risk parity", "Equal", "Custom"], index=0)
    custom_weight_text = st.text_input("Custom weights", value="AAPL:40, MSFT:35, SPY:25")

    run_button = st.form_submit_button("Run analysis", type="primary")

config = AnalysisConfig(
    price_policy=price_policy,
    label_policy=label_policy,
    lookahead_days=int(lookahead_days),
    volatility_window=DEFAULT_ANALYSIS_CONFIG.volatility_window,
    volatility_barrier_multiplier=DEFAULT_ANALYSIS_CONFIG.volatility_barrier_multiplier,
    walk_forward_splits=DEFAULT_ANALYSIS_CONFIG.walk_forward_splits,
    top_n_features=DEFAULT_ANALYSIS_CONFIG.top_n_features,
    risk_free_rate=float(risk_free_rate),
    monte_carlo_horizon_days=int(monte_carlo_horizon),
    monte_carlo_simulations=int(monte_carlo_sims),
    monte_carlo_stop_loss=float(stop_loss) / 100,
    costs=CostAssumptions(transaction_cost_bps=float(cost_bps), slippage_bps=float(slippage_bps)),
)
investor_profile = InvestorProfile(
    account_value=float(account_value),
    max_single_position_pct=float(max_position_pct) / 100,
    risk_budget_pct=float(risk_budget_pct) / 100,
)

with st.expander("Current assumptions", expanded=False):
    for assumption in describe_assumptions(config):
        st.write(assumption)

if not run_button:
    st.write("Choose a ticker and run an analysis from the sidebar.")
    st.stop()

if not ticker:
    st.error("Ticker cannot be empty.")
    st.stop()

tickers = [item.strip().upper() for item in watchlist.split(",") if item.strip()]
if ticker not in tickers:
    tickers.insert(0, ticker)

with st.spinner("Running research, fundamentals, and portfolio analysis..."):
    report = build_research_report(
        ticker,
        period=period,
        config=config,
        model_name=model_name,
        include_fundamentals=include_fundamentals,
        investor_profile=investor_profile,
    )
    portfolio_bundle, portfolio_errors = build_portfolio_bundle(tickers, period, config, weight_mode, custom_weight_text)

portfolio_context = {}
if portfolio_bundle and portfolio_bundle["diagnostics"]["warnings"]:
    portfolio_context["concentration_warning"] = portfolio_bundle["diagnostics"]["warnings"][0]

decision = build_decision_summary(
    rule_result=report.rule_result,
    ml_prediction=report.ml_prediction,
    model_metrics=report.training_result.metrics,
    backtest_metrics=report.backtest.metrics,
    risk_summary=report.risk_summary,
    monte_carlo_metrics=report.monte_carlo.metrics,
    fundamentals=report.fundamentals,
    portfolio_context=portfolio_context,
    investor_profile=investor_profile,
)
report.decision_summary.update(decision)
payload = research_report_to_payload(report, config)
payload["decision_summary"] = decision
saved_markdown_path = save_markdown_report(payload, output_dir="reports")

if report.warnings:
    for warning in report.warnings:
        st.warning(warning)
if portfolio_errors:
    for error in portfolio_errors:
        st.warning(error)

st.subheader(f"{report.ticker} Research Run")
st.caption(f"Saved Markdown report: {saved_markdown_path}")
overview_cols = st.columns(5)
overview_cols[0].metric("Decision", decision["action"])
overview_cols[1].metric("Conviction", decision["conviction"])
overview_cols[2].metric("Suggested Position", money(decision["suggested_position_value"]))
overview_cols[3].metric("ML Signal", report.ml_prediction["signal"])
overview_cols[4].metric("Rule Signal", report.rule_result["signal"])

decision_tab, fundamentals_tab, model_tab, risk_tab, portfolio_tab, report_tab = st.tabs(
    ["Decision", "Fundamentals", "Model", "Risk", "Portfolio", "Report"]
)

with decision_tab:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Action Translation")
        st.dataframe(
            compact_metric_frame(
                {
                    "score": decision["score"],
                    "suggested_allocation": pct(decision["suggested_allocation_pct"]),
                    "max_single_position": money(decision["max_single_position_value"]),
                    "risk_budget": money(decision["risk_budget_value"]),
                    "order_readiness": decision["order_readiness"]["status"],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Dollar Risk")
        translations = decision["translations"].copy()
        translations["probability_of_loss"] = pct(translations["probability_of_loss"])
        translations["probability_stop_loss_hit"] = pct(translations["probability_stop_loss_hit"])
        for key in [
            "position_value_used",
            "expected_return_value",
            "var_95_value",
            "cvar_95_value",
            "historical_max_drawdown_value",
        ]:
            translations[key] = money(translations[key])
        st.dataframe(compact_metric_frame(translations), use_container_width=True, hide_index=True)

    with right:
        st.subheader("Drivers")
        if decision["drivers"]:
            for item in decision["drivers"]:
                st.success(item)
        else:
            st.info("No positive drivers were strong enough to highlight.")

        st.subheader("Cautions")
        if decision["cautions"]:
            for item in decision["cautions"]:
                st.warning(item)
        else:
            st.success("No major cautions were raised by the decision layer.")

        if decision["order_readiness"]["blockers"]:
            st.subheader("Order Blockers")
            for blocker in decision["order_readiness"]["blockers"]:
                st.warning(blocker)

with fundamentals_tab:
    st.subheader("Financial Statement Quality")
    if report.fundamentals is None:
        st.info("Fundamentals were not loaded for this run.")
    else:
        fundamentals = report.fundamentals.to_dict()
        metric_cols = st.columns(4)
        metric_cols[0].metric("Rating", fundamentals["rating"])
        metric_cols[1].metric("Score", fundamentals["score"])
        metric_cols[2].metric("Revenue Growth", pct(fundamentals["metrics"].get("revenue_growth_yoy")))
        metric_cols[3].metric("FCF Yield", pct(fundamentals["metrics"].get("free_cash_flow_yield")))

        display_metrics = {
            key: value
            for key, value in fundamentals["metrics"].items()
            if key
            in {
                "revenue",
                "revenue_growth_yoy",
                "gross_margin",
                "operating_margin",
                "net_margin",
                "free_cash_flow",
                "free_cash_flow_margin",
                "free_cash_flow_yield",
                "debt_to_equity",
                "cash_to_debt",
                "return_on_equity",
                "trailing_pe",
                "forward_pe",
                "price_to_sales",
                "price_to_book",
            }
        }
        st.dataframe(compact_metric_frame(display_metrics), use_container_width=True, hide_index=True)

        positive_col, concern_col = st.columns(2)
        with positive_col:
            st.subheader("Strengths")
            for item in fundamentals["positives"] or ["No major strengths were identified."]:
                st.success(item)
        with concern_col:
            st.subheader("Concerns")
            for item in fundamentals["concerns"] or ["No major concerns were identified."]:
                st.warning(item)

with model_tab:
    st.subheader("Model Diagnostics")
    model_cols = st.columns(4)
    model_cols[0].metric("Mean Accuracy", pct(report.training_result.metrics["mean_accuracy"]))
    model_cols[1].metric("Mean F1", f"{report.training_result.metrics['mean_f1']:.3f}")
    model_cols[2].metric("Std Accuracy", pct(report.training_result.metrics["std_accuracy"]))
    model_cols[3].metric("Classes", ", ".join(report.training_result.metrics["class_names"]))

    probability_col, importance_col = st.columns(2)
    with probability_col:
        st.subheader("Class Probabilities")
        if report.ml_prediction["probabilities"]:
            st.bar_chart(pd.Series(report.ml_prediction["probabilities"]))
        else:
            st.write("The selected model did not provide class probabilities.")
    with importance_col:
        st.subheader("Top Feature Importance")
        st.bar_chart(report.feature_importance.head(15))

    st.subheader("Walk-Forward Folds")
    st.dataframe(pd.DataFrame(report.training_result.fold_results), use_container_width=True)

with risk_tab:
    st.subheader("Backtest And Risk")
    backtest_col, risk_col = st.columns(2)
    with backtest_col:
        st.dataframe(
            compact_metric_frame(report.backtest.metrics),
            use_container_width=True,
            hide_index=True,
        )
    with risk_col:
        st.dataframe(
            compact_metric_frame(report.risk_summary),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Monte Carlo Percentile Paths")
    st.line_chart(report.monte_carlo.percentiles)
    st.subheader("Monte Carlo Metrics")
    st.dataframe(compact_metric_frame(report.monte_carlo.metrics), use_container_width=True, hide_index=True)

with portfolio_tab:
    st.subheader("Portfolio Fit")
    if not portfolio_bundle:
        st.info("Add at least two valid tickers to the watchlist for portfolio analytics.")
    else:
        diagnostics = portfolio_bundle["diagnostics"]
        portfolio_cols = st.columns(5)
        portfolio_cols[0].metric("Volatility", pct(diagnostics["annualized_volatility"]))
        portfolio_cols[1].metric("Max Drawdown", pct(diagnostics["max_drawdown"]))
        portfolio_cols[2].metric("Diversification", f"{diagnostics['diversification_score']:.3f}")
        portfolio_cols[3].metric("Effective Positions", f"{diagnostics['effective_positions']:.2f}")
        portfolio_cols[4].metric("Top Risk", diagnostics["top_risk_symbol"] or "n/a")

        if diagnostics["warnings"]:
            for warning in diagnostics["warnings"]:
                st.warning(warning)

        weights_col, risk_contrib_col = st.columns(2)
        with weights_col:
            st.subheader("Weights")
            st.bar_chart(pd.Series(diagnostics["weights"]))
        with risk_contrib_col:
            st.subheader("Risk Contribution")
            st.bar_chart(pd.Series(diagnostics["risk_contribution"]))

        st.subheader("Correlation Matrix")
        st.dataframe(correlation_matrix(portfolio_bundle["prices"]), use_container_width=True)

        st.subheader("Efficient Frontier Sample")
        frontier = efficient_frontier(portfolio_bundle["prices"], n_portfolios=300)
        st.scatter_chart(frontier, x="volatility", y="expected_return")

        st.subheader("Portfolio Monte Carlo Paths")
        st.line_chart(
            portfolio_monte_carlo(
                portfolio_bundle["prices"],
                portfolio_bundle["weights"],
                horizon_days=int(monte_carlo_horizon),
                n_simulations=300,
            ).iloc[:, :30]
        )

with report_tab:
    st.subheader("Downloadable Report")
    st.download_button(
        "Download JSON report",
        data=json.dumps(payload, indent=2),
        file_name=f"{ticker}_research_report.json",
        mime="application/json",
    )
    st.download_button(
        "Download Markdown report",
        data=report_payload_to_markdown(payload),
        file_name=f"{ticker}_research_report.md",
        mime="text/markdown",
    )
    st.json(payload)
