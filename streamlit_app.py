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

from config import DEFAULT_ANALYSIS_CONFIG, describe_assumptions
from data_loader import get_stock_data
from portfolio import (
    correlation_matrix,
    diversification_score,
    efficient_frontier,
    portfolio_monte_carlo,
    portfolio_summary,
    risk_parity_weights,
)
from research import build_research_report
from reports import report_payload_to_markdown, research_report_to_payload


st.set_page_config(page_title="Investment Research Dashboard", layout="wide")

st.title("Investment Research Dashboard")
st.caption("Educational decision-support tool. Not financial advice or a real-money trading system.")

with st.sidebar:
    st.header("Analysis Controls")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    watchlist = st.text_input("Watchlist", value="AAPL,MSFT,SPY")
    period = st.selectbox("History period", ["1y", "2y", "5y", "10y"], index=2)
    model_name = st.selectbox("Model", ["random_forest", "logistic_regression", "gradient_boosting"], index=0)
    run_button = st.button("Run analysis", type="primary")

for assumption in describe_assumptions(DEFAULT_ANALYSIS_CONFIG):
    st.info(assumption)

if run_button:
    with st.spinner("Running research analysis..."):
        report = build_research_report(ticker, period=period, model_name=model_name)

    st.header(f"{report.ticker} Overview")
    st.write(f"Rows: {report.data_rows}")
    st.write(f"Date range: {report.date_range[0]} to {report.date_range[1]}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rule Signal", report.rule_result["signal"])
    col2.metric("ML Signal", report.ml_prediction["signal"])
    col3.metric("Mean F1", f"{report.training_result.metrics['mean_f1']:.3f}")

    if report.warnings:
        for warning in report.warnings:
            st.warning(warning)

    st.subheader("Class Probabilities")
    if report.ml_prediction["probabilities"]:
        st.bar_chart(pd.Series(report.ml_prediction["probabilities"]))
    else:
        st.write("The selected model did not provide class probabilities.")

    st.subheader("Feature Importance")
    st.bar_chart(report.feature_importance.head(15))

    st.subheader("Backtest Metrics")
    st.dataframe(pd.DataFrame([report.backtest.metrics]).T.rename(columns={0: "value"}))

    st.subheader("Risk Metrics")
    st.dataframe(pd.DataFrame([report.risk_summary]).T.rename(columns={0: "value"}))

    st.subheader("Monte Carlo Percentile Paths")
    st.line_chart(report.monte_carlo.percentiles)

    st.subheader("Monte Carlo Metrics")
    st.dataframe(pd.DataFrame([report.monte_carlo.metrics]).T.rename(columns={0: "value"}))

    payload = research_report_to_payload(report, DEFAULT_ANALYSIS_CONFIG)
    st.subheader("Saved Analysis Report")
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

    st.header("Portfolio / Watchlist")
    tickers = [item.strip().upper() for item in watchlist.split(",") if item.strip()]
    if len(tickers) >= 2:
        prices = {}
        for symbol in tickers:
            try:
                prices[symbol] = get_stock_data(
                    symbol,
                    period=period,
                    price_policy=DEFAULT_ANALYSIS_CONFIG.price_policy,
                ).set_index("Date")["Close"]
            except Exception as exc:
                st.warning(f"Could not load {symbol}: {exc}")

        if len(prices) >= 2:
            price_frame = pd.DataFrame(prices).dropna()
            rp_weights = risk_parity_weights(price_frame)
            st.subheader("Correlation Matrix")
            st.dataframe(correlation_matrix(price_frame))
            st.subheader("Portfolio Summary")
            st.dataframe(
                pd.DataFrame([portfolio_summary(price_frame, rp_weights)]).T.rename(columns={0: "value"})
            )
            st.write(f"Diversification score: {diversification_score(price_frame):.3f}")
            st.subheader("Risk Parity Weights")
            st.bar_chart(rp_weights)
            st.subheader("Efficient Frontier Sample")
            frontier = efficient_frontier(price_frame, n_portfolios=200)
            st.scatter_chart(frontier, x="volatility", y="expected_return")
            st.subheader("Portfolio Monte Carlo Paths")
            st.line_chart(
                portfolio_monte_carlo(price_frame, rp_weights, horizon_days=60, n_simulations=200).iloc[:, :25]
            )
else:
    st.write("Choose a ticker and run an analysis from the sidebar.")
