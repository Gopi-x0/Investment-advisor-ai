import pandas as pd

from fundamentals import analyze_statement_fundamentals, score_fundamentals


def test_analyze_statement_fundamentals_extracts_growth_margins_and_balance_sheet():
    columns = [pd.Timestamp("2024-12-31"), pd.Timestamp("2023-12-31")]
    income = pd.DataFrame(
        {
            columns[0]: [120.0, 72.0, 30.0, 24.0],
            columns[1]: [100.0, 55.0, 18.0, 14.0],
        },
        index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income"],
    )
    balance = pd.DataFrame(
        {
            columns[0]: [40.0, 30.0, 100.0],
            columns[1]: [20.0, 45.0, 90.0],
        },
        index=["Cash And Cash Equivalents", "Total Debt", "Stockholders Equity"],
    )
    cashflow = pd.DataFrame(
        {
            columns[0]: [32.0, -8.0],
            columns[1]: [20.0, -6.0],
        },
        index=["Operating Cash Flow", "Capital Expenditure"],
    )

    analysis = analyze_statement_fundamentals(
        "XYZ",
        income_statement=income,
        balance_sheet=balance,
        cashflow_statement=cashflow,
        info={"marketCap": 400.0, "trailingPE": 18.0},
    )

    assert analysis.metrics["revenue_growth_yoy"] == 0.2
    assert analysis.metrics["operating_margin"] == 0.25
    assert analysis.metrics["free_cash_flow"] == 24.0
    assert analysis.metrics["free_cash_flow_yield"] == 0.06
    assert analysis.metrics["debt_to_equity"] == 0.3
    assert analysis.rating in {"Strong", "Constructive"}
    assert analysis.positives


def test_score_fundamentals_flags_expensive_weak_business():
    score, positives, concerns = score_fundamentals(
        {
            "revenue_growth_yoy": -0.12,
            "operating_margin": -0.05,
            "free_cash_flow_margin": -0.10,
            "free_cash_flow_yield": -0.03,
            "debt_to_equity": 3.0,
            "cash_to_debt": 0.05,
            "return_on_equity": -0.20,
            "trailing_pe": 55.0,
        }
    )

    assert score < 0
    assert not positives
    assert any("Revenue" in concern for concern in concerns)
    assert any("Debt" in concern for concern in concerns)
