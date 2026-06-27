from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


INCOME_ROWS = {
    "revenue": ["Total Revenue", "Operating Revenue", "Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income", "Operating Income Or Loss"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
}

BALANCE_ROWS = {
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "total_debt": ["Total Debt", "Long Term Debt And Capital Lease Obligation"],
    "equity": ["Stockholders Equity", "Total Equity Gross Minority Interest"],
    "assets": ["Total Assets"],
    "liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities"],
}

CASHFLOW_ROWS = {
    "operating_cash_flow": ["Operating Cash Flow", "Total Cash From Operating Activities"],
    "capital_expenditure": ["Capital Expenditure", "Capital Expenditures"],
    "free_cash_flow": ["Free Cash Flow"],
}


@dataclass(frozen=True)
class FundamentalAnalysis:
    ticker: str
    source: str
    generated_at: str
    statement_dates: list[str]
    metrics: dict
    score: float
    rating: str
    positives: list[str]
    concerns: list[str]

    def to_dict(self):
        return {
            "ticker": self.ticker,
            "source": self.source,
            "generated_at": self.generated_at,
            "statement_dates": self.statement_dates,
            "metrics": self.metrics,
            "score": self.score,
            "rating": self.rating,
            "positives": self.positives,
            "concerns": self.concerns,
        }


def fetch_yfinance_fundamentals(ticker):
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Ticker must be a non-empty string.")

    yf_ticker = yf.Ticker(ticker)
    income_statement = _get_yfinance_frame(yf_ticker, "income_stmt", "financials")
    balance_sheet = _get_yfinance_frame(yf_ticker, "balance_sheet")
    cashflow_statement = _get_yfinance_frame(yf_ticker, "cashflow")
    info = _get_yfinance_info(yf_ticker)

    return analyze_statement_fundamentals(
        ticker=ticker.upper(),
        income_statement=income_statement,
        balance_sheet=balance_sheet,
        cashflow_statement=cashflow_statement,
        info=info,
        source="yfinance",
    )


def analyze_statement_fundamentals(
    ticker,
    income_statement=None,
    balance_sheet=None,
    cashflow_statement=None,
    info=None,
    source="provided",
):
    info = info or {}
    income_values = _extract_statement_values(income_statement, INCOME_ROWS)
    balance_values = _extract_statement_values(balance_sheet, BALANCE_ROWS)
    cashflow_values = _extract_statement_values(cashflow_statement, CASHFLOW_ROWS)

    latest_revenue = income_values["revenue"]["latest"]
    previous_revenue = income_values["revenue"]["previous"]
    gross_profit = income_values["gross_profit"]["latest"]
    operating_income = income_values["operating_income"]["latest"]
    net_income = income_values["net_income"]["latest"]
    previous_net_income = income_values["net_income"]["previous"]

    operating_cash_flow = cashflow_values["operating_cash_flow"]["latest"]
    capital_expenditure = cashflow_values["capital_expenditure"]["latest"]
    free_cash_flow = cashflow_values["free_cash_flow"]["latest"]
    if free_cash_flow is None:
        free_cash_flow = _derive_free_cash_flow(operating_cash_flow, capital_expenditure)

    cash = _first_number(balance_values["cash"]["latest"], info.get("totalCash"))
    total_debt = _first_number(balance_values["total_debt"]["latest"], info.get("totalDebt"))
    equity = balance_values["equity"]["latest"]
    market_cap = _first_number(info.get("marketCap"))
    enterprise_value = _first_number(info.get("enterpriseValue"))

    metrics = {
        "revenue": latest_revenue,
        "revenue_growth_yoy": _growth(latest_revenue, previous_revenue),
        "gross_margin": _ratio(gross_profit, latest_revenue),
        "operating_margin": _ratio(operating_income, latest_revenue),
        "net_margin": _ratio(net_income, latest_revenue),
        "net_income_growth_yoy": _growth(net_income, previous_net_income),
        "operating_cash_flow": operating_cash_flow,
        "free_cash_flow": free_cash_flow,
        "free_cash_flow_margin": _ratio(free_cash_flow, latest_revenue),
        "free_cash_flow_yield": _ratio(free_cash_flow, market_cap),
        "cash": cash,
        "total_debt": total_debt,
        "debt_to_equity": _ratio(total_debt, equity),
        "cash_to_debt": _ratio(cash, total_debt),
        "net_debt_to_fcf": _ratio((total_debt or 0.0) - (cash or 0.0), free_cash_flow),
        "return_on_equity": _ratio(net_income, equity),
        "trailing_pe": _first_number(info.get("trailingPE")),
        "forward_pe": _first_number(info.get("forwardPE")),
        "price_to_sales": _first_number(info.get("priceToSalesTrailing12Months")),
        "price_to_book": _first_number(info.get("priceToBook")),
        "enterprise_to_ebitda": _first_number(info.get("enterpriseToEbitda")),
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
    }

    score, positives, concerns = score_fundamentals(metrics)
    return FundamentalAnalysis(
        ticker=str(ticker).upper(),
        source=source,
        generated_at=datetime.now(timezone.utc).isoformat(),
        statement_dates=_statement_dates(income_statement, balance_sheet, cashflow_statement),
        metrics={key: _round_metric(value) for key, value in metrics.items()},
        score=round(score, 1),
        rating=_rating_from_score(score),
        positives=positives,
        concerns=concerns,
    )


def score_fundamentals(metrics):
    score = 0.0
    positives = []
    concerns = []

    score += _score_threshold(
        metrics.get("revenue_growth_yoy"),
        positives,
        concerns,
        "Revenue growth is healthy.",
        "Revenue is declining.",
        strong_at=0.10,
        good_at=0.00,
        weak_below=-0.05,
        strong_points=12,
        good_points=5,
        weak_points=-8,
    )
    score += _score_threshold(
        metrics.get("operating_margin"),
        positives,
        concerns,
        "Operating margin is strong.",
        "Operating margin is negative.",
        strong_at=0.20,
        good_at=0.10,
        weak_below=0.00,
        strong_points=12,
        good_points=6,
        weak_points=-12,
    )
    score += _score_threshold(
        metrics.get("free_cash_flow_margin"),
        positives,
        concerns,
        "Free cash flow margin is strong.",
        "Free cash flow is negative or weak.",
        strong_at=0.10,
        good_at=0.00,
        weak_below=0.00,
        strong_points=12,
        good_points=5,
        weak_points=-15,
    )
    score += _score_threshold(
        metrics.get("free_cash_flow_yield"),
        positives,
        concerns,
        "Free cash flow yield is attractive.",
        "Free cash flow yield is negative.",
        strong_at=0.05,
        good_at=0.02,
        weak_below=0.00,
        strong_points=10,
        good_points=4,
        weak_points=-8,
    )

    debt_to_equity = metrics.get("debt_to_equity")
    if debt_to_equity is not None:
        if debt_to_equity <= 0.50:
            score += 8
            positives.append("Balance sheet leverage is modest.")
        elif debt_to_equity >= 2.0:
            score -= 12
            concerns.append("Debt is high relative to equity.")

    cash_to_debt = metrics.get("cash_to_debt")
    if cash_to_debt is not None:
        if cash_to_debt >= 1.0:
            score += 6
            positives.append("Cash covers total debt.")
        elif cash_to_debt < 0.20:
            score -= 6
            concerns.append("Cash cushion is thin relative to debt.")

    return_on_equity = metrics.get("return_on_equity")
    if return_on_equity is not None:
        if return_on_equity >= 0.15:
            score += 8
            positives.append("Return on equity is strong.")
        elif return_on_equity < 0:
            score -= 8
            concerns.append("Return on equity is negative.")

    trailing_pe = metrics.get("trailing_pe")
    if trailing_pe is not None and trailing_pe > 0:
        if trailing_pe <= 20:
            score += 6
            positives.append("Trailing P/E is not stretched.")
        elif trailing_pe >= 40:
            score -= 8
            concerns.append("Trailing P/E is expensive.")

    if not positives and not concerns:
        concerns.append("Insufficient financial-statement data to score fundamentals.")

    return score, positives, concerns


def _extract_statement_values(statement, row_map):
    return {
        key: _latest_previous(_row_series(statement, names))
        for key, names in row_map.items()
    }


def _row_series(statement, row_names):
    if statement is None or getattr(statement, "empty", True):
        return pd.Series(dtype=float)

    normalized_index = {_normalize_label(index): index for index in statement.index}
    for row_name in row_names:
        matched_index = normalized_index.get(_normalize_label(row_name))
        if matched_index is not None:
            return pd.to_numeric(statement.loc[matched_index], errors="coerce")
    return pd.Series(dtype=float)


def _latest_previous(series):
    if series is None or series.empty:
        return {"latest": None, "previous": None}

    ordered = series.reindex(_ordered_columns(series.index)).dropna()
    if ordered.empty:
        return {"latest": None, "previous": None}

    latest = _first_number(ordered.iloc[0])
    previous = _first_number(ordered.iloc[1]) if len(ordered) > 1 else None
    return {"latest": latest, "previous": previous}


def _ordered_columns(columns):
    columns = list(columns)
    try:
        return sorted(columns, reverse=True)
    except TypeError:
        return columns


def _statement_dates(*statements):
    dates = []
    for statement in statements:
        if statement is None or getattr(statement, "empty", True):
            continue
        for column in _ordered_columns(statement.columns):
            text = str(column.date()) if hasattr(column, "date") else str(column)
            if text not in dates:
                dates.append(text)
    return dates


def _derive_free_cash_flow(operating_cash_flow, capital_expenditure):
    if operating_cash_flow is None or capital_expenditure is None:
        return None
    if capital_expenditure < 0:
        return operating_cash_flow + capital_expenditure
    return operating_cash_flow - capital_expenditure


def _score_threshold(
    value,
    positives,
    concerns,
    positive_text,
    concern_text,
    strong_at,
    good_at,
    weak_below,
    strong_points,
    good_points,
    weak_points,
):
    if value is None:
        return 0.0
    if value >= strong_at:
        positives.append(positive_text)
        return strong_points
    if value >= good_at:
        positives.append(positive_text.replace("strong", "positive"))
        return good_points
    if value < weak_below:
        concerns.append(concern_text)
        return weak_points
    return 0.0


def _rating_from_score(score):
    if score >= 45:
        return "Strong"
    if score >= 20:
        return "Constructive"
    if score >= -10:
        return "Mixed"
    if score >= -35:
        return "Weak"
    return "High Risk"


def _growth(latest, previous):
    if latest is None or previous in (None, 0):
        return None
    return (latest - previous) / abs(previous)


def _ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _round_metric(value):
    if value is None:
        return None
    return round(float(value), 6)


def _first_number(*values):
    for value in values:
        try:
            if value is None or pd.isna(value):
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _normalize_label(value):
    return "".join(character for character in str(value).lower() if character.isalnum())


def _get_yfinance_frame(yf_ticker, *attributes):
    for attribute in attributes:
        try:
            frame = getattr(yf_ticker, attribute)
        except Exception:
            continue
        if frame is not None and not frame.empty:
            return frame
    return pd.DataFrame()


def _get_yfinance_info(yf_ticker):
    try:
        return yf_ticker.info or {}
    except Exception:
        return {}
