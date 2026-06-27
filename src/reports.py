import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import DEFAULT_ANALYSIS_CONFIG, describe_assumptions


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, pd.Series):
        return make_json_safe(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return make_json_safe(value.to_dict(orient="records"))
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def build_report_payload(
    ticker,
    config=DEFAULT_ANALYSIS_CONFIG,
    rule_result=None,
    ml_prediction=None,
    model_metrics=None,
    model_metadata=None,
    feature_importance=None,
    backtest_metrics=None,
    risk_summary=None,
    monte_carlo_metrics=None,
    warnings=None,
    data_summary=None,
):
    return make_json_safe(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "disclaimer": describe_assumptions(config)[0],
            "assumptions": describe_assumptions(config),
            "data_summary": data_summary or {},
            "rule_result": rule_result or {},
            "ml_prediction": ml_prediction or {},
            "model_metrics": model_metrics or {},
            "model_metadata": model_metadata or {},
            "top_feature_importance": feature_importance.head(15).to_dict() if isinstance(feature_importance, pd.Series) else feature_importance or {},
            "backtest_metrics": backtest_metrics or {},
            "risk_summary": risk_summary or {},
            "monte_carlo_metrics": monte_carlo_metrics or {},
            "warnings": warnings or [],
            "limitations": [
                "Educational research only; not financial advice.",
                "Market data may be delayed, revised, incomplete, or provider-dependent.",
                "Backtests and Monte Carlo scenarios are assumption-driven and do not predict guaranteed outcomes.",
                "No brokerage integration or real-money trading readiness is provided.",
            ],
        }
    )


def research_report_to_payload(report, config=DEFAULT_ANALYSIS_CONFIG):
    return build_report_payload(
        ticker=report.ticker,
        config=config,
        rule_result=report.rule_result,
        ml_prediction=report.ml_prediction,
        model_metrics=report.training_result.metrics,
        model_metadata=report.training_result.metadata,
        feature_importance=report.feature_importance,
        backtest_metrics=report.backtest.metrics,
        risk_summary=report.risk_summary,
        monte_carlo_metrics=report.monte_carlo.metrics,
        warnings=report.warnings,
        data_summary={
            "rows": report.data_rows,
            "date_start": report.date_range[0],
            "date_end": report.date_range[1],
        },
    )


def report_payload_to_markdown(payload):
    lines = [
        f"# Investment Research Report: {payload['ticker']}",
        "",
        payload["disclaimer"],
        "",
        "## Assumptions",
    ]
    lines.extend(f"- {item}" for item in payload["assumptions"])

    for title, key in [
        ("Data Summary", "data_summary"),
        ("Rule-Based Result", "rule_result"),
        ("ML Prediction", "ml_prediction"),
        ("Model Metrics", "model_metrics"),
        ("Backtest Metrics", "backtest_metrics"),
        ("Risk Summary", "risk_summary"),
        ("Monte Carlo Metrics", "monte_carlo_metrics"),
    ]:
        lines.extend(["", f"## {title}"])
        section = payload.get(key, {})
        if isinstance(section, dict):
            lines.extend(f"- {name}: {value}" for name, value in section.items())
        else:
            lines.append(str(section))

    lines.extend(["", "## Warnings"])
    if payload["warnings"]:
        lines.extend(f"- {item}" for item in payload["warnings"])
    else:
        lines.append("- No guardrail warnings were generated for this run.")

    lines.extend(["", "## Limitations"])
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.append("")
    return "\n".join(lines)


def save_report_payload(payload, output_dir="reports", stem=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_ticker = str(payload["ticker"]).replace("/", "_").replace("\\", "_")
    stem = stem or f"{safe_ticker}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(report_payload_to_markdown(payload), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
