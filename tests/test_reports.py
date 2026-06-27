import json

import pandas as pd

from reports import build_report_payload, report_payload_to_markdown, save_report_payload


def test_build_report_payload_is_json_serializable():
    payload = build_report_payload(
        ticker="AAPL",
        feature_importance=pd.Series({"momentum": 0.2}),
        warnings=["Test warning"],
    )

    encoded = json.dumps(payload)

    assert "AAPL" in encoded
    assert payload["warnings"] == ["Test warning"]
    assert payload["limitations"]


def test_report_payload_to_markdown_includes_disclaimer_and_sections():
    payload = build_report_payload(ticker="AAPL", warnings=[])

    markdown = report_payload_to_markdown(payload)

    assert "Investment Research Report" in markdown
    assert "not financial advice" in markdown
    assert "Limitations" in markdown


def test_save_report_payload_writes_json_and_markdown(tmp_path):
    payload = build_report_payload(ticker="AAPL")

    paths = save_report_payload(payload, output_dir=tmp_path, stem="sample")

    assert paths["json"].exists()
    assert paths["markdown"].exists()
