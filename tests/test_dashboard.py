import json

import pytest

from hallucination_detector.dashboard import (
    flatten_claim_rows,
    highlighted_response_html,
    load_report_data,
)
from hallucination_detector.evaluation import ModelComparisonReport


def test_evaluation_report_saves_versioned_json_for_dashboard(tmp_path):
    report = ModelComparisonReport(
        responses=(),
        model_metrics=(),
        paired_comparisons=(),
        dataset_item_ids=("item-1",),
    )
    output_path = tmp_path / "reports" / "evaluation.json"

    report.save_json(output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["dataset_item_ids"] == ["item-1"]
    assert load_report_data(output_path.read_bytes()) == payload
    assert not list(output_path.parent.glob("*.tmp"))


def test_dashboard_flattens_claim_verification_and_evidence():
    report = {
        "schema_version": 1,
        "dataset_item_ids": ["item-1"],
        "model_metrics": [],
        "paired_comparisons": [],
        "responses": [
            {
                "response": {
                    "model_name": "model-a",
                    "dataset_item_id": "item-1",
                    "text": "Canberra is Australia's capital.",
                },
                "trust_score": {"score": 0.88},
                "claims": [
                    {
                        "verification": {
                            "claim": {"id": "claim-1", "text": "Canberra is Australia's capital."},
                            "label": "supported",
                            "similarity": 0.95,
                            "support_score": 0.9,
                            "contradiction_score": 0.01,
                            "explanation": "Evidence entails the claim.",
                            "evidence_scores": [],
                        },
                        "severity": {"severity": None, "rationale": "Supported."},
                        "retrieved_evidence": [
                            {
                                "evidence": {
                                    "id": "source-1",
                                    "text": "Canberra is the capital city of Australia.",
                                    "source": "reference",
                                },
                                "rank": 1,
                                "bm25_score": 2.1,
                                "embedding_score": 0.9,
                                "combined_score": 0.95,
                            }
                        ],
                    }
                ],
            }
        ],
    }

    rows = flatten_claim_rows(load_report_data(report))

    assert len(rows) == 1
    assert rows[0]["model_name"] == "model-a"
    assert rows[0]["verification_label"] == "supported"
    assert rows[0]["evidence_ids"] == ["source-1"]
    assert rows[0]["trust_score"]["score"] == 0.88


def test_dashboard_highlighting_checks_offsets_and_escapes_response_html():
    response = "Answer: <script>alert(1)</script>"
    claim = "<script>alert(1)</script>"

    rendered = highlighted_response_html(response, claim, 8, len(response), "contradicted")

    assert rendered is not None
    assert "<script>" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "<mark" in rendered
    assert highlighted_response_html(response, claim, 0, 3, "contradicted") is None


@pytest.mark.parametrize(
    "source, message",
    [
        (b"not utf-8: \xff", "UTF-8"),
        ("{bad json", "Invalid report JSON"),
        ("[]", "top level"),
        ({"schema_version": 2}, "schema_version"),
        ({"schema_version": 1, "dataset_item_ids": []}, "responses"),
    ],
)
def test_dashboard_rejects_invalid_report_files(source, message):
    with pytest.raises(ValueError, match=message):
        load_report_data(source)