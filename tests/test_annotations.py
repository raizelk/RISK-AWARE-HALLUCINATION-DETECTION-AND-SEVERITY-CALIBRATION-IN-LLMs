import json

import pytest

from hallucination_detector.annotations import (
    compare_independent_annotations,
    load_adjudicated_annotations,
)
from hallucination_detector.baseline import extract_claims
from hallucination_detector.domain import ModelResponse, Severity, VerificationLabel
from hallucination_detector.generation import JsonlResponseStore
from scripts.export_annotation_template import export_template


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _responses():
    return (ModelResponse("model-a", "item-1", "Canberra is capital. Sydney is not."),)


def _row(claim, *, status, annotator_id, label, severity, evidence_ids):
    return {
        "schema_version": 1,
        "annotation_status": status,
        "annotator_id": annotator_id,
        "claim_review_status": "accepted",
        "model_name": "model-a",
        "dataset_item_id": "item-1",
        "claim_id": claim.id,
        "claim_text": claim.text,
        "claim_start": claim.start,
        "claim_end": claim.end,
        "verification_label": label,
        "evidence_ids": evidence_ids,
        "severity": severity,
        "rationale": "Reviewed against the cited reference.",
    }


def test_adjudicated_annotations_convert_to_evaluator_gold_maps(tmp_path):
    claims = extract_claims(_responses()[0].text)
    rows = [
        _row(
            claims[0],
            status="adjudicated",
            annotator_id="adjudicator-1",
            label="supported",
            severity=None,
            evidence_ids=["evidence-1"],
        ),
        _row(
            claims[1],
            status="adjudicated",
            annotator_id="adjudicator-1",
            label="contradicted",
            severity="moderate",
            evidence_ids=["evidence-2"],
        ),
    ]
    path = tmp_path / "adjudicated.jsonl"
    _write_jsonl(path, rows)

    gold = load_adjudicated_annotations(path, _responses(), {"evidence-1", "evidence-2"}, require_complete=True)

    first_key = ("model-a", "item-1", "claim-1")
    second_key = ("model-a", "item-1", "claim-2")
    assert gold.labels == {
        first_key: VerificationLabel.SUPPORTED,
        second_key: VerificationLabel.CONTRADICTED,
    }
    assert gold.evidence_ids[second_key] == {"evidence-2"}
    assert gold.severity[second_key] is Severity.MODERATE


def test_annotations_reject_changed_claim_spans_and_unreviewed_claims(tmp_path):
    claim = extract_claims(_responses()[0].text)[0]
    row = _row(
        claim,
        status="adjudicated",
        annotator_id="adjudicator-1",
        label="supported",
        severity=None,
        evidence_ids=["evidence-1"],
    )
    path = tmp_path / "annotation.jsonl"

    row["claim_start"] += 1
    _write_jsonl(path, [row])
    with pytest.raises(ValueError, match="claim text and offsets"):
        load_adjudicated_annotations(path, _responses(), {"evidence-1"})

    row["claim_start"] = claim.start
    row["claim_review_status"] = "pending"
    _write_jsonl(path, [row])
    with pytest.raises(ValueError, match="reviewed and marked 'accepted'"):
        load_adjudicated_annotations(path, _responses(), {"evidence-1"})


def test_independent_annotation_agreement_reports_disagreements_and_kappa(tmp_path):
    claims = extract_claims(_responses()[0].text)
    annotator_a = tmp_path / "annotator-a.jsonl"
    annotator_b = tmp_path / "annotator-b.jsonl"
    _write_jsonl(
        annotator_a,
        [
            _row(claims[0], status="independent", annotator_id="a", label="supported", severity=None, evidence_ids=["evidence-1"]),
            _row(claims[1], status="independent", annotator_id="a", label="contradicted", severity="moderate", evidence_ids=["evidence-2"]),
        ],
    )
    _write_jsonl(
        annotator_b,
        [
            _row(claims[0], status="independent", annotator_id="b", label="supported", severity=None, evidence_ids=["evidence-1"]),
            _row(claims[1], status="independent", annotator_id="b", label="unknown", severity="mild", evidence_ids=[]),
        ],
    )

    agreement = compare_independent_annotations(
        annotator_a,
        annotator_b,
        _responses(),
        {"evidence-1", "evidence-2"},
    )

    assert agreement.compared_claims == 2
    assert agreement.exact_agreement == 0.5
    assert agreement.cohen_kappa == pytest.approx(1 / 3)
    assert agreement.disagreement_keys == (("model-a", "item-1", "claim-2"),)


def test_exporter_emits_prediction_free_worksheet_with_exact_claim_spans(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    responses = tmp_path / "responses.jsonl"
    output = tmp_path / "annotator-a.jsonl"
    _write_jsonl(
        prompts,
        [
            {
                "dataset_item_id": "item-1",
                "domain": "healthcare",
                "topic": "topic-a",
                "task_type": "factual",
                "risk_level": "high",
                "prompt": "What is true?",
                "reference_evidence_ids": ["evidence-1"],
            }
        ],
    )
    _write_jsonl(
        evidence,
        [
            {
                "evidence_id": "evidence-1",
                "topic": "topic-a",
                "text": "Canberra is capital.",
                "source": "Example source",
                "source_url": "https://example.test/source",
                "publication_date": "2025-01-01",
            }
        ],
    )
    JsonlResponseStore().save(
        responses,
        (
            ModelResponse("model-a", "item-1", "Canberra is capital. Sydney is not."),
            ModelResponse("model-b", "item-1", "Canberra is capital. Sydney is not."),
        ),
    )

    exported_count = export_template(
        responses,
        prompts,
        evidence,
        output,
        annotator_id="annotator-a",
        allow_unfrozen_pilot=True,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert exported_count == 4
    assert rows[0]["claim_text"] == "Canberra is capital."
    assert rows[0]["response_text"][rows[0]["claim_start"]:rows[0]["claim_end"]] == rows[0]["claim_text"]
    assert rows[0]["risk_level"] == "high"
    assert rows[0]["verification_label"] is None
    assert rows[0]["evidence_candidates"][0]["evidence_id"] == "evidence-1"


def test_exporter_requires_frozen_benchmark_outside_pilot_mode(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    evidence = tmp_path / "evidence.jsonl"
    responses = tmp_path / "responses.jsonl"
    _write_jsonl(
        prompts,
        [{"dataset_item_id": "item-1", "topic": "topic-a", "prompt": "Question?", "reference_evidence_ids": ["evidence-1"]}],
    )
    _write_jsonl(
        evidence,
        [{"evidence_id": "evidence-1", "topic": "topic-a", "text": "Evidence.", "source": "Source"}],
    )
    JsonlResponseStore().save(responses, (ModelResponse("model-a", "item-1", "Claim."),))

    with pytest.raises(ValueError, match="no valid freeze manifest"):
        export_template(
            responses,
            prompts,
            evidence,
            tmp_path / "out.jsonl",
            annotator_id="annotator-a",
        )