"""Interactive Streamlit dashboard for Stage 9 research evaluation reports.

Run from the repository root after installing the optional dashboard extra:
``streamlit run src/hallucination_detector/dashboard.py``.
"""

from __future__ import annotations

import importlib
from html import escape
import json
from pathlib import Path
from typing import Any

_DEFAULT_REPORT = Path("data/evaluation_report.json")


def load_report_data(source: str | bytes | dict[str, Any]) -> dict[str, Any]:
    """Parse and validate the minimal versioned Stage 9 report structure."""
    if isinstance(source, bytes):
        try:
            source = source.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Report file must be UTF-8 encoded JSON.") from error
    if isinstance(source, str):
        try:
            report = json.loads(source)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid report JSON: {error.msg}.") from error
    else:
        report = source

    if not isinstance(report, dict):
        raise ValueError("Report JSON must contain an object at the top level.")
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported or missing evaluation report schema_version.")
    for key in ("dataset_item_ids", "responses", "model_metrics", "paired_comparisons"):
        if not isinstance(report.get(key), list):
            raise ValueError(f"Report field {key!r} must be a JSON array.")
    return report


def flatten_claim_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten nested report records into rows for filtering and CSV export."""
    rows: list[dict[str, Any]] = []
    for evaluated_response in report["responses"]:
        response = evaluated_response.get("response", {})
        for evaluated_claim in evaluated_response.get("claims", []):
            verification = evaluated_claim.get("verification", {})
            claim = verification.get("claim", {})
            severity_assessment = evaluated_claim.get("severity", {})
            retrieved = evaluated_claim.get("retrieved_evidence", [])
            evidence_scores = verification.get("evidence_scores", [])
            rows.append(
                {
                    "model_name": response.get("model_name", ""),
                    "dataset_item_id": response.get("dataset_item_id", ""),
                    "claim_id": claim.get("id", ""),
                    "claim_start": claim.get("start"),
                    "claim_end": claim.get("end"),
                    "response_text": response.get("text", ""),
                    "claim_text": claim.get("text", ""),
                    "verification_label": verification.get("label", "unknown"),
                    "severity": severity_assessment.get("severity"),
                    "similarity": verification.get("similarity"),
                    "support_score": verification.get("support_score"),
                    "contradiction_score": verification.get("contradiction_score"),
                    "verification_reason": verification.get("explanation", ""),
                    "severity_reason": severity_assessment.get("rationale", ""),
                    "evidence_ids": [
                        item.get("evidence", {}).get("id", "") for item in retrieved
                    ],
                    "evidence": retrieved,
                    "evidence_scores": evidence_scores,
                    "trust_score": evaluated_response.get("trust_score"),
                }
            )
    return rows


def highlighted_response_html(
    response_text: str,
    claim_text: str,
    start: Any,
    end: Any,
    verification_label: str,
) -> str | None:
    """Return an escaped response with its valid claim span marked, if present."""
    if (
        not isinstance(start, int)
        or not isinstance(end, int)
        or not 0 <= start < end <= len(response_text)
        or response_text[start:end] != claim_text
    ):
        return None
    highlight_color = {
        "supported": "#d7f5e8",
        "contradicted": "#ffe0e0",
        "unknown": "#fff0ce",
    }.get(verification_label, "#e8eef5")
    return (
        escape(response_text[:start])
        + f'<mark style="background:{highlight_color};padding:.12rem .22rem;border-radius:.25rem">'
        + escape(response_text[start:end])
        + "</mark>"
        + escape(response_text[end:])
    )


def _load_dashboard_packages() -> tuple[Any, Any, Any]:
    try:
        streamlit = importlib.import_module("streamlit")
        pandas = importlib.import_module("pandas")
        plotly_express = importlib.import_module("plotly.express")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            'Dashboard packages are optional. Install them with '
            '`pip install -e ".[dashboard]"`.'
        ) from error
    return streamlit, pandas, plotly_express


def _apply_styles(st: Any) -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --ink: #17243a; --muted: #718096; --line: #e6edf5; }
        .stApp { background: linear-gradient(180deg, #f5f8fc 0%, #f8fafc 36%, #f3f6fa 100%); }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink); }
        .hero { padding: 1.6rem 1.8rem; border-radius: 22px; color: #f8fbff;
                background: linear-gradient(125deg, #102a43 0%, #174e70 55%, #337d8f 100%);
                box-shadow: 0 14px 32px rgba(16,42,67,.16); margin: .25rem 0 1.4rem; }
        .hero h1 { color: white !important; margin: 0 0 .3rem; font-size: 2.2rem; }
        .hero p { color: #d6eaf0; margin: 0; font-size: 1rem; }
        div[data-testid="stMetric"] { background: white; padding: 1rem 1.1rem;
            border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 5px 18px rgba(23,36,58,.04); }
        div[data-testid="stMetricLabel"] { color: #708090; }
        div[data-testid="stMetricValue"] { color: var(--ink); font-family: 'Space Grotesk', sans-serif; }
        div[data-testid="stTabs"] button { font-weight: 600; }
        .subtle { color: var(--muted); font-size: .92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _percent(value: Any) -> str:
    return "—" if value is None else f"{value:.1%}"


def _render_dashboard(report: dict[str, Any], raw_json: bytes, st: Any, pd: Any, px: Any) -> None:
    rows = flatten_claim_rows(report)
    all_models = [metric.get("model_name", "") for metric in report["model_metrics"]]
    all_items = [str(item_id) for item_id in report["dataset_item_ids"]]

    st.sidebar.markdown("## Research filters")
    selected_models = st.sidebar.multiselect("Models", all_models, default=all_models)
    selected_items = st.sidebar.multiselect("Dataset items", all_items, default=all_items)
    filtered_rows = [
        row
        for row in rows
        if row["model_name"] in selected_models
        and row["dataset_item_id"] in selected_items
    ]
    selected_responses = [
        item
        for item in report["responses"]
        if item.get("response", {}).get("model_name") in selected_models
        and item.get("response", {}).get("dataset_item_id") in selected_items
    ]

    total_claims = len(filtered_rows)
    contradicted = sum(row["verification_label"] == "contradicted" for row in filtered_rows)
    unknown = sum(row["verification_label"] == "unknown" for row in filtered_rows)
    high_severity = sum(row["severity"] in {"severe", "critical"} for row in filtered_rows)
    filtered_dataset_count = len({row["dataset_item_id"] for row in filtered_rows})

    st.markdown(
        """
        <section class="hero">
          <h1>Reliability, made inspectable.</h1>
          <p>Compare model factuality, risk severity, evidence, and confidence from one shared evaluation pipeline.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Stage 10 · Risk-aware hallucination research dashboard")
    kpi_columns = st.columns(5)
    kpi_columns[0].metric("Dataset items", filtered_dataset_count)
    kpi_columns[1].metric("Responses", len(selected_responses))
    kpi_columns[2].metric("Claims", total_claims)
    kpi_columns[3].metric("Contradicted", contradicted)
    kpi_columns[4].metric("Severe + critical", high_severity)

    tab_overview, tab_models, tab_risk, tab_claims, tab_evidence, tab_trust = st.tabs(
        ["Overview", "Model comparison", "Risk & severity", "Claim explorer", "Evidence explorer", "Trust score"]
    )

    with tab_overview:
        st.subheader("Benchmark snapshot")
        if not selected_models:
            st.info("Select at least one model in the sidebar to display charts.")
        elif not filtered_rows:
            st.info("No claims match the selected filters yet.")
        else:
            st.markdown(
                f"<p class='subtle'>Across the current selection, {contradicted} claims are contradicted "
                f"and {unknown} remain unknown. Unknown is kept separate from proven errors.</p>",
                unsafe_allow_html=True,
            )
            selected_metrics = [
                item for item in report["model_metrics"]
                if item.get("model_name") in selected_models
            ]
            if selected_metrics:
                rates = pd.DataFrame(
                    [
                        {
                            "Model": item["model_name"],
                            "Supported": item.get("supported_rate") or 0.0,
                            "Contradicted": item.get("contradicted_rate") or 0.0,
                            "Unknown": item.get("unknown_rate") or 0.0,
                        }
                        for item in selected_metrics
                    ]
                ).melt(id_vars="Model", var_name="Outcome", value_name="Claim rate")
                chart = px.bar(
                    rates,
                    x="Model",
                    y="Claim rate",
                    color="Outcome",
                    barmode="group",
                    color_discrete_map={
                        "Supported": "#21a179",
                        "Contradicted": "#e05a5a",
                        "Unknown": "#e4a83a",
                    },
                    title="Claim outcome rates · full benchmark",
                )
                chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_tickformat=".0%")
                st.plotly_chart(chart, use_container_width=True)
        st.download_button(
            "Download evaluation report (JSON)",
            data=raw_json,
            file_name="evaluation_report.json",
            mime="application/json",
        )

    with tab_models:
        st.subheader("Model comparison")
        selected_metrics = [
            item for item in report["model_metrics"]
            if item.get("model_name") in selected_models
        ]
        if not selected_metrics:
            st.info("Select one or more models to compare.")
        else:
            metrics_table = pd.DataFrame(
                [
                    {
                        "Model": item.get("model_name"),
                        "Responses": item.get("response_count"),
                        "Claims": item.get("claim_count"),
                        "Supported": item.get("supported_rate"),
                        "Contradicted": item.get("contradicted_rate"),
                        "Unknown": item.get("unknown_rate"),
                        "Severity-weighted risk": item.get("severity_weighted_error_rate"),
                        "Average trust": item.get("average_trust_score"),
                        "Verification accuracy": item.get("verification_accuracy"),
                        "Macro F1": item.get("verification_macro_f1"),
                        f"Recall@{selected_metrics[0].get('retrieval_k', '?')}": item.get("retrieval_recall_at_k"),
                    }
                    for item in selected_metrics
                ]
            )
            st.dataframe(
                metrics_table.style.format(
                    {
                        "Supported": _percent,
                        "Contradicted": _percent,
                        "Unknown": _percent,
                        "Severity-weighted risk": _percent,
                        "Average trust": _percent,
                        "Verification accuracy": _percent,
                        "Macro F1": _percent,
                    },
                    na_rep="—",
                ),
                use_container_width=True,
                hide_index=True,
            )

        comparisons = [
            item for item in report["paired_comparisons"]
            if item.get("model_a") in selected_models and item.get("model_b") in selected_models
        ]
        st.markdown("#### Paired item-level comparisons")
        comparison_rows = []
        for item in comparisons:
            interval = item.get("supported_rate_difference") or {}
            comparison_rows.append(
                {
                    "Model A": item.get("model_a"),
                    "Model B": item.get("model_b"),
                    "Paired items": item.get("paired_item_count"),
                    "Supported-rate Δ (A − B)": interval.get("mean_difference"),
                    "CI lower": interval.get("lower"),
                    "CI upper": interval.get("upper"),
                    "Exact McNemar p": item.get("mcnemar_exact_p_value"),
                }
            )
        if comparison_rows:
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No pairwise comparison is available for the selected models.")
        st.caption("Confidence intervals and p-values are descriptive; small benchmark samples may be inconclusive.")

    with tab_risk:
        st.subheader("Risk profile")
        severity_rows = []
        for row in filtered_rows:
            if row["severity"]:
                severity_rows.append({"Model": row["model_name"], "Severity": row["severity"].title(), "Claims": 1})
        if severity_rows:
            severity_frame = pd.DataFrame(severity_rows).groupby(["Model", "Severity"], as_index=False)["Claims"].sum()
            chart = px.bar(
                severity_frame,
                x="Model",
                y="Claims",
                color="Severity",
                barmode="stack",
                category_orders={"Severity": ["Mild", "Moderate", "Severe", "Critical"]},
                color_discrete_map={"Mild": "#9bb8d3", "Moderate": "#e4a83a", "Severe": "#e97843", "Critical": "#a83255"},
                title="Severity distribution for non-supported claims",
            )
            chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(chart, use_container_width=True)
            critical = [row for row in filtered_rows if row["severity"] == "critical"]
            st.markdown("#### Critical claims requiring review")
            if critical:
                critical_frame = pd.DataFrame(
                    [
                        {
                            "Model": row["model_name"],
                            "Dataset item": row["dataset_item_id"],
                            "Claim": row["claim_text"],
                            "Reason": row["severity_reason"],
                        }
                        for row in critical
                    ]
                )
                st.dataframe(critical_frame, use_container_width=True, hide_index=True)
            else:
                st.success("No critical claims in the current filter.")
        else:
            st.info("No non-supported claims have severity assessments in this selection.")

    row, selected_claim_index = _claim_picker(filtered_rows, st)
    with tab_claims:
        st.subheader("Claim-level analysis")
        if row is None:
            st.info("Choose filters that include evaluated claims to inspect details.")
        else:
            st.caption(f"{row['model_name']} · {row['dataset_item_id']} · {row['claim_id']}")
            st.markdown("**Extracted claim**")
            st.info(row["claim_text"])
            with st.expander("Original model response", expanded=False):
                response_text = row["response_text"]
                highlighted = highlighted_response_html(
                    response_text,
                    row["claim_text"],
                    row["claim_start"],
                    row["claim_end"],
                    row["verification_label"],
                )
                if highlighted is not None:
                    st.markdown(highlighted, unsafe_allow_html=True)
                else:
                    st.code(response_text, language=None)
            outcome_col, severity_col, similarity_col = st.columns(3)
            outcome_col.metric("Verification", str(row["verification_label"]).title())
            severity_col.metric("Severity", str(row["severity"]).title() if row["severity"] else "None")
            similarity_col.metric("Semantic similarity", _percent(row["similarity"]))
            st.markdown("**Verification explanation**")
            st.write(row["verification_reason"] or "No explanation recorded.")
            if row["severity_reason"]:
                st.markdown("**Severity rationale**")
                st.write(row["severity_reason"])
            csv_data = pd.DataFrame(filtered_rows).drop(
                columns=["evidence", "evidence_scores", "trust_score"], errors="ignore"
            ).to_csv(index=False).encode("utf-8")
            st.download_button("Download filtered claim data (CSV)", csv_data, "claim_analysis.csv", "text/csv")

    with tab_evidence:
        st.subheader("Evidence explorer")
        if row is None:
            st.info("Select a claim in the Claim explorer tab after choosing models/items.")
        else:
            st.markdown(f"**Claim:** {row['claim_text']}")
            if not row["evidence"]:
                st.warning("No retrieved passages were stored for this claim.")
            else:
                score_by_id = {
                    item.get("evidence", {}).get("id"): item
                    for item in row["evidence_scores"]
                }
                for rank, candidate in enumerate(row["evidence"], start=1):
                    evidence = candidate.get("evidence", {})
                    score = score_by_id.get(evidence.get("id"), {})
                    with st.expander(
                        f"#{candidate.get('rank', rank)} · {evidence.get('id', 'source')} · "
                        f"combined {candidate.get('combined_score', 0.0):.3f}",
                        expanded=rank == 1,
                    ):
                        st.write(evidence.get("text", ""))
                        st.caption(f"Source: {evidence.get('source') or 'not specified'}")
                        left, middle, right = st.columns(3)
                        left.metric("BM25", f"{candidate.get('bm25_score', 0.0):.3f}")
                        middle.metric("Embedding", f"{candidate.get('embedding_score', 0.0):.3f}")
                        right.metric("Combined retrieval", f"{candidate.get('combined_score', 0.0):.3f}")
                        if score:
                            st.write(
                                {
                                    "Similarity": score.get("similarity"),
                                    "Entailment probability": score.get("entailment_probability"),
                                    "Contradiction probability": score.get("contradiction_probability"),
                                    "Neutral probability": score.get("neutral_probability"),
                                    "Adjusted support": score.get("support_score"),
                                    "Adjusted contradiction": score.get("contradiction_score"),
                                }
                            )

    with tab_trust:
        st.subheader("Trust score components")
        trust_rows = []
        for response in selected_responses:
            score = response.get("trust_score")
            if score:
                trust_rows.append(
                    {
                        "Model": response["response"].get("model_name"),
                        "Dataset item": response["response"].get("dataset_item_id"),
                        "Overall trust": score.get("score"),
                        "Factuality": score.get("factual_score"),
                        "Semantic alignment": score.get("semantic_alignment"),
                        "Consistency": score.get("consistency_score"),
                        "Confidence calibration": score.get("confidence_calibration"),
                        "Severity penalty": score.get("severity_penalty"),
                    }
                )
        if trust_rows:
            trust_frame = pd.DataFrame(trust_rows)
            st.dataframe(
                trust_frame.style.format(
                    {column: _percent for column in trust_frame.columns if column not in {"Model", "Dataset item"}},
                    na_rep="—",
                ),
                use_container_width=True,
                hide_index=True,
            )
            component_frame = trust_frame.drop(columns=["Dataset item"]).melt(
                id_vars="Model", var_name="Signal", value_name="Score"
            )
            chart = px.line(
                component_frame,
                x="Signal",
                y="Score",
                color="Model",
                markers=True,
                title="Trust signals by model and response",
            )
            chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 1])
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Trust scores require external consistency scores; none were saved for these responses.")


def _claim_picker(rows: list[dict[str, Any]], st: Any) -> tuple[dict[str, Any] | None, int | None]:
    if not rows:
        return None, None
    choices = list(range(len(rows)))
    selected = st.sidebar.selectbox(
        "Claim to inspect",
        choices,
        format_func=lambda index: (
            f"{rows[index]['model_name']} / {rows[index]['dataset_item_id']} · "
            f"{rows[index]['claim_text'][:72]}"
        ),
    )
    return rows[selected], selected


def main() -> None:
    """Launch the interactive dashboard app."""
    try:
        st, pd, px = _load_dashboard_packages()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error

    st.set_page_config(
        page_title="LLM Reliability Lab",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_styles(st)
    st.sidebar.markdown("# LLM Reliability Lab")
    uploaded = st.sidebar.file_uploader("Load Stage 9 evaluation report", type=["json"])

    if uploaded is not None:
        raw_json = uploaded.getvalue()
    elif _DEFAULT_REPORT.exists():
        raw_json = _DEFAULT_REPORT.read_bytes()
        st.sidebar.caption(f"Loaded `{_DEFAULT_REPORT.as_posix()}`")
    else:
        st.title("Start with an evaluation report")
        st.markdown(
            "Upload a JSON report exported from `ModelComparisonReport.save_json()`. "
            "You can also save one to `data/evaluation_report.json` and refresh this page."
        )
        st.code(
            'report = evaluator.evaluate(responses, consistency_scores=consistency_scores)\n'
            'report.save_json("data/evaluation_report.json")',
            language="python",
        )
        st.stop()

    try:
        report = load_report_data(raw_json)
    except ValueError as error:
        st.error(str(error))
        st.stop()
        return

    _render_dashboard(report, raw_json, st, pd, px)


if __name__ == "__main__":
    main()