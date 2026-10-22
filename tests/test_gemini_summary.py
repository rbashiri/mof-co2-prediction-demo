import io
import json
from urllib.error import HTTPError, URLError

import pytest

from src.gemini_summary import (
    GeminiSummaryError,
    FEATURE_LABELS,
    build_prompt,
    generate_gemini_summary,
    validate_summary,
)

UPTAKES = [0.02, 0.07, 0.22, 0.88, 4.33]
SECTIONS = {
    "Pressure comparison": "The endpoint difference is +4.31 mol/kg.",
    "Low-pressure SHAP (0.01 bar)": (
        "The strongest positive contribution is surface_area_m2g. "
        "The strongest negative contribution is total_ring_count."
    ),
    "Higher-pressure SHAP (2.5 bar)": (
        "The strongest positive contribution is total_ring_count. "
        "The strongest negative contribution is surface_area_m2g."
    ),
    "Limitations": "Simulated data; SHAP attribution is not chemical causation.",
}
TEXT = (
    "ADSORPTION BEHAVIOR\n"
    "Predicted uptake increases consistently across the evaluated pressure range. "
    "The predictions show no decrease between adjacent modeled pressures. "
    "This describes the model outputs, not measured adsorption behavior.\n\n"
    "STRUCTURE-PROPERTY INSIGHTS\n"
    "At the lowest pressure, surface area gives the strongest upward push relative "
    "to the corresponding model baseline, while the ring count in extracted linkers "
    "gives the strongest downward push. At the highest pressure, these leading "
    "contributors switch directions. These SHAP attributions explain the model "
    "prediction, not established chemical mechanisms. Their raw magnitudes should "
    "not be compared as relative importance across separate pressure models.\n\n"
    "RESEARCH IMPLICATIONS\n"
    "A hypothesis to test is whether the predicted pressure trend is reproduced "
    "by independent adsorption measurements. Such measurements could investigate "
    "agreement with the model without assuming that the identified descriptors "
    "cause adsorption or establish a performance advantage."
)


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("GEMINI_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-flash")


def payload(text=TEXT):
    return {"candidates": [{
        "finishReason": "STOP", "content": {"parts": [{"text": text}]},
    }]}


def test_mock_request_and_validation(configured, monkeypatch):
    calls = []

    def fake_open(request, timeout):
        calls.append(request)
        assert timeout == 25
        assert request.get_header("X-goog-api-key") == "test-only-key"
        body = json.loads(request.data)
        assert body["generationConfig"]["temperature"] == 0.1
        assert "scientific_sections" in body["contents"][0]["parts"][0]["text"]
        return io.BytesIO(json.dumps(payload()).encode())

    monkeypatch.setattr("src.gemini_summary.urlopen", fake_open)
    assert generate_gemini_summary(UPTAKES, SECTIONS) == TEXT
    assert len(calls) == 1


@pytest.mark.parametrize("status", [400, 403, 404, 429, 500, 503])
def test_http_errors_are_not_retried(configured, monkeypatch, status):
    calls = []

    def fail(request, timeout):
        calls.append(request)
        raise HTTPError("https://example.invalid", status, "Rejected", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr("src.gemini_summary.urlopen", fail)
    with pytest.raises(GeminiSummaryError):
        generate_gemini_summary(UPTAKES, SECTIONS)
    assert len(calls) == 1


def test_invalid_draft_is_not_retried(configured, monkeypatch):
    calls = []

    def fake_open(request, timeout):
        calls.append(request)
        return io.BytesIO(json.dumps(payload("too short")).encode())

    monkeypatch.setattr("src.gemini_summary.urlopen", fake_open)
    with pytest.raises(GeminiSummaryError, match="word requirement"):
        generate_gemini_summary(UPTAKES, SECTIONS)
    assert len(calls) == 1


def test_disabled_never_calls_network(monkeypatch):
    monkeypatch.delenv("GEMINI_ENABLED", raising=False)
    monkeypatch.setattr("src.gemini_summary.urlopen", lambda *a, **k: pytest.fail("Network call"))
    with pytest.raises(GeminiSummaryError, match="disabled"):
        generate_gemini_summary(UPTAKES, SECTIONS)


def test_missing_credentials(configured, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(GeminiSummaryError, match="GEMINI_API_KEY"):
        generate_gemini_summary(UPTAKES, SECTIONS)


@pytest.mark.parametrize("response", [
    {}, [], {"promptFeedback": None}, {"promptFeedback": "invalid"},
    {"promptFeedback": {"blockReason": "SAFETY"}},
    {"candidates": []},
    {"candidates": [{"finishReason": "MAX_TOKENS"}]},
    {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": None}]}}]},
])
def test_invalid_provider_responses(configured, monkeypatch, response):
    monkeypatch.setattr(
        "src.gemini_summary.urlopen",
        lambda *a, **k: io.BytesIO(json.dumps(response).encode()),
    )
    with pytest.raises(GeminiSummaryError):
        generate_gemini_summary(UPTAKES, SECTIONS)


@pytest.mark.parametrize("text", [
    "too short",
    TEXT + " 2.5 bar: 9.99 mol/kg.",
    TEXT + " An invented value is 123.456.",
    TEXT.replace("surface area", "feature_a"),
    TEXT.replace("surface area", "feature_0"),
    TEXT + " 2.5 bar: 4.33 mol/kg; 2.5 bar: 4.33 mol/kg.",
    TEXT.replace("RESEARCH IMPLICATIONS", "CONCLUSIONS"),
    TEXT.replace("ADSORPTION BEHAVIOR", "RESEARCH IMPLICATIONS"),
    TEXT + " extra" * 180,
])
def test_invalid_drafts(text):
    with pytest.raises(GeminiSummaryError):
        validate_summary(text, UPTAKES, build_prompt(UPTAKES, SECTIONS))


@pytest.mark.parametrize("error", [
    URLError("private provider diagnostic"),
    TimeoutError("private diagnostic"),
    HTTPError("https://example.invalid", 429, "private diagnostic", {}, None),
])
def test_network_failures_are_public_safe(configured, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr("src.gemini_summary.urlopen", fail)
    with pytest.raises(GeminiSummaryError) as caught:
        generate_gemini_summary(UPTAKES, SECTIONS)
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("body, expected", [
    ({"error": {"message": "private key or input", "details": [{"reason": "API_KEY_INVALID"}]}}, "rejected the API key"),
    ({"error": {"details": [{"reason": "SERVICE_DISABLED"}]}}, "not enabled"),
    ({"error": {"details": [{"reason": "API_KEY_SERVICE_BLOCKED"}]}}, "restrictions block"),
    ({"error": {"status": "FAILED_PRECONDITION"}}, "unmet account or project requirement"),
    ({"error": {"details": None, "message": "private input"}}, "Google rejected the request"),
    ([], "Google rejected the request"),
])
def test_http_400_diagnostics_are_safe(configured, monkeypatch, body, expected):
    def fail(*args, **kwargs):
        raise HTTPError(
            "https://example.invalid", 400, "private diagnostic", {},
            io.BytesIO(json.dumps(body).encode()),
        )
    monkeypatch.setattr("src.gemini_summary.urlopen", fail)
    with pytest.raises(GeminiSummaryError, match=expected) as caught:
        generate_gemini_summary(UPTAKES, SECTIONS)
    assert "private" not in str(caught.value)


def test_prompt_extrema_include_all_ties():
    prompt = build_prompt([1, 3, 3, 1, 2], SECTIONS)
    facts = json.loads(prompt.split("instructions:\n")[1])
    assert facts["highest_uptake_pressures_bar"] == [0.05, 0.1]
    assert facts["lowest_uptake_pressures_bar"] == [0.01, 0.5]


def test_interpretation_needs_no_repeated_prediction_table():
    assert "mol/kg" not in TEXT
    assert 120 <= len(TEXT.split()) <= 180
    validate_summary(TEXT, UPTAKES, build_prompt(UPTAKES, SECTIONS))


def test_optional_numeric_pair_must_match_its_pressure():
    prompt = build_prompt(UPTAKES, SECTIONS)
    validate_summary(TEXT + " 2.5 bar: 4.33 mol/kg.", UPTAKES, prompt)
    with pytest.raises(GeminiSummaryError, match="pressure/uptake pair"):
        validate_summary(TEXT + " 0.01 bar: 4.33 mol/kg.", UPTAKES, prompt)


def test_complete_table_is_rejected():
    pairs = "; ".join(f"{p:g} bar: {u:.6g} mol/kg" for p, u in zip(
        [0.01, 0.05, 0.1, 0.5, 2.5], UPTAKES, strict=True,
    ))
    with pytest.raises(GeminiSummaryError, match="instead of interpreting"):
        validate_summary(TEXT + " " + pairs, UPTAKES, build_prompt(UPTAKES, SECTIONS))


def test_instruction_numbers_are_not_evidence():
    with pytest.raises(GeminiSummaryError, match="unverified numeric"):
        validate_summary(TEXT + " A value is 180.", UPTAKES, build_prompt(UPTAKES, SECTIONS))


def test_prompt_labels_cover_frozen_feature_schema():
    from pathlib import Path
    import yaml

    manifest = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "docs/PHASE3/phase3-model-b-v1.yaml").read_text()
    )
    assert set(FEATURE_LABELS) == set(manifest["feature_schema"]["ordered_names"])


def test_prompt_separates_vocabulary_from_facts():
    prompt = build_prompt(UPTAKES, SECTIONS)
    facts = json.loads(prompt.split("instructions:\n")[1])
    assert facts["scientific_sections"] == SECTIONS
    assert facts["feature_labels"]["total_ring_count"] == "ring count in extracted linkers"
    assert "vocabulary, not evidence" in prompt
    assert "do not force a feature-switch story" in prompt
    assert "separate fixed limitations note" in prompt
    assert facts["feature_labels"]["void_fraction"] == "void fraction"
    assert facts["feature_labels"]["pld"] == "pore limiting diameter"
    assert "retaining exact scientific" in prompt
    assert "Do not infer selectivity, stability, or recyclability" in prompt
    assert "hypothesis to test, not a finding" in prompt
    assert facts["predictions"] == [
        {"pressure_bar": p, "uptake_mol_kg": u}
        for p, u in zip([0.01, 0.05, 0.1, 0.5, 2.5], UPTAKES, strict=True)
    ]


def test_prompt_contributors_follow_actual_shap_values():
    import numpy as np
    import pandas as pd
    import shap
    from src.scientific_summary import PRESSURES, generate_pressure_comparison

    names = list(FEATURE_LABELS)
    features = pd.DataFrame([[0.0] * 48], columns=names)
    predictions = pd.DataFrame({
        "Pressure (bar)": PRESSURES,
        "Predicted CO2 uptake (mol/kg)": UPTAKES,
    })

    def explanation(pressure_index, positive_feature, negative_feature):
        values = np.zeros(48)
        values[names.index(positive_feature)] = 0.01
        values[names.index(negative_feature)] = -0.005
        return shap.Explanation(
            values=values,
            data=features.iloc[0].to_numpy(),
            base_values=UPTAKES[pressure_index] - values.sum(),
            feature_names=names,
        )

    low = explanation(0, "surface_area_m2g", "total_ring_count")
    high = explanation(4, "total_ring_count", "surface_area_m2g")
    sections = generate_pressure_comparison(predictions, features, low, high)
    facts = json.loads(build_prompt(UPTAKES, sections).split("instructions:\n")[1])
    assert "positive contribution is surface_area_m2g" in facts["scientific_sections"]["Low-pressure SHAP (0.01 bar)"]
    assert "positive contribution is total_ring_count" in facts["scientific_sections"]["Higher-pressure SHAP (2.5 bar)"]

    high = explanation(4, "density", "total_HBA")
    changed = generate_pressure_comparison(predictions, features, low, high)
    changed_facts = json.loads(build_prompt(UPTAKES, changed).split("instructions:\n")[1])
    high_section = changed_facts["scientific_sections"]["Higher-pressure SHAP (2.5 bar)"]
    assert "positive contribution is density" in high_section
    assert "negative contribution is total_HBA" in high_section
    assert "total_ring_count" not in high_section
