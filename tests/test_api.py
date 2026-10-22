import base64
import io
import math
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.paths import MODELS_DIR
from src.train import GEOMETRY_COLUMNS


def geometry_form(values=None):
    base = {name: 1.0 for name in GEOMETRY_COLUMNS}
    if values:
        base.update(values)
    return base


def cif_file(content: bytes = b"data_test\n_cell_length_a 10\n"):
    return ("test.cif", io.BytesIO(content))


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_health_returns_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["pressures"] == [0.01, 0.05, 0.1, 0.5, 2.5]
    assert data["sbu_path"] is True or data["sbu_path"] is False


def test_health_returns_model_info(client):
    response = client.get("/api/health")
    data = response.json()
    assert data["status"] == "ok"
    assert "model_label" in data


def test_empty_cif_returns_error(client):
    response = client.post(
        "/api/predict",
        data=geometry_form(),
        files={"cif": ("empty.cif", io.BytesIO(b""))},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]["error"].lower()


def test_missing_geometry_field_returns_error(client):
    response = client.post(
        "/api/predict",
        data={"lcd": 10.0, "pld": 9.0, "void_fraction": 0.5},
        files={"cif": cif_file()},
    )
    assert response.status_code == 422


def test_void_fraction_out_of_range_returns_error(client):
    response = client.post(
        "/api/predict",
        data=geometry_form({"void_fraction": 1.5}),
        files={"cif": cif_file()},
    )
    assert response.status_code == 422


def test_predict_returns_five_finite_pressures(client, monkeypatch):
    from src.api import _state
    names = _state["feature_names"]

    def build(path, geometry, *, feature_names, **kwargs):
        return pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)

    monkeypatch.setattr("src.api.build_phase3_features", build)

    response = client.post(
        "/api/predict",
        data=geometry_form(),
        files={"cif": cif_file(b"data_test\n_cell_length_a 10\n")},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    assert len(data["pressures"]) == 5
    for point in data["pressures"]:
        assert 0.01 <= point["pressure"] <= 2.5
        assert isinstance(point["uptake"], float)

    assert len(data["features"]) == 48
    assert set(data["shap"].keys()) == {"0.01", "0.05", "0.1", "0.5", "2.5"}
    for shap_data in data["shap"].values():
        assert "baseline" in shap_data
        assert "contributions" in shap_data
        assert "feature_names" in shap_data
        assert len(shap_data["feature_names"]) == 48

    assert "Pressure comparison" in data["summary"]
    assert "Low-pressure SHAP (0.01 bar)" in data["summary"]
    assert "Higher-pressure SHAP (2.5 bar)" in data["summary"]
    assert data["gemini_summary"] is None
    assert data["gemini_error"] is None

    csv_bytes = base64.b64decode(data["predictions_csv"])
    assert b"Pressure" in csv_bytes


@pytest.mark.parametrize("fails", [False, True])
def test_optional_gemini_preserves_prediction_results(client, monkeypatch, fails):
    from src.gemini_summary import GeminiSummaryError

    def build(path, geometry, *, feature_names, **kwargs):
        return pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)

    calls = []

    def generate(uptakes, sections):
        calls.append((uptakes, sections))
        if fails:
            raise GeminiSummaryError("Mock provider unavailable")
        return "Mock validated draft"

    monkeypatch.setattr("src.api.build_phase3_features", build)
    monkeypatch.setattr("src.api.generate_gemini_summary", generate)
    baseline = client.post(
        "/api/predict", data=geometry_form(), files={"cif": cif_file()},
    ).json()
    assert calls == []
    response = client.post(
        "/api/predict", data={**geometry_form(), "include_gemini": "true"},
        files={"cif": cif_file()},
    )
    assert response.status_code == 200
    result = response.json()
    for key in ("pressures", "shap", "features", "summary", "predictions_csv"):
        assert result[key] == baseline[key]
    assert len(calls) == 1
    assert calls[0][0] == [point["uptake"] for point in result["pressures"]]
    assert calls[0][1] == result["summary"]
    if fails:
        assert result["gemini_summary"] is None
        assert result["gemini_error"] == "Mock provider unavailable"
    else:
        assert result["gemini_summary"] == "Mock validated draft"
        assert result["gemini_error"] is None


def test_temp_cif_file_is_removed(client, monkeypatch):
    from src.api import _state
    names = _state["feature_names"]
    paths = []

    def build(path, geometry, *, feature_names, **kwargs):
        paths.append(Path(path))
        assert path.read_bytes() == b"CIF"
        raise ValueError("Extraction rejected")

    monkeypatch.setattr("src.api.build_phase3_features", build)

    response = client.post(
        "/api/predict",
        data=geometry_form(),
        files={"cif": ("test.cif", io.BytesIO(b"CIF"))},
    )
    assert response.status_code == 400
    assert paths
    assert not paths[0].exists()


def test_shap_additivity_for_zero_features(client, monkeypatch):
    from src.api import _state
    names = _state["feature_names"]

    def build(path, geometry, *, feature_names, **kwargs):
        return pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)

    monkeypatch.setattr("src.api.build_phase3_features", build)

    response = client.post(
        "/api/predict",
        data=geometry_form(),
        files={"cif": cif_file(b"data_test\n_cell_length_a 10\n")},
    )
    assert response.status_code == 200
    data = response.json()
    for pressure_str, shap_data in data["shap"].items():
        pressure = float(pressure_str)
        uptake = next(p["uptake"] for p in data["pressures"] if p["pressure"] == pressure)
        total = shap_data["baseline"] + sum(shap_data["contributions"])
        assert abs(total - uptake) < 1e-4


def test_cif_only_request_extracts_geometry_before_features(client, monkeypatch):
    geometry = {"lcd": 11.76, "pld": 7.89, "void_fraction": 0.36, "surface_area_m2g": 3544.0}
    paths = []

    def extract(path, **kwargs):
        paths.append(path)
        assert path.read_bytes() == b"data_test\n"
        return geometry

    def build(path, supplied, *, feature_names, **kwargs):
        assert path == paths[0]
        assert supplied == geometry
        return pd.DataFrame(
            [[geometry.get(name, 0.0) for name in feature_names]], columns=feature_names,
        )

    monkeypatch.setattr("src.api.extract_geometry", extract)
    monkeypatch.setattr("src.api.build_phase3_features", build)
    response = client.post("/api/predict", files={"cif": cif_file(b"data_test\n")})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["geometry"] == geometry
    assert result["geometry_source"] == "zeopp"
    assert result["geometry_settings"]["surface_probe_radius_angstrom"] == 1.65
    assert result["geometry_settings"]["volume_probe_radius_angstrom"] == 1.2
    assert result["geometry_settings"]["profile"] == "experimental-sample-comparison-v1"
    assert result["geometry_warning"]
    assert len(result["features"]) == 48
    assert len(result["pressures"]) == 5
    assert not paths[0].exists()


def test_manual_geometry_does_not_call_zeopp(client, monkeypatch):
    def extract(*args, **kwargs):
        pytest.fail("Manual requests must not require Zeo++.")

    def build(path, geometry, *, feature_names, **kwargs):
        return pd.DataFrame(
            [[geometry.get(name, 0.0) for name in feature_names]], columns=feature_names,
        )

    monkeypatch.setattr("src.api.extract_geometry", extract)
    monkeypatch.setattr("src.api.build_phase3_features", build)
    response = client.post(
        "/api/predict", data=geometry_form(), files={"cif": cif_file()},
    )
    assert response.status_code == 200
    assert response.json()["geometry_source"] == "manual"
    assert response.json()["geometry_warning"] is None


@pytest.mark.parametrize("error", [
    FileNotFoundError("Zeo++ network not found"),
    ValueError("Zeo++ timed out"),
    ValueError("Zeo++ malformed output"),
])
def test_automatic_geometry_failure_stops_prediction(client, monkeypatch, error):
    paths = []

    def extract(path, **kwargs):
        paths.append(path)
        raise error

    def build(*args, **kwargs):
        pytest.fail("Features must not be built after geometry fails.")

    monkeypatch.setattr("src.api.extract_geometry", extract)
    monkeypatch.setattr("src.api.build_phase3_features", build)
    response = client.post("/api/predict", files={"cif": cif_file()})
    assert response.status_code == 400
    assert str(error) in response.json()["detail"]["error"]
    assert not paths[0].exists()


@pytest.mark.parametrize("filename", ["hMOF-0", "hMOF-1"])
def test_real_cif_only_prediction(client, filename):
    if not all(os.environ.get(name) for name in ("ZEO_NETWORK_PATH", "MOF_SBU_PATH")):
        pytest.skip("Set ZEO_NETWORK_PATH and MOF_SBU_PATH for the native pipeline test.")
    path = Path(__file__).resolve().parents[1] / "notebook/sample/cif_files" / f"{filename}.cif"
    response = client.post(
        "/api/predict", files={"cif": (path.name, path.read_bytes(), "application/octet-stream")},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["geometry_source"] == "zeopp"
    assert set(result["geometry"]) == set(GEOMETRY_COLUMNS)
    assert len(result["features"]) == 48
    assert all(math.isfinite(value) for value in result["features"].values())
    assert len(result["pressures"]) == 5
    assert all(math.isfinite(point["uptake"]) for point in result["pressures"])
    for point in result["pressures"]:
        explanation = result["shap"][str(point["pressure"])]
        assert explanation["baseline"] + sum(explanation["contributions"]) == pytest.approx(
            point["uptake"], abs=1e-4,
        )


@pytest.mark.parametrize("fails", [False, True])
def test_deferred_explanation_does_not_repeat_prediction(client, monkeypatch, fails):
    from src.gemini_summary import GeminiSummaryError

    builds = []
    calls = []

    def build(path, geometry, *, feature_names, **kwargs):
        builds.append(path)
        return pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)

    def generate(uptakes, sections):
        calls.append((uptakes, sections))
        if fails:
            raise GeminiSummaryError("Provider unavailable")
        return "Explanation of existing results"

    monkeypatch.setattr("src.api.build_phase3_features", build)
    monkeypatch.setattr("src.api.generate_gemini_summary", generate)
    response = client.post("/api/predict", data=geometry_form(), files={"cif": cif_file()})
    assert response.status_code == 200
    result = response.json()
    assert calls == []
    assert result["gemini_summary"] is None
    explained = client.post(f"/api/explanations/{result['explanation_token']}")
    assert len(builds) == 1
    assert calls == [([point["uptake"] for point in result["pressures"]], result["summary"])]
    if fails:
        assert explained.status_code == 503
        assert explained.json()["detail"]["error"] == "Provider unavailable"
    else:
        assert explained.status_code == 200
        assert explained.json()["gemini_summary"] == "Explanation of existing results"


def test_explanation_expiry_and_cache_bound(client, monkeypatch):
    from src.api import EXPLANATION_CACHE_SIZE, EXPLANATION_TTL_SECONDS, _remember_explanation, _state

    monkeypatch.setattr("src.api.time.monotonic", lambda: 100.0)
    token = _remember_explanation([1.0] * 5, {"summary": "validated"})
    assert client.post("/api/explanations/unknown").status_code == 410
    monkeypatch.setattr("src.api.time.monotonic", lambda: 100.0 + EXPLANATION_TTL_SECONDS)
    assert client.post(f"/api/explanations/{token}").status_code == 410
    assert token not in _state["explanation_requests"]
    first = _remember_explanation([1.0] * 5, {})
    for _ in range(EXPLANATION_CACHE_SIZE):
        _remember_explanation([1.0] * 5, {})
    assert len(_state["explanation_requests"]) == EXPLANATION_CACHE_SIZE
    assert client.post(f"/api/explanations/{first}").status_code == 410
