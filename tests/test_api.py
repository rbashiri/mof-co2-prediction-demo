import base64
import io
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

    csv_bytes = base64.b64decode(data["predictions_csv"])
    assert b"Pressure" in csv_bytes


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
