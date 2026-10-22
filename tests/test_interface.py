import io
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np
import pytest
import shap
from streamlit.testing.v1 import AppTest

from src.app import features_from_upload
from src.gemini_summary import GeminiSummaryError
from src.paths import MODELS_DIR
from src.scientific_summary import generate_pressure_comparison

APP = str(Path(__file__).resolve().parents[1] / "src" / "app.py")
GEOMETRY = {
    "lcd": 11.76,
    "pld": 7.89,
    "void_fraction": 0.75,
    "surface_area_m2g": 3543.89,
}


def prediction_table(app):
    return next(
        table.value for table in app.table
        if "Predicted CO2 uptake (mol/kg)" in table.value.columns
    )


def test_app_starts_without_upload():
    app = AppTest.from_file(APP).run(timeout=60)
    assert not app.exception
    assert not app.error
    assert not app.number_input
    assert app.title[0].value == "Welcome to the MOF CO₂ Uptake Predictor"
    assert not app.radio
    assert app.expander[0].label == "About these predictions"
    assert not app.expander[0].proto.expanded
    assert not app.table
    assert not app.dataframe
    assert any(header.value == "Meet Our Team" for header in app.header)


def test_empty_cif_reports_error():
    with patch("streamlit.file_uploader", return_value=io.BytesIO(b"")):
        app = AppTest.from_file(APP).run(timeout=60)
        app.button[0].click().run(timeout=60)
        assert not app.exception
        assert any("empty" in error.value for error in app.error)
        assert not app.dataframe


def test_upload_tempfile_is_removed(monkeypatch):
    paths = []

    def extract(path, **kwargs):
        paths.append(path)
        assert path.read_bytes() == b"CIF"
        return GEOMETRY

    def build(path, geometry, **kwargs):
        assert geometry == GEOMETRY
        assert path.read_bytes() == b"CIF"
        raise ValueError("Extraction rejected")

    monkeypatch.setattr("src.app.extract_geometry", extract)
    monkeypatch.setattr("src.app.build_phase3_features", build)
    with pytest.raises(ValueError, match="Extraction rejected"):
        features_from_upload(b"CIF", [], None, 120)
    assert not paths[0].exists()


def test_predictions_use_static_table_and_preserve_download_and_shap():
    def build(path, geometry, *, feature_names, **kwargs):
        row = {name: 0.0 for name in feature_names}
        row.update(geometry)
        return pd.DataFrame([row], columns=feature_names)

    with (
        patch("streamlit.file_uploader", side_effect=lambda *a, **k: io.BytesIO(b"CIF")),
        patch("src.zeopp_geometry.extract_geometry", return_value=GEOMETRY) as extract,
        patch("src.phase3_features.build_phase3_features", side_effect=build),
    ):
        app = AppTest.from_file(APP).run(timeout=60)
        app.button[0].click().run(timeout=120)
        assert not app.exception
        assert not app.error
        assert extract.call_count == 1
        predictions = prediction_table(app)
        assert predictions["Pressure (bar)"].tolist() == [0.01, 0.05, 0.1, 0.5, 2.5]
        assert predictions.shape == (5, 2)
        geometry_table = next(
            table.value for table in app.table
            if "Largest cavity diameter (angstrom)" in table.value.columns
        )
        assert geometry_table.iloc[0, 0] == pytest.approx(GEOMETRY["lcd"])
        assert app.dataframe[0].value.shape == (1, 48)
        assert len(app.get("download_button")) == 1
        assert app.get("download_button")[0].proto.label == "Download predictions"
        assert len(app.get("image")) >= 1
        app.selectbox[0].set_value(0.5).run(timeout=60)
        assert not app.exception
        assert not app.error
        assert prediction_table(app).shape == (5, 2)
        assert extract.call_count == 1


def test_summary_click_generation_invalidation_failure_and_retry(tmp_path):
    upload = [b"CIF"]

    def build(path, geometry, *, feature_names, **kwargs):
        row = {name: 0.0 for name in feature_names}
        row.update(geometry)
        return pd.DataFrame([row], columns=feature_names)

    with (
        patch("streamlit.file_uploader", side_effect=lambda *a, **k: io.BytesIO(upload[0])),
        patch("src.zeopp_geometry.extract_geometry", return_value=GEOMETRY),
        patch("src.phase3_features.build_phase3_features", side_effect=build),
        patch(
            "src.scientific_summary.generate_pressure_comparison",
            wraps=generate_pressure_comparison,
        ) as generate,
    ):
        app = AppTest.from_file(APP).run(timeout=60)
        app.button[0].click().run(timeout=120)
        assert not generate.called
        assert not app.text
        app.button(key="generate_scientific_summary").click().run(timeout=60)
        assert not app.exception
        assert not app.error
        assert generate.call_count == 1
        assert len(app.text) == 4
        assert "endpoint difference" in app.text[0].value
        assert "At 0.01 bar" in app.text[1].value
        assert "At 2.5 bar" in app.text[2].value
        assert "own baseline and uptake scale" in app.text[3].value
        predictions = prediction_table(app).iloc[:, 1]
        difference = predictions.iloc[-1] - predictions.iloc[0]
        assert f"{difference:+.6g} mol/kg" in app.text[0].value
        low, high = generate.call_args.args[2:]
        assert not np.array_equal(low.values, high.values)
        assert any("not AI-generated" in caption.value for caption in app.caption)
        app.selectbox[0].set_value(0.5).run(timeout=60)
        assert not app.text
        app.selectbox[0].set_value(0.1).run(timeout=60)
        assert not app.text
        generate.side_effect = ValueError("SHAP additivity failed")
        app.button(key="generate_scientific_summary").click().run(timeout=60)
        assert any("Summary generation failed" in error.value for error in app.error)
        assert prediction_table(app).shape == (5, 2)
        assert len(app.get("image")) >= 1
        assert not app.text
        generate.side_effect = None
        tree_explainer = shap.TreeExplainer
        calls = []

        def fail_endpoint(*args, **kwargs):
            calls.append(args[0])
            if len(calls) == 2:
                raise ValueError("Endpoint failed")
            return tree_explainer(*args, **kwargs)

        with patch("shap.TreeExplainer", side_effect=fail_endpoint):
            app.button(key="generate_scientific_summary").click().run(timeout=60)
            assert not app.exception
            assert any("Endpoint failed" in error.value for error in app.error)
            assert prediction_table(app).shape == (5, 2)
            assert len(app.get("image")) >= 1
            assert not app.text
        app.button(key="generate_scientific_summary").click().run(timeout=60)
        assert not app.error
        assert app.text
        upload[0] = b"ANOTHER CIF"
        app.run(timeout=60)
        assert not app.text
        assert not app.table
        upload[0] = b"CIF"
        app.run(timeout=60)
        assert not app.text
        app.button(key="generate_scientific_summary").click().run(timeout=60)
        assert app.text
        with patch("shap.TreeExplainer", side_effect=ValueError("SHAP unavailable")):
            app.run(timeout=60)
            assert any("SHAP explanation failed" in error.value for error in app.error)
            assert app.button(key="generate_scientific_summary").disabled
            assert not app.text
            assert prediction_table(app).shape == (5, 2)
        app.run(timeout=60)
        app.button(key="generate_scientific_summary").click().run(timeout=60)
        assert app.text
        for model in MODELS_DIR.glob("*.joblib"):
            (tmp_path / model.name).symlink_to(model)
        with patch("src.paths.MODELS_DIR", tmp_path):
            app.run(timeout=60)
            assert not app.exception
            assert not app.text
            assert not app.table
        app.run(timeout=60)
        assert not app.text
        app.button(key="generate_scientific_summary").click().run(timeout=60)
        assert app.text
        with patch("streamlit.file_uploader", return_value=None):
            app.run(timeout=60)
            assert not app.text
        app.run(timeout=60)
        assert not app.text


def test_zeopp_failure_stops_prediction():
    with (
        patch("streamlit.file_uploader", return_value=io.BytesIO(b"CIF")),
        patch(
            "src.zeopp_geometry.extract_geometry",
            side_effect=ValueError("Zeo++ rejected CIF"),
        ),
    ):
        app = AppTest.from_file(APP).run(timeout=60)
        app.button[0].click().run(timeout=60)
    assert not app.exception
    assert any("Zeo++ rejected CIF" in error.value for error in app.error)
    assert not app.table
    assert not app.dataframe


def test_gemini_is_deferred_and_does_not_repeat_extraction(monkeypatch):
    monkeypatch.setenv("GEMINI_ENABLED", "true")

    def build(path, geometry, *, feature_names, **kwargs):
        row = {name: 0.0 for name in feature_names}
        row.update(geometry)
        return pd.DataFrame([row], columns=feature_names)

    gemini_text = "Validated Gemini draft"
    with (
        patch("streamlit.file_uploader", return_value=io.BytesIO(b"CIF")),
        patch("src.zeopp_geometry.extract_geometry", return_value=GEOMETRY) as extract,
        patch("src.phase3_features.build_phase3_features", side_effect=build),
        patch(
            "src.gemini_summary.generate_gemini_summary",
            return_value=gemini_text,
        ) as gemini,
    ):
        app = AppTest.from_file(APP).run(timeout=60)
        app.button[0].click().run(timeout=120)
        assert not gemini.called
        app.button(key="generate_scientific_summary").click().run(timeout=60)
        assert not gemini.called
        app.button(key="generate_gemini_summary").click().run(timeout=60)
        assert gemini.call_count == 1
        assert extract.call_count == 1
        assert any(text.value == gemini_text for text in app.text)
        assert prediction_table(app).shape == (5, 2)

        gemini.side_effect = GeminiSummaryError("Provider unavailable")
        app.button(key="generate_gemini_summary").click().run(timeout=60)
        assert any("Provider unavailable" in error.value for error in app.error)
        assert prediction_table(app).shape == (5, 2)
        assert extract.call_count == 1


def test_real_cif_predictions_shap_and_stale_results():
    cif_dir = os.environ.get("MOF_REFERENCE_CIF_DIR")
    if (
        not cif_dir
        or not os.environ.get("MOF_SBU_PATH")
        or not os.environ.get("ZEO_NETWORK_PATH")
    ):
        pytest.skip("Local CIFs, MOF_SBU_PATH, and ZEO_NETWORK_PATH required.")
    cif = (Path(cif_dir) / "hMOF-0.cif").read_bytes()
    with patch("streamlit.file_uploader", side_effect=lambda *a, **k: io.BytesIO(cif)):
        app = AppTest.from_file(APP).run(timeout=60)
        app.button[0].click().run(timeout=120)
        assert not app.exception
        assert not app.error
        predictions = prediction_table(app)
        assert predictions["Pressure (bar)"].tolist() == [0.01, 0.05, 0.1, 0.5, 2.5]
        assert len(predictions) == 5
        assert np.isfinite(predictions.iloc[:, 1]).all()
        assert app.dataframe[0].value.shape == (1, 48)
        app.selectbox[0].set_value(0.5).run(timeout=60)
        assert not app.exception
        assert not app.error