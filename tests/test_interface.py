import io
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import joblib
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.app import features_from_upload
from src.paths import PHASE3_DATA_PATH, MODELS_DIR
from src.scientific_summary import generate_summary
from src.train import GEOMETRY_COLUMNS

APP = str(Path(__file__).resolve().parents[1] / "src" / "app.py")


def test_app_starts_without_upload():
    app = AppTest.from_file(APP).run(timeout=60)
    assert not app.exception
    assert not app.error
    assert len(app.number_input) == 4
    assert app.title[0].value == "Welcome to the MOF CO₂ Uptake Predictor"
    assert not app.radio
    assert not app.caption
    assert not app.info
    assert app.expander[0].label == "About these predictions"
    assert not app.expander[0].proto.expanded
    assert not app.table
    assert not app.dataframe


def test_empty_cif_reports_error():
    with patch("streamlit.file_uploader", return_value=io.BytesIO(b"")):
        app = AppTest.from_file(APP).run(timeout=60)
        app.button[0].click().run(timeout=60)
        assert not app.exception
        assert any("empty" in error.value for error in app.error)
        assert not app.dataframe


def test_upload_tempfile_is_removed(monkeypatch):
    paths = []

    def build(path, geometry, **kwargs):
        paths.append(path)
        assert path.read_bytes() == b"CIF"
        raise ValueError("Extraction rejected")

    monkeypatch.setattr("src.app.build_phase3_features", build)
    with pytest.raises(ValueError, match="Extraction rejected"):
        features_from_upload(b"CIF", {}, [], None, 120)
    assert not paths[0].exists()


def test_predictions_use_static_table_and_preserve_download_and_shap():
    def build(path, geometry, *, feature_names, **kwargs):
        return pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)

    with (
        patch("streamlit.file_uploader", side_effect=lambda *a, **k: io.BytesIO(b"CIF")),
        patch("src.phase3_features.build_phase3_features", side_effect=build),
    ):
        app = AppTest.from_file(APP).run(timeout=60)
        app.button[0].click().run(timeout=120)
        assert not app.exception
        assert not app.error
        assert app.table[0].value["Pressure (bar)"].tolist() == [0.01, 0.05, 0.1, 0.5, 2.5]
        assert app.table[0].value.shape == (5, 2)
        assert app.dataframe[0].value.shape == (1, 48)
        assert len(app.get("download_button")) == 1
        assert app.get("download_button")[0].proto.label == "Download predictions"
        assert len(app.get("image")) == 1
        app.selectbox[0].set_value(0.5).run(timeout=60)
        assert not app.exception
        assert not app.error
        assert len(app.table) == 1
        assert len(app.get("image")) == 1
        app.number_input[0].set_value(12.0).run(timeout=60)
        assert not app.table
        assert not app.dataframe
        assert not app.get("download_button")


def test_summary_click_generation_invalidation_failure_and_retry(tmp_path):
    upload = [b"CIF"]

    def build(path, geometry, *, feature_names, **kwargs):
        return pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)

    with (
        patch("streamlit.file_uploader", side_effect=lambda *a, **k: io.BytesIO(upload[0])),
        patch("src.phase3_features.build_phase3_features", side_effect=build),
        patch(
            "src.scientific_summary.generate_summary",
            wraps=generate_summary,
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
        assert 80 <= len(app.text[0].value.split()) <= 150
        assert any("not AI-generated" in caption.value for caption in app.caption)
        app.selectbox[0].set_value(0.5).run(timeout=60)
        assert not app.text
        app.selectbox[0].set_value(0.1).run(timeout=60)
        assert not app.text
        generate.side_effect = ValueError("SHAP additivity failed")
        app.button(key="generate_scientific_summary").click().run(timeout=60)
        assert any("Summary generation failed" in error.value for error in app.error)
        assert len(app.table) == 1
        assert len(app.get("image")) == 1
        assert not app.text
        generate.side_effect = None
        app.button(key="generate_scientific_summary").click().run(timeout=60)
        assert not app.error
        assert app.text
        app.number_input[0].set_value(12.0).run(timeout=60)
        assert not app.text
        app.number_input[0].set_value(0.0).run(timeout=60)
        assert not app.text
        app.button(key="generate_scientific_summary").click().run(timeout=60)
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
            assert len(app.table) == 1
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


def test_real_cif_predictions_shap_and_stale_results():
    cif_dir = os.environ.get("MOF_REFERENCE_CIF_DIR")
    if not cif_dir or not os.environ.get("MOF_SBU_PATH"):
        pytest.skip("Local CIFs and MOF_SBU_PATH required.")
    cif = (Path(cif_dir) / "hMOF-0.cif").read_bytes()
    reference = pd.read_csv(PHASE3_DATA_PATH, low_memory=False)
    row = reference.loc[reference.filename.eq("hMOF-0")].iloc[0]
    with patch("streamlit.file_uploader", side_effect=lambda *a, **k: io.BytesIO(cif)):
        app = AppTest.from_file(APP).run(timeout=60)
        for widget, name in zip(app.number_input, GEOMETRY_COLUMNS):
            widget.set_value(float(row[name]))
        app.button[0].click().run(timeout=120)
        assert not app.exception
        assert not app.error
        predictions = app.table[0].value
        assert predictions["Pressure (bar)"].tolist() == [0.01, 0.05, 0.1, 0.5, 2.5]
        assert len(predictions) == 5
        assert app.dataframe[0].value.shape == (1, 48)
        for index, pressure in enumerate([0.01, 0.05, 0.1, 0.5, 2.5]):
            model = joblib.load(
                MODELS_DIR /
                f"phase_3_model_b_geometry_plus_linkers_plus_cif_{pressure}bar.joblib"
            )
            names = model.get_booster().feature_names
            assert names is not None
            expected = model.predict(pd.DataFrame([row[names]]))[0]
            np.testing.assert_allclose(
                predictions.iloc[index, 1], expected, rtol=1e-6, atol=1e-6,
            )
        app.selectbox[0].set_value(0.5).run(timeout=60)
        assert not app.exception
        assert not app.error
        app.number_input[0].set_value(12.0).run(timeout=60)
        assert not app.table
        assert not app.dataframe