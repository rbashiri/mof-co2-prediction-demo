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
from src.train import GEOMETRY_COLUMNS

APP = str(Path(__file__).resolve().parents[1] / "src" / "app.py")


def test_app_starts_without_upload():
    app = AppTest.from_file(APP).run(timeout=60)
    assert not app.exception
    assert not app.error
    assert len(app.number_input) == 4
    assert "Phase 3 Model B" in app.caption[0].value
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


def test_prepared_csv_reports_missing_features():
    with patch("streamlit.file_uploader", side_effect=lambda *a, **k: io.BytesIO(b"lcd\n1\n")):
        app = AppTest.from_file(APP).run(timeout=60)
        app.sidebar.radio[0].set_value("Prepared feature CSV").run(timeout=60)
        assert not app.exception
        assert any("Missing model features" in error.value for error in app.error)


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
        predictions = app.dataframe[0].value
        assert predictions["Pressure (bar)"].tolist() == [0.01, 0.05, 0.1, 0.5, 2.5]
        assert len(predictions) == 5
        assert app.dataframe[1].value.shape == (1, 48)
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
        assert not app.dataframe