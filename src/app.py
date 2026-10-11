"""Streamlit Phase 3 demo using supplied geometry and CIF-derived descriptors."""

import logging
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import yaml
from xgboost import XGBRegressor

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import CONFIG_PATH, MODELS_DIR, PROJECT_ROOT
from src.phase3_features import build_phase3_features
from src.train import GEOMETRY_COLUMNS

LOGGER = logging.getLogger(__name__)
PRESSURES = [0.01, 0.05, 0.1, 0.5, 2.5]
GEOMETRY_LABELS = {
    "lcd": "Largest cavity diameter (angstrom)",
    "pld": "Pore limiting diameter (angstrom)",
    "void_fraction": "Void fraction (0-1)",
    "surface_area_m2g": "Surface area (m2/g)",
}


def load_app_settings() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("app"), dict):
        raise ValueError("config.yaml requires an app mapping.")
    settings = config["app"]
    model_files = settings.get("models")
    if (
        not isinstance(model_files, dict)
        or set(model_files) != {str(p) for p in PRESSURES}
        or not all(isinstance(value, str) and value for value in model_files.values())
    ):
        raise ValueError("app.models must specify a model file for each of the five pressures.")
    timeout = settings.get("extraction_timeout_seconds")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("app.extraction_timeout_seconds must be finite and positive.")
    if not isinstance(settings.get("model_label"), str):
        raise ValueError("app.model_label must be a string.")
    return settings


@st.cache_resource
def load_models(model_paths: tuple[str, ...]) -> dict[float, XGBRegressor]:
    models = {}
    schema = None
    for pressure, path in zip(PRESSURES, model_paths):
        model = joblib.load(path)
        if not isinstance(model, XGBRegressor):
            raise ValueError(f"Expected an XGBoost regressor in {path}")
        names = model.get_booster().feature_names
        if (
            names is None
            or len(names) != 48
            or len(set(names)) != 48
            or any("oxidation" in name.lower() for name in names)
        ):
            raise ValueError(f"Model at {path} does not have the corrected 48-feature schema.")
        if schema is not None and names != schema:
            raise ValueError("The five models must use identical feature names and order.")
        schema = names
        models[pressure] = model
    return models


def features_from_upload(
    cif_bytes: bytes,
    geometry: dict[str, float],
    names: list[str],
    sbu_path: str | None,
    timeout: float,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="mof-co2-upload-") as directory:
        path = Path(directory) / "structure.cif"
        path.write_bytes(cif_bytes)
        return build_phase3_features(
            path, geometry, feature_names=names, sbu_path=sbu_path, timeout=timeout,
        )


def predict_all(models: dict[float, XGBRegressor], features: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Pressure (bar)": PRESSURES,
        "Predicted CO2 uptake (mol/kg)": [
            float(models[pressure].predict(features)[0]) for pressure in PRESSURES
        ],
    })


def explain(model: XGBRegressor, features: pd.DataFrame):
    explanation = shap.TreeExplainer(model)(features)
    shap.plots.waterfall(explanation[0], show=False)
    return plt.gcf()


def report_error(message: str, error: Exception) -> None:
    LOGGER.exception(message)
    st.error(f"{message}: {error}")


def main() -> None:
    st.set_page_config(page_title="MOF CO2 Uptake Predictor", layout="wide")
    st.title("MOF CO₂ Uptake Predictor")
    try:
        settings = load_app_settings()
        paths = tuple(str(MODELS_DIR / settings["models"][str(p)]) for p in PRESSURES)
        models = load_models(paths)
    except (OSError, ValueError, EOFError) as error:
        report_error("Unable to load configured models", error)
        return
    names = models[PRESSURES[0]].get_booster().feature_names
    assert names is not None
    st.caption(settings["model_label"])
    st.info(
        "Supply geometry for the same structure as the CIF. Geometry is not "
        "calculated here (no Zeo++). These models predict simulated hMOF uptake."
    )
    mode = st.sidebar.radio("Input mode", ["Geometry + CIF", "Prepared feature CSV"])

    if mode == "Prepared feature CSV":
        with st.sidebar:
            uploaded_csv = st.file_uploader(
                "CSV containing all 48 Phase 3 model features", type=["csv"],
            )
        if uploaded_csv is None:
            st.info("Upload prepared features to obtain batch predictions.")
            return
        try:
            data = pd.read_csv(uploaded_csv)
            missing = sorted(set(names) - set(data.columns))
            if missing:
                raise ValueError(f"Missing model features: {missing}")
            if data.empty:
                raise ValueError("The CSV contains no rows.")
            features = data[names].apply(pd.to_numeric, errors="raise")
            if not np.isfinite(features.to_numpy(dtype=float)).all():
                raise ValueError("All 48 features must contain finite numeric values.")
            out = data.copy()
            for pressure in PRESSURES:
                out[f"CO2_uptake_{pressure}bar_molkg"] = models[pressure].predict(features)
        except (ValueError, pd.errors.ParserError, UnicodeError) as error:
            report_error("Invalid prepared-feature CSV", error)
            return
        st.dataframe(out)
        st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv")
        return

    with st.sidebar:
        geometry = {
            name: st.number_input(
                GEOMETRY_LABELS[name], min_value=0.0,
                max_value=1.0 if name == "void_fraction" else None,
                value=0.0, format="%.6f",
            )
            for name in GEOMETRY_COLUMNS
        }
        uploaded = st.file_uploader("CIF for this MOF", type=["cif"])
        submitted = st.button("Extract descriptors and predict")
    if uploaded is None:
        st.info("Enter your four geometry values and upload a CIF to begin.")
        return
    cif_bytes = uploaded.getvalue()
    request = (cif_bytes, tuple(geometry.items()), paths)
    previous = st.session_state.get("phase3_result")
    if submitted:
        st.session_state.pop("phase3_result", None)
        try:
            if not cif_bytes:
                raise ValueError("The uploaded CIF is empty.")
            sbu_path = os.environ.get("MOF_SBU_PATH") or settings.get("sbu_path")
            if sbu_path is not None:
                if not isinstance(sbu_path, str) or not sbu_path:
                    raise ValueError("app.sbu_path must be a non-empty path or null.")
                path = Path(sbu_path).expanduser()
                sbu_path = str(path if path.is_absolute() else PROJECT_ROOT / path)
            with st.spinner("Extracting linker and metal/lattice descriptors..."):
                features = features_from_upload(
                    cif_bytes, geometry, names, sbu_path,
                    settings["extraction_timeout_seconds"],
                )
                predictions = predict_all(models, features)
            previous = (request, features, predictions)
            st.session_state["phase3_result"] = previous
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report_error("CIF processing or prediction failed", error)
            return
    if previous is None or previous[0] != request:
        st.info("Click 'Extract descriptors and predict' for the current inputs.")
        return
    _, features, predictions = previous
    left, right = st.columns(2)
    with left:
        st.subheader("Predicted CO₂ uptake")
        st.dataframe(predictions, hide_index=True)
        st.line_chart(predictions, x="Pressure (bar)", y="Predicted CO2 uptake (mol/kg)")
        st.download_button("Download predictions", predictions.to_csv(index=False), "predictions.csv")
        with st.expander("Inspect the 48 model input features"):
            st.dataframe(features)
    with right:
        st.subheader("SHAP explanation")
        pressure = st.selectbox("Pressure (bar)", PRESSURES, index=2)
        try:
            fig = explain(models[pressure], features)
            st.pyplot(fig)
            plt.close(fig)
        except (ValueError, RuntimeError) as error:
            report_error("SHAP explanation failed", error)
        st.caption("SHAP describes model contributions, not proven chemical mechanisms.")


if __name__ == "__main__":
    main()
