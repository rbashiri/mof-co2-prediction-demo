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
import pandas as pd
import shap
import streamlit as st
import yaml
from xgboost import XGBRegressor

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import CONFIG_PATH, MODELS_DIR, PROJECT_ROOT
from src.phase3_features import build_phase3_features
from src.scientific_summary import PRESSURES, SUMMARY_LABEL, generate_summary
from src.train import GEOMETRY_COLUMNS

LOGGER = logging.getLogger(__name__)
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


def explain(explanation: shap.Explanation):
    shap.plots.waterfall(explanation, show=False)
    return plt.gcf()


def report_error(message: str, error: Exception) -> None:
    LOGGER.exception(message)
    st.error(f"{message}: {error}")


def main() -> None:
    st.set_page_config(page_title="MOF CO2 Uptake Predictor", layout="wide")
    st.title("Welcome to the MOF CO₂ Uptake Predictor")
    st.write("Upload a CIF and enter its four matching geometry values.")
    with st.expander("About these predictions"):
        st.write(
            "Geometry values must describe the uploaded structure and are supplied "
            "by you, not calculated by this app. The models predict simulated "
            "hMOF uptake; predictions are not experimental measurements and "
            "accuracy on other MOF families has not been established."
        )
    try:
        settings = load_app_settings()
        paths = tuple(str(MODELS_DIR / settings["models"][str(p)]) for p in PRESSURES)
        models = load_models(paths)
    except (OSError, ValueError, EOFError) as error:
        report_error("Unable to load configured models", error)
        return
    names = models[PRESSURES[0]].get_booster().feature_names
    assert names is not None
    names = list(names)
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
        st.session_state.pop("phase3_summary", None)
        return
    cif_bytes = uploaded.getvalue()
    request = (cif_bytes, tuple(geometry.items()), paths, settings["model_label"])
    previous = st.session_state.get("phase3_result")
    if submitted:
        st.session_state.pop("phase3_summary", None)
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
        st.session_state.pop("phase3_summary", None)
        st.write("Click 'Extract descriptors and predict' for the current inputs.")
        return
    _, features, predictions = previous
    left, right = st.columns(2)
    with left:
        st.subheader("Predicted CO₂ uptake")
        st.table(predictions, hide_index=True)
        st.line_chart(predictions, x="Pressure (bar)", y="Predicted CO2 uptake (mol/kg)")
        st.download_button("Download predictions", predictions.to_csv(index=False), "predictions.csv")
        with st.expander("Inspect the 48 model input features"):
            st.dataframe(features)
    with right:
        st.subheader("SHAP explanation")
        pressure = st.selectbox("Pressure (bar)", PRESSURES, index=2)
        explanation = None
        try:
            explanation = shap.TreeExplainer(models[pressure])(features)[0]
            fig = explain(explanation)
            st.pyplot(fig)
            plt.close(fig)
        except (ValueError, RuntimeError) as error:
            explanation = None
            report_error("SHAP explanation failed", error)
        st.caption("SHAP describes model contributions, not proven chemical mechanisms.")

    context = (request, pressure)
    saved_summary = st.session_state.get("phase3_summary")
    if explanation is None or (saved_summary is not None and saved_summary[0] != context):
        st.session_state.pop("phase3_summary", None)
    st.subheader("Scientific summary")
    st.caption(f"{SUMMARY_LABEL}. Runs locally; no API key or external data sharing.")
    if st.button(
        "Generate scientific summary", key="generate_scientific_summary",
        disabled=explanation is None,
    ):
        st.session_state.pop("phase3_summary", None)
        try:
            if explanation is None:
                raise ValueError("A matching SHAP explanation is required.")
            with st.spinner("Summarizing predictions and SHAP..."):
                summary = generate_summary(predictions, features, explanation, pressure)
            st.session_state["phase3_summary"] = (context, summary)
        except (ValueError, TypeError, RuntimeError) as error:
            report_error("Summary generation failed", error)
    saved_summary = st.session_state.get("phase3_summary")
    if saved_summary is not None:
        st.text(saved_summary[1])
        st.caption("Draft for scientific review, not an approved scientific interpretation.")


if __name__ == "__main__":
    main()
