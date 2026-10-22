"""Streamlit Phase 3 demo using CIF-derived geometry and chemistry descriptors."""

import logging
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
import yaml
from xgboost import XGBRegressor

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import CONFIG_PATH, MODELS_DIR, PROJECT_ROOT
from src.gemini_summary import GeminiSummaryError, generate_gemini_summary
from src.phase3_features import build_phase3_features
from src.scientific_summary import PRESSURES, SUMMARY_LABEL, generate_pressure_comparison
from src.train import GEOMETRY_COLUMNS
from src.zeopp_geometry import (
    GEOMETRY_WARNING,
    SURFACE_PROBE_RADIUS,
    SURFACE_SAMPLES,
    VOLUME_PROBE_RADIUS,
    VOLUME_SAMPLES,
    extract_geometry,
)

LOGGER = logging.getLogger(__name__)
GEOMETRY_LABELS = {
    "lcd": "Largest cavity diameter (angstrom)",
    "pld": "Pore limiting diameter (angstrom)",
    "void_fraction": "Void fraction (0-1)",
    "surface_area_m2g": "Surface area (m2/g)",
}
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TEAM_MEMBERS = (
    {
        "image": "team-robabeh.webp",
        "name": "Robabeh Bashiri, PhD",
        "role": "Project Lead",
        "expertise": "Materials Scientist · Data Scientist",
        "organization": "California State University, Chico",
        "secondary": "Department of Chemistry and Biochemistry",
        "location": "Chico, California, USA",
        "links": (
            ("GitHub", "https://github.com/rbashiri"),
            ("LinkedIn", "https://www.linkedin.com/in/robabehbashiri"),
        ),
    },
    {
        "image": "team-ripunjay.webp",
        "name": "Ripunjay Kumar",
        "role": "Team Member",
        "expertise": "Data Scientist · Machine Learning",
        "organization": "Syngenta Group",
        "secondary": "Indian Institute of Technology Madras",
        "location": "Pune, India",
        "links": (
            ("GitHub", "https://github.com/Ripu0902"),
            ("LinkedIn", "https://www.linkedin.com/in/ripu0902/"),
        ),
    },
    {
        "image": "team-umer.webp",
        "name": "Umer Farooq",
        "role": "Team Member",
        "expertise": "Materials Research · Energy Storage",
        "organization": "Bahauddin Zakariya University",
        "secondary": "Undergraduate Student · Volunteer Researcher",
        "location": "Multan, Pakistan",
        "links": (
            ("LinkedIn", "https://www.linkedin.com/in/umerchem/"),
        ),
    },
)


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
    env_timeout = os.environ.get("MOF_EXTRACTION_TIMEOUT_SECONDS")
    if env_timeout is not None:
        try:
            timeout = float(env_timeout)
        except ValueError as exc:
            raise ValueError("MOF_EXTRACTION_TIMEOUT_SECONDS must be finite and positive.") from exc
        settings["extraction_timeout_seconds"] = timeout
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
    names: list[str],
    sbu_path: str | None,
    timeout: float,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="mof-co2-upload-") as directory:
        path = Path(directory) / "structure.cif"
        path.write_bytes(cif_bytes)
        geometry = extract_geometry(path, timeout=timeout)
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


def render_team_contact() -> None:
    st.divider()
    st.caption("OUR COLLABORATORS")
    st.header("Meet Our Team")
    st.write(
        "Bringing together materials science, machine learning, and "
        "scientific interpretation."
    )
    columns = st.columns(3)
    for column, member in zip(columns, TEAM_MEMBERS, strict=True):
        with column:
            image = ASSETS_DIR / member["image"]
            if image.is_file():
                st.image(str(image), width=128)
            st.caption(member["role"])
            st.subheader(member["name"])
            st.write(member["expertise"])
            st.write(member["organization"])
            st.caption(member["secondary"])
            st.caption(member["location"])
            for label, url in member["links"]:
                st.link_button(label, url)
    st.subheader("Contact Us")
    contact_columns = st.columns((1, 2))
    campus = ASSETS_DIR / "csu-chico-campus.png"
    with contact_columns[0]:
        if campus.is_file():
            st.image(str(campus), caption="California State University, Chico")
    with contact_columns[1]:
        st.write(
            "Have a research question, collaboration idea, or feedback about the "
            "MOF CO₂ Predictor? We'd love to hear from you."
        )
        st.link_button("Email the team", "mailto:robabeh.bashiri@gmail.com")
        st.link_button("Project GitHub", "https://github.com/rbashiri/mof-co2-prediction-demo/tree/osi/app")


def resolve_sbu_path(configured: str | None) -> str | None:
    candidate = os.environ.get("MOF_SBU_PATH") or configured
    if candidate is not None:
        if not isinstance(candidate, str) or not candidate:
            raise ValueError("app.sbu_path must be a non-empty path or null.")
        path = Path(candidate).expanduser()
        return str(path if path.is_absolute() else PROJECT_ROOT / path)
    default = Path.home() / "mofid-tools" / "mofid" / "bin" / "sbu"
    return str(default) if default.is_file() and os.access(default, os.X_OK) else None


def main() -> None:
    st.set_page_config(
        page_title="PRISM — Pore Regression & Interpretability for Simulated MOFs",
        layout="wide",
    )
    st.title("PRISM — Pore Regression & Interpretability for Simulated MOFs")
    st.write(
        "Upload one CIF. Zeo++ calculates its pore geometry, while "
        "MOFid/RDKit and pymatgen derive the remaining model descriptors."
    )
    with st.expander("About these predictions"):
        st.write(
            f"{GEOMETRY_WARNING} The models predict simulated hMOF uptake; "
            "predictions are not experimental measurements and accuracy on other "
            "MOF families has not been established."
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
        uploaded = st.file_uploader("CIF for this MOF", type=["cif"])
        submitted = st.button("Extract descriptors and predict")
    if uploaded is None:
        st.session_state.pop("phase3_summary", None)
        st.session_state.pop("gemini_summary", None)
        render_team_contact()
        return
    cif_bytes = uploaded.getvalue()
    request = (cif_bytes, paths, settings["model_label"])
    previous = st.session_state.get("phase3_result")
    if submitted:
        st.session_state.pop("phase3_summary", None)
        st.session_state.pop("gemini_summary", None)
        st.session_state.pop("phase3_result", None)
        try:
            if not cif_bytes:
                raise ValueError("The uploaded CIF is empty.")
            sbu_path = resolve_sbu_path(settings.get("sbu_path"))
            with st.spinner("Extracting Zeo++ geometry and chemistry descriptors..."):
                features = features_from_upload(
                    cif_bytes, names, sbu_path, settings["extraction_timeout_seconds"],
                )
                predictions = predict_all(models, features)
            previous = (request, features, predictions)
            st.session_state["phase3_result"] = previous
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            report_error("CIF processing or prediction failed", error)
            return
    if previous is None or previous[0] != request:
        st.session_state.pop("phase3_summary", None)
        st.session_state.pop("gemini_summary", None)
        st.write("Click 'Extract descriptors and predict' for the current inputs.")
        render_team_contact()
        return
    _, features, predictions = previous
    st.subheader("Zeo++ geometry")
    geometry = pd.DataFrame([{
        GEOMETRY_LABELS[name]: float(features.iloc[0][name]) for name in GEOMETRY_COLUMNS
    }])
    st.table(geometry, hide_index=True)
    st.warning(GEOMETRY_WARNING)
    st.caption(
        f"Profile: surface probe/channel radius {SURFACE_PROBE_RADIUS} Å "
        f"({SURFACE_SAMPLES} samples per atom); volume probe/channel radius "
        f"{VOLUME_PROBE_RADIUS} Å ({VOLUME_SAMPLES} samples); high-accuracy mode."
    )
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
        st.session_state.pop("gemini_summary", None)
    st.subheader("Scientific summary")
    st.caption(f"{SUMMARY_LABEL}. Runs locally; no API key or external data sharing.")
    st.caption("Compares 0.01 and 2.5 bar, independently of the waterfall pressure above.")
    if st.button(
        "Generate scientific summary", key="generate_scientific_summary",
        disabled=explanation is None,
    ):
        st.session_state.pop("phase3_summary", None)
        try:
            if explanation is None:
                raise ValueError("A matching SHAP explanation is required.")
            with st.spinner("Comparing predictions and endpoint SHAP explanations..."):
                low_explanation = (
                    explanation if pressure == PRESSURES[0]
                    else shap.TreeExplainer(models[PRESSURES[0]])(features)[0]
                )
                high_explanation = (
                    explanation if pressure == PRESSURES[-1]
                    else shap.TreeExplainer(models[PRESSURES[-1]])(features)[0]
                )
                summary = generate_pressure_comparison(
                    predictions, features, low_explanation, high_explanation,
                )
            st.session_state["phase3_summary"] = (context, summary)
        except (ValueError, TypeError, RuntimeError) as error:
            report_error("Summary generation failed", error)
    saved_summary = st.session_state.get("phase3_summary")
    if saved_summary is not None:
        for title, text in saved_summary[1].items():
            st.markdown(f"**{title}**")
            st.text(text)
        st.caption("Draft for scientific review, not an approved scientific interpretation.")
        st.subheader("Optional Gemini explanation")
        st.caption(
            "Runs only when requested. Prediction values and validated SHAP summary "
            "text are sent to Google; the uploaded CIF is not sent."
        )
        gemini_enabled = os.environ.get("GEMINI_ENABLED", "").lower() == "true"
        if not gemini_enabled:
            st.info("Gemini is disabled. Set GEMINI_ENABLED=true and configure its credentials.")
        if st.button(
            "Generate Gemini explanation",
            key="generate_gemini_summary",
            disabled=not gemini_enabled,
        ):
            st.session_state.pop("gemini_summary", None)
            try:
                with st.spinner("Requesting Gemini explanation..."):
                    draft = generate_gemini_summary(
                        predictions.iloc[:, 1].tolist(), saved_summary[1],
                    )
                st.session_state["gemini_summary"] = (context, draft)
            except GeminiSummaryError as error:
                report_error("Gemini explanation failed", error)
        saved_gemini = st.session_state.get("gemini_summary")
        if saved_gemini is not None and saved_gemini[0] == context:
            st.text(saved_gemini[1])
            st.caption("AI-generated draft for scientific review.")
    render_team_contact()


if __name__ == "__main__":
    main()
