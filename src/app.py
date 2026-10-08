"""Streamlit demo: predict MOF CO2 uptake at several pressures and explain with SHAP."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "results" / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data_clean_v2"

FEATURES = ["lcd", "pld", "void_fraction", "surface_area_m2g"]
PRESSURES = [0.01, 0.05, 0.1, 0.5, 2.5]


def target_name(pressure):
    return f"CO2_uptake_{pressure}bar_molkg"


@st.cache_resource
def load_models():
    return {
        p: joblib.load(MODELS_DIR / f"phase1_xgboost_{target_name(p)}.joblib")
        for p in PRESSURES
    }


@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_PATH, usecols=FEATURES)


def predict_all(models, features):
    return pd.DataFrame({
        "Pressure (bar)": PRESSURES,
        "Predicted CO2 uptake (mol/kg)": [
            float(models[p].predict(features)[0]) for p in PRESSURES
        ],
    })


def explain(model, features):
    explanation = shap.TreeExplainer(model)(features)
    fig = plt.figure()
    shap.plots.waterfall(explanation[0], show=False)
    return fig


def main():
    st.set_page_config(page_title="MOF CO2 Uptake Predictor", layout="wide")
    st.title("MOF CO₂ Uptake Predictor")
    st.caption("XGBoost models trained on hMOF geometric descriptors (Phase 1).")

    if not MODELS_DIR.exists() or not any(MODELS_DIR.glob("phase1_xgboost_*.joblib")):
        st.error("Models not found. Run notebook/phase 1/modeling/modeling_xgboost_all_pressures.ipynb first.")
        st.stop()

    models = load_models()
    reference = load_reference_data() if DATA_PATH.exists() else None

    mode = st.sidebar.radio("Input mode", ["Enter features", "Upload CSV"])

    if mode == "Upload CSV":
        uploaded = st.sidebar.file_uploader("CSV with columns: " + ", ".join(FEATURES), type="csv")
        if uploaded is None:
            st.info("Upload a CSV to get predictions.")
            return
        data = pd.read_csv(uploaded)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return
        out = data.copy()
        for p in PRESSURES:
            out[target_name(p)] = models[p].predict(data[FEATURES])
        st.dataframe(out)
        st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv")
        return

    defaults = reference.median() if reference is not None else pd.Series(
        {"lcd": 8.0, "pld": 5.0, "void_fraction": 0.5, "surface_area_m2g": 2000.0}
    )
    values = {}
    for name in FEATURES:
        if reference is not None:
            lo, hi = float(reference[name].min()), float(reference[name].max())
        else:
            lo, hi = 0.0, float(defaults[name]) * 5
        values[name] = st.sidebar.number_input(
            name, min_value=lo, max_value=hi, value=float(defaults[name])
        )
    features = pd.DataFrame([values], columns=FEATURES)

    left, right = st.columns(2)
    with left:
        st.subheader("Predicted CO₂ uptake")
        predictions = predict_all(models, features)
        st.dataframe(predictions, hide_index=True)
        st.line_chart(predictions, x="Pressure (bar)", y="Predicted CO2 uptake (mol/kg)")
    with right:
        st.subheader("SHAP explanation")
        pressure = st.selectbox("Pressure (bar)", PRESSURES, index=2)
        st.pyplot(explain(models[pressure], features))


main()
