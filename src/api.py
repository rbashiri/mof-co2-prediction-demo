"""FastAPI backend for the MOF CO2 uptake predictor."""

import base64
import io
import logging
import math
import os
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import shap
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from xgboost import XGBRegressor

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import CONFIG_PATH, MODELS_DIR, PROJECT_ROOT
from src.gemini_summary import GeminiSummaryError, generate_gemini_summary
from src.phase3_features import build_phase3_features
from src.scientific_summary import PRESSURES, generate_pressure_comparison
from src.train import GEOMETRY_COLUMNS

LOGGER = logging.getLogger(__name__)

GEOMETRY_LABELS = {
    "lcd": "Largest cavity diameter (angstrom)",
    "pld": "Pore limiting diameter (angstrom)",
    "void_fraction": "Void fraction (0-1)",
    "surface_area_m2g": "Surface area (m2/g)",
}

_state: dict[str, Any] = {}


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


def load_models(settings: dict) -> dict[float, XGBRegressor]:
    models = {}
    schema = None
    paths = tuple(str(MODELS_DIR / settings["models"][str(p)]) for p in PRESSURES)
    for pressure, path in zip(PRESSURES, paths):
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        settings = load_app_settings()
        models = load_models(settings)
        names = list(models[PRESSURES[0]].get_booster().feature_names)
        _state["settings"] = settings
        _state["models"] = models
        _state["feature_names"] = names
        _state["sbu_path"] = os.environ.get("MOF_SBU_PATH") or settings.get("sbu_path")
        LOGGER.info("Models loaded: %s", settings["model_label"])
    except Exception as exc:
        LOGGER.exception("Failed to load models during startup")
        _state["startup_error"] = str(exc)
    yield
    _state.clear()


app = FastAPI(title="MOF CO2 Uptake Predictor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    if "startup_error" in _state:
        raise HTTPException(status_code=503, detail={"error": _state["startup_error"]})
    return {
        "status": "ok",
        "model_label": _state.get("settings", {}).get("model_label"),
        "pressures": PRESSURES,
        "sbu_path": _state.get("sbu_path") is not None,
    }


def _resolve_sbu_path(settings: dict) -> str | None:
    sbu_path = os.environ.get("MOF_SBU_PATH") or settings.get("sbu_path")
    if sbu_path is None:
        return None
    if not isinstance(sbu_path, str) or not sbu_path:
        raise ValueError("app.sbu_path must be a non-empty path or null.")
    path = Path(sbu_path).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)


def _features_from_upload(
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


def _predict_all(models: dict[float, XGBRegressor], features: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Pressure (bar)": PRESSURES,
        "Predicted CO2 uptake (mol/kg)": [
            float(models[pressure].predict(features)[0]) for pressure in PRESSURES
        ],
    })


def _shap_to_dict(explanation: shap.Explanation, feature_names: list[str]) -> dict[str, Any]:
    values = explanation.values
    data = explanation.data
    base = explanation.base_values
    return {
        "baseline": float(base.item() if hasattr(base, "item") else base),
        "feature_names": feature_names,
        "feature_values": [float(v) for v in data],
        "contributions": [float(v) for v in values],
    }


def _generate_summary(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    explanations: dict[float, shap.Explanation],
) -> dict[str, str]:
    return generate_pressure_comparison(
        predictions,
        features,
        explanations[PRESSURES[0]],
        explanations[PRESSURES[-1]],
    )


@app.post("/api/predict")
async def predict(
    lcd: float = Form(..., ge=0),
    pld: float = Form(..., ge=0),
    void_fraction: float = Form(..., ge=0, le=1),
    surface_area_m2g: float = Form(..., ge=0),
    cif: UploadFile = File(...),
    include_gemini: bool = Form(False),
) -> JSONResponse:
    if "startup_error" in _state:
        raise HTTPException(status_code=503, detail={"error": _state["startup_error"]})

    models = _state.get("models")
    settings = _state.get("settings")
    names = _state.get("feature_names")
    if models is None or settings is None or names is None:
        raise HTTPException(status_code=503, detail={"error": "Models not loaded"})

    geometry = {
        "lcd": lcd,
        "pld": pld,
        "void_fraction": void_fraction,
        "surface_area_m2g": surface_area_m2g,
    }

    try:
        cif_bytes = await cif.read()
        if not cif_bytes:
            raise ValueError("The uploaded CIF is empty.")

        sbu_path = _resolve_sbu_path(settings)
        features = _features_from_upload(
            cif_bytes, geometry, names, sbu_path,
            settings["extraction_timeout_seconds"],
        )
        predictions = _predict_all(models, features)

        explanations: dict[float, shap.Explanation] = {}
        for pressure in PRESSURES:
            explanations[pressure] = shap.TreeExplainer(models[pressure])(features)[0]

        shap_by_pressure = {
            str(p): _shap_to_dict(explanations[p], names) for p in PRESSURES
        }
        summary = _generate_summary(predictions, features, explanations)
        gemini_summary = None
        gemini_error = None
        if include_gemini:
            try:
                gemini_summary = await run_in_threadpool(
                    generate_gemini_summary,
                    predictions.iloc[:, 1].tolist(),
                    summary,
                )
            except GeminiSummaryError as exc:
                LOGGER.warning("Gemini summary failed: %s", exc)
                gemini_error = str(exc)

        csv_buffer = io.StringIO()
        predictions.to_csv(csv_buffer, index=False)
        csv_base64 = base64.b64encode(csv_buffer.getvalue().encode("utf-8")).decode("ascii")

        features_row = features.iloc[0].to_dict()

        return JSONResponse({
            "pressures": [
                {"pressure": float(row["Pressure (bar)"]), "uptake": float(row["Predicted CO2 uptake (mol/kg)"])}
                for _, row in predictions.iterrows()
            ],
            "features": {name: float(value) for name, value in features_row.items()},
            "shap": shap_by_pressure,
            "summary": summary,
            "gemini_summary": gemini_summary,
            "gemini_error": gemini_error,
            "predictions_csv": csv_base64,
        })
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=400, detail={"error": f"CIF processing or prediction failed: {exc}"}) from exc


# Serve the React frontend if it has been built.
_frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")

    @app.get("/{catch_all:path}")
    async def serve_react(catch_all: str) -> FileResponse:
        return FileResponse(_frontend_dist / "index.html")
