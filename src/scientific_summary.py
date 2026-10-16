"""Local, deterministic descriptions of predictions and matching SHAP results."""

import math

import numpy as np
import pandas as pd
import shap

PRESSURES = [0.01, 0.05, 0.1, 0.5, 2.5]
SUMMARY_LABEL = "Rule-based summary - not AI-generated"


def generate_summary(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    explanation: shap.Explanation,
    pressure: float,
) -> str:
    """Validate one model row and describe its outputs without changing them."""
    if (
        predictions.shape != (5, 2)
        or list(predictions.columns)
        != ["Pressure (bar)", "Predicted CO2 uptake (mol/kg)"]
        or predictions.iloc[:, 0].tolist() != PRESSURES
    ):
        raise ValueError("Summary requires predictions at exactly the five trained pressures.")
    uptake = predictions.iloc[:, 1].to_numpy(dtype=float)
    if not np.isfinite(uptake).all() or pressure not in PRESSURES:
        raise ValueError("Summary requires finite predictions and a trained SHAP pressure.")
    if (
        features.shape != (1, 48)
        or not features.columns.is_unique
        or not all(isinstance(name, str) and name.strip() for name in features.columns)
    ):
        raise ValueError("Summary requires one row of 48 uniquely named features.")
    row = features.iloc[0].to_numpy(dtype=float)
    values = np.asarray(explanation.values, dtype=float)
    data = np.asarray(explanation.data, dtype=float)
    baseline = np.asarray(explanation.base_values, dtype=float)
    if (
        values.shape != (48,)
        or data.shape != (48,)
        or baseline.size != 1
        or explanation.feature_names is None
        or list(explanation.feature_names) != list(features.columns)
        or not np.isfinite(row).all()
        or not np.isfinite(values).all()
        or not np.isfinite(data).all()
        or not np.isfinite(baseline).all()
        or not np.array_equal(data, row)
    ):
        raise ValueError("SHAP must contain all 48 finite contributions for the same feature row.")
    base = float(baseline.item())
    if not math.isclose(
        base + float(values.sum()), float(uptake[PRESSURES.index(pressure)]),
        rel_tol=1e-5, abs_tol=1e-5,
    ):
        raise ValueError("SHAP additivity failed for the selected pressure prediction.")

    pairs = "; ".join(
        f"{p:g} bar: {value:.6g}" for p, value in zip(PRESSURES, uptake)
    )
    contributions = []
    for mask, direction, sign in (
        (values > 0, "positive", "+"), (values < 0, "negative", "-"),
    ):
        indices = np.flatnonzero(mask)
        if indices.size:
            index = int(indices[np.argmax(np.abs(values[indices]))])
            contributions.append(
                f"The strongest {direction} contribution is {features.columns[index]} "
                f"({sign}{abs(values[index]):.6g} mol/kg)."
            )
        else:
            contributions.append(f"There are no {direction} SHAP contributions.")
    warnings = []
    if (uptake < 0).any():
        warnings.append("Negative predicted uptake requires review.")
    if (np.diff(uptake) < 0).any():
        warnings.append("Non-monotonic predictions require review.")
    summary = " ".join([
        f"Predicted CO2 uptake (mol/kg) is {pairs}.",
        f"At {pressure:g} bar, SHAP contributions are relative to a model baseline "
        f"of {base:.6g} mol/kg.",
        *contributions,
        "All 48 contributions plus the baseline reproduce the selected prediction "
        "within numerical tolerance.",
        *warnings,
        "These models were evaluated on simulated hMOF data, not experimental "
        "measurements. SHAP describes model attribution, not chemical causation. "
        "No per-material uncertainty interval or experimental validation is "
        "established. Predictions are reported unchanged; scientific review is required.",
    ])
    if not 80 <= len(summary.split()) <= 150:
        raise ValueError("Summary must contain 80-150 words; review the feature labels.")
    return summary
