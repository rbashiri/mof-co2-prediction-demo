"""Local, deterministic descriptions of predictions and matching SHAP results."""

import math

import numpy as np
import pandas as pd
import shap

PRESSURES = [0.01, 0.05, 0.1, 0.5, 2.5]
SUMMARY_LABEL = "Rule-based summary - not AI-generated"


def describe_contributions(features: pd.DataFrame, values: np.ndarray) -> str:
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
    return " ".join(contributions)


def validate_summary_inputs(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    explanation: shap.Explanation,
    pressure: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate predictions and all SHAP contributions for one pressure and row."""
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
    return uptake, values, base


def generate_summary(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    explanation: shap.Explanation,
    pressure: float,
) -> str:
    """Validate one model row and describe its outputs without changing them."""
    uptake, values, base = validate_summary_inputs(
        predictions, features, explanation, pressure,
    )

    pairs = "; ".join(
        f"{p:g} bar: {value:.6g}" for p, value in zip(PRESSURES, uptake)
    )
    warnings = []
    if (uptake < 0).any():
        warnings.append("Negative predicted uptake requires review.")
    if (np.diff(uptake) < 0).any():
        warnings.append("Non-monotonic predictions require review.")
    summary = " ".join([
        f"Predicted CO2 uptake (mol/kg) is {pairs}.",
        f"At {pressure:g} bar, SHAP contributions are relative to a model baseline "
        f"of {base:.6g} mol/kg.",
        describe_contributions(features, values),
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


def generate_pressure_comparison(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    low_explanation: shap.Explanation,
    high_explanation: shap.Explanation,
) -> dict[str, str]:
    """Compare the trained endpoints using independently validated SHAP results."""
    for pressure, explanation in (
        (PRESSURES[0], low_explanation), (PRESSURES[-1], high_explanation),
    ):
        validate_summary_inputs(predictions, features, explanation, pressure)

    uptake = predictions.iloc[:, 1].to_numpy(dtype=float)
    difference = float(uptake[-1] - uptake[0])
    if not math.isfinite(difference):
        raise ValueError("The endpoint uptake difference must be finite.")
    pairs = "; ".join(f"{p:g} bar: {value:.6g}" for p, value in zip(PRESSURES, uptake))
    warnings = []
    if (uptake < 0).any():
        warnings.append("Negative predicted uptake requires review.")
    if (np.diff(uptake) < 0).any():
        warnings.append("Non-monotonic predictions require review.")
    sections = {
        "Pressure comparison": (
            f"Predicted CO2 uptake (mol/kg) is {pairs}. "
            f"The endpoint difference (2.5 bar minus 0.01 bar) is {difference:+.6g} mol/kg. "
            "These are the lowest and highest trained pressures, not a general "
            "definition of low- and high-pressure adsorption. "
            + " ".join(warnings)
        ).strip(),
    }
    for title, pressure, explanation in (
        ("Low-pressure SHAP (0.01 bar)", PRESSURES[0], low_explanation),
        ("Higher-pressure SHAP (2.5 bar)", PRESSURES[-1], high_explanation),
    ):
        base = float(np.asarray(explanation.base_values).item())
        sections[title] = (
            f"At {pressure:g} bar, the model baseline is {base:.6g} mol/kg. "
            + describe_contributions(features, np.asarray(explanation.values, dtype=float))
            + " All 48 contributions plus this baseline reproduce the prediction "
            "within numerical tolerance."
        )
    sections["Limitations"] = (
        "Each pressure model has its own baseline and uptake scale. Larger absolute "
        "SHAP values at higher pressure do not alone establish greater relative "
        "importance. These are local explanations for this MOF, not global feature "
        "rankings. These models were evaluated on simulated hMOF data, not experimental "
        "measurements. SHAP describes model attribution, not chemical causation. "
        "No per-material uncertainty interval or experimental validation is established. "
        "Predictions are unchanged; scientific review is required."
    )
    return sections
