import numpy as np
import pandas as pd
import pytest
import shap

from src.app import load_app_settings, load_models, predict_all
from src.paths import MODELS_DIR, PHASE3_DATA_PATH
from src.scientific_summary import PRESSURES, generate_summary


def example():
    features = pd.DataFrame(
        [[float(i) for i in range(48)]], columns=[f"feature_{i}" for i in range(48)],
    )
    predictions = pd.DataFrame({
        "Pressure (bar)": PRESSURES,
        "Predicted CO2 uptake (mol/kg)": [0.02, 0.07, 0.23, 0.9, 4.33],
    })
    explanation = shap.Explanation(
        values=np.array([0.4, 0.1, -0.15, -0.05] + [0.0] * 44),
        base_values=0.6, data=features.iloc[0].to_numpy(),
        feature_names=features.columns.tolist(),
    )
    return predictions, features, explanation


def test_summary_numbers_signs_length_and_no_mutation():
    predictions, features, explanation = example()
    original = predictions.copy(deep=True)
    summary = generate_summary(predictions, features, explanation, 0.5)
    assert 80 <= len(summary.split()) <= 150
    for pressure, uptake in zip(PRESSURES, predictions.iloc[:, 1]):
        assert f"{pressure:g} bar: {uptake:.6g}" in summary
    assert "feature_0 (+0.4 mol/kg)" in summary
    assert "feature_2 (-0.15 mol/kg)" in summary
    assert "baseline of 0.6 mol/kg" in summary
    assert "not chemical causation" in summary
    assert "not experimental measurements" in summary
    pd.testing.assert_frame_equal(predictions, original)


@pytest.mark.parametrize("case", [
    "missing_pressure", "wrong_pressure", "nan_prediction", "missing_feature",
    "wrong_row", "wrong_order", "nan_shap", "wrong_baseline", "missing_shap",
])
def test_inconsistent_inputs_are_rejected(case):
    predictions, features, explanation = example()
    if case == "missing_pressure":
        predictions = predictions.iloc[:-1]
    elif case == "wrong_pressure":
        predictions.iloc[0, 0] = 0.02
    elif case == "nan_prediction":
        predictions.iloc[0, 1] = np.nan
    elif case == "missing_feature":
        features = features.iloc[:, :-1]
    elif case == "wrong_row":
        explanation.data = explanation.data + 1
    elif case == "wrong_order":
        explanation.feature_names = explanation.feature_names[::-1]
    elif case == "nan_shap":
        explanation.values[0] = np.nan
    elif case == "wrong_baseline":
        explanation.base_values = 0.8
    elif case == "missing_shap":
        explanation.values = explanation.values[:-1]
    with pytest.raises(ValueError):
        generate_summary(predictions, features, explanation, 0.5)


def test_negative_nonmonotonic_outputs_are_flagged_not_changed():
    predictions, features, explanation = example()
    predictions.iloc[0, 1] = -0.1
    predictions.iloc[2, 1] = 0.04
    original = predictions.copy(deep=True)
    summary = generate_summary(predictions, features, explanation, 0.5)
    assert "Negative predicted uptake requires review" in summary
    assert "Non-monotonic predictions require review" in summary
    assert 80 <= len(summary.split()) <= 150
    pd.testing.assert_frame_equal(predictions, original)


@pytest.mark.parametrize("sign", [-1, 0, 1])
def test_one_sided_and_zero_contributions(sign):
    predictions, features, explanation = example()
    explanation.values[:] = sign * 0.01
    explanation.base_values = 0.9 - explanation.values.sum()
    summary = generate_summary(predictions, features, explanation, 0.5)
    assert 80 <= len(summary.split()) <= 150
    if sign <= 0:
        assert "no positive SHAP contributions" in summary
    if sign >= 0:
        assert "no negative SHAP contributions" in summary


def test_wrong_selected_pressure_is_rejected():
    predictions, features, explanation = example()
    with pytest.raises(ValueError, match="additivity"):
        generate_summary(predictions, features, explanation, 0.1)
    with pytest.raises(ValueError, match="trained SHAP pressure"):
        generate_summary(predictions, features, explanation, 1.0)


@pytest.mark.parametrize("name", ["hMOF-0", "hMOF-10", "hMOF-100"])
def test_real_reference_rows_at_every_pressure(name):
    if not PHASE3_DATA_PATH.exists():
        pytest.skip("Prepared Phase 3 reference dataset required.")
    settings = load_app_settings()
    models = load_models(tuple(
        str(MODELS_DIR / settings["models"][str(p)]) for p in PRESSURES
    ))
    names = models[PRESSURES[0]].get_booster().feature_names
    assert names is not None
    reference = pd.read_csv(PHASE3_DATA_PATH, low_memory=False)
    features = reference.loc[reference.filename.eq(name), list(names)]
    assert features.shape == (1, 48)
    predictions = predict_all(models, features)
    original = predictions.copy(deep=True)
    for pressure in PRESSURES:
        explanation = shap.TreeExplainer(models[pressure])(features)[0]
        summary = generate_summary(predictions, features, explanation, pressure)
        assert 80 <= len(summary.split()) <= 150
        assert f"At {pressure:g} bar" in summary
        for p, uptake in zip(PRESSURES, predictions.iloc[:, 1]):
            assert f"{p:g} bar: {uptake:.6g}" in summary
    pd.testing.assert_frame_equal(predictions, original)
