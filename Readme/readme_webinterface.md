# Part 2: Web interface pipeline

[Back to Part 1: Research and model selection](Readme_model.md)

## Goal

Build a Streamlit demo around the selected **Phase 3 Model B** models:

**Prepared MOF features -> validation -> XGBoost predictions -> SHAP
explanations -> optional LLM scientific summary**

The LLM writes a summary of model outputs; it does not calculate uptake or
replace XGBoost.

## Current state versus target

| Component | Current state | Phase 3 target |
|-----------|---------------|----------------|
| Interface | Streamlit app with manual geometry entry and CSV upload | Inputs compatible with the full Model B feature set |
| Models | Five Phase 1 joblib models | Five Phase 3 Model B joblib models |
| Predictions | Five pressures, table/chart for manual entry; downloadable CSV for uploads | Preserve these outputs using Phase 3 |
| SHAP | Waterfall explanation for manual entry at a selected pressure | Explain the same Phase 3 model and feature row used for prediction |
| LLM summary | Not implemented; provider undecided | Optional, grounded summary of predictions and SHAP |
| Deployment | Local prototype; Dockerfile is only a base image | Tested, reproducible deployment with separately supplied model artifacts |

The current [app](../src/app.py) is a **Phase 1 prototype**, not yet a
Phase 3 web demo. Documentation changes do not switch its models.

## 1. Prepare the Phase 3 model bundle

Train with [src/train.py](../src/train.py) using
`hmof_phase3_structure_descriptors.csv`:

```bash
python src/train.py --phase phase3
```

The script trains both Model A and Model B. Use the five selected Model B
artifacts from the configured models directory:

```text
phase_3_model_b_geometry_plus_linkers_plus_cif_0.01bar.joblib
phase_3_model_b_geometry_plus_linkers_plus_cif_0.05bar.joblib
phase_3_model_b_geometry_plus_linkers_plus_cif_0.1bar.joblib
phase_3_model_b_geometry_plus_linkers_plus_cif_0.5bar.joblib
phase_3_model_b_geometry_plus_linkers_plus_cif_2.5bar.joblib
```

Before integrating them, record their exact feature names/order, descriptor
definitions and units, dependency versions, and evaluation metrics. Only load
trusted joblib artifacts; deserialization can execute code. Data and models
are distributed separately from GitHub.

Download the prepared artifacts from
[data_mof_demo on OneDrive](https://1drv.ms/f/c/ab0bbacec1d386eb/IgDJcymaZGwpR7fG_rqk97GXAcGskJ7--PZfOQ6_GWqbcFA?e=orssqL).
The owner uploads bundle updates manually. Copy `data/processed/` files into
the repository's `data/processed/`, and `models/` files into `results/models/`,
unless alternative paths are configured. Request access from the owner if
needed. With the selected saved models available, retraining is not required
for interface integration.

## 2. Define and validate inputs

Phase 3 requires **geometry + linker + CIF/metal descriptors**, not just four
geometry values. Agree on the input experience before implementing it:

- CSV upload of already prepared feature rows is the simplest initial path.
- Manual entry requires the complete feature schema; alternatively, selecting
  an existing MOF could populate a prepared row.
- The agreed next interface accepts four supplied geometry values and a CIF.
  Geometry is not calculated from the CIF; Zeo++ is not required.

For the planned geometry-plus-CIF workflow,
[phase3_extract_metal_descriptors.py](../src/phase3_extract_metal_descriptors.py)
provides reusable pymatgen extraction of the 15 metal/lattice descriptors.
These outputs matched the saved Phase 3 dataset for five known CIFs
(hMOF-0, hMOF-27262, hMOF-10022, hMOF-10, and hMOF-100), with relative and
absolute tolerances of 1e-8. Oxidation-state estimation is excluded.
The script is not yet connected to the app.

### Combined geometry-plus-CIF feature pipeline

[phase3_features.py](../src/phase3_features.py) now combines:

1. MOFid native fragmentation of the CIF.
2. The training-time RDKit fragment filters in
   [extract_linkers.py](../src/extract_linkers.py).
3. All 29 aggregated linker descriptors from
   [linker_descriptors.py](../src/linker_descriptors.py).
4. The 15 pymatgen metal/lattice descriptors and four supplied geometry values.

The result is one numeric row with 48 Model B inputs. Pass the saved model's
feature names to enforce its exact schema and ordering. Missing/invalid
geometry, failed extraction, rejected fragments, non-finite descriptors,
and incompatible model schemas raise errors; descriptors are not guessed.
Extraction uses temporary output directories that are cleaned automatically.

Set `MOF_SBU_PATH` to the installed MOFid `bin/sbu` executable, or supply
`sbu_path` to the function. If neither is supplied, `sbu` must be on PATH.
This function calls MOFid's native fragmentation executable directly; it
does not need Java topology classification. The MOFid Python wrapper still
requires Java for its separate full-identifier workflow.

From the repository root, in the project environment:

```python
from src.phase3_features import build_phase3_features

features = build_phase3_features(
    "uploaded_structure.cif",
    {
        "lcd": 11.75,
        "pld": 10.75,
        "void_fraction": 0.795539,
        "surface_area_m2g": 3676.3,
    },
    feature_names=model.get_booster().feature_names,
)
predicted_uptake = model.predict(features)
```

Here `model` is a trusted, loaded Phase 3 Model B artifact. Repeat prediction
with the other four models after verifying their feature order agrees.
The supplied geometry must correspond to the uploaded structure; the function
cannot verify that correspondence from the CIF alone.

All 48 inputs matched the reference dataset on the five examples above at
`rtol=1e-8`, `atol=1e-8`; predictions from all five pressure models matched
reference-row predictions at `rtol=1e-6`, `atol=1e-6`. This is an initial
validation on known hMOFs, not certification for arbitrary uploaded structures.

Run the focused tests with:

```bash
python -m pytest tests/test_phase3_features.py tests/test_train_features.py -q
```

To include the real-CIF comparisons, set `MOF_REFERENCE_DATA_DIR`,
`MOF_REFERENCE_CIF_DIR`, `MOF_REFERENCE_MODELS_DIR`, and `MOF_SBU_PATH` to local
prepared datasets, reference CIFs, corrected model artifacts, and the native
executable respectively. Those five integration cases are skipped when the
reference environment is not configured. Native MOFid installation remains
a separate deployment prerequisite, not a Python-only dependency.

The next step is connecting this validated function to the Streamlit app.

Validate required columns, numeric types, missing/non-finite values, and
feature ordering against the trained models. Exclude identifiers and observed
uptake targets from predictors. Report invalid inputs explicitly rather than
silently filling missing descriptors.

## 3. Predict uptake

Load the five Model B models once, then use the same validated feature rows
for all pressures. Display pressure in bar and predicted uptake in mol/kg,
with a results table, pressure curve, and downloadable predictions.

Verify app predictions against direct predictions from the saved artifacts
on known rows before publishing.

## 4. Explain with SHAP

Use the selected pressure's Phase 3 model and the same feature row supplied
to prediction. Show positive and negative contributions and the model's
baseline value. For uploads, define which row the user selects to explain.

Label these as model explanations, not causal evidence or experimentally
verified adsorption mechanisms.

## 5. Add an optional scientific summary

Choose a provider together before implementation. Decide on local versus
hosted execution, costs, credentials, and what information may leave the
computer.

The summary should state the predicted uptake by pressure, describe the
strongest SHAP contributions, and acknowledge that the model was evaluated
on simulated hMOF data. It must not invent mechanisms, citations, or
experimental validation. Keep predictions and SHAP usable if the summary
service is unavailable; show any service failure explicitly.

## 6. Test and deploy

- Add interface tests covering valid inputs, invalid CSVs, missing artifacts,
  and consistency between prediction and explanation.
- Verify the interface in a browser, including uploads and downloads.
- Review runtime dependencies and complete the Dockerfile if using containers.
- Decide hosting, access permissions, model distribution, and any API budget.
- Keep credentials outside source control.

## Current prototype quick start

From the repository root, in an activated Python environment:

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

The prototype still needs the five `phase1_xgboost_*.joblib` artifacts in
the configured models directory. Its reference data is `data_clean_v2`.
Changing the configured Phase 3 dataset alone will **not** migrate the app.

Configure file locations in [configs/config.yaml](../configs/config.yaml).
The first development milestone is to wire the full Phase 3 input schema,
Model B loading, and matching SHAP explanations into the existing app.
