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
- Raw CIF upload would require an additional extraction pipeline and is not
  part of the prepared-data MVP.

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
