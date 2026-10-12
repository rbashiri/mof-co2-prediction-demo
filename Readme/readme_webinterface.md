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
| Interface | Four geometry inputs + CIF upload; prepared 48-feature CSV batch mode | Broader browser and deployment validation |
| Models | Five configured Phase 3 Model B joblib models; baseline frozen as phase3-model-b-v1 | Distribute versioned bundle and complete native deployment recipe |
| Predictions | Five pressures, table/chart/download for geometry + CIF; CSV batch download | Preserve these outputs |
| SHAP | Waterfall for the geometry + CIF row at a selected pressure | Explain the same Phase 3 model and feature row used for prediction |
| LLM summary | Not implemented; provider undecided | Optional, grounded summary of predictions and SHAP |
| Deployment | Local prototype; Dockerfile is only a base image | Tested, reproducible deployment with separately supplied model artifacts |

The current [app](../src/app.py) is a **local Phase 3 demo**. It requires
trusted corrected model artifacts and an installed native MOFid executable.
The LLM summary and reproducible hosted deployment remain unfinished.

## 1. Prepare the Phase 3 model bundle

The corrected baseline is now frozen as **phase3-model-b-v1** in the
[manifest](../docs/PHASE3/phase3-model-b-v1.yaml). Its local distribution
folder is `results/frozen_models/phase3-model-b-v1/`: five unchanged model
copies, `manifest.yaml`, `pre_freeze_environment.txt`, and `validation.md`.
Models remain outside Git; the owner must upload this versioned folder to
the shared storage manually. The online copy has not been verified.

Verify each downloaded model against the manifest SHA-256 before loading.
The existing app paths remain unchanged. To use the versioned local bundle,
set `MOF_MODELS_DIR` to its directory before starting the app. This freeze
identifies the tested baseline, not a completed hosted deployment. Native
MOFid rebuilding and full browser upload/download review remain outstanding.
Do not overwrite the frozen folder or manifest; use a new version for changed
models. Retraining is unnecessary to use this baseline.

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
The owner uploads bundle updates manually. Copy the five frozen
`phase_3_model_b_geometry_plus_linkers_plus_cif_*.joblib` files into
`results/models/`, and copy dataset files into `data/processed/`, unless
alternative paths are configured. Do not copy the other experiment models
into `results/models/`; older experiment outputs are archived locally in
`docs/archive/results_archive/`. Request access from the owner if needed. With
the selected saved models available, retraining is not required for interface
integration.

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
The script is connected through the combined feature function to the app.

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

Install [requirements-dev.txt](../requirements-dev.txt) for notebook and test
tools; it also includes the runtime requirements:

```bash
python -m pip install -r requirements-dev.txt
```

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

The app calls this function after the user clicks **Extract descriptors and
predict**. Selecting another SHAP pressure reuses the extracted row; changing
geometry or the uploaded CIF hides prior results until prediction is requested
again. Extraction failures are displayed rather than replaced with fabricated
features.

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

The team has selected the LLM summary for the hackathon delivery scope.
It remains unimplemented; the provider and data policy must be agreed
together. See the [task and ownership plan](../docs/hackathon_tasks_and_ownership.md)
for its scientific contract, acceptance tests, owners, and deadline.
The core prediction/SHAP workflow remains independent of LLM availability.

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

## Local Phase 3 demo quick start

From the repository root, in an activated Python environment:

```bash
python -m pip install -r requirements.txt
# Set this to your installed MOFid native executable:
export MOF_SBU_PATH="/path/to/mofid/bin/sbu"
streamlit run src/app.py
```

### Dependency declarations

- [requirements.txt](../requirements.txt) contains the Python dependencies for
  prediction, RDKit/pymatgen CIF descriptors, SHAP, and Streamlit. Runtime pins
  match the local environment tested with Python **3.12.3**; `pymatgen-core`
  and SciPy are pinned explicitly to preserve the scientific stack.
- [requirements-dev.txt](../requirements-dev.txt) includes the runtime file
  and adds IPython/ipykernel, Seaborn for the retained EDA notebooks, and
  pytest/ipytest for tests. Use this file for research and development.
- The previous broad package list was reviewed against current source,
  retained notebook imports, and setup instructions. Unused direct declarations
  for MLflow, Databricks, Flask/FastAPI, database tooling, the Docker Python
  SDK, and other unrelated tools were removed. Dependencies required by the
  retained packages are still installed transitively by pip. No packages were
  uninstalled from the existing environment.
- Native MOFid's `sbu` executable and supporting resources remain a separate
  requirement. Adding a PyPI package named `mofid` does not reproduce that
  native installation. CMake is a build tool, not an app runtime dependency.
- These files are dependency declarations, not a complete transitive lock or
  a certification of a fresh installation. Capture a reproducible environment
  and native MOFid setup, then validate them before formally freezing models.

The app needs the five corrected Phase 3 Model B artifacts described above in
the configured models directory. Reference datasets are not required for
geometry + CIF inference: users supply their own geometry values.

1. Select **Geometry + CIF**.
2. Enter LCD and PLD in angstrom, void fraction between 0 and 1, and surface
   area in m2/g. These values must describe the uploaded structure.
3. Upload a CIF and click **Extract descriptors and predict**.
4. Review predictions, download them, and choose a pressure for SHAP.

Alternatively, select **Prepared feature CSV** for batch predictions. Supply
all 48 model features as finite numeric columns. Identifiers can be included
for traceability but are not passed to the models. Batch SHAP is not yet
implemented.

The `app` section of [configs/config.yaml](../configs/config.yaml) controls
model filenames, the displayed model label, an optional `sbu_path`, and
`extraction_timeout_seconds`. `MOF_SBU_PATH` overrides `app.sbu_path`; relative
configured executable paths resolve against the repository root. Model
directories still use the existing `paths.models_dir` / `MOF_MODELS_DIR`.
No personal machine paths are embedded in the app.

Configure file locations in [configs/config.yaml](../configs/config.yaml).
Focused app tests are in [tests/test_interface.py](../tests/test_interface.py).
The real-CIF app test requires `MOF_REFERENCE_CIF_DIR` and `MOF_SBU_PATH`.
Next milestones are broader uploaded-structure validation, reproducible
deployment, and the optional LLM summary.
