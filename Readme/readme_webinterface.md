# Part 2: Web interface pipeline

[Back to Part 1: Research and model selection](Readme_model.md)

## Goal

Run a Streamlit demo around the selected **Phase 3 Model B** models:

**CIF -> Zeo++ geometry + MOFid/RDKit/pymatgen chemistry -> validation ->
XGBoost predictions -> SHAP -> local and optional Gemini explanations**

Gemini narrates validated model outputs only. It does not calculate uptake,
replace XGBoost, or receive the uploaded CIF.

## Current state versus target

| Component | Current state | Phase 3 target |
|-----------|---------------|----------------|
| Interface | CIF-only Streamlit upload with computed geometry and limitations | Broader browser and deployment validation |
| Models | Five configured Phase 3 Model B joblib models; baseline frozen as phase3-model-b-v1 | Distribute versioned bundle and complete native deployment recipe |
| Predictions | Five pressures, non-scrolling table/chart/CSV download for geometry + CIF | Preserve these outputs |
| SHAP | Waterfall for the geometry + CIF row at a selected pressure | Explain the same Phase 3 model and feature row used for prediction |
| Scientific summary | Local rule-based summary plus separately requested Gemini draft | Continue scientific review and provider/data-policy review |
| Deployment | Streamlit Docker image with native MOFid and Zeo++ | Broader hosted deployment validation |

The current [app](../src/app.py) is a **local Phase 3 demo**. It requires
trusted corrected model artifacts plus native MOFid and Zeo++ executables.

## 1. Prepare the Phase 3 model bundle

The corrected baseline is now frozen as **phase3-model-b-v1** in the
[manifest](../docs/PHASE3/phase3-model-b-v1.yaml). Its local distribution
folder is `results/frozen_models/phase3-model-b-v1/`: five unchanged model
copies, `manifest.yaml`, `pre_freeze_environment.txt`, and `validation.md`.
The complete frozen bundle and app-ready copies of the five final models are
committed in `results/frozen_models/phase3-model-b-v1/` and
`results/models/`, respectively.

Verify each model against the manifest SHA-256 before loading.
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

Before integrating new or retrained models, record their exact feature
names/order, descriptor definitions and units, dependency versions, and
evaluation metrics. Only load trusted joblib artifacts; deserialization can
execute code. Prepared datasets are distributed separately from GitHub.

Download the prepared artifacts from
[data_mof_demo on OneDrive](https://1drv.ms/f/c/ab0bbacec1d386eb/IgDJcymaZGwpR7fG_rqk97GXAcGskJ7--PZfOQ6_GWqbcFA?e=orssqL).
Copy the prepared dataset files into `data/processed/`, unless alternative
paths are configured. The five final `phase_3_model_b_geometry_plus_linkers_plus_cif_*.joblib`
files are already in `results/models/` after cloning; do not copy the other
experiment models into that folder. Older experiment outputs are archived
locally in `docs/archive/results_archive/`. Request access to the OneDrive
dataset from the owner if needed. With the selected saved models available,
retraining is not required for interface integration.

## 2. Define and validate inputs

Phase 3 requires **geometry + linker + CIF/metal descriptors**. The user
supplies one CIF; prepared-feature CSV and manual geometry entry are not
exposed. Zeo++ computes the four geometry values before the existing chemistry
pipeline runs.

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
4. The 15 pymatgen metal/lattice descriptors and four Zeo++ geometry values.

The result is one numeric row with 48 Model B inputs. Pass the saved model's
feature names to enforce its exact schema and ordering. Missing/invalid Zeo++
output, failed chemistry extraction, rejected fragments, non-finite
descriptors, and incompatible model schemas raise errors; values are not
guessed.
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
from src.zeopp_geometry import extract_geometry

geometry = extract_geometry("uploaded_structure.cif")
features = build_phase3_features(
    "uploaded_structure.cif",
    geometry,
    feature_names=model.get_booster().feature_names,
)
predicted_uptake = model.predict(features)
```

Here `model` is a trusted, loaded Phase 3 Model B artifact. Repeat prediction
with the other four models after verifying their feature order agrees.
`extract_geometry` uses the experimental Zeo++ profile documented in the root
README; it is not known to match the original training pipeline exactly.

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

The app calls this pipeline after the user clicks **Extract descriptors and
predict**. Selecting another SHAP pressure reuses the extracted row; changing
the uploaded CIF hides prior results until prediction is requested again.
Extraction failures are displayed rather than replaced with fabricated
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

### Stage 1: local summary (implemented)

After generating predictions, click **Generate scientific summary**. The app
displays four short sections labelled **Rule-based summary - not AI-generated**:

- **Pressure comparison:** all five predictions and the signed endpoint
  difference (2.5 bar minus 0.01 bar), in mol/kg.
- **Low-pressure SHAP (0.01 bar):** baseline and strongest positive/negative
  contributions for this MOF.
- **Higher-pressure SHAP (2.5 bar):** the same information from the separate
  2.5-bar model.
- **Limitations:** local versus global interpretation, different model
  baselines/scales, simulated data, and non-causality.

The comparison always uses the lowest and highest trained pressures; these
are not universal definitions of low/high pressure. The existing pressure
selector still controls the waterfall independently. The multi-section
comparison replaces the previous single 80-150-word paragraph in the app.
No API key, new dependencies,
network request, retraining, or separate backend is needed.

[scientific_summary.py](../src/scientific_summary.py) checks the five pressure
predictions, the matching feature row, all 48 signed contributions, and SHAP
additivity independently at each endpoint before describing the strongest
positive and negative contributions. Endpoint SHAP explanations are calculated
only on the summary-button click, reusing the displayed explanation if it is
already at an endpoint. Larger absolute contributions at 2.5 bar do not alone
establish greater relative importance because the models have different
baselines and uptake scales. No uptake ratios or causal mechanisms are inferred.
Numbers are displayed with six significant digits; the prediction table and
download retain their existing behavior. Negative or non-monotonic predictions
are flagged, not clipped or smoothed.

Generation requires an explicit click. Changing the CIF, configured
model context, or SHAP pressure clears the summary; generate it again for the
new context. A generation error is shown explicitly and the button can be
clicked to retry. Predictions and SHAP remain independent of summary failures.
If the displayed SHAP explanation fails, summary generation is disabled. If an
endpoint explanation fails during generation, an explicit summary error is shown;
the selected-pressure waterfall and predictions remain available.

To review the wording:

1. Start the app using the quick-start instructions below.
2. Upload `hMOF-0.cif` from [sample CIFs](../notebook/sample/cif_files/).
3. Click **Extract descriptors and predict** and inspect the four computed
   geometry values, then click **Generate scientific summary**.
4. Compare all five uptake numbers against the app table and verify the signed
   endpoint difference. Select 0.01 bar in the waterfall and generate the summary
   again to compare the low-pressure contributions, signs, and baseline. Repeat
   with 2.5 bar for the higher-pressure section. Check the simulated-data,
   separate-baseline, local-importance, and non-causality limitations.
5. Repeat for `hMOF-10` and `hMOF-100`. Change pressure or upload to confirm
   the old summary disappears.

Focused validation (headless plotting):

```bash
MPLBACKEND=Agg python -m pytest tests/test_scientific_summary.py tests/test_interface.py -q
```

The three reference-row tests use real frozen models at all five pressures
and validate the endpoint comparison when the prepared Phase 3 dataset is
available. They validate numerical
consistency, not new CIF extraction or scientific approval. Native MOFid is
still required for uploads.

### Stage 2: optional Gemini draft (implemented)

After the local summary exists, **Generate Gemini explanation** makes a
separate request without repeating CIF extraction, prediction, or SHAP.
The button is available only when `GEMINI_ENABLED=true`; the server also
requires `GEMINI_API_KEY` and a `GEMINI_MODEL` beginning with `gemini-`.

The request contains five prediction values, validated local summary sections,
and a feature-label glossary. It does not contain the CIF. Provider responses
must pass heading, length, pressure/value, and numeric-evidence checks before
display. Errors are public-safe and leave all local results intact. The draft
is labelled AI-generated and still requires scientific review.

## 6. Test and deploy

- Add interface tests covering valid inputs, invalid CIFs, missing artifacts,
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
python scripts/install_zeopp.py
# Only needed when Zeo++ is installed outside its default location:
export ZEO_NETWORK_PATH="/path/to/zeo++-0.3/network"
python -m streamlit run src/app.py
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
  for MLflow, Databricks, web API frameworks, database tooling, the Docker Python
  SDK, and other unrelated tools were removed. Dependencies required by the
  retained packages are still installed transitively by pip. No packages were
  uninstalled from the existing environment.
- Native MOFid's `sbu` executable and Zeo++'s `network` executable remain
  separate requirements. Adding a PyPI package named `mofid` does not
  reproduce its native installation.
- These files are dependency declarations, not a complete transitive lock or
  a certification of a fresh installation. Capture a reproducible environment
  and native MOFid setup, then validate them before formally freezing models.

The app needs the five corrected Phase 3 Model B artifacts described above in
the configured models directory. Reference datasets are not required for
CIF-only inference.

1. Upload a CIF and click **Extract descriptors and predict**.
2. Review Zeo++ geometry, predictions, and a selected-pressure SHAP plot.
3. Generate the local summary and optionally request a Gemini explanation.

The page shows a welcome heading and short upload instructions. Scientific
limitations remain available under **About these predictions**, collapsed by
default. All five prediction rows appear in a static table without an internal
scrollbar. Prepared-feature CSV batch input is no longer exposed; prediction
CSV downloads remain available.

The `app` section of [configs/config.yaml](../configs/config.yaml) controls
model filenames, the retained model label metadata, an optional `sbu_path`, and
`extraction_timeout_seconds`. `MOF_SBU_PATH` overrides `app.sbu_path`; relative
configured executable paths resolve against the repository root. Model
directories still use the existing `paths.models_dir` / `MOF_MODELS_DIR`.
No personal machine paths are embedded in the app.

Configure file locations in [configs/config.yaml](../configs/config.yaml).
Focused app tests are in [tests/test_interface.py](../tests/test_interface.py).
The real-CIF app test requires `MOF_REFERENCE_CIF_DIR`, `MOF_SBU_PATH`, and
`ZEO_NETWORK_PATH`. Next milestones are broader uploaded-structure validation
and hosted deployment validation.
