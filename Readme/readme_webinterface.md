# Part 2: Web interface pipeline

[Back to Part 1: Research and model selection](Readme_model.md)

## Goal

### Active React/FastAPI workflow

This is a hackathon MVP focused on demonstrating the working application,
with broader CIF support and scientific validation planned as future work.
The [MVP test results and limitations](../README.md#hackathon-mvp-scope-tests-and-next-steps)
include both successful uploads and the failed additional CIF; failures are
not treated as completed predictions.

Upload a CIF without entering geometry. The backend now runs Zeo++ for
LCD, PLD, total probe-occupiable void fraction, and total surface area, then
MOFid/RDKit and pymatgen for the remaining descriptors before prediction.
The extracted geometry is displayed alongside the existing five-pressure
predictions, SHAP and CSV download. See the root [README](../README.md) for
installation, exact Zeo++ settings, API compatibility and focused tests.
The geometry card shows a brief explanation of how the probe settings were
chosen from ten sample MOFs. Detailed definitions and limitations remain in
the README and API geometry metadata rather than a second paragraph in the card.
The original training geometry calculation is unknown; automatic results
are not claimed to reproduce it. Extraction failures stop the request.

The sections below also describe the retained legacy Streamlit workflow,
which still accepts manual geometry.

Build a Streamlit demo around the selected **Phase 3 Model B** models:

**Prepared MOF features -> validation -> XGBoost predictions -> SHAP
explanations -> optional LLM scientific summary**

The LLM writes a summary of model outputs; it does not calculate uptake or
replace XGBoost.

### React first-version additions

The bottom team/contact section credits Robabeh Bashiri (project lead),
Ripunjay Kumar, and Umer Farooq, with initial-based avatars. It includes
Robabeh's LinkedIn, GitHub and email links and a CSU Chico campus photo beside
the contact form, without an additional affiliation or address panel.
It is visible before and after a prediction.
Team details and the contact email can be edited in `frontend/src/TeamContact.tsx`;
styles are in `frontend/src/TeamContact.css`, and the supplied campus photo is
in `frontend/public/csu-chico-campus.png`. No university endorsement is implied.
The contact form opens the visitor's email application with a composed message;
it does not send or store messages. Visitors must press Send in their email app.

An always-visible **About these predictions** card above the upload form
introduces the MVP and model estimates. Four colored tool tiles show the roles
of Zeo++, MOFid, RDKit and pymatgen. No click is needed to read the explanation.

The React interface keeps its existing line chart and adds a five-pressure
comparison table. Highest and lowest labels use the actual predictions,
including ties, rather than assuming that uptake increases with pressure.
Labels use six significant figures; the CSV retains the original values.
The uptake line chart uses a labeled logarithmic pressure axis to separate
the low-pressure points. The selected-pressure SHAP chart separates feature
names, signed bars around a zero line, and numerical values into distinct
columns. Both charts scroll horizontally on narrow screens to keep labels
readable; predictions and SHAP values are not changed.

Predictions appear before any optional AI generation. The **Click here to learn
more about the results** button at the end of the results requests a separate
Gemini explanation of the existing predictions. There is no upload-form AI
checkbox and no need to reupload or repeat extraction, predictions or SHAP.
The button sends validated uptake values and endpoint SHAP summary sections
to Google only when clicked.
No CIF file, filename, full dataset, or model file is included. Predictions,
SHAP and CSV downloads remain available if Gemini fails; the UI displays the
failure explicitly. Rule-based sections are retained internally as validated
Gemini input and in the API response for compatibility, but are not displayed
in React. The old Streamlit app is unchanged.

While a prediction request is pending, a decorative molecular animation displays
**Examining your MOF...** and elapsed seconds, not an estimated countdown or
unverified processing stages. The separate explanation button has its own
loading state while prediction results remain visible. The indicator stops on success or failure, resets for each
request, and respects reduced-motion preferences.

The prediction API returns an opaque `explanation_token`. Posting to
`/api/explanations/{token}` generates the explanation from server-retained
validated facts, not client-supplied prompt text. The current single-worker
MVP retains at most 128 contexts for 30 minutes in process memory, without
the CIF or filename. Expired/evicted contexts and server restarts require a
new prediction; HTTP 410 explains this explicitly. Multi-worker deployment
needs a shared context store or sticky routing. The legacy `include_gemini`
API parameter remains available for existing clients but is not sent by React.

Gemini is disabled by default. After your team approves external data sharing,
configure these server-side environment variables (for example, in Render):

- `GEMINI_ENABLED=true`
- `GEMINI_API_KEY`: a private API key, never committed or entered in the browser.
- `GEMINI_MODEL`: a Gemini Flash model ID verified as available to your Google
  project. No model ID or free quota is assumed; check current provider terms.

The implementation adapts Umer's prompt/request/validation approach into
`src/gemini_summary.py`; the local `LLM-umer/` review copy is not imported or
required for deployment. Each provider request has a 25-second network timeout.
Each summary makes one provider request, with no automatic retries or delays.
HTTP errors, network failures, and invalid drafts display an explicit summary
error; no error string is returned as a generated draft. The network timeout
applies to socket operations, not a strict end-to-end deadline.
HTTP failures show fixed guidance based on status and recognized provider error
codes, never raw provider messages, request data, keys, or response metadata.
HTTP 400 requires checking configuration rather than repeatedly retrying.
Output
must be 120-180 words with three ordered plain-text sections: **ADSORPTION
BEHAVIOR**, **STRUCTURE-PROPERTY INSIGHTS**, and **RESEARCH IMPLICATIONS**.
It interprets the pressure trend and endpoint SHAP evidence instead of repeating
the complete prediction table, and suggests one or two follow-up investigations
explicitly labeled as hypotheses. The validator checks section headings and word
count, rejects unknown numeric values, mismatched explicitly formatted
pressure/uptake pairs, repeated pairs or a complete formatted table, and
placeholder feature names. Other numeric phrasing is checked for known values,
not for correct association with a pressure or descriptor. The interpretation uses
plain-language labels for the actual endpoint contributors, explaining upward
and downward pushes on this structure's predicted uptake rather than assuming
the reader understands SHAP. Scientific descriptor terms such as void fraction
and pore limiting diameter are preserved. The label glossary supplies vocabulary, not
evidence of chemistry or importance. General limitations and separate-model
baseline cautions appear in a concise **Scientific limitations** section below
the draft, rather than being required verbatim in every paragraph. Specific
warnings such as negative or non-monotonic uptake remain in the prompt.
These checks cannot
prove correct SHAP interpretation or eliminate unsupported chemical claims;
the limitations identify the interpretation as AI-generated and requiring
experimental validation. Under **Scientific interpretation**, the response is
displayed in three sections: **Predicted adsorption behavior**,
**Structure–property relationships**, and **Suggested next investigation**.
Section headings are bold green on light-blue backgrounds. The previous
Gemini attribution line and long reading guide are replaced by the single
concise **Scientific limitations** section.

Offline regression tests make no provider requests:

```bash
python -m pytest tests/test_gemini_summary.py tests/test_api.py
```

## Current state versus target

| Component | Current state | Phase 3 target |
|-----------|---------------|----------------|
| Interface | Four geometry inputs + CIF upload; scientific limitations in a collapsed section | Broader browser and deployment validation |
| Models | Five configured Phase 3 Model B joblib models; baseline frozen as phase3-model-b-v1 | Distribute versioned bundle and complete native deployment recipe |
| Predictions | Five pressures, non-scrolling table/chart/CSV download for geometry + CIF | Preserve these outputs |
| SHAP | Waterfall for the geometry + CIF row at a selected pressure | Explain the same Phase 3 model and feature row used for prediction |
| Scientific summary | Opt-in Gemini draft in React; disabled by default; validated rule-based facts retained internally | Configure approved provider access and review generated summaries |
| Deployment | Local prototype; Dockerfile is only a base image | Tested, reproducible deployment with separately supplied model artifacts |

The current [app](../src/app.py) is a **local Phase 3 demo**. It requires
trusted corrected model artifacts and an installed native MOFid executable.
The LLM summary and reproducible hosted deployment remain unfinished.

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

Phase 3 requires **geometry + linker + CIF/metal descriptors**, not just four
geometry values. Agree on the input experience before implementing it:

- The demo accepts a CIF and its four matching geometry values; prepared-feature CSV upload is not exposed.
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

Generation requires an explicit click. Changing the CIF, geometry, configured
model context, or SHAP pressure clears the summary; generate it again for the
new context. A generation error is shown explicitly and the button can be
clicked to retry. Predictions and SHAP remain independent of summary failures.
If the displayed SHAP explanation fails, summary generation is disabled. If an
endpoint explanation fails during generation, an explicit summary error is shown;
the selected-pressure waterfall and predictions remain available.

To review the wording:

1. Start the app using the quick-start instructions below.
2. Upload `hMOF-0.cif` from [sample CIFs](../notebook/sample/cif_files/).
   Enter its four geometry values from the matching row of the
   [sample table](../notebook/sample/mof_co2_geometry_10.csv).
   The table's uptake columns are reference data, not new app predictions.
3. Click **Extract descriptors and predict**, then **Generate scientific summary**.
4. Compare all five uptake numbers against the app table and verify the signed
   endpoint difference. Select 0.01 bar in the waterfall and generate the summary
   again to compare the low-pressure contributions, signs, and baseline. Repeat
   with 2.5 bar for the higher-pressure section. Check the simulated-data,
   separate-baseline, local-importance, and non-causality limitations.
5. Repeat for `hMOF-10` and `hMOF-100`, using each structure's own geometry.
   Change pressure and geometry to confirm the old summary disappears.

Focused validation (headless plotting):

```bash
MPLBACKEND=Agg python -m pytest tests/test_scientific_summary.py tests/test_interface.py -q
```

The three reference-row tests use real frozen models at all five pressures
and validate the endpoint comparison when the prepared Phase 3 dataset is
available. They validate numerical
consistency, not new CIF extraction or scientific approval. Native MOFid is
still required for uploads.

### Stage 2: real LLM (not implemented)

Proceed after Susan has reviewed the three examples and accepted the units,
wording, signs, limitations, and unchanged predictions. Agree on provider,
model, budget, credentials owner, and allowed data before adding any API call.
Keep the local mode explicitly labelled, separate from AI-generated text.
Mocked provider-response tests belong to this stage; Stage 1 tests mock local
summary failures, not a nonexistent LLM.

The team has selected the LLM summary for the hackathon delivery scope. See
the [task and ownership plan](../docs/hackathon_tasks_and_ownership.md) for its
scientific contract, acceptance tests, owners, and deadline. The core
prediction/SHAP workflow must remain independent of LLM availability.

Choose a provider together before implementation. Decide on local versus
hosted execution, costs, credentials, and what information may leave the
computer.

The summary should state the predicted uptake by pressure, describe the
strongest SHAP contributions, and acknowledge that the model was evaluated
on simulated hMOF data. It must not invent mechanisms, citations, or
experimental validation. Keep predictions and SHAP usable if the summary
service is unavailable; show any service failure explicitly.

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

1. Enter LCD and PLD in angstrom, void fraction between 0 and 1, and surface
   area in m2/g. These values must describe the uploaded structure.
2. Upload a CIF and click **Extract descriptors and predict**.
3. Review predictions, download them, and choose a pressure for SHAP.

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
The real-CIF app test requires `MOF_REFERENCE_CIF_DIR` and `MOF_SBU_PATH`.
Next milestones are broader uploaded-structure validation, reproducible
deployment, and the optional LLM summary.
