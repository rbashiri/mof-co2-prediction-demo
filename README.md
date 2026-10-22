# PRISM — Pore Regression & Interpretability for Simulated MOFs

The project documentation is organized into two guides:

1. **[Research and model selection](Readme/Readme_model.md)**:
   project structure, Phases 1-3, prepared datasets, and selection of Phase 3
   Model B.
2. **[Web interface pipeline](Readme/readme_webinterface.md)**:
   inputs, model integration, predictions, SHAP, optional LLM summaries,
   testing, and deployment.

**PRISM** (React + FastAPI) uses Phase 3 Model B with an interactive **3Dmol.js
crystal structure viewer**, pore geometry extracted automatically by **Zeo++**,
and remaining chemical/crystallographic descriptors extracted by
**MOFid/RDKit** and **pymatgen** from a single uploaded CIF.
See the web guide for local MOFid setup and usage.

## Hackathon MVP: scope, tests and next steps

This application is a **minimum viable product (MVP)** for the hackathon.
The objective at this stage is to demonstrate a working end-to-end flow:
**CIF upload -> automatic geometry and chemical descriptors -> CO2 predictions
at five pressures -> SHAP explanations and CSV output**.
It is not a production-ready or experimentally validated prediction service.

### Tests completed

- All ten sample CIFs completed the automatic prediction workflow. These
  samples were also used to select the Zeo++ settings, so their results are
  development checks, not independent validation.
- Ten additional CIFs were preselected from a reconstructed test split,
  excluding the setting-selection samples. Nine completed, producing 45
  predictions. On those nine structures, mean absolute uptake error against
  simulated reference values decreased from 1.286 to 0.245 mol/kg with the
  updated settings; negative predictions decreased from 24/45 to 0/45.
  The original training invocation is unrecorded, so test-split membership
  was reconstructed from the frozen dataset and documented split procedure.
- The tenth additional CIF, `hMOF-33132`, failed because RDKit rejected a
  nitrogen valence in a MOFid-extracted linker. This failure is part of the
  test results, was not replaced with another CIF, and is excluded from the
  45-prediction metrics. The application reported the failure rather than
  fabricating missing features.
- The Docker image built successfully. Two sample CIF uploads inside the
  container passed automatic extraction, prediction, SHAP consistency and
  CSV checks. Invalid-input handling and frontend loading were also checked;
  see [Docker validation](docker/Docker.md#validation-and-remaining-deployment-work).

### Planned improvements after the MVP

Support for unfamiliar CIFs is not yet comprehensive: MOFid may not identify
usable linkers, and RDKit may reject extracted fragments. Future work will
investigate these cases, improve user-facing diagnostics, and expand testing
across more structures without bypassing chemical validation.

Further work will also investigate the original geometry definitions,
especially PLD, and evaluate or retrain models using a consistent extraction
pipeline. The current results demonstrate application functionality and
initial comparisons on simulated data, not guaranteed accuracy for arbitrary
uploaded structures. Broader validation and deployment hardening remain
planned improvements.

## Hackathon: reproducible Zeo++ installation

On Linux with Python 3.12+, GNU Make and `g++` available, run:

```bash
python3 scripts/install_zeopp.py
```

The script downloads the official Zeo++ 0.3 source, checks its pinned SHA-256,
builds bundled Voro++ and Zeo++, and verifies that `network` runs. By default
it installs outside the repository at `~/zeopp-tools/zeo++-0.3`. It never
overwrites an existing directory; use `--destination /path/to/new/directory`
for a separate installation. Build/download failures exit with an explicit error.
The checksum pins the archive used in the hackathon setup, not an independently
published upstream signature. The source includes Zeo++ licensing information.

The application finds this default installation automatically. For another
location, set `ZEO_NETWORK_PATH` to the absolute path of the `network` binary.
MOFid is also required: set `MOF_SBU_PATH` to its native `bin/sbu` executable.
Then start the app from the repository root:

```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
# In another terminal:
cd frontend
npm ci
npm run dev
```

Upload one CIF and select **Extract descriptors and predict**. No manual geometry
is required. The app displays the four geometry values, five predictions, SHAP,
and the existing CSV download. `POST /api/predict` accepts a multipart `cif`
alone; existing clients may still supply all four manual geometry values.
Partial manual geometry is rejected.

Predictions do not wait for AI generation. After results appear, the bottom
**Click here to learn more about the results** button requests an optional
Gemini explanation separately, without repeating CIF extraction, predictions
or SHAP. The former upload-form Gemini checkbox is removed. Prediction results
remain available if explanation generation fails. Gemini must be configured
on the server; the optional explanation sends prediction and SHAP information
to Google, not the CIF file.
See the [web guide](Readme/readme_webinterface.md) for the explanation endpoint
and the current single-worker, 30-minute result-context lifetime.

Zeo++ runs in high-accuracy mode with the experimental
`experimental-sample-comparison-v1` profile:

- LCD and PLD: direct `-res` outputs in angstroms.
- Surface area: `ASA_m^2/g + NASA_m^2/g` from `-sa`, including inaccessible
  pockets, with a 1.65 angstrom channel/probe radius and 10,000 samples per atom.
- Void fraction: `POAV_Volume_fraction + PONAV_Volume_fraction` from `-volpo`,
  total probe-occupiable volume, with a separate 1.20 angstrom channel/probe
  radius and 50,000 samples.

This profile was selected after comparing radii 0, 1.20, 1.65 and 1.86
angstrom and accessible/total volume and surface definitions on all ten sample
CIFs. Mean absolute differences from the sample table changed from 0.418 to
0.035 for void fraction and 332 to 72 m2/g for surface area. LCD and PLD
differences remained 0.083 and 1.049 angstrom respectively. These are
sample-selection results, not independent validation or guarantees of improved
predictions on other structures. Monte Carlo results can vary.
These settings are not established as the original training-data settings;
the app reports that limitation, not a claim of prediction accuracy.
Failed extraction stops the request rather than substituting geometry.

Tests (no external AI requests):

```bash
python -m pytest tests/test_zeopp_geometry.py tests/test_api.py tests/test_phase3_features.py -q
# Also run a real Zeo++ sample:
ZEO_NETWORK_PATH="$HOME/zeopp-tools/zeo++-0.3/network" \
  python -m pytest tests/test_zeopp_geometry.py -q
# Full native upload-to-prediction checks (use your own MOFid path):
ZEO_NETWORK_PATH="$HOME/zeopp-tools/zeo++-0.3/network" \
  MOF_SBU_PATH="/absolute/path/to/mofid/bin/sbu" \
  python -m pytest tests/test_api.py -k real_cif_only_prediction -q
```

The legacy Streamlit app retains its manual-geometry workflow.

## Sample files

The [sample folder](notebook/sample/) includes a
[notebook](notebook/sample/sample.ipynb), a
[10-MOF geometry table](notebook/sample/mof_co2_geometry_10.csv), and the
corresponding [CIF structures](notebook/sample/cif_files/).

## Phase 3 model and notebook

The selected model is the frozen **phase3-model-b-v1** baseline. Its
[manifest](docs/PHASE3/phase3-model-b-v1.yaml) records the model files,
checksums, feature order, metrics, dataset identity, and tested environment.
The [Phase 3 validation report](docs/PHASE3/pre_freeze_validation.md) records
the checks and remaining deployment limitations, and the
[Phase 3 comparison notebook](notebook/phase_3/phase3D_xgboost_controlled_comparison.ipynb)
shows the model comparison.

**Where to get the model files:** the five final `.joblib` files are included
in GitHub under `results/models/` and are ready to use after cloning the
repository. Verify their SHA-256 values against the manifest before loading.
The complete frozen bundle, including its manifest, environment, and
validation records, is included in GitHub at
`results/frozen_models/phase3-model-b-v1/`. `results/models/` contains the
five app-ready copies. Old experiment outputs are kept locally in the ignored
`docs/archive/results_archive/` folder. The prepared datasets are shared
separately through
[data_mof_demo on OneDrive](https://1drv.ms/f/c/ab0bbacec1d386eb/IgDJcymaZGwpR7fG_rqk97GXAcGskJ7--PZfOQ6_GWqbcFA?e=orssqL).

The [hackathon task and ownership plan](docs/hackathon_tasks_and_ownership.md)
targets a prepared setup by October 17, 2026, including an LLM summary,
scientific review, deployment, and presentation preparation.

The [original README](docs/archive/README_original.md) is preserved unchanged
for reference.
