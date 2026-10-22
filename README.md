# MOF CO2 Uptake Prediction

The project documentation is organized into two guides:

1. **[Research and model selection](Readme/Readme_model.md)**:
   project structure, Phases 1-3, prepared datasets, and selection of Phase 3
   Model B.
2. **[Web interface pipeline](Readme/readme_webinterface.md)**:
   inputs, model integration, predictions, SHAP, optional LLM summaries,
   testing, and deployment.

The Streamlit app uses Phase 3 Model B with geometry extracted by Zeo++ and
the remaining descriptors extracted by MOFid/RDKit and pymatgen from one CIF.
See the web guide for setup and usage.

## Hackathon MVP: scope, validation, and limitations

This application is a minimum viable product demonstrating:

**CIF upload -> automatic geometry and chemical descriptors -> CO2
predictions at five pressures -> SHAP and CSV output -> optional Gemini
explanation.**

It is not a production-ready or experimentally validated prediction service.
All ten sample CIFs completed the automatic workflow, but those samples were
also used to select the Zeo++ settings and are development checks rather than
independent validation.

Ten additional CIFs were preselected from a reconstructed test split. Nine
completed, producing 45 predictions. On those nine, mean absolute uptake error
against simulated reference values changed from 1.286 to 0.245 mol/kg with the
updated settings, and negative predictions changed from 24/45 to 0/45. The
original training invocation is unrecorded, so test-split membership was
reconstructed from the frozen dataset and documented split procedure.
`hMOF-33132` was retained as a failure: RDKit rejected a nitrogen valence in a
MOFid-extracted linker. It is excluded from the 45-prediction metrics. The app
reports extraction failures instead of fabricating missing features.

The Zeo++ profile is experimental. The original model-training geometry
settings remain unknown, especially for PLD. The current comparisons establish
application functionality on simulated data, not accuracy for arbitrary MOFs.
Unsupported MOFid fragmentation and RDKit valence cases remain known limits.

## Install and run

Install Python dependencies and native MOFid, then install Zeo++ 0.3 with
Python 3.12+, GNU Make, and `g++`:

```bash
python -m pip install -r requirements.txt
python scripts/install_zeopp.py
export MOF_SBU_PATH=/absolute/path/to/mofid/bin/sbu
python -m streamlit run src/app.py
```

The app finds the default Zeo++ installation under
`~/zeopp-tools/zeo++-0.3/network`. Set `ZEO_NETWORK_PATH` for another location.
The checksum-pinned installer never overwrites an existing destination.

Zeo++ uses the `experimental-sample-comparison-v1` profile:

- LCD and PLD come directly from `-res`.
- Surface area is total `ASA_m^2/g + NASA_m^2/g`, using a 1.65 angstrom
  channel/probe radius and 10,000 samples per atom.
- Void fraction is total
  `POAV_Volume_fraction + PONAV_Volume_fraction`, using a 1.20 angstrom
  channel/probe radius and 50,000 samples.
- High-accuracy mode is enabled. Monte Carlo results can vary.

Predictions do not wait for AI generation. The local rule-based scientific
summary runs first and remains available without external calls. A separate
Gemini button appears after that summary. To enable it:

```bash
export GEMINI_ENABLED=true
export GEMINI_API_KEY=your-key
export GEMINI_MODEL=gemini-3.5-flash
```

Only prediction values and validated SHAP summary text are sent to Google when
the user explicitly clicks the Gemini button; the CIF is not sent. Prediction
and SHAP results remain available if Gemini fails.

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
