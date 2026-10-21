# MOF CO2 Uptake Prediction

The project documentation is organized into two guides:

1. **[Research and model selection](Readme/Readme_model.md)**:
   project structure, Phases 1-3, prepared datasets, and selection of Phase 3
   Model B.
2. **[Web interface pipeline](Readme/readme_webinterface.md)**:
   inputs, model integration, predictions, SHAP, optional LLM summaries,
   testing, and deployment.

The app uses Phase 3 Model B with supplied geometry and CIF-derived
descriptors. See the web guide for local MOFid setup and usage.

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

This installs the tool only; automatic geometry extraction is not yet wired
into the app. Probe settings and agreement with training geometry still need
validation before using its outputs for validated predictions.

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
