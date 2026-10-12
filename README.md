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

## Phase 3 model and notebook

The selected model is the frozen **phase3-model-b-v1** baseline. Its
[manifest](docs/PHASE3/phase3-model-b-v1.yaml) records the model files,
checksums, feature order, metrics, dataset identity, and tested environment.
The [Phase 3 validation report](docs/PHASE3/pre_freeze_validation.md) records
the checks and remaining deployment limitations, and the
[Phase 3 comparison notebook](notebook/phase_3/phase3D_xgboost_controlled_comparison.ipynb)
shows the model comparison.

**Where to get the model files:** the five `.joblib` files are not stored in
GitHub. Download them from the shared
[data_mof_demo OneDrive folder](https://1drv.ms/f/c/ab0bbacec1d386eb/IgDJcymaZGwpR7fG_rqk97GXAcGskJ7--PZfOQ6_GWqbcFA?e=orssqL)
and put them in `results/models/`, or set `MOF_MODELS_DIR` to their folder.
Verify their SHA-256 values against the manifest before loading. The frozen
local bundle is kept at `results/frozen_models/phase3-model-b-v1/`; it is
excluded from Git. `results/models/` contains only the five files needed by
the app; old experiment outputs are kept locally in the ignored
`docs/archive/results_archive/` folder. The online copy of this versioned
bundle has not yet been verified, so contact the owner if the model files are
not available in the shared folder.

The [hackathon task and ownership plan](docs/hackathon_tasks_and_ownership.md)
targets a prepared setup by October 17, 2026, including an LLM summary,
scientific review, deployment, and presentation preparation.

The [original README](docs/archive/README_original.md) is preserved unchanged
for reference.
