# MOF CO2 Uptake Prediction

## Part 1: Research and model selection

This project uses XGBoost to predict simulated CO2 uptake in hypothetical
metal-organic frameworks (hMOFs) at five pressures: **0.01, 0.05, 0.1, 0.5,
and 2.5 bar**, with uptake expressed in **mol/kg**.

The research workflow compares geometry-only features with progressively
richer linker and crystallographic descriptors. The selected model will
support an interactive web demo with predictions and SHAP explanations.

## Research phases

| Phase | Inputs | Purpose |
|-------|--------|---------|
| **1: Geometry baseline** | LCD, PLD, void fraction, and surface area | Establish baseline performance and examine structure-property relationships |
| **2: Linker chemistry** | Geometry plus numeric linker descriptors | Evaluate the contribution of linker chemistry |
| **3: Controlled comparison** | Model A: geometry + linkers; Model B: geometry + linkers + CIF/metal descriptors | Compare the two feature sets using identical rows and train/validation/test splits |

A separate XGBoost regressor is trained for each pressure. The training
script uses a reproducible 60%/20%/20% train/validation/test split and reports
MAE, RMSE, and R2. Identifiers and target columns are excluded from predictors;
Phase 3 descriptors are also excluded from Model A's linker feature set.

### Selected model: Phase 3 Model B

**Frozen baseline:** `phase3-model-b-v1`. The
[versioned manifest](../docs/PHASE3/phase3-model-b-v1.yaml) records the exact
five artifacts, ordered 48-feature schema, dataset checksum, recomputed
metrics, and tested environment. The complete versioned bundle is committed
under `results/frozen_models/phase3-model-b-v1/`. Never overwrite this
version; future retraining must produce a new version.

**How to find and download the final model from GitHub**

1. Open the project repository on GitHub and select **Code** > **Download ZIP**
   (or clone the repository).
2. In the downloaded project, open
   `results/frozen_models/phase3-model-b-v1/`. This is the frozen final
   **Phase 3 Model B** bundle.
3. The bundle contains five `.joblib` files, one for each pressure
   (0.01, 0.05, 0.1, 0.5, and 2.5 bar), along with `manifest.yaml`,
   `pre_freeze_environment.txt`, and `validation.md`. Keep these files
   together; the manifest lists each model's SHA-256 checksum.
4. To run the app, copy the five `.joblib` files into
   `results/models/`. That folder already contains app-ready copies after
   cloning or unzipping the repository, so no copy is needed in that case.
5. Do not use older experiment models from the local archive. They are not
   part of the frozen final bundle.

Check the model checksums against the
[Phase 3 Model B manifest](../docs/PHASE3/phase3-model-b-v1.yaml) before
loading. Older experiment outputs are archived locally in
`docs/archive/results_archive/` and are not needed by collaborators.

We selected **Phase 3 Model B (geometry + linkers + CIF descriptors)** for
the web demo because it achieved higher test R2 than Model A across all five
pressures in the evaluated controlled comparison.

Its additional inputs include metal composition, unit-cell volume, density,
site count, and lattice parameters. These features must be supplied alongside
geometry and linker descriptors; the four Phase 1 geometry inputs alone are
not sufficient.

This selection reflects performance on the evaluated simulated dataset. It
does not establish experimental accuracy or generalization to unseen MOF
families. SHAP explains model contributions, not proven chemical mechanisms.

**Implementation status:** the Streamlit app now loads Phase 3 Model B.
Users supply four geometry values and a CIF; local MOFid, RDKit, and pymatgen
assemble the remaining descriptors for predictions and SHAP.

## Data for collaborators

The prepared datasets are shared separately from GitHub. To find and set up
the data:

1. Open the [data_mof_demo OneDrive folder](https://1drv.ms/f/c/ab0bbacec1d386eb/IgDJcymaZGwpR7fG_rqk97GXAcGskJ7--PZfOQ6_GWqbcFA?e=orssqL).
   Sign in with an account that has access; if access is denied, ask the
   project owner to share the folder with you.
2. Open the `data/processed/` folder in the shared files. Download the dataset
   you need, or download the folder if you need several datasets.
3. In your local project copy, place the downloaded files under
   `data/processed/`, keeping the exact filenames shown in the table below.
   Create the folder if it does not already exist.
4. For Phase 3 training, use
   `hmof_phase3_structure_descriptors.csv`. You do not need to download the
   dataset just to run the app with the already-provided final models.
5. Do not commit or push downloaded datasets; they are excluded from Git.

The prepared files are maintained manually by the project owner. The final
five Phase 3 Model B files are already included in GitHub under
`results/models/`. The `phase1_xgboost_*.joblib` files are retained only as
baseline artifacts and are not used by the current app.

| File | Use |
|------|-----|
| `data_clean_v2` (CSV without an extension) | Phase 1 baseline |
| `hmof_linker_extended_descriptors.csv` | Phase 2 training and Phase 3D linker comparison |
| `hmof_phase3_structure_descriptors.csv` | Phase 3 training and the selected Model B |
| `hmof_phase3_metal_descriptors.csv` | Intermediate metal-descriptor dataset, retained for reference |
| `hmof_linker_descriptors.pkl` | Used by the retained Phase 2 modeling notebook |

Datasets are excluded from Git and must be obtained separately; the final
trained Phase 3 models are committed in the repository. The datasets
originate from the hMOF/mofdb workflow described in the
[original project README](../docs/archive/README_original.md).

## Repository structure

```text
mof-co2-prediction-demo/
├── configs/config.yaml       # Configurable data and output paths
├── data/processed/           # Prepared datasets (not committed)
├── docs/
│   ├── archive/README_original.md
│   └── ...                   # Research and descriptor-method notes
├── Readme/
│   ├── Readme_model.md
│   └── readme_webinterface.md
├── notebook/
│   ├── EDA/
│   ├── phase 1/              # Baseline modeling and SHAP
│   ├── phase_2/              # Linker modeling and preprocessing
│   └── phase_3/              # Controlled Phase 3D comparison
├── src/                      # Training, paths, evaluation, and Streamlit app
├── mof_models.py             # Reusable model utilities
├── results/                  # Models and evaluation outputs (not committed)
├── tests/
└── requirements.txt
```

Configure data and output locations in [configs/config.yaml](../configs/config.yaml).
[src/paths.py](../src/paths.py) resolves those paths and supports environment
overrides, so collaborators need not edit machine-specific paths in code.

## Research entry points

For notebooks and tests, install
[requirements-dev.txt](../requirements-dev.txt) from the repository root:

```bash
python -m pip install -r requirements-dev.txt
```

This includes the app's [runtime requirements](../requirements.txt), plus
notebook and testing tools. The locally validated runtime uses Python 3.12.3.
Native MOFid is installed separately for geometry-plus-CIF inference; these
dependency declarations are not yet a full frozen environment.

- [Phase 1 modeling](../notebook/phase%201/modeling/modeling_xgboost_all_pressures.ipynb)
  and [SHAP analysis](../notebook/phase%201/shap/shap_all_pressures.ipynb).
- [Phase 2 modeling](../notebook/phase_2/geometry_linker_modeling.ipynb).
- [Phase 3 controlled comparison notebook](../notebook/phase_3/phase3D_xgboost_controlled_comparison.ipynb).
- [Training script](../src/train.py): run `python src/train.py --phase phase3`
  from the repository root in an environment with the project dependencies.
  This writes models and evaluation results to the configured output folders.

## Continue to Part 2

For the interface workflow, Phase 3 integration requirements, and deployment
milestones, go to **[Part 2: Web interface pipeline](readme_webinterface.md)**.
