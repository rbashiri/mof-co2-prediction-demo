## Project Structure

```text
mof-co2-prediction-demo/
├── .venv/                                # Virtual environment, not committed
├── configs/
│   └── config.yaml
├── data/
│   ├── processed/                        # Cleaned and feature-engineered datasets
│   └── raw/                              # Original hMOF JSON/CIF files, not committed
├── docs/
│   ├── project_objectives.md
│   ├── SHAP_conceptual.md
│   ├── PHASE_2_*.md
│   ├── PHASE3/
│   └── linker_geometry_report.md
├── notebook/
│   ├── phase 1/                          # Geometry-only baseline modeling and SHAP
│   ├── phase_2/                          # Geometry + linker descriptor modeling
│   │   └── geometry_linker_modeling.ipynb
│   └── phase_3/                          # CIF matching, metal descriptors, controlled comparison
│       ├── cif_matching_metal_descriptors.ipynb
│       └── phase3D_xgboost_controlled_comparison.ipynb
├── src/
│   ├── app.py                            # App or future model interface
│   ├── evaluate.py                       # Model evaluation utilities
│   ├── linker_descriptors.py             # Linker chemistry descriptor 
│   ├── preprocess.py                     # Data cleaning and feature 
│   └── train.py                          # Phases 1 +  2 + 3 training
        ├── Phase 1: geometry-only baseline
        ├── Phase 2: geometry + linker descriptors
        └── Phase 3: geometry + linker + CIF descriptors
├── tests/
│   ├── test_interface.py
│   ├── test_model.py
│   └── test_preprocess.py
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```
The current project now includes Phase 2 RDKit-based linker chemistry descriptors and Phase 3 CIF-derived metal-node descriptors. The `src/` folder stores reusable Python scripts for preprocessing, linker extraction, descriptor generation, training, evaluation, and future deployment.

## Configure data and model paths

Training, the Streamlit app, and notebooks read their paths from
`configs/config.yaml`. By default, processed datasets are expected in
`data/processed/`, raw datasets in `data/raw/`, and models and training
results under `results/`. If those folders move, update the corresponding
entry under `paths` in that YAML file; the code does not need to change.
Use an absolute path for data stored outside the repository. You can also
override paths with environment variables such as `MOF_DATA_DIR`,
`MOF_RAW_DATA_DIR`, `MOF_PHASE1_DATA_PATH`, `MOF_RESULTS_ROOT`,
`MOF_MODELS_DIR`, and `MOF_RESULTS_DIR`.
====================================================
## Project Objectives
### Overview
This project predicts and analyzes CO2 adsorption behavior in Metal-Organic
Frameworks (MOFs) based on their physical and chemical properties, using
machine learning trained on simulated data and validated against real,
experimentally synthesized structures where possible.

###  Phase 1 — Structure-Property Understanding
Use the hMOF simulated dataset (~32,768+ hypothetical MOF structures) to
identify which physical properties most strongly influence CO2 adsorption:
- Largest cavity diameter (LCD)
- Pore limiting diameter (PLD)
- Void fraction
- Surface area (m2/g)

This replicates and validates the structure-property relationships reported
in Wilmer et al., *Energy Environ. Sci.*, 2012 (DOI: 10.1039/C2EE23201D).


======================================================
## project Dataset

**Source:** hMOF (hypothetical Metal-Organic Framework) database,
originating from Wilmer et al., *Energy Environ. Sci.*, 2012
(DOI: 10.1039/C2EE23201D), accessed via the mofdb bulk download.

**File type used:** `.json` (not `.cif`)

**Why JSON over CIF:**
- The `.cif` files contain only raw crystallographic data (unit cell
  parameters and atomic coordinates) — using them would require running
  separate pore-analysis software (e.g. Zeo++) to compute structural
  descriptors from scratch.
- The `.json` files already include the pre-computed structural properties
  (`lcd`, `pld`, `void_fraction`, `surface_area_m2g`) **and** simulated
  gas adsorption isotherms (CO2, N2, CH4) at multiple pressures — giving
  both the input features (X) and the prediction target (Y) needed for
  this project without extra preprocessing.
- `.cif` files are retained on external storage in case future work
  requires custom structural feature extraction beyond what's in the JSON.

**Fields extracted per MOF:**
| Field | Description |
|---|---|
| `lcd` | Largest cavity diameter (Å) |
| `pld` | Pore limiting diameter (Å) |
| `void_fraction` | Fraction of accessible void volume |
| `surface_area_m2g` | Gravimetric surface area (m²/g) |
| `elements` | Constituent atoms in the framework |
| `isotherms` | CO2 (and N2/CH4) adsorption uptake (mol/kg) at multiple pressures, 298 K |

**Dataset size:** 

The CO₂-specific hMOF dataset used for the baseline analysis contains:

- **32,768 JSON files** containing structural descriptors and CO₂ adsorption data
- **32,767 CIF files** containing crystallographic structures
- **65,535 JSON/CIF files in total**

======================================================
## CO2 Uptake Columns

The dataset includes CO2 adsorption values measured at five different
gas pressures, from the same simulated MOF structure — not five
different properties.

| Column | Meaning |
|---|---|
| `CO2_uptake_0.01bar_molkg` | CO2 adsorbed at very low pressure (0.01 bar) |
| `CO2_uptake_0.05bar_molkg` | CO2 adsorbed at 0.05 bar |
| `CO2_uptake_0.1bar_molkg` | CO2 adsorbed at 0.1 bar |
| `CO2_uptake_0.5bar_molkg` | CO2 adsorbed at 0.5 bar |
| `CO2_uptake_2.5bar_molkg` | CO2 adsorbed at 2.5 bar (highest pressure tested) |

**Why pressure matters:** at low pressure, only MOFs with strong CO2-binding sites (e.g. open metal sites, polar functional groups) adsorb meaningfully weaker materials adsorb almost nothing. At high pressure, CO2 is pushed into any available pore space, so uptake becomes driven more by physical capacity (surface area, pore volume) than by binding strength.

This mirrors why Wilmer et al. (2012) evaluated four separate industrial
cases at different pressures: flue gas capture occurs at low CO2 partial
pressure (~0.1 bar), while natural gas purification occurs at higher
pressure — so the "best" MOF depends on which pressure regime matters
for the target application.

**Implication for feature importance:** the most influential structural or
chemical property can differ depending on which pressure column is analyzed.
For example, void fraction may matter more at 2.5 bar (physical space
dominates), while chemistry-based features (e.g. fluorine content, once
added via RDKit) may matter more at 0.01 bar (binding strength dominates).
Correlation and feature-importance analysis should therefore be checked
across all five pressure columns, not just one.

======================================================
## Data Cleaning Summary

The team works from the already cleaned dataset `data_clean_v2` (column names standardized, duplicates and zero-surface-area rows without a mofid removed). Cleaning is not repeated in this repository. hMOF entries with a null mofid were kept.

## Feature and Target Selection

The project uses structural properties to predict CO₂ uptake.
The variables are assigned the following roles:

| Role | Variables | Purpose |
|------|-----------|---------|
| **Features (`X`)** | `lcd`, `pld`, `void_fraction`, `surface_area_m2g` | Describe the pore structure and physical properties of each MOF |
| **Targets (`y`)** | Five CO₂-uptake columns | Measure CO₂ adsorption at different pressures |
| **Identifiers** | `filename`, `mofid` | Identify and trace individual records |

> `filename` and `mofid` are excluded from model features — they are
> identifiers, not physical MOF properties.  
> Missing `mofid` values do not affect the current analysis because all
> required feature and target values are complete.

---

### Recommended Modeling Approach: Separate Regression by Pressure

A separate regression model will be trained for each CO₂-pressure target:

| Model | Target |
|-------|--------|
| Model 1 | `CO2_uptake_0.01bar_molkg` |
| Model 2 | `CO2_uptake_0.05bar_molkg` |
| Model 3 | `CO2_uptake_0.1bar_molkg` |
| Model 4 | `CO2_uptake_0.5bar_molkg` |
| Model 5 | `CO2_uptake_2.5bar_molkg` |

This approach makes it possible to compare how the influence of `lcd`,
`pld`, `void_fraction`, and `surface_area_m2g` changes across pressure
regimes:

- At **low pressure** (0.01 – 0.1 bar), CO₂ adsorption is more sensitive
  to pore accessibility and chemical interactions — relevant to **flue gas
  capture** (~0.1 bar CO₂ partial pressure).
- At **high pressure** (0.5 – 2.5 bar), physical capacity dominates —
  void fraction and surface area become more influential — relevant to
  **natural gas purification**.

XGBoost is the currently selected regression model for these targets.
The reusable evaluation workflow uses the same model settings for each
pressure; baseline estimators remain available for comparison.

> ⚠️ These relationships will be tested using **correlation analysis** and
> **model-based feature importance** rather than assumed in advance.
> Feature importance should be checked across all five pressure targets,
> not just one, consistent with the approach in Wilmer et al. (2012).
