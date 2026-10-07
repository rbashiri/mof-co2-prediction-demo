# Phase 3 — Metal-Node and Structure-Derived Descriptors

> **Question:** Can descriptors extracted directly from MOF structures improve CO₂ uptake prediction beyond the Phase 2 geometry + linker model?

## 3A — Match Structures to the Modeling Dataset

- [x] Locate the CIF corresponding to each Phase 2 MOF record.
- [x] Verify CIF–dataset matches using filename and available identifiers.
- [x] Record missing, unreadable, and unmatched CIFs.
- [x] Define the same matched MOF subset for baseline and new models.

## 3B — Extract Metal-Node Descriptors

- [x] Identify metal elements present in each structure (for example, Zn, Cu, Co, Ni).
- [x]  Define and record a reproducible metal-count measure; distinguish atom count from number of metal types.
- [x] Add selected atomic properties, including atomic number and atomic mass.
- [x] Record the number of distinct metal elements per MOF.

`Conclusion:` Phase 3B basic metal-node descriptors were completed, including metal identity, metal counts, metal fraction, average metal atomic number, and average metal atomic mass. Oxidation state and possible open metal site descriptors remain investigation items because they require a separate reliability check.

## Remaining Phase 3B Investigation Items

The following descriptors were not added to the Phase 3B dataset yet because they require additional reliability checks:

| Descriptor | Reason for delay | Next action |
|---|---|---|
| `estimated_oxidation_state` | Oxidation states are not always reliable from CIF alone | Test pymatgen oxidation-state assignment on sample CIFs |
| `possible_open_metal_site` | Open metal sites require coordination-environment analysis | Investigate pymatgen/ChemEnv or MOF-specific methods |
Conclusion: These two descriptors are kept as Phase 3B extension tasks and should be revisited after Phase 3C or before final Phase 3 modeling.

### 3B Extension — Oxidation State and Open Metal Site Investigation
- [ x] Investigate oxidation state assignment using pymatgen or another reliable method.
- [x ] Record oxidation-state success/failure for each CIF.
- [x ] Investigate possible open metal site indicators from coordination environment.
- [x ] Check extracted oxidation-state and open-metal-site values against a manual sample of CIFs.
- [ x] Include oxidation state or open metal site descriptors only if the method is reliable and documented.


## 3C — Extract Additional Pore Descriptors

### 3C-1 — pymatgen Structure Descriptors

- [x] Select structure-derived descriptors that add information beyond LCD, PLD, void fraction, and surface area.
- [x] Calculate selected descriptors from CIFs using pymatgen.
- [x] Check units and descriptor meanings.
- [x] Extract descriptors consistently for the matched subset.

`Conclusion:` Phase 3C-1 was completed using pymatgen. Unit cell volume, density, number of sites, and lattice parameters were extracted from matched CIF structures.
### 3C-2 — Zeo++ Pore Descriptor Pipeline for Future CIF Uploads

- [ ] Use Zeo++ later in the AI agent/backend pipeline to calculate `lcd`, `pld`, `void_fraction`, and `surface_area_m2g` from newly uploaded CIF files.
- [ ] Match Zeo++ output format to the training dataset feature columns.
- [ ] Use Zeo++ only when predicting from new CIF files that do not already have precomputed geometry descriptors.

`Status:` Deferred. The current hMOF dataset already includes `lcd`, `pld`, `void_fraction`, and `surface_area_m2g`, so Zeo++ is not required for the current modeling dataset.

## 3D — Controlled XGBoost Comparison

- [x] Load the Phase 3 final descriptor dataset.
- [x] Define the matched CIF subset used for both models.
- [x] Build Model A: Phase 2 baseline using geometry + linker descriptors.
- [x] Build Model B: Phase 3 model using geometry + linker descriptors + Phase 3 descriptors.
- [x] Use identical rows for both models.
- [x] Use identical train/validation/test splits for both models.
- [x] Use identical XGBoost model settings for both models.
- [x] Compare MAE, RMSE, and R² across all five CO₂ pressures.
- [x] Quantify improvement from Phase 3 descriptors.
- [x] Summarize whether Phase 3 descriptors add predictive value.

## 3E — Interpretation and Literature

- [x] Examine the importance of metal and new pore descriptors across pressures.
- [x] Compare metal, linker, and geometry feature groups carefully.
- [x] Investigate whether important descriptors have a scientifically plausible relationship to adsorption.
- [ ] Compare the findings with published MOF/CO₂ adsorption literature after metal testing.
- [ ] Document uncertainties, including unreliable oxidation-state or open-metal-site assignments.