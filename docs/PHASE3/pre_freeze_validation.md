# Phase 3 Model B pre-freeze validation

Validation date: 2026-10-10. Code evaluated: `25b3327`.

This records validation of the existing corrected Model B artifacts, not a
formal model freeze or hosted-deployment certification. No retraining or
dataset changes were performed.

## Results

| Check | Result |
|-------|--------|
| Clean Python 3.12.3 environment installed from requirements-dev.txt | Passed |
| Package consistency (`python -m pip check`) | Passed |
| Focused tests in existing environment | 20 passed, no skips |
| Same focused tests in clean environment | 20 passed, no skips |
| Five known CIFs: all 48 features versus reference rows | Passed at rtol=1e-8, atol=1e-8 |
| 25,928 dataset rows at all five pressures, between environments | Maximum absolute prediction difference: 0.0 |
| SHAP additivity on hMOF-0 at all five pressures | Passed at rtol=1e-5, atol=1e-6 |
| Model schemas | Identical ordered 48-feature schemas; no oxidation fields |

Reference CIFs: hMOF-0, hMOF-27262, hMOF-10022, hMOF-10, and hMOF-100.
Pressures: 0.01, 0.05, 0.1, 0.5, and 2.5 bar; uptake in mol/kg.
Largest absolute feature error was 4.547473508864641e-13. Largest absolute
SHAP additivity error was 4.76837158203125e-7.

The focused tests were:

```bash
python -m pytest tests/test_phase3_features.py tests/test_train_features.py tests/test_interface.py -vv
```

Integration checks used `MOF_REFERENCE_DATA_DIR`, `MOF_REFERENCE_CIF_DIR`,
`MOF_REFERENCE_MODELS_DIR`, and `MOF_SBU_PATH` pointing to local references.
`OMP_NUM_THREADS=2` and `OPENBLAS_NUM_THREADS=2` bounded test resource use.
The app checks covered startup, empty CIF handling, temporary-file cleanup,
missing CSV features, real-CIF predictions, SHAP pressure selection, and
hiding stale results after geometry changes.

Each environment emitted three upstream SHAP plotting deprecation warnings.
These did not fail the tests. The retained notebooks and legacy tests were
not rerun as part of this focused validation.

## Tested Python environment

[pre_freeze_environment.txt](pre_freeze_environment.txt) records the full
resolved package versions from the clean environment, including development
tools. Some transitive packages were newer than in the original environment;
the prediction comparisons above still matched exactly.

For a separate Linux/Python 3.12.3 environment:

```bash
python -m pip install -r docs/PHASE3/pre_freeze_environment.txt
python -m pip check
```

This is a tested version snapshot, not a hash-locked, cross-platform lock.
Installation directly from the snapshot has not been separately rerun.
Native MOFid and trusted model artifacts must still be supplied separately.

## Artifact integrity

All filenames below are relative to `results/models/`.

| Model filename | SHA-256 |
|----------------|---------|
| phase_3_model_b_geometry_plus_linkers_plus_cif_0.01bar.joblib | 355c06a6282a66236409ad36c993b10496a6bd29d0c4de347513fc5ed2478464 |
| phase_3_model_b_geometry_plus_linkers_plus_cif_0.05bar.joblib | 20890006cc7d0e81660b2bf8df8650c29b3435764fb4cb8c241298705dc71812 |
| phase_3_model_b_geometry_plus_linkers_plus_cif_0.1bar.joblib | d42cba45be590a12aa852ee72a9072ce9651c98e58639f410f47624d2bd865d5 |
| phase_3_model_b_geometry_plus_linkers_plus_cif_0.5bar.joblib | ff640607cdb81780f5ea045f78f92f310021b7dcecf9c9813b75052ad2699d42 |
| phase_3_model_b_geometry_plus_linkers_plus_cif_2.5bar.joblib | 50300d95631690ecbfb3fa98ff537862ea9629917fd1f7c90b9745f6db27b0b0 |

Reference dataset: `data/processed/hmof_phase3_structure_descriptors.csv`.
SHA-256: `213bb407c1286bbeebb39df86c8c092a1b18d70c5410a724f32397400a84942c`.
At the time of this validation, models and datasets were distributed separately
and were not added to Git. The five final Phase 3 Model B app copies are now
committed under `results/models/`; prepared datasets and the full frozen bundle
remain outside Git.

## Native and browser limitations

- Native MOFid source revision:
  `36873683083c7bc62c1da2062df2833adc35b48a`.
- Bundled Open Babel version: 3.1.0. The native executable links to the
  locally built Open Babel library; copying only `sbu` is insufficient.
- The clean Python environment reused the existing native installation.
  A fresh native build and reproducible deployment recipe remain unvalidated.
- The local server health endpoint, browser startup, and geometry controls
  were checked. Browser upload/download review remains incomplete because
  the browser tool could not select Linux reference files from its Windows
  environment. Real-CIF upload behavior passed through Streamlit AppTest.
- Five known CIFs do not certify arbitrary uploaded structures or
  generalization to unseen MOF families.

The corrected model baseline passed the core pre-freeze checks. It was
subsequently frozen in the [phase3-model-b-v1 manifest](phase3-model-b-v1.yaml).
Deployment validation remains a separate next step.
