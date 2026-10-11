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

The corrected baseline is frozen as **phase3-model-b-v1**. Its
[manifest](docs/PHASE3/phase3-model-b-v1.yaml) records artifact checksums,
feature order, metrics, dataset identity, and the tested environment.
The [pre-freeze validation report](docs/PHASE3/pre_freeze_validation.md)
records the checks and remaining deployment limitations.

The [hackathon task and ownership plan](docs/hackathon_tasks_and_ownership.md)
targets a prepared setup by October 17, 2026, including an LLM summary,
scientific review, deployment, and presentation preparation.

The [original README](docs/archive/README_original.md) is preserved unchanged
for reference.
