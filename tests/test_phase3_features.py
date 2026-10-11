import os
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.linker_descriptors import aggregate_linker_descriptors
from src.phase3_features import build_phase3_features, extract_cif_linkers
from src.train import GEOMETRY_COLUMNS


def test_linker_aggregation_preserves_fragment_multiplicity():
    smiles = "O=C([O-])c1ccc(C(=O)[O-])cc1"
    single = aggregate_linker_descriptors([smiles])
    double = aggregate_linker_descriptors([smiles, smiles])
    assert len(single) == 29
    assert single["total_C_count"] == 8
    assert single["total_O_count"] == 4
    assert single["total_carboxylate_or_carboxylic_acid_count"] == 2
    for name, value in single.items():
        expected = value if name.startswith(("mean_", "max_")) or name == "heteroatom_fraction" else 2 * value
        assert double[name] == pytest.approx(expected)


@pytest.mark.parametrize("smiles", [[], ["not-smiles"], ["*"]])
def test_invalid_linkers_fail_explicitly(smiles):
    with pytest.raises(ValueError):
        aggregate_linker_descriptors(smiles)


def test_invalid_geometry_fails_before_extraction():
    geometry = dict.fromkeys(GEOMETRY_COLUMNS, 1.0)
    geometry["void_fraction"] = 1.5
    with pytest.raises(ValueError, match="void_fraction"):
        build_phase3_features("does-not-exist.cif", geometry)


def test_native_extraction_applies_filters_and_cleans_output(tmp_path, monkeypatch):
    cif = tmp_path / "example.cif"
    cif.write_text("placeholder")
    output_directories = []

    def run(command, **kwargs):
        output_directories.append(Path(command[2]))
        assert output_directories[-1].exists()
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(
            command, 0,
            "# Nodes:\n[Zn]\n# Linkers:\nCCO\n# Found 1 simplified net(s)\n", "",
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert extract_cif_linkers(cif, sbu_path="configured-sbu") == ["CCO"]
    assert not output_directories[0].exists()


def test_unrecognized_mofid_output_fails(tmp_path, monkeypatch):
    cif = tmp_path / "example.cif"
    cif.touch()
    monkeypatch.setattr(
        subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "*", "failed"),
    )
    with pytest.raises(ValueError, match="unrecognized"):
        extract_cif_linkers(cif, sbu_path="configured-sbu")


@pytest.mark.parametrize("error", [
    subprocess.CalledProcessError(1, ["sbu"], stderr="invalid CIF"),
    subprocess.TimeoutExpired(["sbu"], 120),
])
def test_native_failures_are_not_hidden(tmp_path, monkeypatch, error):
    cif = tmp_path / "example.cif"
    cif.touch()

    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(type(error)):
        extract_cif_linkers(cif, sbu_path="configured-sbu")


@pytest.mark.parametrize("name", [
    "hMOF-0", "hMOF-27262", "hMOF-10022", "hMOF-10", "hMOF-100",
])
def test_known_cif_features_and_predictions(name):
    data_dir = os.environ.get("MOF_REFERENCE_DATA_DIR")
    cif_dir = os.environ.get("MOF_REFERENCE_CIF_DIR")
    models_dir = os.environ.get("MOF_REFERENCE_MODELS_DIR")
    if not all((data_dir, cif_dir, models_dir, os.environ.get("MOF_SBU_PATH"))):
        pytest.skip("Local reference datasets, CIFs, models, and MOF_SBU_PATH required.")
    assert data_dir is not None and cif_dir is not None and models_dir is not None
    reference = pd.read_csv(
        Path(data_dir) / "hmof_phase3_structure_descriptors.csv", low_memory=False,
    )
    row = reference.loc[reference.filename.eq(name)]
    assert len(row) == 1
    geometry = {column: float(row[column].iloc[0]) for column in GEOMETRY_COLUMNS}
    for pressure in ("0.01", "0.05", "0.1", "0.5", "2.5"):
        model = joblib.load(
            Path(models_dir) /
            f"phase_3_model_b_geometry_plus_linkers_plus_cif_{pressure}bar.joblib"
        )
        names = model.get_booster().feature_names
        assert names is not None
        if pressure == "0.01":
            features = build_phase3_features(
                Path(cif_dir) / f"{name}.cif", geometry, feature_names=names,
            )
            assert features.shape == (1, 48)
            np.testing.assert_allclose(
                features.to_numpy(), row[names].to_numpy(dtype=float),
                rtol=1e-8, atol=1e-8,
            )
        assert list(features.columns) == names
        np.testing.assert_allclose(
            model.predict(features), model.predict(row[names]), rtol=1e-6, atol=1e-6,
        )
