import os
import subprocess
from pathlib import Path

import pytest

from src.zeopp_geometry import extract_geometry, resolve_zeopp_path


@pytest.fixture
def cif(tmp_path):
    path = tmp_path / "input with spaces.cif"
    path.write_text("data_test\n")
    return path


def mock_network(monkeypatch, *, res="structure.cif 11.76 7.89 11.75",
                 area="ASA_m^2/g: 3543.89 NASA_m^2/g: 0",
                 volume="POAV_Volume_fraction: 0.75 PONAV_Volume_fraction: 0"):
    directories = []
    monkeypatch.setattr("src.zeopp_geometry.resolve_zeopp_path", lambda _: "/zeopp/network")

    def run(command, **kwargs):
        work = kwargs["cwd"]
        directories.append(work)
        assert (work / "structure.cif").read_text() == "data_test\n"
        assert kwargs["check"] and kwargs["timeout"] == 120
        assert "-ha" in command
        assert command[command.index("-sa") + 1:command.index("-sa") + 4] == [
            "1.65", "1.65", "10000",
        ]
        assert "-vol" not in command
        assert command[command.index("-volpo") + 1:command.index("-volpo") + 4] == [
            "1.2", "1.2", "50000",
        ]
        for name, content in (("geometry.res", res), ("area.sa", area), ("volume.vol", volume)):
            if content is not None:
                (work / name).write_text(content)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("src.zeopp_geometry.subprocess.run", run)
    return directories


def test_extracts_correct_fields_and_cleans_workdir(cif, monkeypatch):
    directories = mock_network(
        monkeypatch, area="NASA_m^2/g: 999 ASA_m^2/g: 3543.89",
        volume="PONAV_Volume_fraction: 0.1 POAV_Volume_fraction: 0.65 AV_Volume_fraction: 0.01",
    )
    assert extract_geometry(cif) == pytest.approx({
        "lcd": 11.76, "pld": 7.89, "void_fraction": 0.75,
        "surface_area_m2g": 4542.89,
    })
    assert not directories[0].exists()


def test_zero_total_porosity_is_valid(cif, monkeypatch):
    mock_network(
        monkeypatch, res="structure.cif 2 0 0",
        area="ASA_m^2/g: 0 NASA_m^2/g: 0",
        volume="POAV_Volume_fraction: 0 PONAV_Volume_fraction: 0",
    )
    assert extract_geometry(cif) == {
        "lcd": 2.0, "pld": 0.0, "void_fraction": 0.0, "surface_area_m2g": 0.0,
    }


@pytest.mark.parametrize("outputs", [
    {"res": "malformed"},
    {"res": "structure.cif nan 1 1"},
    {"res": "structure.cif 1 2 1"},
    {"volume": "POAV_Volume_fraction: 0.9 PONAV_Volume_fraction: 0.2"},
    {"volume": "POAV_Volume_fraction: -0.1 PONAV_Volume_fraction: 0.5"},
    {"volume": "AV_Volume_fraction: 0.5 NAV_Volume_fraction: 0"},
    {"volume": "POAV_Volume_fraction: 0.5"},
    {"area": "NASA_m^2/g: 500"},
    {"area": "ASA_m^2/g: 500"},
    {"area": "ASA_m^2/g: inf NASA_m^2/g: 0"},
    {"area": "ASA_m^2/g: -1 NASA_m^2/g: 100"},
    {"area": "ASA_m^2/g: 1 ASA_m^2/g: 2 NASA_m^2/g: 0"},
    {"area": None},
])
def test_bad_output_is_not_replaced_with_defaults(cif, monkeypatch, outputs):
    directories = mock_network(monkeypatch, **outputs)
    with pytest.raises(ValueError):
        extract_geometry(cif)
    assert not directories[0].exists()


@pytest.mark.parametrize("error, message", [
    (subprocess.CalledProcessError(1, ["network"], stderr="Invalid CIF"), "Invalid CIF"),
    (subprocess.TimeoutExpired(["network"], 120), "timed out"),
])
def test_process_errors_are_explicit_and_cleaned(cif, monkeypatch, error, message):
    directories = []
    monkeypatch.setattr("src.zeopp_geometry.resolve_zeopp_path", lambda _: "/zeopp/network")

    def run(*args, **kwargs):
        directories.append(kwargs["cwd"])
        raise error

    monkeypatch.setattr("src.zeopp_geometry.subprocess.run", run)
    with pytest.raises(ValueError, match=message):
        extract_geometry(cif)
    assert not directories[0].exists()


def test_missing_executable_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="ZEO_NETWORK_PATH"):
        resolve_zeopp_path(tmp_path / "missing-network")


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout(cif, timeout):
    with pytest.raises(ValueError, match="timeout"):
        extract_geometry(cif, timeout=timeout)


def test_real_zeopp_sample():
    executable = os.environ.get("ZEO_NETWORK_PATH")
    if not executable:
        pytest.skip("Set ZEO_NETWORK_PATH to run real Zeo++ extraction.")
    geometry = extract_geometry(
        Path(__file__).resolve().parents[1] / "notebook/sample/cif_files/hMOF-0.cif",
        executable=executable,
    )
    assert set(geometry) == {"lcd", "pld", "void_fraction", "surface_area_m2g"}
    assert geometry["lcd"] >= geometry["pld"] > 0
    assert 0 < geometry["void_fraction"] < 1
    assert geometry["surface_area_m2g"] > 0
