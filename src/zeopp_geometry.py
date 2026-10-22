"""Extract geometry with Zeo++ 0.3; lengths are in angstroms."""

import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

SURFACE_PROBE_RADIUS = 1.65
VOLUME_PROBE_RADIUS = 1.2
SURFACE_SAMPLES = 10000
VOLUME_SAMPLES = 50000
GEOMETRY_WARNING = (
    "Experimental Zeo++ profile selected by comparison with ten sample structures; "
    "the original training settings remain unknown. "
    "Predictions using this geometry are not validated against that pipeline."
)


def resolve_zeopp_path(executable: str | Path | None = None) -> str:
    candidate = (
        executable
        or os.environ.get("ZEO_NETWORK_PATH")
        or shutil.which("network")
        or Path.home() / "zeopp-tools" / "zeo++-0.3" / "network"
    )
    path = Path(candidate).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(
            f"Zeo++ network executable not found or not executable: {path}. "
            "Run python3 scripts/install_zeopp.py or set ZEO_NETWORK_PATH."
        )
    return str(path.resolve())


def _field(text: str, label: str) -> float:
    matches = re.findall(rf"(?<!\S){re.escape(label)}:\s*(\S+)", text)
    if len(matches) != 1:
        raise ValueError(f"Zeo++ output must contain exactly one {label} value.")
    value = float(matches[0])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Zeo++ {label} must be finite and non-negative.")
    return value


def extract_geometry(
    cif_path: str | Path,
    *,
    executable: str | Path | None = None,
    timeout: float = 120,
) -> dict[str, float]:
    """Return LCD, PLD, total probe-occupiable fraction and total m2/g area."""
    cif = Path(cif_path).resolve()
    if not cif.is_file() or cif.stat().st_size == 0:
        raise ValueError("Zeo++ requires a non-empty CIF file.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Zeo++ timeout must be finite and positive.")
    network = resolve_zeopp_path(executable)
    with tempfile.TemporaryDirectory(prefix="mof-zeopp-") as directory:
        work = Path(directory)
        # A fixed local name keeps uploaded filenames out of commands and outputs.
        local_cif = work / "structure.cif"
        shutil.copyfile(cif, local_cif)
        res, area, volume = (work / name for name in ("geometry.res", "area.sa", "volume.vol"))
        surface_radius = str(SURFACE_PROBE_RADIUS)
        volume_radius = str(VOLUME_PROBE_RADIUS)
        command = [
            network, "-ha", "-res", str(res),
            "-sa", surface_radius, surface_radius, str(SURFACE_SAMPLES), str(area),
            "-volpo", volume_radius, volume_radius, str(VOLUME_SAMPLES), str(volume),
            str(local_cif),
        ]
        try:
            completed = subprocess.run(
                command, cwd=work, capture_output=True, text=True,
                check=True, timeout=timeout,
            )
        except subprocess.CalledProcessError as exc:
            diagnostics = (exc.stderr or exc.stdout or "No diagnostics")[-2000:]
            raise ValueError(f"Zeo++ extraction failed: {diagnostics}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"Zeo++ extraction timed out after {timeout} seconds.") from exc
        if not all(path.is_file() for path in (res, area, volume)):
            diagnostics = (completed.stderr or completed.stdout)[-2000:]
            raise ValueError(f"Zeo++ did not produce all geometry output files: {diagnostics}")
        fields = res.read_text().split()
        if len(fields) != 4:
            raise ValueError("Unrecognized Zeo++ pore diameter output.")
        geometry = {
            "lcd": float(fields[1]),
            "pld": float(fields[2]),
            "void_fraction": (
                _field(volume.read_text(), "POAV_Volume_fraction")
                + _field(volume.read_text(), "PONAV_Volume_fraction")
            ),
            "surface_area_m2g": (
                _field(area.read_text(), "ASA_m^2/g") + _field(area.read_text(), "NASA_m^2/g")
            ),
        }
        if any(not math.isfinite(value) or value < 0 for value in geometry.values()):
            raise ValueError("Zeo++ produced non-finite or negative geometry.")
        if geometry["void_fraction"] > 1:
            raise ValueError("Zeo++ void fraction is outside 0-1.")
        if geometry["pld"] > geometry["lcd"]:
            raise ValueError("Zeo++ PLD exceeds LCD.")
        return geometry
