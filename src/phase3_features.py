"""Assemble Phase 3 inputs from supplied geometry and a local CIF."""

import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from .extract_linkers import extract_linkers
from .linker_descriptors import aggregate_linker_descriptors
from .phase3_extract_metal_descriptors import extract_phase3_descriptors
from .train import GEOMETRY_COLUMNS, PHASE3_DESCRIPTOR_COLUMNS


def extract_cif_linkers(
    cif_path: str | Path,
    *,
    sbu_path: str | Path | None = None,
    timeout: float = 120,
) -> list[str]:
    """Run MOFid's native fragmentation and apply training-time RDKit filters."""
    cif = Path(cif_path).resolve()
    if not cif.is_file():
        raise FileNotFoundError(f"CIF file not found: {cif}")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Extraction timeout must be finite and positive.")
    executable = sbu_path or os.environ.get("MOF_SBU_PATH") or shutil.which("sbu")
    if not executable:
        raise FileNotFoundError(
            "MOFid sbu executable not found. Supply sbu_path or set MOF_SBU_PATH."
        )

    # Only chemical fragments are needed; topology classification and Java are not.
    with tempfile.TemporaryDirectory(prefix="mof-co2-fragments-") as output:
        completed = subprocess.run(
            [str(executable), str(cif), output],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if (
        not lines
        or lines[0] != "# Nodes:"
        or lines.count("# Linkers:") != 1
    ):
        raise ValueError(
            f"MOFid returned unrecognized fragment output: {completed.stdout!r}; "
            f"diagnostics: {completed.stderr}"
        )
    fragments = [line for line in lines[1:] if not line.startswith("#")]
    if not fragments or "*" in fragments:
        raise ValueError(f"MOFid did not extract usable fragments: {completed.stderr}")
    linkers, rejected = extract_linkers(".".join(sorted(fragments)))
    if rejected:
        raise ValueError(f"RDKit rejected extracted fragments: {rejected}")
    if not linkers:
        raise ValueError("No valid linkers were extracted from the CIF.")
    return linkers


def build_phase3_features(
    cif_path: str | Path,
    geometry: Mapping[str, float],
    *,
    feature_names: Sequence[str] | None = None,
    sbu_path: str | Path | None = None,
    timeout: float = 120,
) -> pd.DataFrame:
    """Return one complete numeric feature row; optionally enforce model order."""
    if set(geometry) != set(GEOMETRY_COLUMNS):
        raise ValueError(f"Supply exactly these geometry fields: {GEOMETRY_COLUMNS}")
    values: dict[str, float] = {}
    for name in GEOMETRY_COLUMNS:
        value = geometry[name]
        if isinstance(value, (bool, str)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite, non-negative number.")
        values[name] = float(value)
    if values["void_fraction"] > 1:
        raise ValueError("void_fraction must be between 0 and 1.")

    linkers = extract_cif_linkers(cif_path, sbu_path=sbu_path, timeout=timeout)
    values.update(aggregate_linker_descriptors(linkers))
    structure = extract_phase3_descriptors(cif_path)
    for name in PHASE3_DESCRIPTOR_COLUMNS:
        value = structure[name]
        if value is None or not math.isfinite(value):
            raise ValueError(f"CIF produced a missing or non-finite descriptor: {name}")
        values[name] = float(value)

    names = list(feature_names) if feature_names is not None else list(values)
    if len(names) != len(set(names)) or set(names) != set(values):
        raise ValueError(
            "Model feature schema does not match the extracted Phase 3 descriptors. "
            f"Missing: {sorted(set(names) - set(values))}; "
            f"unexpected: {sorted(set(values) - set(names))}."
        )
    return pd.DataFrame([[values[name] for name in names]], columns=names)
