"""
Phase 3 CIF-derived descriptor extraction.

This module contains reusable functions for:
1. matching MOF filenames to CIF paths,
2. extracting metal-node descriptors from CIF files,
3. extracting unit-cell / structure descriptors from CIF files.
"""

from pathlib import Path

import pandas as pd
from pymatgen.core import Structure


def make_cif_path(cif_filename, raw_structure_dir):
    """
    Create the full CIF file path using the raw structure folder.
    """
    return Path(raw_structure_dir) / cif_filename


def check_cif_exists(cif_path):
    """
    Check whether a CIF file exists at the given path.
    """
    return Path(cif_path).exists()


def add_cif_paths(df, raw_structure_dir, filename_col="filename"):
    """
    Add CIF filename, CIF path, and CIF-found columns to the dataframe.
    """
    df = df.copy()

    df["cif_filename"] = df[filename_col].astype(str) + ".cif"
    df["cif_path"] = df["cif_filename"].apply(
        lambda cif_name: make_cif_path(cif_name, raw_structure_dir)
    )
    df["cif_found"] = df["cif_path"].apply(check_cif_exists)

    return df


def extract_basic_metal_descriptors(cif_path):
    """
    Extract basic metal-node descriptors from one CIF structure.
    """
    structure = Structure.from_file(cif_path)
    composition = structure.composition
    elements = composition.elements

    metal_elements = []

    for element in elements:
        if element.is_metal:
            metal_elements.append(element)

    total_metal_atoms = 0

    for metal in metal_elements:
        total_metal_atoms += composition[metal]

    total_atoms = composition.num_atoms

    if total_atoms > 0:
        metal_fraction = total_metal_atoms / total_atoms
    else:
        metal_fraction = None

    if total_metal_atoms > 0:
        weighted_atomic_number_sum = 0
        weighted_atomic_mass_sum = 0

        for metal in metal_elements:
            metal_count = composition[metal]
            weighted_atomic_number_sum += metal.Z * metal_count
            weighted_atomic_mass_sum += float(metal.atomic_mass) * metal_count

        avg_metal_atomic_number = weighted_atomic_number_sum / total_metal_atoms
        avg_metal_atomic_mass = weighted_atomic_mass_sum / total_metal_atoms
    else:
        avg_metal_atomic_number = None
        avg_metal_atomic_mass = None

    return {
        "metal_elements": [str(metal) for metal in metal_elements],
        "num_metal_types": len(metal_elements),
        "total_metal_atoms": total_metal_atoms,
        "total_atoms": total_atoms,
        "metal_fraction": metal_fraction,
        "avg_metal_atomic_number": avg_metal_atomic_number,
        "avg_metal_atomic_mass": avg_metal_atomic_mass,
    }


def extract_structure_descriptors(cif_path):
    """
    Extract unit-cell and structure-derived descriptors from one CIF structure.
    """
    structure = Structure.from_file(cif_path)
    lattice = structure.lattice

    return {
        "unit_cell_volume": structure.volume,
        "density": structure.density,
        "num_sites": len(structure),
        "lattice_a": lattice.a,
        "lattice_b": lattice.b,
        "lattice_c": lattice.c,
        "lattice_alpha": lattice.alpha,
        "lattice_beta": lattice.beta,
        "lattice_gamma": lattice.gamma,
    }


def extract_phase3_descriptors(cif_path):
    """
    Extract both metal-node and structure descriptors from one CIF file.
    """
    metal_descriptors = extract_basic_metal_descriptors(cif_path)
    structure_descriptors = extract_structure_descriptors(cif_path)

    return {
        **metal_descriptors,
        **structure_descriptors,
    }


def add_phase3_descriptors(df, cif_path_col="cif_path"):
    """
    Extract Phase 3 descriptors for every CIF path and add them to the dataframe.
    """
    df = df.copy()

    descriptor_results = []

    for cif_path in df[cif_path_col]:
        descriptors = extract_phase3_descriptors(cif_path)
        descriptor_results.append(descriptors)

    descriptor_df = pd.DataFrame(descriptor_results)

    df[descriptor_df.columns] = descriptor_df[descriptor_df.columns]

    return df