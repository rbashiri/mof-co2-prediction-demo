"""
Phase 2 linker descriptor extraction.

This module contains reusable RDKit functions for:
1. calculating the original 15 linker descriptors,
2. counting elements in linker SMILES,
3. counting functional groups,
4. calculating extended linker descriptors.
"""

import math
from collections.abc import Sequence

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors


def calculate_all_descriptors(smiles):
    """Return the original 15 RDKit descriptors for one linker SMILES."""

    missing_descriptors = {
        "MolWt": float("nan"),
        "TPSA": float("nan"),
        "NumHAcceptors": float("nan"),
        "NumHDonors": float("nan"),
        "NumAromaticRings": float("nan"),
        "MaxPartialCharge": float("nan"),
        "MinPartialCharge": float("nan"),
        "NOCount": float("nan"),
        "NumHeteroatoms": float("nan"),
        "LogP": float("nan"),
        "LabuteASA": float("nan"),
        "RotatableBonds": float("nan"),
        "FractionCSP3": float("nan"),
        "PEOE_VSA1": float("nan"),
        "PEOE_VSA2": float("nan"),
    }

    if pd.isna(smiles) or str(smiles).strip() == "":
        return missing_descriptors

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print("RDKit could not read this SMILES:", smiles)
        return missing_descriptors

    return {
        "MolWt": Descriptors.MolWt(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "MaxPartialCharge": Descriptors.MaxPartialCharge(mol),
        "MinPartialCharge": Descriptors.MinPartialCharge(mol),
        "NOCount": Descriptors.NOCount(mol),
        "NumHeteroatoms": Descriptors.NumHeteroatoms(mol),
        "LogP": Descriptors.MolLogP(mol),
        "LabuteASA": Descriptors.LabuteASA(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "PEOE_VSA1": Descriptors.PEOE_VSA1(mol),
        "PEOE_VSA2": Descriptors.PEOE_VSA2(mol),
    }


FUNCTIONAL_GROUP_SMARTS = {
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "hydroxyl": "[OX2H]",
    "carboxylate_or_carboxylic_acid": "[CX3](=O)[O-,OH]",
    "fluorinated_group": "[F]",
    "sulfonyl_or_sulfate": "[SX4](=O)(=O)",
}


FUNCTIONAL_GROUP_PATTERNS = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in FUNCTIONAL_GROUP_SMARTS.items()
}


def count_element(smiles, element_symbol):
    """Count one element in one linker SMILES."""

    if pd.isna(smiles) or str(smiles).strip() == "":
        return 0

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return 0

    count = 0

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == element_symbol:
            count += 1

    return count


def count_functional_group(smiles, pattern):
    """Count one functional group pattern in one linker SMILES."""

    if pd.isna(smiles) or str(smiles).strip() == "":
        return 0

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return 0

    return len(mol.GetSubstructMatches(pattern))


def calculate_extended_linker_descriptors(smiles):
    """Calculate extended physicochemical descriptors for one linker SMILES."""

    missing_descriptors = {
        "MolWt": float("nan"),
        "TPSA": float("nan"),
        "HBD": float("nan"),
        "HBA": float("nan"),
        "aromatic_atom_count": float("nan"),
        "aromatic_fraction": float("nan"),
        "heavy_atom_count": float("nan"),
        "ring_count": float("nan"),
        "rotatable_bond_count": float("nan"),
    }

    if pd.isna(smiles) or str(smiles).strip() == "":
        return missing_descriptors

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return missing_descriptors

    heavy_atom_count = mol.GetNumHeavyAtoms()

    aromatic_atom_count = 0

    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            aromatic_atom_count += 1

    if heavy_atom_count > 0:
        aromatic_fraction = aromatic_atom_count / heavy_atom_count
    else:
        aromatic_fraction = 0

    return {
        "MolWt": Descriptors.MolWt(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "aromatic_atom_count": aromatic_atom_count,
        "aromatic_fraction": aromatic_fraction,
        "heavy_atom_count": heavy_atom_count,
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "rotatable_bond_count": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }


def aggregate_linker_descriptors(linker_smiles: Sequence[str]) -> dict[str, float]:
    """Reproduce Phase 2 MOF-level summaries over retained linker fragments."""
    if not linker_smiles:
        raise ValueError("No valid linkers were extracted from the CIF.")
    descriptors = []
    for smiles in linker_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            raise ValueError(f"Invalid linker SMILES: {smiles!r}")
        descriptor = calculate_extended_linker_descriptors(smiles)
        if not all(math.isfinite(value) for value in descriptor.values()):
            raise ValueError(f"Non-finite linker descriptors for {smiles!r}")
        descriptors.append(descriptor)

    # Training counts these eight elements, rather than all possible elements.
    elements = ("C", "N", "O", "F", "S", "Cl", "Br", "I")
    result = {
        f"total_{element}_count": float(
            sum(count_element(smiles, element) for smiles in linker_smiles)
        )
        for element in elements
    }
    result["total_heteroatom_count"] = sum(
        result[f"total_{element}_count"] for element in elements[1:]
    )
    result["total_heavy_atom_count"] = sum(
        result[f"total_{element}_count"] for element in elements
    )
    heavy_count = result["total_heavy_atom_count"]
    result["heteroatom_fraction"] = (
        result["total_heteroatom_count"] / heavy_count if heavy_count else 0.0
    )
    for group, pattern in FUNCTIONAL_GROUP_PATTERNS.items():
        result[f"total_{group}_count"] = float(
            sum(count_functional_group(smiles, pattern) for smiles in linker_smiles)
        )
    for name in ("MolWt", "TPSA"):
        values = [descriptor[name] for descriptor in descriptors]
        result[f"total_{name}"] = sum(values)
        result[f"mean_{name}"] = sum(values) / len(values)
        result[f"max_{name}"] = max(values)
    for name in ("HBD", "HBA", "aromatic_atom_count"):
        result[f"total_{name}"] = sum(descriptor[name] for descriptor in descriptors)
    result["mean_aromatic_fraction"] = sum(
        descriptor["aromatic_fraction"] for descriptor in descriptors
    ) / len(descriptors)
    for name, column in (
        ("heavy_atom_count", "total_heavy_atom_count_rdkit"),
        ("ring_count", "total_ring_count"),
        ("rotatable_bond_count", "total_rotatable_bond_count"),
    ):
        result[column] = sum(descriptor[name] for descriptor in descriptors)
    return result