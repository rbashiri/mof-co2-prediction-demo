import pandas as pd

from src.train import (
    GEOMETRY_COLUMNS,
    PHASE3_DESCRIPTOR_COLUMNS,
    TARGET_COLUMNS,
    get_numeric_linker_columns,
)


def test_linker_selection_excludes_phase3_and_processing_columns():
    excluded = GEOMETRY_COLUMNS + TARGET_COLUMNS + PHASE3_DESCRIPTOR_COLUMNS + [
        "oxidation_error",
        "oxidation_assignment_success",
        "mean_metal_oxidation_state",
        "min_metal_oxidation_state",
        "max_metal_oxidation_state",
        "cif_found",
    ]
    df = pd.DataFrame({column: [1.0] for column in excluded})
    df["oxidation_error"] = float("nan")
    df["total_C_count"] = 8
    df["mean_MolWt"] = 164.116
    df["linker_smiles"] = "O=C([O-])c1ccc(C(=O)[O-])cc1"

    assert get_numeric_linker_columns(df) == ["total_C_count", "mean_MolWt"]
