"""
Training script for MOF CO2 uptake prediction.

This script includes:
1. Phase 1: geometry-only training on cleaned data.
2. Phase 2: geometry vs geometry + linker descriptors.
3. Phase 3: controlled comparison of Model A and Model B.

SHAP plots and scientific interpretation stay in the notebooks.
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

if __package__:
    from .paths import (
        MODELS_DIR as CONFIGURED_MODELS_DIR,
        PHASE1_DATA_PATH as CONFIGURED_PHASE1_DATA_PATH,
        PHASE2_DATA_PATH as CONFIGURED_PHASE2_DATA_PATH,
        PHASE3_DATA_PATH as CONFIGURED_PHASE3_DATA_PATH,
        RESULTS_DIR as CONFIGURED_RESULTS_DIR,
    )
else:
    from paths import (
        MODELS_DIR as CONFIGURED_MODELS_DIR,
        PHASE1_DATA_PATH as CONFIGURED_PHASE1_DATA_PATH,
        PHASE2_DATA_PATH as CONFIGURED_PHASE2_DATA_PATH,
        PHASE3_DATA_PATH as CONFIGURED_PHASE3_DATA_PATH,
        RESULTS_DIR as CONFIGURED_RESULTS_DIR,
    )

SEED = 12345

PHASE1_DATA_PATH = str(CONFIGURED_PHASE1_DATA_PATH)
PHASE2_DATA_PATH = str(CONFIGURED_PHASE2_DATA_PATH)
PHASE3_DATA_PATH = str(CONFIGURED_PHASE3_DATA_PATH)
RESULTS_DIR = str(CONFIGURED_RESULTS_DIR)
MODELS_DIR = str(CONFIGURED_MODELS_DIR)

GEOMETRY_COLUMNS = [
    "lcd",
    "pld",
    "void_fraction",
    "surface_area_m2g",
]

TARGET_COLUMNS = [
    "CO2_uptake_0.01bar_molkg",
    "CO2_uptake_0.05bar_molkg",
    "CO2_uptake_0.1bar_molkg",
    "CO2_uptake_0.5bar_molkg",
    "CO2_uptake_2.5bar_molkg",
]

PHASE3_DESCRIPTOR_COLUMNS = [
    "num_metal_types",
    "total_metal_atoms",
    "total_atoms",
    "metal_fraction",
    "avg_metal_atomic_number",
    "avg_metal_atomic_mass",
    "unit_cell_volume",
    "density",
    "num_sites",
    "lattice_a",
    "lattice_b",
    "lattice_c",
    "lattice_alpha",
    "lattice_beta",
    "lattice_gamma",
]

XGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_child_weight": 1,
    "objective": "reg:squarederror",
    "random_state": SEED,
    "n_jobs": -1,
}


def pressure_label(target_column):
    """Create a short pressure label from a target column name."""
    return target_column.replace("CO2_uptake_", "").replace("_molkg", "")


def calculate_metrics(y_true, predictions):
    """Calculate MAE, RMSE, and R2."""
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions),
    }


def split_dataframe(df, seed=SEED):
    """Split full dataframe into 60% train, 20% validation, and 20% test."""
    df_train, df_temp = train_test_split(
        df,
        test_size=0.40,
        random_state=seed,
        shuffle=True,
    )

    df_valid, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
    )

    return df_train, df_valid, df_test


def get_numeric_linker_columns(df):
    """Select numeric Phase 2 linker descriptor columns."""
    excluded_columns = set(
        GEOMETRY_COLUMNS + TARGET_COLUMNS + PHASE3_DESCRIPTOR_COLUMNS
    )

    metadata_columns = {
        "filename",
        "mofid",
        "mofkey",
        "chemical",
        "linker_smiles",
        "cif_filename",
        "cif_path",
        "cif_found",
    }

    excluded_columns = excluded_columns.union(metadata_columns)

    return [
        column for column in df.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(df[column])
    ]
def train_xgboost_model(X_train, y_train):
    """Train one XGBoost model."""
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, X_valid, X_test, y_train, y_valid, y_test):
    """Evaluate one trained model on train, validation, and test sets."""
    train_predictions = model.predict(X_train)
    valid_predictions = model.predict(X_valid)
    test_predictions = model.predict(X_test)

    train_metrics = calculate_metrics(y_train, train_predictions)
    valid_metrics = calculate_metrics(y_valid, valid_predictions)
    test_metrics = calculate_metrics(y_test, test_predictions)

    return train_metrics, valid_metrics, test_metrics, valid_predictions, test_predictions


def train_feature_set(
    df_train,
    df_valid,
    df_test,
    feature_columns,
    model_name,
    phase_name,
    results_dir,
    models_dir,
):
    """Train one feature set across all five CO2 pressure targets."""
    results = []

    X_train = df_train[feature_columns]
    X_valid = df_valid[feature_columns]
    X_test = df_test[feature_columns]

    for target in TARGET_COLUMNS:
        label = pressure_label(target)

        y_train = df_train[target]
        y_valid = df_valid[target]
        y_test = df_test[target]

        model = train_xgboost_model(X_train, y_train)

        train_metrics, valid_metrics, test_metrics, valid_predictions, test_predictions = evaluate_model(
            model,
            X_train,
            X_valid,
            X_test,
            y_train,
            y_valid,
            y_test,
        )

        results.append({
            "phase": phase_name,
            "model": model_name,
            "target": target,
            "pressure": label,
            "n_features": len(feature_columns),
            "train_MAE": train_metrics["MAE"],
            "train_RMSE": train_metrics["RMSE"],
            "train_R2": train_metrics["R2"],
            "valid_MAE": valid_metrics["MAE"],
            "valid_RMSE": valid_metrics["RMSE"],
            "valid_R2": valid_metrics["R2"],
            "test_MAE": test_metrics["MAE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_R2": test_metrics["R2"],
        })

        safe_phase = phase_name.lower().replace(" ", "_")
        safe_model = model_name.lower().replace(" ", "_").replace("+", "plus")

        model_path = os.path.join(
            models_dir,
            f"{safe_phase}_{safe_model}_{label}.joblib",
        )
        joblib.dump(model, model_path)

        pd.DataFrame({
            "actual": y_valid,
            "predicted": valid_predictions,
        }).to_csv(
            os.path.join(
                results_dir,
                f"predictions_valid_{safe_phase}_{safe_model}_{label}.csv",
            ),
            index=False,
        )

        pd.DataFrame({
            "actual": y_test,
            "predicted": test_predictions,
        }).to_csv(
            os.path.join(
                results_dir,
                f"predictions_test_{safe_phase}_{safe_model}_{label}.csv",
            ),
            index=False,
        )

        pd.DataFrame({
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }).to_csv(
            os.path.join(
                results_dir,
                f"feature_importance_{safe_phase}_{safe_model}_{label}.csv",
            ),
            index=False,
        )

    return pd.DataFrame(results)


def train_phase1_geometry(
    data_path=PHASE1_DATA_PATH,
    results_dir=RESULTS_DIR,
    models_dir=MODELS_DIR,
):
    """Phase 1: train geometry-only XGBoost models."""
    df = pd.read_csv(data_path)
    df_train, df_valid, df_test = split_dataframe(df)

    return train_feature_set(
        df_train=df_train,
        df_valid=df_valid,
        df_test=df_test,
        feature_columns=GEOMETRY_COLUMNS,
        model_name="Geometry only",
        phase_name="Phase 1",
        results_dir=results_dir,
        models_dir=models_dir,
    )


def train_phase2_geometry_vs_linkers(
    data_path=PHASE2_DATA_PATH,
    results_dir=RESULTS_DIR,
    models_dir=MODELS_DIR,
):
    """Phase 2: compare geometry-only model with geometry + linker descriptors."""
    df = pd.read_csv(data_path, low_memory=False)
    df_train, df_valid, df_test = split_dataframe(df)

    linker_columns = get_numeric_linker_columns(df)
    geometry_linker_columns = GEOMETRY_COLUMNS + linker_columns

    phase2_geometry = train_feature_set(
        df_train=df_train,
        df_valid=df_valid,
        df_test=df_test,
        feature_columns=GEOMETRY_COLUMNS,
        model_name="Geometry only",
        phase_name="Phase 2",
        results_dir=results_dir,
        models_dir=models_dir,
    )

    phase2_linkers = train_feature_set(
        df_train=df_train,
        df_valid=df_valid,
        df_test=df_test,
        feature_columns=geometry_linker_columns,
        model_name="Geometry + linkers",
        phase_name="Phase 2",
        results_dir=results_dir,
        models_dir=models_dir,
    )

    return pd.concat([phase2_geometry, phase2_linkers], ignore_index=True)


def train_phase3_controlled_comparison(
    data_path=PHASE3_DATA_PATH,
    results_dir=RESULTS_DIR,
    models_dir=MODELS_DIR,
):
    """Phase 3: compare Model A and Model B using identical rows and splits."""
    df = pd.read_csv(data_path, low_memory=False)
    df_train, df_valid, df_test = split_dataframe(df)

    linker_columns = get_numeric_linker_columns(df)

    model_a_columns = GEOMETRY_COLUMNS + linker_columns
    model_b_columns = model_a_columns + [
        column for column in PHASE3_DESCRIPTOR_COLUMNS
        if column in df.columns
    ]

    model_a_results = train_feature_set(
        df_train=df_train,
        df_valid=df_valid,
        df_test=df_test,
        feature_columns=model_a_columns,
        model_name="Model A geometry + linkers",
        phase_name="Phase 3",
        results_dir=results_dir,
        models_dir=models_dir,
    )

    model_b_results = train_feature_set(
        df_train=df_train,
        df_valid=df_valid,
        df_test=df_test,
        feature_columns=model_b_columns,
        model_name="Model B geometry + linkers + CIF",
        phase_name="Phase 3",
        results_dir=results_dir,
        models_dir=models_dir,
    )

    return pd.concat([model_a_results, model_b_results], ignore_index=True)


def train_all_phases(
    phase1_data_path=PHASE1_DATA_PATH,
    phase2_data_path=PHASE2_DATA_PATH,
    phase3_data_path=PHASE3_DATA_PATH,
    results_dir=RESULTS_DIR,
    models_dir=MODELS_DIR,
):
    """Train Phase 1, Phase 2, and Phase 3 models."""
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    all_results = []

    print("Training Phase 1 models...")
    all_results.append(
        train_phase1_geometry(
            data_path=phase1_data_path,
            results_dir=results_dir,
            models_dir=models_dir,
        )
    )

    print("Training Phase 2 models...")
    all_results.append(
        train_phase2_geometry_vs_linkers(
            data_path=phase2_data_path,
            results_dir=results_dir,
            models_dir=models_dir,
        )
    )

    print("Training Phase 3 models...")
    all_results.append(
        train_phase3_controlled_comparison(
            data_path=phase3_data_path,
            results_dir=results_dir,
            models_dir=models_dir,
        )
    )

    summary = pd.concat(all_results, ignore_index=True)

    summary_path = os.path.join(results_dir, "training_summary_all_phases.csv")
    summary.to_csv(summary_path, index=False)

    print("Saved full training summary to:", summary_path)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MOF CO2 uptake models for Phase 1, Phase 2, and Phase 3."
    )

    parser.add_argument("--phase1-data-path", default=PHASE1_DATA_PATH)
    parser.add_argument("--phase2-data-path", default=PHASE2_DATA_PATH)
    parser.add_argument("--phase3-data-path", default=PHASE3_DATA_PATH)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--models-dir", default=MODELS_DIR)
    parser.add_argument(
        "--phase",
        choices=["all", "phase1", "phase2", "phase3"],
        default="all",
    )

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    if args.phase == "phase1":
        summary_df = train_phase1_geometry(
            data_path=args.phase1_data_path,
            results_dir=args.results_dir,
            models_dir=args.models_dir,
        )
    elif args.phase == "phase2":
        summary_df = train_phase2_geometry_vs_linkers(
            data_path=args.phase2_data_path,
            results_dir=args.results_dir,
            models_dir=args.models_dir,
        )
    elif args.phase == "phase3":
        summary_df = train_phase3_controlled_comparison(
            data_path=args.phase3_data_path,
            results_dir=args.results_dir,
            models_dir=args.models_dir,
        )
    else:
        summary_df = train_all_phases(
            phase1_data_path=args.phase1_data_path,
            phase2_data_path=args.phase2_data_path,
            phase3_data_path=args.phase3_data_path,
            results_dir=args.results_dir,
            models_dir=args.models_dir,
        )

    print(summary_df)