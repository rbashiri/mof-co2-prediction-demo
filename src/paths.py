"""Resolve project data and artifact paths from one editable configuration."""

import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _resolve_path(value, relative_to):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


with CONFIG_PATH.open(encoding="utf-8") as config_file:
    _config = yaml.safe_load(config_file)

if not isinstance(_config, dict) or not isinstance(_config.get("paths"), dict):
    raise ValueError(f"Expected a 'paths' mapping in {CONFIG_PATH}")

_path_config = _config["paths"]

DATA_DIR = _resolve_path(
    os.environ.get("MOF_DATA_DIR", _path_config["data_dir"]),
    PROJECT_ROOT,
)
RAW_DATA_DIR = _resolve_path(
    os.environ.get("MOF_RAW_DATA_DIR", _path_config["raw_data_dir"]),
    PROJECT_ROOT,
)


def _configured_path(setting, environment_variable, base_directory):
    value = os.environ.get(environment_variable, _path_config[setting])
    return _resolve_path(value, base_directory)


PHASE1_DATA_PATH = _configured_path(
    "phase1_data", "MOF_PHASE1_DATA_PATH", DATA_DIR
)
PHASE2_DATA_PATH = _configured_path(
    "phase2_data", "MOF_PHASE2_DATA_PATH", DATA_DIR
)
PHASE3_DATA_PATH = _configured_path(
    "phase3_data", "MOF_PHASE3_DATA_PATH", DATA_DIR
)
RESULTS_ROOT = _configured_path("results_root", "MOF_RESULTS_ROOT", PROJECT_ROOT)
RESULTS_DIR = _configured_path("results_dir", "MOF_RESULTS_DIR", PROJECT_ROOT)
MODELS_DIR = _configured_path("models_dir", "MOF_MODELS_DIR", PROJECT_ROOT)
