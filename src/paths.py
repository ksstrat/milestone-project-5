# src/paths.py
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core directories
INPUTS_DIR = PROJECT_ROOT / "inputs"
DATA_DIR = INPUTS_DIR / "cherry_leaves_dataset"
MANIFESTS_DIR = INPUTS_DIR / "manifests" / "v1"
PLOTS_DIR = PROJECT_ROOT / "plots" / "v1"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Ensure important folders exist
for d in [PLOTS_DIR, ARTIFACTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)