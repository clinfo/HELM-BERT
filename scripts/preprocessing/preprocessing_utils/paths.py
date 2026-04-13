"""Shared path constants for preprocessing scripts."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_DATA_DIR = REPO_ROOT / "local_data"
RAW_DATA_DIR = LOCAL_DATA_DIR / "raw"
INTERMEDIATE_PRODUCT_DIR = LOCAL_DATA_DIR / "intermediate_product"
DATA_DIR = REPO_ROOT / "data"
MONOMER_LIBRARY_PATH = DATA_DIR / "monomer_library/helm_monomer_library.csv"
PREPROCESSING_OUTPUT_DIR = REPO_ROOT / "outputs/preprocessing"

