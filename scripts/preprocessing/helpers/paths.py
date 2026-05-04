"""Shared path constants for preprocessing scripts."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_DATA_DIR = REPO_ROOT / "local_data"
RAW_DATA_DIR = LOCAL_DATA_DIR / "raw"
INTERMEDIATE_PRODUCT_DIR = LOCAL_DATA_DIR / "intermediate_product"
DATA_DIR = REPO_ROOT / "data"
PREPROCESSING_OUTPUT_DIR = REPO_ROOT / "outputs/preprocessing"

# Monomer library — produced by the build pipeline (01–03_data_build_*),
# consumed by 07_data_normalize_helm and downstream scripts. The canonical
# copy lives under intermediate_product/ alongside other staged outputs;
# data/monomer_library/ is kept as a tracked snapshot.
MONOMER_LIBRARY_DIR = INTERMEDIATE_PRODUCT_DIR / "monomer_library"
MONOMER_LIBRARY_PATH = MONOMER_LIBRARY_DIR / "helm_monomer_library.csv"

# Hand-curated extensions to the auto-built monomer library.
# These layer on top of the raw-source build (02_data_build_helm_monomer_library)
# so adding curated monomers does NOT require modifying the build pipeline.
# Both files are absent by default — the pipeline silently skips merging when missing.
MANUAL_MONOMER_ADDITIONS_PATH = MONOMER_LIBRARY_DIR / "manual_additions.csv"
HELM_CORRECTIONS_PATH = MONOMER_LIBRARY_DIR / "helm_corrections.csv"

# Staged warehouse layout (01_ingested → 05_final). Each pipeline stage
# writes one CSV per dataset key (e.g. 05_final/chembl_ppi.csv). Stages
# skipped by a given dataset have no file in that stage's directory.
INGESTED_DIR = INTERMEDIATE_PRODUCT_DIR / "01_ingested"
CONVERTED_DIR = INTERMEDIATE_PRODUCT_DIR / "02_converted"
HELM_NORMALIZED_DIR = INTERMEDIATE_PRODUCT_DIR / "03_helm_normalized"
SMILES_NORMALIZED_DIR = INTERMEDIATE_PRODUCT_DIR / "04_smiles_normalized"
FINAL_DIR = INTERMEDIATE_PRODUCT_DIR / "05_final"

STAGE_DIRS: dict[str, Path] = {
    "ingested": INGESTED_DIR,
    "converted": CONVERTED_DIR,
    "helm_normalized": HELM_NORMALIZED_DIR,
    "smiles_normalized": SMILES_NORMALIZED_DIR,
    "final": FINAL_DIR,
}

# Pipeline log directory (used by 07/08/09_data_*).
LOG_DIR = PREPROCESSING_OUTPUT_DIR

# Per-dataset raw inputs. The build pipeline reads these directly; rerun
# the pipeline by re-pointing here if you swap source vintages.
CYCPEPT_RAW = RAW_DATA_DIR / "CycPeptMPDB_Peptide_All_V1.2.csv"
CHEMBL_COMPOUNDS_RAW = RAW_DATA_DIR / "chembl36_helm_compounds.csv"
CHEMBL_PPI_RAW = RAW_DATA_DIR / "helm_ppi_dataset.csv"
CHEMBL_PPI_ALL_RAW = RAW_DATA_DIR / "helm_ppi_all_activities.csv"
CREMP_RAW = RAW_DATA_DIR / "CREMP_v1.1.csv"
PROPEDIA_RAW_DIR = RAW_DATA_DIR / "propedia_v2"


def ensure_dirs() -> None:
    """Create stage + log directories if missing. Idempotent."""
    for d in (LOG_DIR, MONOMER_LIBRARY_DIR, *STAGE_DIRS.values()):
        d.mkdir(parents=True, exist_ok=True)
