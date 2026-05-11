"""Path constants for the local preprocessing pipeline.

The warehouse layout under ``local_data/`` mirrors the HELM_datasets
internal structure:

    local_data/
      raw/{CycPeptMPDB,ChEMBL,CREMP,Propedia,monomer_library,curation}/
      processed/{01_ingested,02_converted,03_helm_normalized,
                 04_smiles_normalized,05_final,monomer_library}/

All scripts depend on this single source of truth for layout.
"""
from __future__ import annotations

from pathlib import Path

# Resolve from this file's location:
# helpers/paths.py -> helpers -> preprocessing -> scripts -> REPO_ROOT
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

# Warehouse root.
BASE: Path = REPO_ROOT / "local_data"
RAW_DIR: Path = BASE / "raw"
PROCESSED_DIR: Path = BASE / "processed"
LOG_DIR: Path = REPO_ROOT / "outputs" / "preprocessing"

MONOMER_LIBRARY_PATH: Path = PROCESSED_DIR / "monomer_library" / "helm_monomer_library.csv"

# Hand-curated inputs that layer on top of the raw-source build
# (02_build_helm_monomer_library.py) and the HELM normalize step
# (07_normalize_helm.py). Both files are absent by default — the
# pipeline silently skips merging / correcting when missing.
MANUAL_MONOMER_ADDITIONS_PATH: Path = RAW_DIR / "monomer_library" / "manual_monomer_additions.csv"
HELM_CORRECTIONS_PATH: Path = RAW_DIR / "curation" / "helm_corrections.csv"

# Per-dataset input files (raw or upstream-processed).
CYCPEPT_RAW: Path = RAW_DIR / "CycPeptMPDB" / "CycPeptMPDB_Peptide_All.csv"
CHEMBL_COMPOUNDS_RAW: Path = RAW_DIR / "ChEMBL" / "chembl36_helm_compounds.csv"
CHEMBL_PPI_RAW: Path = RAW_DIR / "ChEMBL" / "helm_ppi_dataset.csv"
CHEMBL_PPI_ALL_RAW: Path = RAW_DIR / "ChEMBL" / "helm_ppi_all_activities.csv"
CREMP_RAW: Path = RAW_DIR / "CREMP" / "summary.csv"
PROPEDIA_RAW_DIR: Path = RAW_DIR / "Propedia" / "raw"

# Stage-numbered directories. Each pipeline stage writes to its own
# directory; filenames within a stage are just the dataset key (e.g.
# ``processed/05_final/cycpept_permeability_compounds.csv``). Stages
# skipped by a given dataset simply don't have a file in that directory.
INGESTED_DIR: Path = PROCESSED_DIR / "01_ingested"
CONVERTED_DIR: Path = PROCESSED_DIR / "02_converted"
HELM_NORMALIZED_DIR: Path = PROCESSED_DIR / "03_helm_normalized"
SMILES_NORMALIZED_DIR: Path = PROCESSED_DIR / "04_smiles_normalized"
FINAL_DIR: Path = PROCESSED_DIR / "05_final"

STAGE_DIRS: dict[str, Path] = {
    "ingested": INGESTED_DIR,
    "converted": CONVERTED_DIR,
    "helm_normalized": HELM_NORMALIZED_DIR,
    "smiles_normalized": SMILES_NORMALIZED_DIR,
    "final": FINAL_DIR,
}

# Tracked downstream artifacts read directly by training (data/ is in git).
DATA_DIR: Path = REPO_ROOT / "data"
MLM_DIR: Path = DATA_DIR / "mlm"
DOWNSTREAM_DIR: Path = DATA_DIR / "downstream"

# Warehouse-level intermediates derived from PROCESSED_DIR but not consumed
# by training directly. Kept under local_data/processed/ so they share the
# ephemeral / regenerable life-cycle of the rest of the pipeline state.
SIGNATURES_DIR: Path = PROCESSED_DIR / "signatures_acsm_all"


def ensure_dirs() -> None:
    """Create output directories if missing. Idempotent."""
    for d in (LOG_DIR, *STAGE_DIRS.values()):
        d.mkdir(parents=True, exist_ok=True)
