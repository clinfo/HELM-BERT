"""Per-dataset configuration shared across the pipeline.

A single source of truth for input paths, column names, ID columns, and
any dataset-specific filtering. Every script (normalize_helm,
normalize_smiles, dedup) reads from this dict so we never have
column-name drift between stages.

Stage layout (under ``processed/``):

    01_ingested/<key>.csv               raw -> CSV (PDB walks etc.)
    02_converted/<key>.csv              sequence -> HELM/SMILES
    03_helm_normalized/<key>.csv        HELM canonicalized
    04_smiles_normalized/<key>.csv      adds normalized_smiles column
    05_final/<key>.csv                  deduplicated, ready for downstream

A dataset that skips a stage simply has no file in that stage's
directory; the next stage reads from the most recent prior stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from helpers.paths import (
    CHEMBL_COMPOUNDS_RAW,
    CHEMBL_PPI_ALL_RAW,
    CHEMBL_PPI_RAW,
    CYCPEPT_RAW,
    STAGE_DIRS,
)


@dataclass(frozen=True)
class DatasetConfig:
    """Per-dataset wiring used by all pipeline stages.

    Attributes:
        key: short name; the only string used in output filenames.
        raw_input: original CSV (raw or upstream-pipeline output).
        helm_col: HELM column name in the *output* CSVs (after rename).
        smiles_col: source SMILES column name in the output CSVs.
        id_cols: columns that uniquely identify a row at ingestion time.
                 Used by dedup to preserve provenance via Source_IDs.
        dedup_extra_keys: columns to include in the dedup grouping key
                          beyond (normalized_helm, normalized_smiles).
                          Empty for compound-level dedup; populated when
                          measurement rows must be preserved (chembl_ppi).
        usecols: optional column subset to read from raw_input.
        filter_in: optional row filter {column: allowed_values}.
        drop_cols: columns to drop after filtering.
        rename: column renames applied immediately after reading.
    """
    key: str
    raw_input: Path
    helm_col: str
    smiles_col: str
    id_cols: tuple[str, ...]
    dedup_extra_keys: tuple[str, ...] = ()
    usecols: tuple[str, ...] | None = None
    filter_in: dict[str, list[str]] = field(default_factory=dict)
    drop_cols: tuple[str, ...] = ()
    rename: dict[str, str] = field(default_factory=dict)

    def stage_path(self, stage: str) -> Path:
        """Return the output CSV path for a given stage.

        stage ∈ {'ingested', 'converted', 'helm_normalized',
                 'smiles_normalized', 'final'}
        Files live at ``processed/<stage_dir>/<key>.csv``.
        """
        if stage not in STAGE_DIRS:
            raise ValueError(
                f"unknown stage {stage!r}; expected one of {list(STAGE_DIRS)}"
            )
        return STAGE_DIRS[stage] / f"{self.key}.csv"


# Convert-step outputs feed normalize_helm. The convert/ingest stages
# (01_ingested, 02_converted) are *source-level* — one file per upstream
# source — because they're upstream of any view-level branching. View
# names (e.g. propedia_ppi vs propedia_compounds) only appear from
# 03_helm_normalized onward, where each view runs the pipeline
# independently.
CREMP_CONVERTED = STAGE_DIRS["converted"] / "cremp.csv"
PROPEDIA_CONVERTED = STAGE_DIRS["converted"] / "propedia.csv"


DATASETS: dict[str, DatasetConfig] = {
    "cycpept_permeability_compounds": DatasetConfig(
        # CycPeptMPDB cyclic peptides. One row per unique molecule;
        # PAMPA / Caco2 / MDCK / RRCK permeability columns ride along
        # as compound attributes.
        key="cycpept_permeability_compounds",
        raw_input=CYCPEPT_RAW,
        helm_col="HELM",
        smiles_col="SMILES",
        id_cols=("ID",),
    ),
    "chembl_compounds": DatasetConfig(
        key="chembl_compounds",
        raw_input=CHEMBL_COMPOUNDS_RAW,
        helm_col="helm_notation",
        smiles_col="canonical_smiles",
        id_cols=("compound_chembl_id",),
        usecols=(
            "md_chembl_id",
            "md_molecule_type",
            "bt_helm_notation",
            "cs_canonical_smiles",
        ),
        filter_in={"md_molecule_type": ["Protein", "Small molecule"]},
        drop_cols=("md_molecule_type",),
        rename={
            "md_chembl_id": "compound_chembl_id",
            "bt_helm_notation": "helm_notation",
            "cs_canonical_smiles": "canonical_smiles",
        },
    ),
    "chembl_ppi": DatasetConfig(
        key="chembl_ppi",
        raw_input=CHEMBL_PPI_RAW,
        helm_col="helm_notation",
        smiles_col="canonical_smiles",
        id_cols=("compound_chembl_id",),
        # Each (compound, target, measurement type/value) row is a distinct
        # observation; collapse only when ALL of these match. Compound-level
        # canonicalization still happens, but measurement rows are preserved.
        dedup_extra_keys=(
            "target_chembl_id",
            "standard_type",
            "standard_value",
        ),
    ),
    "chembl_ppi_measurements": DatasetConfig(
        # Same source as chembl_ppi but unaggregated: every individual
        # bioactivity measurement is a separate row (preserves `relation`,
        # `assay_type`, raw value distribution). Use chembl_ppi for ML;
        # use this for inspecting measurement variance / audit.
        key="chembl_ppi_measurements",
        raw_input=CHEMBL_PPI_ALL_RAW,
        helm_col="helm_notation",
        smiles_col="canonical_smiles",
        id_cols=("compound_chembl_id",),
        dedup_extra_keys=(
            "target_chembl_id",
            "standard_type",
            "standard_value",
            "relation",
        ),
    ),
    "cremp_conformer_compounds": DatasetConfig(
        # CREMP v1.1 designed macrocyclic peptides. One row per unique
        # molecule; 3D conformer ensemble metrics (totalconfs,
        # uniqueconfs, lowestenergy, ensemblefreeenergy, …) ride along
        # as compound attributes.
        key="cremp_conformer_compounds",
        raw_input=CREMP_CONVERTED,
        helm_col="helm",
        smiles_col="smiles",
        # CREMP rows have no synthetic ID; the source sequence is the
        # natural identifier (one row = one designed peptide).
        id_cols=("sequence",),
    ),
    "propedia_ppi": DatasetConfig(
        # Propedia v2: peptide-receptor complex structures derived from
        # PDB. Linear peptides bound to protein receptors.
        key="propedia_ppi",
        raw_input=PROPEDIA_CONVERTED,
        helm_col="Peptide_HELM",
        smiles_col="Peptide_SMILES",
        id_cols=("PDB", "Peptide_Chain", "Receptor_Chain"),
        # A peptide that binds two different receptors is two distinct
        # PPI rows; preserve receptor identity in the dedup key.
        dedup_extra_keys=("Receptor_Sequence",),
    ),
    "propedia_compounds": DatasetConfig(
        # Compound-only view of Propedia: unique peptide chains stripped
        # of receptor pairing. Mirrors chembl_compounds in purpose.
        # Source row count: 31k (PPI pairs) → 9k unique peptides.
        key="propedia_compounds",
        raw_input=PROPEDIA_CONVERTED,
        helm_col="Peptide_HELM",
        smiles_col="Peptide_SMILES",
        id_cols=("PDB", "Peptide_Chain"),
        # No dedup_extra_keys → collapse all rows sharing the same
        # peptide molecule, regardless of which receptor it bound.
    ),
}
