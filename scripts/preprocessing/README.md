# HELM Datasets — Unified HELM notation workspace

## Purpose

Warehouse for all HELM (Hierarchical Editing Language for Macromolecules)
peptide datasets. Ingests raw sources, normalizes HELM strings and
SMILES into a canonical form, and dedups within each source. Train/test
splits and MLM assembly are out of scope (handled downstream by
HELM-BERT).

## Directory Structure

```
HELM_datasets/
├── raw/                                    # Source inputs (upstream + hand-curated)
│   ├── monomer_library/
│   │   ├── CycPeptMPDB_Monomer_All.csv     (385 monomers; upstream name preserved)
│   │   ├── chembl_36_monomer_library.xml   (2,851 peptide monomers; upstream name preserved)
│   │   ├── cremp_monomer_library.csv       (55 monomers, 3 generated)
│   │   └── manual_monomer_additions.csv    (hand-curated overlay; merged by 02_build)
│   ├── curation/
│   │   └── helm_corrections.csv            (per-compound HELM token rewrites; applied by 07_normalize_helm)
│   ├── CREMP/summary.csv                   (36,198 peptides)
│   ├── ChEMBL/
│   │   ├── chembl36_helm_compounds.csv     (22,122 HELM compounds)
│   │   └── chembl36_helm_bioactivities.csv (112,430 activities)
│   ├── CycPeptMPDB/CycPeptMPDB_Peptide_All.csv  (8,466 peptides)
│   └── Propedia/raw/                       (PDB structures)
│
├── scripts/                          # Numeric prefix == execution order
│   ├── helpers/
│   │   ├── paths.py                  Path constants for the warehouse
│   │   ├── datasets.py               Per-dataset config (input / cols / id_cols / dedup keys)
│   │   ├── helm_utils.py             HELM parse / normalize utilities
│   │   ├── smiles_utils.py           RDKit canonical SMILES only — no salt/charge/isotope normalization
│   │   └── logging_utils.py          Stage-aware file + stdout logging
│   ├── tests/                        unittest suite for helpers
│   ├── 01_build_cremp_monomer_library.py    CREMP-specific monomer lib (raw/)
│   ├── 02_build_helm_monomer_library.py     unified library (processed/)
│   ├── 03_build_monomer_library_residues.py residue_1letter column on the unified lib
│   ├── 04_ingest_propedia.py                PDB -> CSV (sequences)
│   ├── 05_convert_propedia_to_helm.py       Propedia sequence -> HELM/SMILES
│   ├── 06_convert_cremp_to_helm.py          CREMP sequence -> HELM
│   ├── 07_normalize_helm.py                 stage 1: HELM canonicalization
│   ├── 08_normalize_smiles.py               stage 2: SMILES canonicalization
│   └── 09_dedup.py                          stage 3: per-source dedup
│
└── processed/                        Build / pipeline outputs
    ├── monomer_library/
    │   ├── helm_monomer_library.csv          (3,098 unified monomers)
    │   ├── helm_monomer_library_residues.csv (residue_1letter column)
    │   ├── merge_log.csv                     (57 merged pairs)
    │   ├── build_report.txt
    │   └── residue_mapping_report.txt
    ├── 01_ingested/                  PDB → CSV (source-level)
    │   └── propedia.csv
    ├── 02_converted/                 Sequence → HELM/SMILES (source-level)
    │   ├── cremp.csv
    │   └── propedia.csv
    ├── 03_helm_normalized/           Stage 1 (view-level): HELM canonicalized
    │   ├── chembl_compounds.csv
    │   ├── chembl_ppi.csv
    │   ├── chembl_ppi_measurements.csv
    │   ├── cremp_conformer_compounds.csv
    │   ├── cycpept_permeability_compounds.csv
    │   ├── propedia_compounds.csv
    │   └── propedia_ppi.csv
    ├── 04_smiles_normalized/         Stage 2: normalized_smiles column added
    │   └── (same 7 files)
    └── 05_final/                     Stage 3: deduplicated
        └── (same 7 files)
```

Filename convention:
- Stages **01_ingested** / **02_converted** are *source-level*: one file per
  upstream source (e.g. ``propedia.csv``), regardless of how many views
  consume it downstream.
- Stages **03_helm_normalized** / **04_smiles_normalized** / **05_final** are
  *view-level*: one file per dataset key
  (e.g. ``propedia_ppi.csv`` and ``propedia_compounds.csv``).
- Dataset key follows ``<source>_<purpose>`` everywhere (chembl_ppi,
  cycpept_permeability, …).

## Pipeline

Three pipeline stages plus three monomer-library and three ingest/convert
scripts, all numbered by execution order.

| # | Script | Output | Purpose |
|---|---|---|---|
| 01 | `01_build_cremp_monomer_library.py` | `raw/monomer_library/cremp_monomer_library.csv` | Generate D/Me variant monomers + reuse ChEMBL/CycPept SMILES |
| 02 | `02_build_helm_monomer_library.py` | `processed/monomer_library/helm_monomer_library.csv` | Merge CycPept + ChEMBL + CREMP → 3,098 unique monomers |
| 03 | `03_build_monomer_library_residues.py` | `processed/monomer_library/helm_monomer_library_residues.csv` | Add residue_1letter column to the unified library |
| 04 | `04_ingest_propedia.py` | `01_ingested/propedia.csv` | Walk PDB structures → peptide/receptor sequence CSV |
| 05 | `05_convert_propedia_to_helm.py` | `02_converted/propedia.csv` | Sequence → HELM + SMILES |
| 06 | `06_convert_cremp_to_helm.py` | `02_converted/cremp.csv` | Sequence → HELM (head-to-tail cyclic) |
| 07 | `07_normalize_helm.py` | `03_helm_normalized/<key>.csv` | Single-chain renumber, alt-symbol remap, connection canonicalize, monomer validate |
| 08 | `08_normalize_smiles.py` | `04_smiles_normalized/<key>.csv` | RDKit canonical SMILES (no structural mutation) |
| 09 | `09_dedup.py` | `05_final/<key>.csv` | Group by `(helm, normalized_smiles, *extra)`; lex-min id survives; `Source_IDs` preserved |

### Run

```bash
cd scripts/

# Monomer library (only when raw/monomer_library/* changes)
python 01_build_cremp_monomer_library.py
python 02_build_helm_monomer_library.py
python 03_build_monomer_library_residues.py

# Propedia ingest + convert (only when raw/Propedia/raw/* changes)
python 04_ingest_propedia.py
python 05_convert_propedia_to_helm.py

# CREMP convert (only when raw/CREMP/summary.csv changes)
python 06_convert_cremp_to_helm.py

# Main pipeline (run on every change in any input)
python 07_normalize_helm.py
python 08_normalize_smiles.py
python 09_dedup.py
```

Stages 07–09 accept `--datasets <key> [<key> ...]` for partial reruns.
All stages are idempotent; rerunning produces bit-identical output.

### Tests

```bash
cd scripts/
python -m unittest discover tests
```

54 tests cover stereo preservation, multi-chain protection,
canonicalization correctness, and idempotency.

## Normalization details

### HELM (`helpers/helm_utils.py`)

1. `normalize_single_chain` — `PEPTIDEn{...}` → `PEPTIDE1{...}` only when
   exactly one PEPTIDE chain is present. Multi-chain (homo/heterodimer)
   inputs are left untouched.
2. `remap_alt_symbols` — alt monomer aliases mapped to canonical symbols
   via the merged monomer library.
3. `canonicalize_connections` — bond endpoints sorted to a canonical
   `(chain_type, chain_idx, position, R_idx)` order. Safe for
   single-chain and multi-chain; chain blocks are never modified.
4. `validate_monomers` — drop rows whose monomers aren't in the library.

Cyclic residue rotation is intentionally not canonicalized at the HELM
layer; SMILES canonicalization handles rotational equivalence at the
chemistry level.

### SMILES (`helpers/smiles_utils.py`)

```
canonical SMILES        → MolToSmiles(canonical=True, isomericSmiles=True)
```

The SMILES *string* is rewritten into RDKit canonical form. The
molecule itself is preserved exactly as registered — salts,
counter-ions, protonation states, and isotope labels untouched.
Stereo (`@`/`@@`/`/`/`\\`) is preserved; D and L amino-acids stay
distinct.

### Dedup (`scripts/dedup.py`)

Per-dataset key from `helpers/datasets.py`:

| Dataset | Dedup key beyond `(helm, normalized_smiles)` | Note |
|---|---|---|
| `cycpept` | — | compound-level |
| `chembl_compounds` | — | compound-level |
| `chembl_ppi` | `(target_chembl_id, standard_type, standard_value)` | preserves measurement rows |
| `chembl_ppi_all` | `(target_chembl_id, standard_type, standard_value, relation)` | full activity rows |
| `cremp` | — | sequence-level |
| `propedia` | `(Receptor_Sequence)` | per peptide-receptor pair |

Within a duplicate group, the row with the lex-smallest `id_cols`
tuple survives. The collapsed group's full set of original IDs is
joined into a `Source_IDs` column.

## Datasets

Naming pattern: ``<source>[_<attribute>]_<row_granularity>``. The
``_compounds`` suffix means *one row per unique molecule*; the optional
attribute (``permeability``, ``conformer``) flags non-trivial extra
columns that ride along with the compound.

| Key | Row granularity | Final rows | Notes |
|---|---|---:|---|
| `chembl_compounds` | one unique peptide compound | 21,755 | ChEMBL compound table; no activity |
| `chembl_ppi` | one (compound, target, type) triple | 16,993 | Median pchembl; ML training default |
| `chembl_ppi_measurements` | one individual bioactivity measurement | 20,148 | Includes `relation`, `assay_type` |
| `cremp_conformer_compounds` | one unique peptide compound | 36,198 | + 3D conformer-ensemble columns |
| `cycpept_permeability_compounds` | one unique peptide compound | 8,032 | + PAMPA / Caco2 / MDCK / RRCK columns |
| `propedia_compounds` | one unique peptide compound | 9,212 | Derived from PPI by collapsing receptors |
| `propedia_ppi` | one (peptide, receptor) structural pair | 20,057 | From PDB |

Counts are post-dedup (Stage 05).

## Monomer Library Build

`scripts/build_helm_monomer_library.py` merges three sources:

| Source | Priority | Monomers |
|--------|----------|----------|
| CycPeptMPDB | 1 (highest) | 385 |
| ChEMBL v36 | 2 | 2,851 |
| CREMP (generated only) | 3 | 3 |

Merge criterion: identical isomeric canonical SMILES + R-group count +
monomer type. Descriptive symbols preferred over generic IDs
(`Mono\\d+`, `X\\d+`). Result: **3,098 unique monomers**, 57 merged pairs.
