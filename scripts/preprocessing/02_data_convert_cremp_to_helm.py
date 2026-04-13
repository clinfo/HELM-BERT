#!/usr/bin/env python3
"""
Convert CREMP v1.1 sequences to HELM 2.0 notation.

Monomer IDs aligned to CycPeptMPDB + ChEMBL v36 monomer libraries:
  - Standard L-AA: single uppercase letter (A, C, D, ...)
  - D-AA: [dX] (dA, dC, dF, ...)
  - N-methyl L-AA: [meX] (meA, meC, meF, ...) except G -> [Sar]
  - N-methyl D-AA: [Me_dX] (Me_dA, Me_dC, Me_dF, ...)

All sequences are head-to-tail cyclic peptides, connection: 1:R1-N:R2

Usage:
    python scripts/preprocessing/02_data_convert_cremp_to_helm.py
"""

import sys
import argparse
import csv
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.preprocessing.preprocessing_utils.helm_utils import (
    load_monomer_library,
    validate_helm_monomers,
)
from scripts.preprocessing.preprocessing_utils.paths import (
    INTERMEDIATE_PRODUCT_DIR,
    MONOMER_LIBRARY_PATH,
    RAW_DATA_DIR,
)

DEFAULT_INPUT = RAW_DATA_DIR / "CREMP_v1.1.csv"
DEFAULT_OUTPUT = INTERMEDIATE_PRODUCT_DIR / "CREMP_v1.1_helm.csv"
MONOMER_LIBRARY = MONOMER_LIBRARY_PATH

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

CREMP_TO_HELM = {}
for _aa in STANDARD_AA:
    CREMP_TO_HELM[_aa] = _aa
    CREMP_TO_HELM[_aa.lower()] = f"[d{_aa}]"
    CREMP_TO_HELM[f"Me{_aa}"] = f"[me{_aa}]"
    CREMP_TO_HELM[f"Me{_aa.lower()}"] = f"[Me_d{_aa}]"

CREMP_TO_HELM["MeG"] = "[Sar]"


def monomer_to_helm(monomer: str) -> str:
    return CREMP_TO_HELM.get(monomer, f"[{monomer}]")


def sequence_to_helm(sequence: str) -> str:
    monomers = sequence.split(".")
    helm_monomers = [monomer_to_helm(m) for m in monomers]
    n = len(helm_monomers)
    polymer = "PEPTIDE1{" + ".".join(helm_monomers) + "}"
    connection = f"PEPTIDE1,PEPTIDE1,1:R1-{n}:R2"
    return f"{polymer}${connection}$$$V2.0"


def main():
    parser = argparse.ArgumentParser(description="Convert CREMP sequences to HELM 2.0")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    with open(args.input) as f:
        reader = csv.DictReader(f)
        orig_fields = reader.fieldnames
        rows = list(reader)

    smiles_idx = orig_fields.index("smiles")
    new_fields = orig_fields[:smiles_idx] + ["helm"] + orig_fields[smiles_idx:]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        for row in rows:
            row["helm"] = sequence_to_helm(row["sequence"])
            writer.writerow(row)

    # Validate generated HELM against monomer library
    valid_symbols, _ = load_monomer_library(MONOMER_LIBRARY)
    invalid = []
    for row in rows:
        if not validate_helm_monomers(row["helm"], valid_symbols):
            invalid.append((row["sequence"], row["helm"]))

    print(f"Input:  {args.input} ({len(rows)} rows)")
    print(f"Output: {args.output} ({len(new_fields)} columns)")
    if invalid:
        print(f"WARNING: {len(invalid)} rows with invalid monomers:")
        for seq, helm in invalid[:10]:
            print(f"  {seq} → {helm}")
    else:
        print(f"Validation: all {len(rows)} HELM strings pass monomer library check")


if __name__ == "__main__":
    main()
