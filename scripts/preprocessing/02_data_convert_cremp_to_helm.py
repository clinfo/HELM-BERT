#!/usr/bin/env python3
"""
Convert CREMP v1.1 sequences to HELM 2.0 notation.

Monomer IDs follow the Pistoia Alliance HELMCoreLibrary:
  - Standard L-AA: single uppercase letter (A, C, D, ...)
  - D-AA: [dX] (dA, dC, dF, ...)
  - N-methyl L-AA: [meX] (meA, meC, meF, ...)
  - N-methyl D-AA: [d-meX] — not in the official library; conventional notation

All sequences are head-to-tail cyclic peptides, connection: 1:R1-N:R2

Usage:
    python scripts/preprocessing/02_data_convert_cremp_to_helm.py
"""

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "local_data/raw/CREMP_v1.1.csv"
DEFAULT_OUTPUT = REPO_ROOT / "local_data/intermediate_product/CREMP_v1.1_helm.csv"

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

CREMP_TO_HELM = {}
for _aa in STANDARD_AA:
    CREMP_TO_HELM[_aa] = _aa
    CREMP_TO_HELM[_aa.lower()] = f"[d{_aa}]"
    CREMP_TO_HELM[f"Me{_aa}"] = f"[me{_aa}]"
    CREMP_TO_HELM[f"Me{_aa.lower()}"] = f"[d-me{_aa}]"


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

    print(f"Input:  {args.input} ({len(rows)} rows)")
    print(f"Output: {args.output} ({len(new_fields)} columns)")
    print(f"Columns: {new_fields}")


if __name__ == "__main__":
    main()
