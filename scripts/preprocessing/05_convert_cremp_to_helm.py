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
    python 05_convert_cremp_to_helm.py
"""

import argparse
import csv
import sys
from pathlib import Path

# Allow direct invocation from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers.paths import CREMP_RAW, STAGE_DIRS

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
    parser.add_argument("--input", default=str(CREMP_RAW))
    parser.add_argument(
        "--output", default=str(STAGE_DIRS["converted"] / "cremp.csv")
    )
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

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
