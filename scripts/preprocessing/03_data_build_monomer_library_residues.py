#!/usr/bin/env python3
"""
Build HELM monomer library with single-letter residue mapping.

Adds a 'residue_1letter' column for downstream use (ESM2, PepLink, etc.).

Mapping rules (based on library metadata — no heuristics):
  1. natural_analog ∈ standard 20 AA  →  analog letter
  2. n_rgroups == 1 & analog == "X"   →  "DROP"  (terminal cap or cap+AA;
     these always appear at sequence endpoints, never in the middle)
  3. Everything else                  →  "X"

Output: processed/monomer_library/helm_monomer_library_residues.csv

Usage:
    python scripts/03_build_monomer_library_residues.py
"""

import csv
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers.paths import MONOMER_LIBRARY_DIR

STANDARD_20 = set("ACDEFGHIKLMNPQRSTVWY")

INPUT = str(MONOMER_LIBRARY_DIR / "helm_monomer_library.csv")
OUTPUT = str(MONOMER_LIBRARY_DIR / "helm_monomer_library_residues.csv")
REPORT = str(MONOMER_LIBRARY_DIR / "residue_mapping_report.txt")


def assign_residue(row):
    analog = row["natural_analog"]
    n_rg = int(row["n_rgroups"])

    if analog in STANDARD_20:
        return analog
    if n_rg == 1 and analog == "X":
        return "DROP"
    return "X"


def main():
    with open(INPUT) as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)

    if "residue_1letter" not in fields:
        fields.append("residue_1letter")

    counts = {"analog": 0, "DROP": 0, "X": 0}
    for row in rows:
        res = assign_residue(row)
        row["residue_1letter"] = res
        if res == "DROP":
            counts["DROP"] += 1
        elif res == "X":
            counts["X"] += 1
        else:
            counts["analog"] += 1

    with open(OUTPUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

    # Per-letter distribution
    letter_dist = Counter(row["residue_1letter"] for row in rows)

    # DROP examples
    drop_rows = [r for r in rows if r["residue_1letter"] == "DROP"]

    report = f"""Residue Mapping Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

════════════════════════════════════════════════════════════════
1. MAPPING RULES
════════════════════════════════════════════════════════════════

  Rule 1: natural_analog ∈ standard 20 AA  →  analog letter
  Rule 2: n_rgroups == 1 & analog == "X"   →  DROP
  Rule 3: everything else                  →  X

════════════════════════════════════════════════════════════════
2. SUMMARY
════════════════════════════════════════════════════════════════

  Input:  {INPUT}
  Output: {OUTPUT}
  Total monomers: {len(rows)}

  Mapped to analog: {counts['analog']}
  Mapped to X:      {counts['X']}
  Mapped to DROP:   {counts['DROP']}

════════════════════════════════════════════════════════════════
3. PER-LETTER DISTRIBUTION
════════════════════════════════════════════════════════════════

"""
    for letter, cnt in sorted(letter_dist.items(), key=lambda x: (-x[1] if x[0] not in ("X", "DROP") else 0, x[0])):
        report += f"  {letter:5s}: {cnt:>5}\n"

    report += f"""
════════════════════════════════════════════════════════════════
4. DROP MONOMERS ({len(drop_rows)})
════════════════════════════════════════════════════════════════

"""
    for r in sorted(drop_rows, key=lambda x: x["symbol"]):
        report += f"  {r['symbol']:20s} type={r['monomer_type']:10s} source={r['source']}\n"

    with open(REPORT, "w") as f:
        f.write(report)

    print(report)
    print(f"  Report: {REPORT}")


if __name__ == "__main__":
    main()
