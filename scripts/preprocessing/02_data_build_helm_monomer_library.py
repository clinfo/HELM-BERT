"""
Build the unified HELM monomer library from three sources:
  1. CycPeptMPDB  (highest priority)
  2. ChEMBL v36
  3. CREMP v1.1   (generated monomers only)

Merge criterion: canonical isomeric SMILES + n_rgroups + monomer_type must
ALL be identical.  CycPeptMPDB symbol is kept as the primary when a merge
occurs; the other symbol goes into alt_symbols.

Outputs (all under processed/monomer_library/):
  helm_monomer_library.csv   — final unified library
  merge_log.csv              — evidence for every merged pair
  build_report.txt           — human-readable build log
"""

import csv
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers.paths import RAW_DATA_DIR, MONOMER_LIBRARY_DIR

RAW_MONO = RAW_DATA_DIR / "monomer_library"
OUT_DIR = MONOMER_LIBRARY_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHEMBL_XML = RAW_MONO / "chembl_36_monomer_library.xml"
CYCPEPT_CSV = RAW_MONO / "CycPeptMPDB_Monomer_All.csv"
CREMP_CSV = RAW_MONO / "CREMP_monomer_library.csv"

OUTPUT = OUT_DIR / "helm_monomer_library.csv"
MERGE_LOG = OUT_DIR / "merge_log.csv"
REPORT = OUT_DIR / "build_report.txt"

# Hand-curated layer merged on top of the auto-built library.
# Optional — if absent, build proceeds without manual additions.
MANUAL_ADDITIONS = OUT_DIR / "manual_additions.csv"

FIELDS = [
    "symbol", "name", "natural_analog", "smiles",
    "monomer_type", "n_rgroups", "source", "alt_symbols",
]

MERGE_FIELDS = [
    "primary_symbol", "alt_symbol", "primary_source", "alt_source",
    "canonical_smiles", "primary_raw_smiles", "alt_raw_smiles",
    "n_rgroups", "monomer_type", "chirality", "verification",
]


def normalize_smi(smi):
    return smi.split("|")[0].strip()


def canonical(smi):
    mol = Chem.MolFromSmiles(normalize_smi(smi))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def count_rg(smi):
    mol = Chem.MolFromSmiles(normalize_smi(smi))
    if mol is None:
        return normalize_smi(smi).count("[*]")
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)


def chirality_str(smi):
    mol = Chem.MolFromSmiles(normalize_smi(smi))
    if mol is None:
        return ""
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    return str(centers) if centers else "none"


_GENERIC_RE = re.compile(r'^(Mono|X)\d+$')


def is_generic(sym):
    """True for auto-generated identifiers like Mono55, X249."""
    return bool(_GENERIC_RE.match(sym))


# ── 1. Load sources ──────────────────────────────────────────────

cycpept = {}
with open(CYCPEPT_CSV) as f:
    for row in csv.DictReader(f):
        sym = row["Symbol"].strip()
        cycpept[sym] = {
            "symbol": sym,
            "name": row.get("Compound_Name", "").strip(),
            "natural_analog": row.get("Natural_Analog", "").strip(),
            "smiles": (row.get("CXSMILES", "") or "").strip(),
            "monomer_type": row.get("Monomer_Type", "").strip(),
            "source": "CycPeptMPDB",
        }

ns = {"lmr": "lmr"}
tree = ET.parse(CHEMBL_XML)
chembl = {}
for poly in tree.getroot().findall(".//lmr:Polymer[@polymerType='PEPTIDE']", ns):
    for mono in poly.findall("lmr:Monomer", ns):
        mid = mono.findtext("lmr:MonomerID", namespaces=ns)
        smi = mono.findtext("lmr:MonomerSmiles", namespaces=ns) or ""
        mtype = mono.findtext("lmr:MonomerType", namespaces=ns) or ""
        mname = mono.findtext("lmr:MonomerName", namespaces=ns) or ""
        analog = mono.findtext("lmr:NaturalAnalog", namespaces=ns) or ""
        if mid:
            chembl[mid.strip()] = {
                "symbol": mid.strip(),
                "name": mname.strip(),
                "natural_analog": analog.strip(),
                "smiles": smi.strip(),
                "monomer_type": mtype.strip(),
                "source": "ChEMBL",
            }

cremp_gen = {}
with open(CREMP_CSV) as f:
    for row in csv.DictReader(f):
        if "generated" in row.get("source", ""):
            sym = row["symbol"].strip()
            cremp_gen[sym] = {
                "symbol": sym,
                "name": row.get("name", "").strip(),
                "natural_analog": row.get("natural_analog", "").strip(),
                "smiles": row.get("smiles", "").strip(),
                "monomer_type": row.get("monomer_type", "").strip(),
                "source": "CREMP(" + row.get("source", "") + ")",
            }


# ── 2. Merge ─────────────────────────────────────────────────────

unified = {}
seen = {}              # (canonical, n_rg, type) -> primary symbol
merge_log = []         # records for merge_log.csv
raw_smiles_map = {}    # symbol -> original raw smiles (for merge log)


def try_add(sym, entry, source_dict, source_label):
    """Attempt to add a monomer; merge if canonical match exists."""
    smi = entry["smiles"]
    can = canonical(smi)
    nrg = count_rg(smi)
    mtype = entry["monomer_type"]
    key = (can, nrg, mtype)

    entry["n_rgroups"] = nrg
    entry["alt_symbols"] = ""
    raw_smiles_map[sym] = smi

    if sym in unified:
        return

    if can and key in seen:
        primary = seen[key]
        existing = unified[primary].get("alt_symbols", "")
        unified[primary]["alt_symbols"] = (existing + ";" + sym).lstrip(";")
        if source_label not in unified[primary]["source"]:
            unified[primary]["source"] += "+" + source_label

        merge_log.append({
            "primary_symbol": primary,
            "alt_symbol": sym,
            "primary_source": unified[primary]["source"].split("+")[0],
            "alt_source": source_label,
            "canonical_smiles": can,
            "primary_raw_smiles": raw_smiles_map.get(primary, ""),
            "alt_raw_smiles": smi,
            "n_rgroups": nrg,
            "monomer_type": mtype,
            "chirality": chirality_str(smi),
            "verification": "isomeric_canonical_match",
        })
        return

    unified[sym] = entry
    if can:
        seen[key] = sym


for sym, entry in sorted(cycpept.items(), key=lambda x: (is_generic(x[0]), x[0])):
    try_add(sym, entry, cycpept, "CycPeptMPDB")

for sym, entry in sorted(chembl.items(), key=lambda x: (is_generic(x[0]), x[0])):
    try_add(sym, entry, chembl, "ChEMBL")

for sym, entry in sorted(cremp_gen.items(), key=lambda x: (is_generic(x[0]), x[0])):
    try_add(sym, entry, cremp_gen, "CREMP")


# ── 2.5 Merge hand-curated additions ─────────────────────────────
#
# Layered on top of raw-source build. Manual entries are *appended*:
# they don't displace existing symbols and they don't trigger merges.
# A symbol-collision is a hard error — manual curation must not silently
# overwrite an entry produced from raw sources. Extra audit columns
# (added_by, reason) are accepted in the input file but stripped here
# so the output schema stays identical to the raw build.

manual_appended = 0
if MANUAL_ADDITIONS.exists():
    with open(MANUAL_ADDITIONS) as f:
        # Skip leading comment lines starting with '#'.
        rdr = csv.DictReader(line for line in f if not line.lstrip().startswith("#"))
        for row in rdr:
            sym = (row.get("symbol") or "").strip()
            if not sym:
                continue
            if sym in unified:
                raise ValueError(
                    f"manual_additions.csv: symbol {sym!r} collides with an "
                    "auto-built entry. Manual layer must not overwrite raw "
                    "sources — pick a different symbol or remove the conflict."
                )
            unified[sym] = {k: row.get(k, "") for k in FIELDS}
            manual_appended += 1


# ── 3. Write library ─────────────────────────────────────────────

rows = sorted(unified.values(), key=lambda r: (r["source"], r["symbol"]))

with open(OUTPUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in FIELDS})


# ── 4. Write merge log ───────────────────────────────────────────

merge_log.sort(key=lambda r: r["primary_symbol"])
with open(MERGE_LOG, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=MERGE_FIELDS)
    w.writeheader()
    w.writerows(merge_log)


# ── 5. Write build report ────────────────────────────────────────

source_counts = {}
for r in rows:
    s = r["source"]
    source_counts[s] = source_counts.get(s, 0) + 1

has_alts = sum(1 for r in rows if r.get("alt_symbols"))

report = f"""HELM Monomer Library — Build Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

════════════════════════════════════════════════════════════════
1. SOURCE DATASETS
════════════════════════════════════════════════════════════════

  CycPeptMPDB
    File:     raw/monomer_library/CycPeptMPDB_Monomer_All.csv
    Monomers: {len(cycpept)}

  ChEMBL v36  (PEPTIDE polymer type only)
    File:     raw/monomer_library/chembl_36_monomer_library.xml
    Monomers: {len(chembl)}

  CREMP v1.1  (generated monomers only — not present in above DBs)
    File:     raw/monomer_library/CREMP_monomer_library.csv
    Generated: {len(cremp_gen)}

════════════════════════════════════════════════════════════════
2. MERGE CRITERIA
════════════════════════════════════════════════════════════════

  Two monomers are considered identical when ALL of the following match:
    a) RDKit isomeric canonical SMILES  (includes @/@@ chirality, /\\ E/Z)
    b) Number of R-groups  (dummy atom count, atomic number 0)
    c) Monomer type  (Backbone / Terminal)

  When a merge occurs:
    - CycPeptMPDB symbol is kept as the primary symbol
    - Descriptive names preferred over generic IDs (Mono\\d+, X\\d+)
    - The other symbol is recorded in alt_symbols
    - Source field is updated (e.g. "CycPeptMPDB+ChEMBL")

  Tautomers and E/Z variants on freely-rotating bonds are NOT merged.
  They are treated as distinct monomers.

════════════════════════════════════════════════════════════════
3. MERGE RESULTS
════════════════════════════════════════════════════════════════

  Merged pairs: {len(merge_log)}
    (see merge_log.csv for full evidence with raw SMILES from both sources)

  Hand-curated additions: {manual_appended}
    (from manual_additions.csv if present)

════════════════════════════════════════════════════════════════
4. FINAL LIBRARY
════════════════════════════════════════════════════════════════

  Total unique monomers: {len(rows)}

  By source:
"""

for s, c in sorted(source_counts.items(), key=lambda x: -x[1]):
    report += f"    {s}: {c}\n"

report += f"""
  Monomers with alt symbols: {has_alts}

  Output files:
    helm_monomer_library.csv   — {len(rows)} entries
    merge_log.csv              — {len(merge_log)} merged pairs
    build_report.txt           — this file
"""

with open(REPORT, "w") as f:
    f.write(report)

print(report)
