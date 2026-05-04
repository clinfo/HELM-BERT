"""
Build a HELM monomer library for CREMP v1.1.

Pulls SMILES from ChEMBL v36 / CycPeptMPDB monomer libraries where available.
For missing monomers (Me_dN, Me_dQ, Me_dS etc.), generates SMILES by
N-methylation + stereochemistry inversion of the corresponding L-amino acid.

Output: HELM_datasets/raw/monomer_library/CREMP_monomer_library.csv
"""

import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers.paths import RAW_DATA_DIR, MONOMER_LIBRARY_DIR

# Source monomer libraries (raw drops) and build output. The build pipeline
# reads ChEMBL/CycPept monomer libraries from raw/monomer_library/ and emits
# CREMP_monomer_library.csv next to them; 02_data_build_helm_monomer_library
# then merges all three into the unified library under MONOMER_LIBRARY_DIR.
RAW_MONO = RAW_DATA_DIR / "monomer_library"
CHEMBL_XML = RAW_MONO / "chembl_36_monomer_library.xml"
CYCPEPT_CSV = RAW_MONO / "CycPeptMPDB_Monomer_All.csv"
OUTPUT = RAW_MONO / "CREMP_monomer_library.csv"

CREMP_MONOMERS = [
    "A","C","D","E","F","G","H","I","K","L","N","P","Q","R","S","T","V","W","Y",
    "dA","dC","dF","dI","dL","dN","dP","dQ","dS","dT","dW","dY",
    "meA","meC","meF","meI","meL","meN","meQ","meS","meT","meV","meW","meY",
    "Sar",
    "Me_dC","Me_dF","Me_dI","Me_dL","Me_dN","Me_dQ","Me_dS","Me_dT","Me_dV","Me_dW","Me_dY",
]

AA_NAMES = {
    "A":"Alanine","C":"Cysteine","D":"Aspartate","E":"Glutamate","F":"Phenylalanine",
    "G":"Glycine","H":"Histidine","I":"Isoleucine","K":"Lysine","L":"Leucine",
    "M":"Methionine","N":"Asparagine","P":"Proline","Q":"Glutamine","R":"Arginine",
    "S":"Serine","T":"Threonine","V":"Valine","W":"Tryptophan","Y":"Tyrosine",
}


def load_chembl():
    ns = {"lmr": "lmr"}
    tree = ET.parse(CHEMBL_XML)
    lib = {}
    for mono in tree.getroot().findall(".//lmr:Polymer[@polymerType='PEPTIDE']/lmr:Monomer", ns):
        mid = mono.findtext("lmr:MonomerID", namespaces=ns)
        smi = mono.findtext("lmr:MonomerSmiles", namespaces=ns) or ""
        mtype = mono.findtext("lmr:MonomerType", namespaces=ns) or ""
        if mid:
            lib[mid.strip()] = {"smiles": smi.strip(), "type": mtype.strip()}
    return lib


def load_cycpept():
    lib = {}
    with open(CYCPEPT_CSV) as f:
        for row in csv.DictReader(f):
            sym = row["Symbol"].strip()
            smi = (row.get("CXSMILES", "") or "").strip()
            mtype = row.get("Monomer_Type", "").strip()
            lib[sym] = {"smiles": smi, "type": mtype}
    return lib


def invert_stereo(smi: str) -> str:
    """Invert all stereocenters: @@ <-> @"""
    mol = Chem.MolFromSmiles(smi.split("|")[0].strip())
    if mol is None:
        return smi
    for atom in mol.GetAtoms():
        chi = atom.GetChiralTag()
        if chi == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif chi == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return Chem.MolToSmiles(mol)


def n_methylate(smi: str) -> str:
    """Add methyl to backbone nitrogen (the N bonded to a dummy [*] that has H)."""
    core = smi.split("|")[0].strip()
    mol = Chem.MolFromSmiles(core)
    if mol is None:
        return smi

    mol_rw = Chem.RWMol(mol)
    for atom in mol_rw.GetAtoms():
        if atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() > 0:
            if any(n.GetAtomicNum() == 0 for n in atom.GetNeighbors()):
                idx = mol_rw.AddAtom(Chem.Atom(6))
                mol_rw.AddBond(atom.GetIdx(), idx, Chem.BondType.SINGLE)
                break

    try:
        Chem.SanitizeMol(mol_rw)
        return Chem.MolToSmiles(mol_rw)
    except Exception:
        return smi


def base_aa(symbol: str) -> str:
    """Extract base amino acid letter from symbol."""
    if symbol == "Sar":
        return "G"
    m = re.search(r"[A-Z]$", symbol)
    return m.group(0) if m else ""


def describe(symbol: str) -> str:
    aa = base_aa(symbol)
    name = AA_NAMES.get(aa, aa)
    if symbol == "Sar":
        return "Sarcosine (N-methyl Glycine)"
    if symbol.startswith("Me_d"):
        return f"N-methyl D-{name}"
    if symbol.startswith("me"):
        return f"N-methyl L-{name}"
    if symbol.startswith("d"):
        return f"D-{name}"
    return f"L-{name}"


def main():
    chembl = load_chembl()
    cycpept = load_cycpept()

    rows = []
    for sym in CREMP_MONOMERS:
        source = ""
        smi = ""
        mtype = "Backbone"

        # 1. Try ChEMBL
        if sym in chembl:
            smi = chembl[sym]["smiles"]
            mtype = chembl[sym]["type"]
            source = "ChEMBL"
        # 2. Try CycPeptMPDB
        elif sym in cycpept:
            smi = cycpept[sym]["smiles"]
            mtype = cycpept[sym]["type"]
            source = "CycPeptMPDB"
        # 3. Generate
        else:
            aa = base_aa(sym)
            if sym.startswith("Me_d"):
                # N-methyl D-AA: take L-AA, invert stereo, N-methylate
                if aa in chembl:
                    base_smi = chembl[aa]["smiles"]
                    inv = invert_stereo(base_smi)
                    smi = n_methylate(inv)
                    source = "generated(invert+methylate)"
            elif sym.startswith("me"):
                if aa in chembl:
                    base_smi = chembl[aa]["smiles"]
                    smi = n_methylate(base_smi.split("|")[0].strip())
                    source = "generated(methylate)"
            elif sym.startswith("d"):
                if aa in chembl:
                    base_smi = chembl[aa]["smiles"]
                    smi = invert_stereo(base_smi)
                    source = "generated(invert)"

        rows.append({
            "symbol": sym,
            "name": describe(sym),
            "natural_analog": base_aa(sym),
            "smiles": smi,
            "monomer_type": mtype,
            "source": source,
        })

    with open(OUTPUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Summary
    sources = {}
    for r in rows:
        s = r["source"]
        sources[s] = sources.get(s, 0) + 1

    print(f"Output: {OUTPUT}")
    print(f"Total monomers: {len(rows)}")
    for s, c in sorted(sources.items()):
        print(f"  {s}: {c}")

    missing = [r for r in rows if not r["smiles"]]
    if missing:
        print(f"\nMissing SMILES: {[r['symbol'] for r in missing]}")
    else:
        print("\nAll monomers have SMILES.")


if __name__ == "__main__":
    main()
