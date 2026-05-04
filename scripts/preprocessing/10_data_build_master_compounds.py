#!/usr/bin/env python3
"""Build master compound table across all warehouse compound sources.

Equivalence relation (permissive — Option B):
    Two compounds belong to the same master entity if EITHER their HELM
    matches OR their normalized_smiles matches. Union-find over the
    bipartite graph (HELM nodes ∪ SMILES nodes) yields connected
    components; each component = one master_compound_id.

    This catches both isotope-only HELM-equivalents (different SMILES,
    same HELM — e.g. tritium-labelled variants in chembl_ppi) and
    SMILES-canonical-equivalents whose HELM happens to differ.

Reads (from 05_final/):
    chembl_compounds.csv
    propedia_compounds.csv
    cycpept_permeability_compounds.csv
    cremp_conformer_compounds.csv

Writes:
    05_final/master_compounds.csv with columns:
        master_compound_id              MC0000001 ...
        canonical_helm                  lex-smallest HELM in component
        canonical_normalized_smiles     lex-smallest SMILES in component
        source_datasets                 pipe-joined sorted set
        n_helm_variants                 |HELM aliases|
        n_smiles_variants               |SMILES aliases|
        helm_aliases                    ';'-joined sorted aliases
        smiles_aliases                  ';'-joined sorted aliases

Usage:
    python 10_data_build_master_compounds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers.logging_utils import setup_logger
from helpers.paths import FINAL_DIR


SOURCES: dict[str, tuple[str, str]] = {
    # source key -> (helm column, smiles column).  All four feed the master table.
    "chembl_compounds":               ("helm_notation", "normalized_smiles"),
    "propedia_compounds":             ("Peptide_HELM",  "normalized_smiles"),
    "cycpept_permeability_compounds": ("HELM",          "normalized_smiles"),
    "cremp_conformer_compounds":      ("helm",          "normalized_smiles"),
}


class UnionFind:
    """Tiny union-find. Nodes are arbitrary hashables."""
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        # path compression
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def load_pairs(log) -> list[tuple[str, str, str]]:
    """Return list of (source, helm, smiles) for every compound row.

    All four configured sources MUST be present and have the expected
    HELM/SMILES columns. The master table is meant to be the canonical
    cross-dataset identifier; silently building it from a subset of
    sources would yield incorrect master_compound_ids that downstream
    joins would treat as authoritative. Fail loud instead.
    """
    pairs: list[tuple[str, str, str]] = []
    for src, (h_col, s_col) in SOURCES.items():
        path = FINAL_DIR / f"{src}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"required source missing: {path}. "
                "All compound sources must be built before master_compounds."
            )
        df = pd.read_csv(path, low_memory=False)
        missing_cols = [c for c in (h_col, s_col) if c not in df.columns]
        if missing_cols:
            raise KeyError(
                f"[{src}] required columns missing: {missing_cols}. "
                f"Available columns: {list(df.columns)[:20]}"
            )
        # Drop rows missing both keys (no way to anchor them); keep rows
        # with one key — they still seed their own component via that key.
        sub = df[[h_col, s_col]].copy()
        sub.columns = ["helm", "smiles"]
        sub = sub.dropna(how="all")
        sub["helm"] = sub["helm"].astype(str).where(sub["helm"].notna(), "")
        sub["smiles"] = sub["smiles"].astype(str).where(sub["smiles"].notna(), "")
        for h, s in zip(sub["helm"], sub["smiles"]):
            pairs.append((src, h, s))
        log.info("[%s] %d rows ingested", src, len(sub))
    return pairs


def main() -> int:
    logger, _ = setup_logger("master_compounds")

    pairs = load_pairs(logger)
    if not pairs:
        logger.error("no input pairs found — aborting")
        return 2

    # Build union-find over bipartite (HELM, SMILES) nodes. Empty strings
    # are treated as "no anchor" — they don't participate in their side
    # of the union, but the row's other key still seeds a component.
    uf = UnionFind()
    for _, h, s in pairs:
        h_node = ("h", h) if h else None
        s_node = ("s", s) if s else None
        if h_node is not None:
            uf.find(h_node)
        if s_node is not None:
            uf.find(s_node)
        if h_node is not None and s_node is not None:
            uf.union(h_node, s_node)

    # Collect each component's HELM aliases, SMILES aliases, and the set
    # of source datasets that contributed any row to it.
    components: dict = {}  # root -> dict(helms, smiles, sources)
    for src, h, s in pairs:
        anchors = []
        if h:
            anchors.append(("h", h))
        if s:
            anchors.append(("s", s))
        if not anchors:
            continue
        root = uf.find(anchors[0])
        comp = components.setdefault(root, {"helms": set(), "smiles": set(), "sources": set()})
        if h:
            comp["helms"].add(h)
        if s:
            comp["smiles"].add(s)
        comp["sources"].add(src)

    # Deterministic ordering: sort by (canonical_helm, canonical_smiles)
    rows: list[dict] = []
    for comp in components.values():
        canon_helm   = min(comp["helms"])  if comp["helms"]  else ""
        canon_smiles = min(comp["smiles"]) if comp["smiles"] else ""
        rows.append({
            "canonical_helm": canon_helm,
            "canonical_normalized_smiles": canon_smiles,
            "source_datasets": "|".join(sorted(comp["sources"])),
            "n_helm_variants": len(comp["helms"]),
            "n_smiles_variants": len(comp["smiles"]),
            "helm_aliases": ";".join(sorted(comp["helms"])),
            "smiles_aliases": ";".join(sorted(comp["smiles"])),
        })
    rows.sort(key=lambda r: (r["canonical_helm"], r["canonical_normalized_smiles"]))

    # Assign stable master IDs after sorting so reruns are bit-identical.
    pad = max(7, len(str(len(rows))))
    out = pd.DataFrame(rows)
    out.insert(0, "master_compound_id", [f"MC{i+1:0{pad}d}" for i in range(len(out))])

    out_path = FINAL_DIR / "master_compounds.csv"
    out.to_csv(out_path, index=False)

    # Summary stats — what did the equivalence relation actually collapse?
    naive_helm_count = sum(len(c["helms"]) for c in components.values())
    naive_smi_count = sum(len(c["smiles"]) for c in components.values())
    multi_helm = sum(1 for c in components.values() if len(c["helms"]) > 1)
    multi_smi = sum(1 for c in components.values() if len(c["smiles"]) > 1)
    multi_source = sum(1 for c in components.values() if len(c["sources"]) > 1)

    logger.info("=" * 60)
    logger.info("Master compound table built")
    logger.info("=" * 60)
    logger.info("entries (master_compound_id):       %d", len(out))
    logger.info("HELM aliases collapsed:             %d -> %d", naive_helm_count, len(out))
    logger.info("SMILES aliases collapsed:           %d -> %d", naive_smi_count,  len(out))
    logger.info("master entries with >1 HELM:        %d", multi_helm)
    logger.info("master entries with >1 SMILES:      %d", multi_smi)
    logger.info("master entries spanning >1 source:  %d", multi_source)
    logger.info("written: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
