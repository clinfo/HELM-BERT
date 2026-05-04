"""SMILES standardization utilities.

Pipeline (deterministic, idempotent, stereo-preserving):

    LargestFragmentChooser  — strip salts / counter-ions / minor fragments
    Uncharger               — neutralize protonation states where possible
    isotope strip           — set every atom isotope to 0 (loses [2H] etc.)
    RemoveHs                — drop explicit hydrogens left by isotope strip
    canonical SMILES        — Chem.MolToSmiles(canonical=True, isomericSmiles=True)
    InChIKey                — Chem.MolToInchiKey(mol)   (separate output column)

Tautomer normalization is delegated to InChI's mobile-H layer rather
than RDKit's ``TautomerEnumerator`` because:

    * InChI canonicalizes only well-defined mobile-H tautomers
      (histidine, guanidinium, amide/imidic-acid). It is conservative
      and does not over-merge — preserving truly distinct molecules.
    * InChI is orders of magnitude faster on macrocyclic peptides
      (microseconds vs. seconds per molecule). ``TautomerEnumerator``
      hits ``MaxTransforms`` and emits warnings on ~50-atom peptides.
    * InChIKey is a standard chemical identifier (IUPAC / NIST), useful
      to downstream tooling.

InChIKey is the dedup grouping key in 09_dedup.py; ``normalized_smiles``
remains a human-readable display column.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


logger = logging.getLogger(__name__)


# Stateless workers, instantiated once at import for performance.
_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


@dataclass
class StandardizeStats:
    """Per-row diagnostics for SMILES standardization."""
    parsed: int = 0
    parse_failed: int = 0
    fragment_stripped: int = 0
    uncharged: int = 0
    isotope_stripped: int = 0
    inchikey_failed: int = 0
    fragment_ratio_below_2: int = 0  # potentially ambiguous main fragment
    unchanged: int = 0
    total: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "parsed": self.parsed,
            "parse_failed": self.parse_failed,
            "fragment_stripped": self.fragment_stripped,
            "uncharged": self.uncharged,
            "isotope_stripped": self.isotope_stripped,
            "inchikey_failed": self.inchikey_failed,
            "fragment_ratio_below_2": self.fragment_ratio_below_2,
            "unchanged": self.unchanged,
        }


def _largest_fragment_ratio(smi: str) -> float:
    """Heavy-atom ratio of largest to second-largest fragment in a SMILES.

    Returns ``inf`` for single-component SMILES. Used to flag ambiguous
    multi-component cases where LargestFragmentChooser's pick may not be
    the molecule of interest.
    """
    if "." not in smi:
        return float("inf")
    sizes: list[int] = []
    for part in smi.split("."):
        m = Chem.MolFromSmiles(part)
        if m is not None:
            sizes.append(m.GetNumHeavyAtoms())
    sizes.sort(reverse=True)
    if len(sizes) < 2 or sizes[1] == 0:
        return float("inf")
    return sizes[0] / sizes[1]


def _standardize_mol(smi: str):
    """Internal: parse + apply the standardization pipeline. Returns
    the standardized rdkit Mol or None on failure.
    """
    if smi is None or smi == "" or pd.isna(smi):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smi))
    except Exception:
        return None
    if mol is None:
        return None

    mol = _LARGEST_FRAGMENT.choose(mol)
    mol = _UNCHARGER.uncharge(mol)
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    mol = Chem.RemoveHs(mol)
    return mol


def standardize_smiles(smi: str) -> str:
    """Standardize a single SMILES string. Returns ``""`` on failure.

    Pure function; safe to call concurrently. Idempotent: calling twice
    on the same input yields the same output.
    """
    mol = _standardize_mol(smi)
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def compute_inchikey(smi: str) -> str:
    """Compute the InChIKey of a (pre-standardized) SMILES string.

    Returns ``""`` on failure. Used by 09_dedup.py as the canonical
    same-molecule grouping key — InChI's mobile-H layer normalizes
    common tautomers (histidine, guanidinium, amide/imidic-acid)
    automatically, while preserving stereo (D/L, E/Z) in the second
    block of the key.

    Always pair with :func:`standardize_smiles` upstream — InChI does
    *not* strip salts (`Mol.[Na+]`) or isotope labels.
    """
    mol = _standardize_mol(smi)
    if mol is None:
        return ""
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return ""


def standardize_series(
    s: pd.Series,
    log: logging.Logger | None = None,
    label: str = "smiles",
) -> tuple[pd.Series, pd.Series, StandardizeStats]:
    """Standardize a Series of SMILES and compute InChIKeys in one pass.

    Returns ``(normalized_smiles_series, inchikey_series, stats)``. Both
    output Series share the input's index. Stats are emitted as a
    single info-level log line.

    Computing both outputs in one pass amortizes the cost of parsing +
    standardization (the slow step) — Mol object built once, reused for
    both SMILES canonicalization and InChIKey generation.
    """
    _log = log or logger
    stats = StandardizeStats(total=len(s))

    smiles_out: list[str] = []
    inchikey_out: list[str] = []
    for v in s.astype(object).tolist():
        if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
            stats.parse_failed += 1
            smiles_out.append("")
            inchikey_out.append("")
            continue

        raw = str(v)
        try:
            mol = Chem.MolFromSmiles(raw)
        except Exception:
            mol = None
        if mol is None:
            stats.parse_failed += 1
            smiles_out.append("")
            inchikey_out.append("")
            continue
        stats.parsed += 1

        if "." in raw:
            ratio = _largest_fragment_ratio(raw)
            if ratio < 2.0:
                stats.fragment_ratio_below_2 += 1

        mol_after_lf = _LARGEST_FRAGMENT.choose(mol)
        if mol_after_lf.GetNumHeavyAtoms() != mol.GetNumHeavyAtoms():
            stats.fragment_stripped += 1
        mol = mol_after_lf

        smi_before_uncharge = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        mol = _UNCHARGER.uncharge(mol)
        smi_after_uncharge = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if smi_after_uncharge != smi_before_uncharge:
            stats.uncharged += 1

        if any(a.GetIsotope() != 0 for a in mol.GetAtoms()):
            stats.isotope_stripped += 1
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)
        mol = Chem.RemoveHs(mol)

        try:
            normalized = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            stats.parse_failed += 1
            smiles_out.append("")
            inchikey_out.append("")
            continue

        try:
            inchikey = Chem.MolToInchiKey(mol)
        except Exception:
            inchikey = ""
        if not inchikey:
            stats.inchikey_failed += 1

        if normalized == raw:
            stats.unchanged += 1
        smiles_out.append(normalized)
        inchikey_out.append(inchikey)

    _log.info(
        "SMILES normalize [%s]: total=%d parsed=%d parse_fail=%d "
        "frag_strip=%d uncharged=%d isotope_strip=%d inchikey_fail=%d "
        "ratio_lt_2=%d unchanged=%d",
        label,
        stats.total,
        stats.parsed,
        stats.parse_failed,
        stats.fragment_stripped,
        stats.uncharged,
        stats.isotope_stripped,
        stats.inchikey_failed,
        stats.fragment_ratio_below_2,
        stats.unchanged,
    )
    smiles_series = cast(pd.Series, pd.Series(smiles_out, index=s.index, dtype=object))
    inchikey_series = cast(pd.Series, pd.Series(inchikey_out, index=s.index, dtype=object))
    return smiles_series, inchikey_series, stats
