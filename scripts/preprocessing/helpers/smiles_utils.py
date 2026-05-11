"""SMILES canonicalization utilities.

Rewrites SMILES strings into RDKit canonical form. The molecule
itself is preserved exactly as registered — no salt strip, no charge
neutralization, no isotope strip, no tautomer canonicalization. Only
the *string representation* is canonicalized so that lexically
different inputs for the same molecule collapse to byte-identical
output.

Reproducibility note: canonical SMILES depends on the bundled RDKit
version. Pin via ``environment.yml`` (rdkit=2023.03.3) when reproducing
the warehouse.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import pandas as pd
from rdkit import Chem


logger = logging.getLogger(__name__)


@dataclass
class StandardizeStats:
    """Per-row diagnostics for SMILES canonicalization."""
    parsed: int = 0
    parse_failed: int = 0
    unchanged: int = 0
    total: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "parsed": self.parsed,
            "parse_failed": self.parse_failed,
            "unchanged": self.unchanged,
        }


def standardize_smiles(smi: str) -> str:
    """Canonicalize a single SMILES string. Returns ``""`` on failure.

    Pure function; safe to call concurrently. Idempotent: calling twice
    on the same input yields the same output.
    """
    if smi is None or smi == "" or pd.isna(smi):
        return ""
    try:
        mol = Chem.MolFromSmiles(str(smi))
    except Exception:
        return ""
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def standardize_series(
    s: pd.Series,
    log: logging.Logger | None = None,
    label: str = "smiles",
) -> tuple[pd.Series, StandardizeStats]:
    """Canonicalize a Series of SMILES.

    Returns ``(normalized_smiles_series, stats)``. The output Series
    shares the input's index. Stats are emitted as a single info-level
    log line.
    """
    _log = log or logger
    stats = StandardizeStats(total=len(s))

    out: list[str] = []
    for v in s.astype(object).tolist():
        if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
            stats.parse_failed += 1
            out.append("")
            continue

        raw = str(v)
        try:
            mol = Chem.MolFromSmiles(raw)
        except Exception:
            mol = None
        if mol is None:
            stats.parse_failed += 1
            out.append("")
            continue
        stats.parsed += 1

        try:
            normalized = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            stats.parse_failed += 1
            out.append("")
            continue

        if normalized == raw:
            stats.unchanged += 1
        out.append(normalized)

    _log.info(
        "SMILES canonicalize [%s]: total=%d parsed=%d parse_fail=%d unchanged=%d",
        label,
        stats.total,
        stats.parsed,
        stats.parse_failed,
        stats.unchanged,
    )
    return cast(pd.Series, pd.Series(out, index=s.index, dtype=object)), stats
