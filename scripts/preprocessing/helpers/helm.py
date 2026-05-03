"""Shared HELM monomer mapping utilities for preprocessing scripts.

Provides alt_symbol -> canonical remapping and monomer validation
using the project monomer library.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from scripts.preprocessing.helpers.paths import MONOMER_LIBRARY_PATH

logger = logging.getLogger(__name__)
DEFAULT_MONOMER_LIBRARY = MONOMER_LIBRARY_PATH


def load_monomer_library(
    path: str | Path = DEFAULT_MONOMER_LIBRARY,
    log: logging.Logger | None = None,
) -> tuple[set[str], dict[str, str]]:
    """Load monomer library and build valid symbol set + alt->canonical map."""
    _log = log or logger
    df = pd.read_csv(path)
    valid_symbols = {str(symbol).strip() for symbol in df["symbol"].dropna().tolist()}
    alt_to_canonical: dict[str, str] = {}
    for _, row in df.iterrows():
        alt = str(row.get("alt_symbols", "")).strip()
        if alt and alt != "nan":
            for a in alt.split(";"):
                a = a.strip()
                if a:
                    alt_to_canonical[a] = str(row["symbol"]).strip()
    _log.info(
        f"Monomer library: {len(valid_symbols)} symbols, {len(alt_to_canonical)} alt mappings"
    )
    return valid_symbols, alt_to_canonical


def parse_helm_monomers(helm: str) -> list[str] | None:
    """Extract monomer symbols from all PEPTIDE chains in HELM notation."""
    if not helm or pd.isna(helm):
        return None
    helm = str(helm).strip()
    matches = re.findall(r"PEPTIDE\d*\{([^}]+)\}", helm)
    if not matches:
        return None
    monomers = []
    for seq in matches:
        for token in seq.split("."):
            token = token.strip()
            if not token:
                continue
            if token.startswith("[") and token.endswith("]"):
                token = token[1:-1]
            monomers.append(token)
    return monomers


def normalize_single_chain_helm(helm: str) -> str:
    """Renumber single-chain PEPTIDEn identifiers to PEPTIDE1."""
    if not helm or pd.isna(helm):
        return helm
    chain_ids = set(re.findall(r"PEPTIDE\d+", str(helm)))
    if len(chain_ids) != 1:
        return str(helm)
    chain_id = next(iter(chain_ids))
    if chain_id == "PEPTIDE1":
        return str(helm)
    return str(helm).replace(chain_id, "PEPTIDE1")


_CONNECTION_TOKEN_RE = re.compile(
    r"([A-Z]+)(\d+),([A-Z]+)(\d+),(\d+):R(\d+)-(\d+):R(\d+)"
)


def canonicalize_connections(helm: str) -> str:
    """Canonicalize connection orientation in HELM `$...$` section.

    For each `chainA,chainB,posA:Rx-posB:Ry` token, swap source and target
    so the (chain_type, chain_idx, residue_pos, R_idx) tuple on the left is
    the smaller one. Resolves convention differences such as ChEMBL's
    `N:R2-1:R1` vs CycPeptMPDB's `1:R1-N:R2` for the same H2T-cyclic
    molecule, and generalizes to any R pair (R1-R3 lariat, R3-R3 disulfide)
    and any HELM polymer type (PEPTIDE, RNA, CHEM, BLOB, ...).

    Tokens that don't match the standard `<TYPE><n>,<TYPE><n>,p:Rx-p:Ry`
    form (e.g. wildcard positions `?:R1-3:R2`, ambiguous bond syntax with
    `+`/`,`) are left untouched — the function never produces an invalid
    HELM string from valid input.
    """
    if not helm or pd.isna(helm):
        return helm
    s = str(helm)
    parts = s.split("$")
    if len(parts) < 2 or not parts[1].strip():
        return s

    def _canon(m: re.Match) -> str:
        type_a, idx_a, type_b, idx_b, pos_a, r_a, pos_b, r_b = m.groups()
        key_a = (type_a, int(idx_a), int(pos_a), int(r_a))
        key_b = (type_b, int(idx_b), int(pos_b), int(r_b))
        if key_a > key_b:
            return f"{type_b}{idx_b},{type_a}{idx_a},{pos_b}:R{r_b}-{pos_a}:R{r_a}"
        return m.group(0)

    parts[1] = _CONNECTION_TOKEN_RE.sub(_canon, parts[1])
    return "$".join(parts)


def remap_helm(helm: str, alt_to_canonical: dict) -> str:
    """Remap alt_symbols in HELM notation to canonical symbols."""
    if not alt_to_canonical:
        return helm

    def _replace_in_sequence(match):
        seq = match.group(1)
        prefix = match.group(0)[: match.group(0).index("{") + 1]
        tokens = seq.split(".")
        new_tokens = []
        for t in tokens:
            t_stripped = t.strip()
            is_bracketed = t_stripped.startswith("[") and t_stripped.endswith("]")
            inner = t_stripped[1:-1] if is_bracketed else t_stripped
            if inner in alt_to_canonical:
                canonical = alt_to_canonical[inner]
                new_tokens.append(
                    f"[{canonical}]" if is_bracketed or len(canonical) > 1 else canonical
                )
            else:
                new_tokens.append(t_stripped)
        return prefix + ".".join(new_tokens) + "}"

    return re.sub(r"PEPTIDE\d*\{([^}]+)\}", _replace_in_sequence, helm)


def validate_helm_monomers(helm: str, valid_symbols: set) -> bool:
    """Check if all monomers in HELM are in the valid symbol set."""
    monomers = parse_helm_monomers(helm)
    if monomers is None:
        return False
    return all(m in valid_symbols for m in monomers)


def apply_helm_normalization(
    df: pd.DataFrame,
    helm_col: str,
    valid_symbols: set,
    alt_to_canonical: dict,
    log: logging.Logger | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Normalize HELM strings and validate monomers."""
    _log = log or logger
    original_size = len(df)

    df = df.loc[df[helm_col].notna()].copy()
    df = df.loc[df[helm_col].astype(str).str.strip() != ""].copy()
    null_removed = original_size - len(df)

    renumbered_count = 0
    remapped_count = 0
    connection_swapped_count = 0
    new_helms = []
    for h in df[helm_col]:
        normalized = normalize_single_chain_helm(str(h))
        if normalized != str(h):
            renumbered_count += 1
        remapped = remap_helm(normalized, alt_to_canonical)
        if remapped != normalized:
            remapped_count += 1
        canonicalized = canonicalize_connections(remapped)
        if canonicalized != remapped:
            connection_swapped_count += 1
        new_helms.append(canonicalized)
    df[helm_col] = new_helms

    valid_mask = df[helm_col].map(lambda h: validate_helm_monomers(h, valid_symbols))
    invalid_count = int((~valid_mask).sum())
    df = df.loc[valid_mask].copy()

    stats = {
        "original": original_size,
        "null_removed": null_removed,
        "renumbered": renumbered_count,
        "remapped": remapped_count,
        "connection_swapped": connection_swapped_count,
        "invalid_removed": invalid_count,
        "after_mapping": len(df),
    }

    _log.info(
        f"  HELM mapping: {original_size} -> {len(df)} "
        f"(null: -{null_removed}, renumber: {renumbered_count}, "
        f"remap: {remapped_count}, conn_swap: {connection_swapped_count}, "
        f"invalid: -{invalid_count})"
    )

    return df, stats


apply_helm_mapping = apply_helm_normalization
