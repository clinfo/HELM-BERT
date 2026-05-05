"""HELM string normalization utilities.

Operates purely on HELM strings (no chemistry). Each transformation is a
pure function of (helm, library) so the pipeline is fully deterministic
and idempotent: f(f(x)) == f(x) for every public function.

Pipeline (apply_helm_normalization):
    1. normalize_single_chain   — PEPTIDEn -> PEPTIDE1 *only* if exactly one
                                  PEPTIDE chain is present (multi-chain safe).
    2. remap_alt_symbols        — alt monomer symbols -> canonical via library.
    3. canonicalize_connections — undirected bond endpoints sorted.
                                  Safe for both single- and multi-chain.
    4. validate_monomers        — drop rows whose monomers aren't in library.

Cyclic residue rotation is intentionally NOT done here; canonical SMILES
captures rotational equivalence at the chemistry layer (see smiles_utils).
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import cast

import pandas as pd


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monomer library
# ---------------------------------------------------------------------------

def load_monomer_library(
    path: str | Path,
    log: logging.Logger | None = None,
) -> tuple[set[str], dict[str, str]]:
    """Load canonical monomer symbols + alt-symbol mapping from the library CSV.

    The library CSV has columns ``symbol`` (canonical) and ``alt_symbols``
    (pipe-delimited synonyms). Every row's canonical symbol is included in
    the valid set; each alt symbol maps to that row's canonical.

    Returns:
        (valid_symbols, alt_to_canonical)
    """
    _log = log or logger
    df = cast(pd.DataFrame, pd.read_csv(path))

    valid_symbols: set[str] = {
        str(s).strip() for s in df["symbol"].dropna().tolist() if str(s).strip()
    }

    alt_to_canonical: dict[str, str] = {}
    if "alt_symbols" in df.columns:
        for _, row in df.iterrows():
            canonical = str(row["symbol"]).strip()
            alts = row.get("alt_symbols")
            if pd.isna(alts) or not str(alts).strip():
                continue
            for a in str(alts).split("|"):
                a_clean = a.strip()
                if a_clean and a_clean != canonical:
                    alt_to_canonical[a_clean] = canonical

    _log.info(
        "Monomer library: %d canonical symbols, %d alt mappings (%s)",
        len(valid_symbols),
        len(alt_to_canonical),
        path,
    )
    return valid_symbols, alt_to_canonical


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_PEPTIDE_BLOCK_RE = re.compile(r"PEPTIDE\d*\{([^}]+)\}")
_CHAIN_ID_RE = re.compile(r"PEPTIDE\d+")


def parse_helm_monomers(helm: str) -> list[str] | None:
    """Extract monomer symbols from every PEPTIDE chain in a HELM string.

    Returns ``None`` when no PEPTIDE block is present (the string is not
    a peptide HELM, or is malformed).
    """
    if not helm or pd.isna(helm):
        return None
    matches = _PEPTIDE_BLOCK_RE.findall(str(helm))
    if not matches:
        return None
    monomers: list[str] = []
    for seq in matches:
        for token in seq.split("."):
            t = token.strip()
            if not t:
                continue
            if t.startswith("[") and t.endswith("]"):
                t = t[1:-1]
            monomers.append(t)
    return monomers


def get_peptide_chain_ids(helm: str) -> set[str]:
    """Return the set of distinct ``PEPTIDEn`` chain identifiers in a HELM string."""
    if not helm or pd.isna(helm):
        return set()
    return set(_CHAIN_ID_RE.findall(str(helm)))


# ---------------------------------------------------------------------------
# Step 1 — single-chain renumber (multi-chain SAFE: skipped)
# ---------------------------------------------------------------------------

def normalize_single_chain(helm: str) -> str:
    """Renumber a single PEPTIDE chain to PEPTIDE1.

    No-op (returns input unchanged) when:
      - input is empty / NaN
      - more than one distinct PEPTIDEn chain identifier exists (multi-chain
        molecules, where renaming would collapse two chains into one)
      - the only chain is already named PEPTIDE1.

    Uses a numeric-boundary aware regex so PEPTIDE2 never partially matches
    PEPTIDE20 etc. (defensive — multi-digit conflicts also trip the
    multi-chain guard above).
    """
    if not helm or pd.isna(helm):
        return helm
    s = str(helm)
    chain_ids = _CHAIN_ID_RE.findall(s)
    distinct = set(chain_ids)
    if len(distinct) != 1:
        return s
    chain_id = next(iter(distinct))
    if chain_id == "PEPTIDE1":
        return s
    # Replace only when not followed by another digit (e.g. PEPTIDE2 must not
    # consume PEPTIDE20). Suffix-anchored to avoid trailing-digit ambiguity.
    return re.sub(rf"{re.escape(chain_id)}(?!\d)", "PEPTIDE1", s)


# ---------------------------------------------------------------------------
# Step 2 — alt-symbol remap
# ---------------------------------------------------------------------------

def remap_alt_symbols(helm: str, alt_to_canonical: dict[str, str]) -> str:
    """Replace alt monomer symbols with their canonical symbol.

    Only acts inside ``PEPTIDE\\d*{...}`` blocks. Unrecognized tokens pass
    through unchanged. Bracketed/non-bracketed forms are both handled, and
    output bracketing is chosen so multi-character canonicals are bracketed.
    """
    if not alt_to_canonical or not helm or pd.isna(helm):
        return helm

    def _replace_block(match: re.Match) -> str:
        seq = match.group(1)
        prefix = match.group(0)[: match.group(0).index("{") + 1]
        new_tokens: list[str] = []
        for raw in seq.split("."):
            t = raw.strip()
            is_bracketed = t.startswith("[") and t.endswith("]")
            inner = t[1:-1] if is_bracketed else t
            if inner in alt_to_canonical:
                canonical = alt_to_canonical[inner]
                if is_bracketed or len(canonical) > 1:
                    new_tokens.append(f"[{canonical}]")
                else:
                    new_tokens.append(canonical)
            else:
                new_tokens.append(t)
        return prefix + ".".join(new_tokens) + "}"

    return _PEPTIDE_BLOCK_RE.sub(_replace_block, str(helm))


# ---------------------------------------------------------------------------
# Step 3 — connection orientation canonicalize
# ---------------------------------------------------------------------------

# Strict form: <TYPE><n>,<TYPE><n>,p:Rx-p:Ry
# Wildcard / ambiguous tokens (`?:R1-3:R2`, `1+2:R1`) intentionally don't
# match — they pass through untouched.
_CONNECTION_TOKEN_RE = re.compile(
    r"([A-Z]+)(\d+),([A-Z]+)(\d+),(\d+):R(\d+)-(\d+):R(\d+)"
)


def canonicalize_connections(helm: str) -> str:
    """Sort each connection-token's two endpoints into canonical order.

    HELM bonds are undirected; ``A,B,p:R1-q:R2`` and ``B,A,q:R2-p:R1``
    encode the same bond. We choose the form whose left endpoint
    ``(chain_type, chain_idx, position, R_idx)`` tuple is lexicographically
    smaller. This resolves cross-source convention differences (ChEMBL's
    ``N:R2-1:R1`` vs CycPeptMPDB's ``1:R1-N:R2``) and is safe for both
    single-chain and multi-chain (homo/heterodimer) molecules — the chain
    block contents are never touched.
    """
    if not helm or pd.isna(helm):
        return helm
    s = str(helm)
    parts = s.split("$")
    # Connection list lives in parts[1]; if missing or empty, nothing to do.
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


# ---------------------------------------------------------------------------
# Step 4 — monomer validation
# ---------------------------------------------------------------------------

def validate_monomers(helm: str, valid_symbols: set[str]) -> bool:
    """Return True iff every PEPTIDE-chain monomer is in ``valid_symbols``.

    HELM strings without any PEPTIDE block return False (cannot validate
    a non-peptide entry against the peptide library).

    Note: monomers inside non-PEPTIDE blocks (CHEM, RNA, BLOB, ...) are
    *not* checked because the warehouse library is peptide-scoped. None
    of the warehouse datasets currently contain those polymer types; if
    they're added, extend this function.
    """
    monomers = parse_helm_monomers(helm)
    if monomers is None:
        return False
    return all(m in valid_symbols for m in monomers)


# ---------------------------------------------------------------------------
# Hand-curated HELM token corrections
# ---------------------------------------------------------------------------

def load_helm_corrections(
    path: str | Path,
    log: logging.Logger | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Load per-compound HELM token rewrites from a CSV.

    Returns a mapping ``{compound_id: [(original_token, corrected_token), ...]}``
    where each tuple is a literal token replacement applied to that
    compound's HELM string. Returns ``{}`` if the file is absent (the
    file is optional — pipelines run cleanly without curation).

    Schema (lines starting with ``#`` are skipped as comments):
        compound_chembl_id,original_token,corrected_token,added_by,reason
    """
    _log = log or logger
    p = Path(path)
    if not p.exists():
        _log.info("HELM corrections: %s not found, skipping", p)
        return {}

    corrections: dict[str, list[tuple[str, str]]] = {}
    with open(p) as f:
        rows = list(csv.DictReader(line for line in f if not line.lstrip().startswith("#")))
    for row in rows:
        cid = (row.get("compound_chembl_id") or "").strip()
        orig = (row.get("original_token") or "").strip()
        new = (row.get("corrected_token") or "").strip()
        if not (cid and orig and new):
            continue
        corrections.setdefault(cid, []).append((orig, new))
    n_pairs = sum(len(v) for v in corrections.values())
    _log.info(
        "HELM corrections: %d compounds, %d token rewrites loaded from %s",
        len(corrections),
        n_pairs,
        p,
    )
    return corrections


def apply_helm_corrections(
    df: pd.DataFrame,
    helm_col: str,
    corrections: dict[str, list[tuple[str, str]]],
    id_cols: tuple[str, ...] = ("compound_chembl_id",),
    log: logging.Logger | None = None,
) -> tuple[pd.DataFrame, int]:
    """Apply per-compound HELM token rewrites in place on a DataFrame copy.

    Uses ``id_cols[0]`` as the lookup key. Applied as plain string
    replacement on the ``helm_col`` column — every occurrence of
    ``original_token`` is rewritten. Misses (compound id absent from
    this dataset, or token absent from a found compound's HELM) are
    logged as warnings — these usually indicate the corrections file
    is stale relative to the source data.

    Returns ``(new_df, n_rows_affected)``. ``n_rows_affected`` counts
    each touched row at most once even if multiple token rewrites land
    on the same row.
    """
    _log = log or logger
    if not corrections:
        return df, 0

    primary_id = id_cols[0]
    if primary_id not in df.columns:
        _log.info(
            "HELM corrections: id column %r absent from this dataset, skipping",
            primary_id,
        )
        return df, 0

    work = df.copy()
    touched_idx: set = set()
    compound_misses: list[str] = []
    token_misses: list[str] = []
    for cid, rewrites in corrections.items():
        mask = work[primary_id] == cid
        if not mask.any():
            compound_misses.append(cid)
            continue
        for orig, new in rewrites:
            sub_mask = mask & work[helm_col].astype(str).str.contains(
                re.escape(orig), regex=True
            )
            if not sub_mask.any():
                token_misses.append(f"{cid}: {orig}")
                continue
            work.loc[sub_mask, helm_col] = (
                work.loc[sub_mask, helm_col].astype(str).str.replace(orig, new, regex=False)
            )
            touched_idx.update(work.index[sub_mask].tolist())

    if compound_misses:
        _log.warning(
            "HELM corrections: %d compound_id(s) listed in corrections file "
            "have no rows in this dataset (file may be stale). Examples: %s",
            len(compound_misses),
            compound_misses[:5],
        )
    if token_misses:
        _log.warning(
            "HELM corrections: %d (compound, token) pairs found the compound "
            "but the token was already absent from its HELM. Examples: %s",
            len(token_misses),
            token_misses[:5],
        )
    affected_rows = len(touched_idx)
    _log.info("HELM corrections: rewrote tokens in %d rows", affected_rows)
    return work, affected_rows


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def apply_helm_normalization(
    df: pd.DataFrame,
    helm_col: str,
    valid_symbols: set[str],
    alt_to_canonical: dict[str, str],
    log: logging.Logger | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the full HELM normalization pipeline to a DataFrame.

    Drops rows with empty/null HELM up front and rows whose monomers fail
    validation at the end. All counts logged. Returns a fresh DataFrame
    (the input is not mutated) and a stats dict.

    Stats keys:
        original           — input row count
        null_removed       — rows dropped for null/empty HELM
        renumbered         — rows whose chain id was renumbered
        remapped           — rows whose alt symbols were rewritten
        connection_swapped — rows whose connection token endpoints were sorted
        invalid_removed    — rows dropped by monomer validation
        final              — output row count
    """
    _log = log or logger
    original = len(df)

    # Drop null / empty HELM
    helm_series = cast(pd.Series, df[helm_col])
    mask_present = helm_series.notna() & (helm_series.astype(str).str.strip() != "")
    df = cast(pd.DataFrame, df.loc[mask_present].copy())
    null_removed = original - len(df)

    renumbered = remapped = connection_swapped = 0
    new_helms: list[str] = []
    for h in df[helm_col].astype(str):
        renum = normalize_single_chain(h)
        if renum != h:
            renumbered += 1
        remap = remap_alt_symbols(renum, alt_to_canonical)
        if remap != renum:
            remapped += 1
        canon = canonicalize_connections(remap)
        if canon != remap:
            connection_swapped += 1
        new_helms.append(canon)
    df[helm_col] = new_helms

    valid_mask = df[helm_col].map(lambda h: validate_monomers(h, valid_symbols))
    invalid_removed = int((~valid_mask).sum())
    df = cast(pd.DataFrame, df.loc[valid_mask].copy())

    stats = {
        "original": original,
        "null_removed": null_removed,
        "renumbered": renumbered,
        "remapped": remapped,
        "connection_swapped": connection_swapped,
        "invalid_removed": invalid_removed,
        "final": len(df),
    }

    _log.info(
        "HELM normalize: %d -> %d (null:-%d, renumber:%d, remap:%d, conn_swap:%d, invalid:-%d)",
        original,
        len(df),
        null_removed,
        renumbered,
        remapped,
        connection_swapped,
        invalid_removed,
    )
    return df, stats
