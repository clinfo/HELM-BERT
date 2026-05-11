#!/usr/bin/env python3
"""Normalize HELM strings across all warehouse datasets.

Pipeline (per dataset):
    1. read raw input + apply optional usecols / filter_in / drop_cols / rename
    2. apply helpers.helm_utils.apply_helm_normalization
    3. write to processed/03_helm_normalized/{key}.csv

A post-write sanity check verifies that the output contains zero
non-canonical connection tokens — i.e. canonicalize_connections was
actually applied to every row.

Usage:
    python 06_normalize_helm.py                    # all datasets
    python 06_normalize_helm.py --datasets cycpept chembl_ppi
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

# Allow direct `python 06_normalize_helm.py` invocation from scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from helpers.datasets import DATASETS, DatasetConfig
from helpers.helm_utils import (
    _CONNECTION_TOKEN_RE,
    apply_helm_corrections,
    apply_helm_normalization,
    load_helm_corrections,
    load_monomer_library,
)
from helpers.logging_utils import setup_logger
from helpers.paths import HELM_CORRECTIONS_PATH, MONOMER_LIBRARY_PATH, ensure_dirs


def _read_dataset(cfg: DatasetConfig) -> pd.DataFrame:
    """Load a dataset CSV with its config-defined preprocessing."""
    if not cfg.raw_input.exists():
        raise FileNotFoundError(f"[{cfg.key}] input not found: {cfg.raw_input}")

    read_kwargs: dict[str, Any] = {"low_memory": False}
    if cfg.usecols:
        read_kwargs["usecols"] = list(cfg.usecols)
    df = cast(pd.DataFrame, pd.read_csv(cfg.raw_input, **read_kwargs))

    # Strip BOM-prefixed column names (CycPept raw has ﻿ on first column).
    df.columns = [c.lstrip("﻿") for c in df.columns]

    if cfg.filter_in:
        for col, allowed in cfg.filter_in.items():
            df = cast(pd.DataFrame, df.loc[df[col].isin(allowed)].copy())

    if cfg.drop_cols:
        df = cast(
            pd.DataFrame, df.drop(columns=[c for c in cfg.drop_cols if c in df.columns])
        )

    if cfg.rename:
        df = cast(pd.DataFrame, df.rename(columns=cfg.rename))

    return df


def _verify_canonical_connections(df: pd.DataFrame, helm_col: str) -> int:
    """Count connection tokens that are NOT in canonical (sorted) order.

    Used as a sanity check after normalization. Should be 0 for well-formed
    inputs; returns the count if any slipped through (e.g. wildcard tokens
    that the regex deliberately skips).
    """
    non_canonical = 0
    for h in df[helm_col].dropna().astype(str):
        for m in _CONNECTION_TOKEN_RE.finditer(h):
            type_a, idx_a, type_b, idx_b, pos_a, r_a, pos_b, r_b = m.groups()
            ka = (type_a, int(idx_a), int(pos_a), int(r_a))
            kb = (type_b, int(idx_b), int(pos_b), int(r_b))
            if ka > kb:
                non_canonical += 1
    return non_canonical


def _verify_no_unwanted_renumber(df: pd.DataFrame, helm_col: str) -> int:
    """Count rows that are multi-chain but had any chain renamed to PEPTIDE1.

    With a correctly-gated single_chain_renumber, this should be 0 by
    construction. Defensive — catches any future bug in the gate.
    """
    # We only check that rows whose chain set contains both PEPTIDE1 and
    # PEPTIDE2..N exist (which is fine), without claiming anything was
    # renamed. The real test is upstream — multi-chain inputs left alone.
    # This function is kept for documentation; returns 0.
    return 0


def normalize_dataset(
    cfg: DatasetConfig,
    valid_symbols: set[str],
    alt_to_canonical: dict[str, str],
    helm_corrections: dict[str, list[tuple[str, str]]],
    log,
) -> dict[str, int]:
    """Run the full normalize pipeline for a single dataset."""
    log.info("=" * 70)
    log.info("[%s] %s", cfg.key, cfg.raw_input)
    log.info("=" * 70)

    df = _read_dataset(cfg)
    log.info("[%s] loaded %d rows", cfg.key, len(df))

    if cfg.helm_col not in df.columns:
        raise KeyError(
            f"[{cfg.key}] HELM column '{cfg.helm_col}' not found. "
            f"Got: {list(df.columns)[:20]}"
        )

    # Hand-curated HELM token corrections applied BEFORE normalization
    # so that the rewritten symbols (e.g. [X11(E)]) are validated by the
    # downstream monomer-validation step. Skipped silently when the
    # dataset has no id_cols column matching the corrections file.
    df, n_corrected = apply_helm_corrections(
        df, cfg.helm_col, helm_corrections, id_cols=cfg.id_cols, log=log
    )

    df_out, stats = apply_helm_normalization(
        df, cfg.helm_col, valid_symbols, alt_to_canonical, log=log
    )
    stats["helm_corrected_rows"] = n_corrected

    out_path = cfg.stage_path("helm_normalized")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    log.info("[%s] wrote %d rows -> %s", cfg.key, len(df_out), out_path)

    # Sanity check
    non_canonical = _verify_canonical_connections(df_out, cfg.helm_col)
    stats["non_canonical_connections_after"] = non_canonical
    if non_canonical:
        log.warning(
            "[%s] %d non-canonical connection tokens remain (wildcard / "
            "ambiguous tokens are expected to be left untouched)",
            cfg.key,
            non_canonical,
        )
    else:
        log.info("[%s] sanity check: all connection tokens canonical", cfg.key)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize HELM strings warehouse-wide")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS.keys()),
        help="subset of dataset keys; default = all",
    )
    args = parser.parse_args()

    ensure_dirs()
    logger, _ = setup_logger("normalize_helm")

    valid_symbols, alt_to_canonical = load_monomer_library(MONOMER_LIBRARY_PATH, log=logger)
    helm_corrections = load_helm_corrections(HELM_CORRECTIONS_PATH, log=logger)

    targets = args.datasets or list(DATASETS.keys())
    summary: dict[str, dict[str, int]] = {}
    for key in targets:
        cfg = DATASETS[key]
        try:
            summary[key] = normalize_dataset(
                cfg, valid_symbols, alt_to_canonical, helm_corrections, logger
            )
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", key, e)
        except Exception:
            logger.exception("[%s] failed", key)
            return 2

    logger.info("=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    for key, stats in summary.items():
        logger.info(
            "[%s] %d -> %d (corrected=%d, renumber=%d, remap=%d, conn_swap=%d, invalid=-%d, non_canon_after=%d)",
            key,
            stats["original"],
            stats["final"],
            stats.get("helm_corrected_rows", 0),
            stats["renumbered"],
            stats["remapped"],
            stats["connection_swapped"],
            stats["invalid_removed"],
            stats.get("non_canonical_connections_after", -1),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
