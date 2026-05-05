#!/usr/bin/env python3
"""Add normalized_smiles column to each dataset's HELM-normalized CSV.

Reads ``processed/03_helm_normalized/{key}.csv``, applies
:func:`helpers.smiles_utils.standardize_series` to the SMILES column,
and writes ``processed/04_smiles_normalized/{key}.csv`` with all
original columns plus one new column:

    normalized_smiles  — RDKit canonical SMILES of the SAME molecule.
                         No structural mutation: salts, charges, and
                         isotope labels are preserved as registered.
                         The dedup grouping key in stage 09.

Sanity check emitted to the log:
    - count of rows whose stereo center count decreased (must be 0)

Usage:
    python 08_normalize_smiles.py
    python 08_normalize_smiles.py --datasets cycpept_permeability_compounds chembl_ppi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

import pandas as pd
from rdkit import Chem

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from helpers.datasets import DATASETS, DatasetConfig
from helpers.logging_utils import setup_logger
from helpers.paths import ensure_dirs
from helpers.smiles_utils import standardize_series


def _stereo_center_count(smi: str) -> int:
    """Number of explicitly-chiral atoms in a SMILES (sanity helper)."""
    if not smi:
        return 0
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0
    return sum(
        1 for a in mol.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    )


def _verify_stereo_preserved(in_smis: pd.Series, out_smis: pd.Series, log) -> int:
    """Compare per-row stereo-center counts before/after canonicalization.

    Returns the number of rows where the count *decreased* (stereo loss).
    Canonicalization should never lose stereo; if it does, regression.
    """
    losses = 0
    for in_smi, out_smi in zip(in_smis.astype(str), out_smis.astype(str)):
        if not in_smi or not out_smi:
            continue
        if _stereo_center_count(in_smi) > _stereo_center_count(out_smi):
            losses += 1
    if losses:
        log.warning(
            "Stereo loss detected on %d rows — pipeline regression?", losses
        )
    return losses


def normalize_dataset(cfg: DatasetConfig, log) -> dict[str, int]:
    """Add normalized_smiles column to a dataset's helm-normalized CSV."""
    in_path = cfg.stage_path("helm_normalized")
    out_path = cfg.stage_path("smiles_normalized")
    if not in_path.exists():
        raise FileNotFoundError(
            f"[{cfg.key}] missing input {in_path}; run 07_normalize_helm.py first"
        )

    log.info("=" * 70)
    log.info("[%s] %s", cfg.key, in_path)
    log.info("=" * 70)

    df = cast(pd.DataFrame, pd.read_csv(in_path, low_memory=False))
    log.info("[%s] loaded %d rows", cfg.key, len(df))

    if cfg.smiles_col not in df.columns:
        raise KeyError(
            f"[{cfg.key}] SMILES column '{cfg.smiles_col}' not found. "
            f"Got: {list(df.columns)[:20]}"
        )

    smi_in = cast(pd.Series, df[cfg.smiles_col])
    smi_norm, stats = standardize_series(smi_in, log=log, label=cfg.key)
    df["normalized_smiles"] = smi_norm

    losses = _verify_stereo_preserved(smi_in, smi_norm, log)
    stats_dict = stats.as_dict()
    stats_dict["stereo_losses"] = losses

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("[%s] wrote %d rows -> %s", cfg.key, len(df), out_path)

    return stats_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Add normalized_smiles column")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS.keys()),
        help="subset of dataset keys; default = all",
    )
    args = parser.parse_args()

    ensure_dirs()
    logger, _ = setup_logger("normalize_smiles")

    targets = args.datasets or list(DATASETS.keys())
    summary: dict[str, dict[str, int]] = {}
    for key in targets:
        cfg = DATASETS[key]
        try:
            summary[key] = normalize_dataset(cfg, logger)
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", key, e)
        except Exception:
            logger.exception("[%s] failed", key)
            return 2

    logger.info("=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    for key, s in summary.items():
        logger.info(
            "[%s] total=%d parsed=%d parse_fail=%d unchanged=%d stereo_loss=%d",
            key,
            s["total"],
            s["parsed"],
            s["parse_failed"],
            s["unchanged"],
            s["stereo_losses"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
