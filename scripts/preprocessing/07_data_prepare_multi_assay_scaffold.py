#!/usr/bin/env python
"""Prepare multi-assay permeability data with scaffold-based train/test split.

Keeps PAMPA and Caco2 as separate target columns.
Uses Murcko scaffolds to ensure molecules sharing the same scaffold
never appear in both train and test sets.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import lightning as L
from rdkit.Chem.Scaffolds import MurckoScaffold

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data/mlm/cycpeptmpdb_deduplicated.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/downstream"
DEFAULT_LOG_DIR = REPO_ROOT / "outputs/preprocessing"

SEED = 42
TEST_RATIO = 0.1
INVALID_THRESHOLD = -10

SMILES_COL = "SMILES"
HELM_COL = "HELM"
ASSAY_COLS = ["PAMPA", "Caco2"]

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logger = logging.getLogger(__name__)


def setup_logging(log_base: Path = DEFAULT_LOG_DIR) -> Tuple[logging.Logger, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = log_base / f"multi_assay_scaffold_preparation_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "prepare_multi_assay_scaffold.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)
    logger.handlers = []

    for handler_cls, args in [
        (logging.StreamHandler, (sys.stdout,)),
        (logging.FileHandler, (log_file,)),
    ]:
        h = handler_cls(*args)
        h.setLevel(LOG_LEVEL)
        h.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(h)

    logger.info(f"Log file: {log_file.absolute()}")
    return logger, log_dir


def generate_scaffold(smiles: str) -> str:
    """Generate Murcko scaffold from SMILES. Returns empty string on failure."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return ""


def _greedy_assign(
    groups: List[List[int]], target_test_size: int
) -> Tuple[List[int], List[int]]:
    """Assign scaffold groups to train/test via greedy largest-first."""
    sorted_groups = sorted(groups, key=len, reverse=True)
    test_indices: List[int] = []
    train_indices: List[int] = []
    for group in sorted_groups:
        if len(test_indices) < target_test_size:
            test_indices.extend(group)
        else:
            train_indices.extend(group)
    return train_indices, test_indices


def scaffold_split(
    df: pd.DataFrame, smiles_col: str, assay_cols: List[str], test_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by Murcko scaffold with assay-stratified allocation.

    Scaffolds are partitioned by whether they contain the minority assay.
    Each partition is independently split at the target ratio, ensuring
    both assays are represented in train and test.
    """
    scaffold_to_indices: Dict[str, List[int]] = {}
    for i, smiles in enumerate(df[smiles_col]):
        scaffold = generate_scaffold(smiles)
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    logger.info(f"Found {len(scaffold_to_indices)} unique scaffolds from {len(df)} molecules")

    minority_col = min(assay_cols, key=lambda c: df[c].notna().sum())
    logger.info(f"Minority assay for stratification: {minority_col}")

    minority_groups: List[List[int]] = []
    majority_groups: List[List[int]] = []
    for indices in scaffold_to_indices.values():
        if df.iloc[indices][minority_col].notna().any():
            minority_groups.append(indices)
        else:
            majority_groups.append(indices)

    n_minority = sum(len(g) for g in minority_groups)
    n_majority = sum(len(g) for g in majority_groups)
    logger.info(
        f"Scaffold partitions: {len(minority_groups)} with {minority_col} "
        f"({n_minority} samples), {len(majority_groups)} without ({n_majority} samples)"
    )

    min_train, min_test = _greedy_assign(minority_groups, int(n_minority * test_ratio))
    maj_train, maj_test = _greedy_assign(majority_groups, int(n_majority * test_ratio))

    train_indices = min_train + maj_train
    test_indices = min_test + maj_test

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    logger.info(
        f"Scaffold split: {len(train_df)} train, {len(test_df)} test "
        f"(actual test ratio: {len(test_df)/len(df):.3f})"
    )
    return train_df, test_df


def main():
    global logger

    parser = argparse.ArgumentParser(description="Prepare multi-assay data (scaffold split)")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    logger, log_dir = setup_logging()
    L.seed_everything(args.seed)

    source_file = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Multi-Assay Data Preparation — Scaffold Split (PAMPA + Caco2)")
    logger.info("=" * 60)
    logger.info(f"Source: {source_file}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Test ratio: {args.test_ratio}")

    df = pd.read_csv(source_file, low_memory=False)
    logger.info(f"Loaded {len(df)} samples")

    required_cols = [SMILES_COL, HELM_COL] + ASSAY_COLS
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Keep rows with at least one valid assay value
    has_any_assay = df[ASSAY_COLS].notna().any(axis=1)
    df_filtered = df[has_any_assay].copy()
    logger.info(f"Rows with at least one assay value: {len(df_filtered)}")

    # Filter invalid values per assay
    for col in ASSAY_COLS:
        invalid_mask = df_filtered[col].notna() & (df_filtered[col] <= INVALID_THRESHOLD)
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            df_filtered.loc[invalid_mask, col] = float("nan")
            logger.info(f"Set {n_invalid} invalid {col} values (<= {INVALID_THRESHOLD}) to NaN")

    # Drop rows that lost all assay values
    has_any_assay = df_filtered[ASSAY_COLS].notna().any(axis=1)
    df_filtered = df_filtered[has_any_assay].copy()
    logger.info(f"After invalid filtering: {len(df_filtered)} samples")

    for col in ASSAY_COLS:
        logger.info(f"  {col}: {df_filtered[col].notna().sum()} valid values")
    both_valid = df_filtered[ASSAY_COLS].notna().all(axis=1)
    logger.info(f"  Both PAMPA+Caco2: {both_valid.sum()} rows")

    # Sort for deterministic processing
    df_filtered = df_filtered.sort_values(by=[HELM_COL, SMILES_COL]).reset_index(drop=True)

    # Scaffold-based split (stratified by minority assay)
    train_df, test_df = scaffold_split(df_filtered, SMILES_COL, ASSAY_COLS, args.test_ratio)

    output_cols = [SMILES_COL, HELM_COL] + ASSAY_COLS

    train_file = output_dir / "cycpeptmpdb_multi_assay_scaffold_train.csv"
    test_file = output_dir / "cycpeptmpdb_multi_assay_scaffold_test.csv"

    train_df[output_cols].to_csv(train_file, index=False)
    test_df[output_cols].to_csv(test_file, index=False)

    logger.info(f"\nSaved:")
    logger.info(f"  Train: {train_file} ({len(train_df)} samples)")
    logger.info(f"  Test: {test_file} ({len(test_df)} samples)")

    for col in ASSAY_COLS:
        logger.info(f"\n{col} statistics:")
        for name, data in [("Train", train_df), ("Test", test_df)]:
            valid = data[col].dropna()
            logger.info(f"  {name}: n={len(valid)}, mean={valid.mean():.3f}, std={valid.std():.3f}")

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
