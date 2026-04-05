#!/usr/bin/env python
"""Prepare permeability data with scaffold-based train/test split.

Keeps Permeability, PAMPA, and Caco2 columns.
Uses Murcko scaffolds to ensure molecules sharing the same scaffold
never appear in both train and test sets.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import lightning as L
from src.utils import build_scaffold_groups, flatten_groups, greedy_scaffold_partition

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
    log_dir = log_base / f"permeability_scaffold_preparation_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "prepare_permeability_scaffold.log"

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


def _group_category_counts(df: pd.DataFrame, indices: List[int], assay_cols: List[str]) -> Tuple[int, int, int]:
    """Return counts for (pampa_only, caco2_only, both) within a scaffold group."""
    group = df.iloc[indices].loc[:, assay_cols]
    has_pampa = group.iloc[:, 0].notna()
    has_caco2 = group.iloc[:, 1].notna()
    both = int((has_pampa & has_caco2).sum())
    pampa_only = int((has_pampa & ~has_caco2).sum())
    caco2_only = int((has_caco2 & ~has_pampa).sum())
    return pampa_only, caco2_only, both


def _dataset_category_counts(df: pd.DataFrame, assay_cols: List[str]) -> Tuple[int, int, int]:
    """Return counts for the full dataset."""
    assay_frame = df.loc[:, assay_cols]
    has_pampa = assay_frame.iloc[:, 0].notna()
    has_caco2 = assay_frame.iloc[:, 1].notna()
    both = int((has_pampa & has_caco2).sum())
    pampa_only = int((has_pampa & ~has_caco2).sum())
    caco2_only = int((has_caco2 & ~has_pampa).sum())
    return int(pampa_only), int(caco2_only), int(both)


def _combine_counts(
    current: Tuple[int, int, int], added: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    """Combine category counts."""
    return tuple(a + b for a, b in zip(current, added))


def _composition_key(
    size: int,
    counts: Tuple[int, int, int],
    total_size: int,
    total_counts: Tuple[int, int, int],
    target_test_size: int,
) -> Tuple[float, ...]:
    """Return a deterministic key for how close the test set is to target composition."""
    category_order = sorted(range(len(total_counts)), key=lambda idx: total_counts[idx])
    category_gaps = [
        abs(counts[idx] - ((total_counts[idx] * target_test_size) / total_size))
        if total_counts[idx] > 0 else 0.0
        for idx in category_order
    ]

    return (
        *category_gaps,
        abs(size - target_test_size),
    )


def scaffold_split(
    df: pd.DataFrame, smiles_col: str, assay_cols: List[str], test_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by scaffold while preserving assay composition as closely as possible."""
    groups = build_scaffold_groups(df[smiles_col].tolist())
    logger.info(f"Found {len(groups)} unique scaffolds from {len(df)} molecules")

    total_counts = _dataset_category_counts(df, assay_cols)
    logger.info(
        "Assay composition totals: "
        f"pampa_only={total_counts[0]}, caco2_only={total_counts[1]}, both={total_counts[2]}"
    )
    target_test_size = round(len(df) * test_ratio)
    category_order = sorted(range(len(total_counts)), key=lambda idx: total_counts[idx])
    grouped = [
        (group, _group_category_counts(df, group, assay_cols))
        for group in groups
    ]
    grouped.sort(
        key=lambda item: (
            *[
                (item[1][cat_idx] > 0, item[1][cat_idx])
                for cat_idx in category_order
            ],
            len(item[0]),
            -item[0][0],
        ),
        reverse=True,
    )
    groups = [group for group, _ in grouped]
    group_states = [counts for _, counts in grouped]
    empty_counts = (0, 0, 0)

    test_groups, train_groups, test_size, test_counts = greedy_scaffold_partition(
        groups=groups,
        group_states=group_states,
        target_test_size=target_test_size,
        empty_state=empty_counts,
        combine_states=_combine_counts,
        key_fn=lambda size, counts, target: _composition_key(
            size=size,
            counts=counts,
            total_size=len(df),
            total_counts=total_counts,
            target_test_size=target,
        ),
    )

    test_indices = flatten_groups(test_groups)
    train_indices = flatten_groups(train_groups)

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    logger.info(
        f"Selected test set: {len(test_groups)} scaffold groups, "
        f"{len(test_df)} samples ({len(test_df)/len(df):.3f} of dataset)"
    )
    logger.info(
        "Test composition: "
        f"pampa_only={test_counts[0]}, caco2_only={test_counts[1]}, both={test_counts[2]}"
    )
    logger.info(
        f"Scaffold split: {len(train_df)} train, {len(test_df)} test "
        f"(actual test ratio: {len(test_df)/len(df):.3f})"
    )
    return train_df, test_df


def main():
    global logger

    parser = argparse.ArgumentParser(description="Prepare permeability data (scaffold split)")
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
    logger.info("Permeability Data Preparation (Scaffold Split)")
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

    # Keep rows with valid Permeability value
    has_perm = df["Permeability"].notna()
    df_filtered = df.loc[has_perm].copy()
    logger.info(f"Rows with valid Permeability: {len(df_filtered)}")

    # Filter invalid values
    for col in ["Permeability"] + ASSAY_COLS:
        invalid_mask = df_filtered[col].notna() & (df_filtered[col] <= INVALID_THRESHOLD)
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            df_filtered.loc[invalid_mask, col] = float("nan")
            logger.info(f"Set {n_invalid} invalid {col} values (<= {INVALID_THRESHOLD}) to NaN")

    # Drop rows that lost Permeability after filtering
    has_perm = df_filtered["Permeability"].notna()
    df_filtered = df_filtered.loc[has_perm].copy()
    logger.info(f"After invalid filtering: {len(df_filtered)} samples")

    filtered_assay_frame = df_filtered.loc[:, ASSAY_COLS].copy()
    for col in ASSAY_COLS:
        logger.info(f"  {col}: {filtered_assay_frame.loc[:, col].notna().sum()} valid values")
    both_valid = filtered_assay_frame.notna().all(axis=1)
    logger.info(f"  Both PAMPA+Caco2: {both_valid.sum()} rows")

    # Sort for deterministic processing
    df_filtered = df_filtered.sort_values([HELM_COL, SMILES_COL]).reset_index(drop=True)

    # Scaffold-based split (stratified by minority assay)
    train_df, test_df = scaffold_split(df_filtered, SMILES_COL, ASSAY_COLS, args.test_ratio)

    train_file = output_dir / "cycpeptmpdb_permeability_scaffold_train.csv"
    test_file = output_dir / "cycpeptmpdb_permeability_scaffold_test.csv"

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

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
