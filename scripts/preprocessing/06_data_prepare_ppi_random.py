#!/usr/bin/env python
"""Prepare PPI (Peptide-Protein Interaction) data.

Creates train/test split with negative pair generation.
Train is further split into train/val by datamodule.

Negative generation strategy (same as original):
1. Split positive pairs into train/test
2. Generate negatives for each split using only that split's peptide/protein pool
3. Negatives are random (peptide, protein) pairs that are NOT in global positive set
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
from datetime import datetime
from typing import Set, Tuple, List

import numpy as np
import pandas as pd
import lightning as L
from sklearn.model_selection import train_test_split

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "local_data/intermediate_product/Propedia_v2_unique_ppi_HELM_SMILES.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/downstream"
DEFAULT_LOG_DIR = REPO_ROOT / "outputs/preprocessing"

SEED = 42
TEST_RATIO = 0.1  # 8:1:1 split (10% test)
NEGATIVE_RATIO = 4  # 1:4 (positive:negative)

DRUG_COL = "Peptide_HELM"
TARGET_COL = "Receptor_Sequence"
LABEL_COL = "Label"

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logger = logging.getLogger(__name__)


def setup_logging(log_base: Path = DEFAULT_LOG_DIR) -> Tuple[logging.Logger, Path]:
    """Set up logging to both console and file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = log_base / f"ppi_random_preparation_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "prepare_ppi_random.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(LOG_LEVEL)
    logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Log file: {log_file.absolute()}")

    return logger, log_dir


def generate_negative_pairs(
    n_negative: int,
    peptides: List[str],
    proteins: List[str],
    positive_pairs: Set[Tuple[str, str]],
    existing_negatives: Set[Tuple[str, str]],
    seed: int
) -> List[Tuple[str, str]]:
    """Generate negative pairs from peptide/protein pool.

    Args:
        n_negative: Number of negative pairs to generate
        peptides: List of peptides in this split
        proteins: List of proteins in this split
        positive_pairs: Global set of positive pairs (to exclude)
        existing_negatives: Already generated negatives (to exclude)
        seed: Random seed

    Returns:
        List of (peptide, protein) negative pairs
    """
    peptides_array = np.array(sorted(peptides))
    proteins_array = np.array(sorted(proteins))

    # All pairs to exclude
    excluded = positive_pairs | existing_negatives

    # Max possible negatives
    max_possible = len(peptides_array) * len(proteins_array) - len(excluded)
    if n_negative > max_possible:
        logger.warning(f"Requested {n_negative} negatives but only {max_possible} available")
        n_negative = max_possible

    rng = np.random.default_rng(seed)
    negative_pairs = []
    negative_set = set()

    batch_size = max(10000, n_negative * 10)
    max_attempts = 100
    attempts = 0

    while len(negative_pairs) < n_negative and attempts < max_attempts:
        # Random sampling
        pep_idx = rng.integers(0, len(peptides_array), batch_size)
        prot_idx = rng.integers(0, len(proteins_array), batch_size)

        for p, r in zip(peptides_array[pep_idx], proteins_array[prot_idx]):
            pair = (p, r)
            if pair not in excluded and pair not in negative_set:
                negative_pairs.append(pair)
                negative_set.add(pair)
                if len(negative_pairs) >= n_negative:
                    break

        attempts += 1

    if len(negative_pairs) < n_negative:
        logger.warning(f"Only generated {len(negative_pairs)}/{n_negative} negatives")

    return negative_pairs


def main():
    global logger

    parser = argparse.ArgumentParser(description="Prepare PPI data")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--negative-ratio", type=int, default=NEGATIVE_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    logger, log_dir = setup_logging()

    L.seed_everything(args.seed)

    source_file = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PPI Data Preparation")
    logger.info("=" * 60)
    logger.info(f"Source: {source_file}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Test ratio: {args.test_ratio}")
    logger.info(f"Negative ratio: 1:{args.negative_ratio}")

    # Load data
    logger.info("\nLoading data...")
    df = pd.read_csv(source_file)
    logger.info(f"Loaded {len(df)} rows")

    # Check required columns
    required_cols = [DRUG_COL, TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Sort for deterministic dedup regardless of input order
    df = df.sort_values(by=[DRUG_COL, TARGET_COL]).reset_index(drop=True)

    # Drop duplicate (peptide, protein) pairs - keep first occurrence
    original_len = len(df)
    df = df.drop_duplicates(subset=[DRUG_COL, TARGET_COL], keep='first')
    if len(df) < original_len:
        logger.info(f"Dropped {original_len - len(df)} duplicate pairs")
    logger.info(f"Unique positive pairs: {len(df)}")

    # All positive pairs (global, for exclusion)
    df[LABEL_COL] = 1
    global_positive_pairs = set(zip(df[DRUG_COL], df[TARGET_COL]))

    # Split positive pairs into train/test
    train_pos, test_pos = train_test_split(
        df,
        test_size=args.test_ratio,
        random_state=args.seed,
        shuffle=True
    )
    logger.info(f"\nPositive split: {len(train_pos)} train, {len(test_pos)} test")

    # Get peptide/protein pools for each split
    train_peptides = list(train_pos[DRUG_COL].unique())
    train_proteins = list(train_pos[TARGET_COL].unique())
    test_peptides = list(test_pos[DRUG_COL].unique())
    test_proteins = list(test_pos[TARGET_COL].unique())

    logger.info(f"Train pool: {len(train_peptides)} peptides × {len(train_proteins)} proteins")
    logger.info(f"Test pool: {len(test_peptides)} peptides × {len(test_proteins)} proteins")

    # Generate negatives for train
    logger.info("\nGenerating train negatives...")
    n_train_neg = len(train_pos) * args.negative_ratio
    train_neg_pairs = generate_negative_pairs(
        n_negative=n_train_neg,
        peptides=train_peptides,
        proteins=train_proteins,
        positive_pairs=global_positive_pairs,
        existing_negatives=set(),
        seed=args.seed
    )
    logger.info(f"Generated {len(train_neg_pairs)} train negatives")

    # Track generated negatives
    all_negatives = set(train_neg_pairs)

    # Generate negatives for test
    logger.info("Generating test negatives...")
    n_test_neg = len(test_pos) * args.negative_ratio
    test_neg_pairs = generate_negative_pairs(
        n_negative=n_test_neg,
        peptides=test_peptides,
        proteins=test_proteins,
        positive_pairs=global_positive_pairs,
        existing_negatives=all_negatives,
        seed=args.seed + 1
    )
    logger.info(f"Generated {len(test_neg_pairs)} test negatives")

    # Create negative DataFrames
    train_neg_df = pd.DataFrame({
        DRUG_COL: [p[0] for p in train_neg_pairs],
        TARGET_COL: [p[1] for p in train_neg_pairs],
        LABEL_COL: 0
    })
    test_neg_df = pd.DataFrame({
        DRUG_COL: [p[0] for p in test_neg_pairs],
        TARGET_COL: [p[1] for p in test_neg_pairs],
        LABEL_COL: 0
    })

    # Combine positive + negative
    output_cols = [DRUG_COL, TARGET_COL, LABEL_COL]

    train_df = pd.concat([
        train_pos[output_cols],
        train_neg_df[output_cols]
    ], ignore_index=True)

    test_df = pd.concat([
        test_pos[output_cols],
        test_neg_df[output_cols]
    ], ignore_index=True)

    # Shuffle
    train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=args.seed + 1).reset_index(drop=True)

    # Label distribution
    logger.info("\nFinal label distribution:")
    for name, data in [("Train", train_df), ("Test", test_df)]:
        pos = (data[LABEL_COL] == 1).sum()
        neg = (data[LABEL_COL] == 0).sum()
        ratio = neg / pos if pos > 0 else 0
        logger.info(f"  {name}: {pos} pos, {neg} neg (1:{ratio:.1f})")

    # Save
    train_file = output_dir / "propedia_ppi_random_train.csv"
    test_file = output_dir / "propedia_ppi_random_test.csv"

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    logger.info(f"\nSaved:")
    logger.info(f"  Train: {train_file} ({len(train_df)} samples)")
    logger.info(f"  Test: {test_file} ({len(test_df)} samples)")

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
