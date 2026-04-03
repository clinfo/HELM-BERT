#!/usr/bin/env python
"""Prepare permeability data with scaffold-based train/test split.

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
TARGET_COL = "Permeability"

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


def generate_scaffold(smiles: str) -> str:
    """Generate Murcko scaffold from SMILES. Returns empty string on failure."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(
    df: pd.DataFrame, smiles_col: str, test_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by Murcko scaffold, keeping same scaffold in one set.

    Scaffold groups are sorted largest-first and greedily assigned to the test
    set until the target ratio is reached.
    """
    scaffold_to_indices: Dict[str, List[int]] = {}
    for i, smiles in enumerate(df[smiles_col]):
        scaffold = generate_scaffold(smiles)
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    n_scaffolds = len(scaffold_to_indices)
    logger.info(f"Found {n_scaffolds} unique scaffolds from {len(df)} molecules")

    # Largest groups first for greedy bin-packing
    sorted_groups = sorted(scaffold_to_indices.values(), key=len, reverse=True)

    target_test_size = int(len(df) * test_ratio)
    test_indices, train_indices = [], []

    for group in sorted_groups:
        if len(test_indices) < target_test_size:
            test_indices.extend(group)
        else:
            train_indices.extend(group)

    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

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

    df = pd.read_csv(source_file)
    logger.info(f"Loaded {len(df)} samples")

    # Sort for deterministic processing
    df = df.sort_values(by=[HELM_COL, SMILES_COL]).reset_index(drop=True)

    # Filter invalid samples
    df_filtered = df[df[TARGET_COL] > INVALID_THRESHOLD].copy()
    n_filtered = len(df) - len(df_filtered)
    logger.info(f"Filtered {n_filtered} samples with {TARGET_COL} <= {INVALID_THRESHOLD}")
    logger.info(f"Remaining: {len(df_filtered)} samples")

    required_cols = [SMILES_COL, HELM_COL, TARGET_COL]
    missing_cols = [c for c in required_cols if c not in df_filtered.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Scaffold-based split
    train_df, test_df = scaffold_split(df_filtered, SMILES_COL, args.test_ratio)

    output_cols = [SMILES_COL, HELM_COL, TARGET_COL]

    train_file = output_dir / "cycpeptmpdb_permeability_scaffold_train.csv"
    test_file = output_dir / "cycpeptmpdb_permeability_scaffold_test.csv"

    train_df[output_cols].to_csv(train_file, index=False)
    test_df[output_cols].to_csv(test_file, index=False)

    logger.info(f"\nSaved:")
    logger.info(f"  Train: {train_file} ({len(train_df)} samples)")
    logger.info(f"  Test: {test_file} ({len(test_df)} samples)")

    logger.info(f"\n{TARGET_COL} statistics:")
    for name, data in [("Train", train_df), ("Test", test_df)]:
        logger.info(f"  {name}: mean={data[TARGET_COL].mean():.3f}, std={data[TARGET_COL].std():.3f}")

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
