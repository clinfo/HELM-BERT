#!/usr/bin/env python
"""Prepare permeability data with scaffold-based train/test split.

Keeps Permeability, PAMPA, and Caco2 columns.
Uses Murcko scaffolds to ensure molecules sharing the same scaffold
never appear in both train and test sets.
"""

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import lightning as L
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.utils import flatten_groups, generate_scaffold, greedy_scaffold_partition

from scripts.preprocessing.preprocessing_utils.downstream_utils import (
    aggregate_median_by_canonical_smiles,
    log_mlm_coverage,
)
from scripts.preprocessing.preprocessing_utils.paths import (
    INTERMEDIATE_PRODUCT_DIR,
    PREPROCESSING_OUTPUT_DIR,
    REPO_ROOT,
)

DEFAULT_SOURCE = INTERMEDIATE_PRODUCT_DIR / "cycpeptmpdb_helm_normalized.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/downstream"
DEFAULT_LOG_DIR = PREPROCESSING_OUTPUT_DIR

SEED = 42
TEST_RATIO = 0.1
INVALID_THRESHOLD = -10

SMILES_COL = "SMILES"
HELM_COL = "HELM"
ASSAY_COLS = ["PAMPA", "Caco2"]
TARGET_COLS = ["Permeability"] + ASSAY_COLS

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logger = logging.getLogger(__name__)
MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


@dataclass
class ScaffoldGroupRecord:
    """Metadata for one scaffold-disjoint group."""

    scaffold: str
    indices: List[int]
    counts: Tuple[int, int, int]

    @property
    def size(self) -> int:
        return len(self.indices)


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


def _build_scaffold_group_records(
    df: pd.DataFrame, smiles_col: str, assay_cols: List[str]
) -> List[ScaffoldGroupRecord]:
    """Build scaffold groups and keep scaffold metadata for candidate scoring."""
    scaffold_to_indices: Dict[str, List[int]] = {}
    for idx, smiles in enumerate(df[smiles_col].tolist()):
        scaffold = generate_scaffold(smiles) or "__no_scaffold__"
        scaffold_to_indices.setdefault(scaffold, []).append(idx)

    records: List[ScaffoldGroupRecord] = []
    for scaffold, indices in scaffold_to_indices.items():
        records.append(
            ScaffoldGroupRecord(
                scaffold=scaffold,
                indices=indices,
                counts=_group_category_counts(df, indices, assay_cols),
            )
        )

    return records


def _seeded_tiebreak(record: ScaffoldGroupRecord, seed: int, strategy: str) -> float:
    """Return a deterministic pseudo-random tiebreaker for candidate orderings."""
    key = f"{seed}:{strategy}:{record.scaffold}:{record.indices[0]}"
    return random.Random(key).random()


def _ordered_group_records(
    records: List[ScaffoldGroupRecord],
    category_order: List[int],
    seed: int,
    strategy: str,
    rarity_scores: Dict[str, float],
) -> List[ScaffoldGroupRecord]:
    """Return one deterministic candidate ordering for greedy scaffold partitioning."""
    ordered = records.copy()
    prefer_small = "small" in strategy
    seeded = "seeded" in strategy
    prefer_rare = strategy.startswith("rare")

    if seeded:
        ordered.sort(key=lambda record: _seeded_tiebreak(record, seed, strategy))

    ordered.sort(key=lambda record: record.indices[0], reverse=True)
    ordered.sort(key=lambda record: record.size, reverse=not prefer_small)
    if prefer_rare:
        ordered.sort(
            key=lambda record: rarity_scores.get(record.scaffold, 0.0),
            reverse=True,
        )
    ordered.sort(
        key=lambda record: tuple(
            (record.counts[cat_idx] > 0, record.counts[cat_idx])
            for cat_idx in category_order
        ),
        reverse=True,
    )
    return ordered


def _scaffold_fingerprint(scaffold: str) -> Optional[DataStructs.ExplicitBitVect]:
    """Return a fingerprint for scaffold-level novelty scoring."""
    if scaffold == "__no_scaffold__":
        return None

    mol = Chem.MolFromSmiles(scaffold)
    if mol is None:
        return None
    return MORGAN_GENERATOR.GetFingerprint(mol)


def _scaffold_rarity_scores(records: List[ScaffoldGroupRecord]) -> Dict[str, float]:
    """Estimate how structurally isolated each scaffold is within the full dataset."""
    scaffold_to_fp = {
        record.scaffold: _scaffold_fingerprint(record.scaffold)
        for record in records
    }
    rarity_scores: Dict[str, float] = {}

    valid_items = [(scaffold, fp) for scaffold, fp in scaffold_to_fp.items() if fp is not None]
    for scaffold, fp in valid_items:
        other_fps = [other_fp for other_scaffold, other_fp in valid_items if other_scaffold != scaffold]
        if not other_fps:
            rarity_scores[scaffold] = 1.0
            continue
        nearest_similarity = max(DataStructs.BulkTanimotoSimilarity(fp, other_fps))
        rarity_scores[scaffold] = 1.0 - nearest_similarity

    for record in records:
        rarity_scores.setdefault(record.scaffold, 0.0)
    return rarity_scores


def _novelty_penalties(
    test_records: List[ScaffoldGroupRecord],
    train_records: List[ScaffoldGroupRecord],
) -> Tuple[float, float]:
    """Return row-weighted novelty penalties for a candidate scaffold test fold."""
    train_fps = [
        _scaffold_fingerprint(record.scaffold)
        for record in train_records
    ]
    train_fps = [fp for fp in train_fps if fp is not None]
    if not train_fps:
        return 1.0, 1.0

    weighted_similarity = 0.0
    total_weight = 0
    worst_similarity = 0.0

    for record in test_records:
        fp = _scaffold_fingerprint(record.scaffold)
        best_similarity = 1.0 if fp is None else max(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        weighted_similarity += best_similarity * record.size
        total_weight += record.size
        worst_similarity = max(worst_similarity, best_similarity)

    if total_weight == 0:
        return 1.0, 1.0
    return weighted_similarity / total_weight, worst_similarity


def _candidate_selection_score(
    test_records: List[ScaffoldGroupRecord],
    train_records: List[ScaffoldGroupRecord],
    total_size: int,
    total_counts: Tuple[int, int, int],
    target_test_size: int,
) -> Tuple[float, ...]:
    """Return a structural selection score for one scaffold-fold candidate.

    Greedy assignment still respects assay-composition and size through
    `_composition_key`, but when comparing multiple feasible candidates we
    prioritize structural hardness before small residual composition gaps.
    """
    test_size = sum(record.size for record in test_records)
    test_counts = (0, 0, 0)
    for record in test_records:
        test_counts = _combine_counts(test_counts, record.counts)

    tiny_group_share = (
        sum(record.size for record in test_records if record.size <= 2) / test_size
        if test_size else 1.0
    )
    no_scaffold_share = (
        sum(record.size for record in test_records if record.scaffold == "__no_scaffold__") / test_size
        if test_size else 1.0
    )
    mean_similarity, worst_similarity = _novelty_penalties(test_records, train_records)
    composition_key = _composition_key(
        size=test_size,
        counts=test_counts,
        total_size=total_size,
        total_counts=total_counts,
        target_test_size=target_test_size,
    )

    return (
        composition_key[-1],
        mean_similarity,
        worst_similarity,
        tiny_group_share,
        no_scaffold_share,
        *composition_key[:-1],
    )


def scaffold_split(
    df: pd.DataFrame, smiles_col: str, assay_cols: List[str], test_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by scaffold while preferring structurally harder test-fold candidates."""
    records = _build_scaffold_group_records(df, smiles_col, assay_cols)
    logger.info(f"Found {len(records)} unique scaffolds from {len(df)} molecules")

    total_counts = _dataset_category_counts(df, assay_cols)
    logger.info(
        "Assay composition totals: "
        f"pampa_only={total_counts[0]}, caco2_only={total_counts[1]}, both={total_counts[2]}"
    )
    target_test_size = round(len(df) * test_ratio)
    category_order = sorted(range(len(total_counts)), key=lambda idx: total_counts[idx])
    rarity_scores = _scaffold_rarity_scores(records)
    empty_counts = (0, 0, 0)

    best_candidate: Optional[Tuple[str, List[List[int]], List[List[int]], int, Tuple[int, int, int]]] = None
    best_score: Optional[Tuple[float, ...]] = None
    candidate_strategies = [
        "large_first",
        "small_first",
        "rare_large_first",
        "rare_small_first",
        "seeded_large_first",
        "seeded_small_first",
    ]

    for strategy in candidate_strategies:
        ordered_records = _ordered_group_records(records, category_order, seed, strategy, rarity_scores)
        groups = [record.indices for record in ordered_records]
        group_states = [record.counts for record in ordered_records]
        record_lookup = {tuple(record.indices): record for record in ordered_records}

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

        test_records = [record_lookup[tuple(group)] for group in test_groups]
        train_records = [record_lookup[tuple(group)] for group in train_groups]
        tiny_group_share = (
            sum(record.size for record in test_records if record.size <= 2) / test_size
            if test_size else 1.0
        )
        no_scaffold_share = (
            sum(record.size for record in test_records if record.scaffold == "__no_scaffold__") / test_size
            if test_size else 1.0
        )
        mean_similarity, _ = _novelty_penalties(test_records, train_records)
        candidate_score = _candidate_selection_score(
            test_records=test_records,
            train_records=train_records,
            total_size=len(df),
            total_counts=total_counts,
            target_test_size=target_test_size,
        )
        logger.info(
            "Candidate %s: test_groups=%d, test_size=%d, "
            "tiny_share=%.3f, no_scaffold_share=%.3f, mean_nn_similarity=%.3f",
            strategy,
            len(test_groups),
            test_size,
            tiny_group_share,
            no_scaffold_share,
            mean_similarity,
        )

        if best_score is None or candidate_score < best_score:
            best_score = candidate_score
            best_candidate = (strategy, test_groups, train_groups, test_size, test_counts)

    assert best_candidate is not None
    selected_strategy, test_groups, train_groups, test_size, test_counts = best_candidate
    logger.info("Selected scaffold candidate: %s", selected_strategy)

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


def _save_split_files(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    """Save one train/test pair and log where it was written."""
    train_file = output_dir / f"{stem}_train.csv"
    test_file = output_dir / f"{stem}_test.csv"
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    logger.info(f"  Train: {train_file} ({len(train_df)} samples)")
    logger.info(f"  Test: {test_file} ({len(test_df)} samples)")


def _log_task_statistics(task_name: str, target_col: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Log basic target stats for one task-specific split."""
    logger.info(f"\n{task_name} statistics:")
    for split_name, df in [("Train", train_df), ("Test", test_df)]:
        values = df[target_col].dropna()
        logger.info(
            f"  {split_name}: n={len(values)}, mean={values.mean():.3f}, std={values.std():.3f}"
        )


def _run_task_scaffold_split(
    df: pd.DataFrame,
    task_name: str,
    target_col: str,
    stem: str,
    output_dir: Path,
    test_ratio: float,
    seed: int,
) -> None:
    """Prepare one task-specific scaffold split without deriving it from another task."""
    task_df = df.loc[df[target_col].notna()].copy().reset_index(drop=True)
    logger.info(f"\nPreparing {task_name} scaffold split from {len(task_df)} rows")

    train_df, test_df = scaffold_split(task_df, SMILES_COL, ASSAY_COLS, test_ratio, seed)
    log_mlm_coverage(train_df, HELM_COL, SMILES_COL, REPO_ROOT, logger, f"{stem}/train")
    log_mlm_coverage(test_df, HELM_COL, SMILES_COL, REPO_ROOT, logger, f"{stem}/test")
    _save_split_files(train_df, test_df, output_dir, stem)
    _log_task_statistics(task_name, target_col, train_df, test_df)


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
    logger.info(f"Loaded {len(df)} samples (HELM already normalized in 02_* stage)")

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

    # Aggregate duplicate measurements per molecule using median assay values.
    df_filtered = df_filtered.sort_values([HELM_COL, SMILES_COL]).reset_index(drop=True)
    df_filtered = aggregate_median_by_canonical_smiles(
        df_filtered, SMILES_COL, TARGET_COLS, logger
    )
    logger.info(f"After molecule aggregation: {len(df_filtered)} unique molecules")

    filtered_assay_frame = df_filtered.loc[:, ASSAY_COLS].copy()
    for col in ASSAY_COLS:
        logger.info(f"  {col}: {filtered_assay_frame.loc[:, col].notna().sum()} valid values")
    both_valid = filtered_assay_frame.notna().all(axis=1)
    logger.info(f"  Both PAMPA+Caco2: {both_valid.sum()} rows")

    # Sort for deterministic processing
    df_filtered = df_filtered.sort_values([HELM_COL, SMILES_COL]).reset_index(drop=True)

    logger.info("\nSaved:")
    _run_task_scaffold_split(
        df_filtered,
        "Permeability",
        "Permeability",
        "cycpeptmpdb_permeability_scaffold",
        output_dir,
        args.test_ratio,
        args.seed,
    )
    _run_task_scaffold_split(
        df_filtered,
        "PAMPA",
        "PAMPA",
        "cycpeptmpdb_permeability_pampa_scaffold",
        output_dir,
        args.test_ratio,
        args.seed,
    )
    _run_task_scaffold_split(
        df_filtered,
        "Caco2",
        "Caco2",
        "cycpeptmpdb_permeability_caco2_scaffold",
        output_dir,
        args.test_ratio,
        args.seed,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
