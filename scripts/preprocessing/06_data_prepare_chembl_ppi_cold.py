"""Data preparation for ChEMBL peptide-protein binding — cold split.

Single train/test split where every test target is completely unseen during
training.  Individual targets are assigned to balanced folds and one fold is
selected as test.

Positives: compound-target pairs with binding activity <= 1 μM (t_1u == 1).
Negatives: random pairs generated from each split's own compound × target pool.

Source: local_data/intermediate_product/chembl_ppi_helm_normalized.csv
Output: data/downstream/chembl_ppi_cold_{train,test}.csv
"""

import os

thread_count = str(min(8, os.cpu_count() or 8))
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, thread_count)

import json
import logging
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import lightning as L
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.preprocessing.helpers.downstream import log_mlm_coverage
from scripts.preprocessing.helpers.paths import (
    INTERMEDIATE_PRODUCT_DIR,
    PREPROCESSING_OUTPUT_DIR,
    REPO_ROOT,
)


DEFAULT_SOURCE = INTERMEDIATE_PRODUCT_DIR / "05_final" / "chembl_ppi.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/downstream"
DEFAULT_LOG_DIR = PREPROCESSING_OUTPUT_DIR

SEED = 42
NEGATIVE_RATIO = 4
ACTIVITY_COL = "t_1u"
TEST_RATIO = 0.2

COMPOUND_ID = "compound_chembl_id"
HELM_COL = "helm_notation"
SMILES_COL = "canonical_smiles"
TARGET_ID = "target_chembl_id"
TARGET_NAME = "target_name"
TARGET_ACCESSION = "target_accession"
TARGET_SEQ = "target_sequence"
PROTEIN_CLASS = "protein_class_desc"

LABEL_COL = "Label"
SPLIT_COL = "split"

COMPOUND_COLS = [COMPOUND_ID, HELM_COL, SMILES_COL]
TARGET_COLS = [TARGET_ID, TARGET_NAME, TARGET_ACCESSION, TARGET_SEQ, PROTEIN_CLASS]

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)
log_dir = None


def setup_logging() -> Path:
    global log_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(DEFAULT_LOG_DIR) / f"chembl_ppi_cold_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    logger.handlers = []
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / "prepare.log")]:
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    logger.info(f"Log directory: {log_dir}")
    return log_dir


def get_fold_seed(base: int, fold: int, offset: int = 0) -> int:
    return base + fold * 1000 + offset


def load_positives(source: Path, activity_col: str = ACTIVITY_COL) -> pd.DataFrame:
    df = pd.read_csv(source, low_memory=False)
    logger.info(f"Loaded {len(df)} rows from {source}")

    df[PROTEIN_CLASS] = (
        df[PROTEIN_CLASS].fillna("UNKNOWN").astype(str).str.strip().replace({"": "UNKNOWN"})
    )

    # Pair key is (HELM, target), NOT (compound_chembl_id, target).
    # Same HELM ⇒ same molecule (the model's input is HELM); compound_chembl_id
    # over-distinguishes (e.g. tritium-labelled isomers, salt forms produce
    # different ChEMBL IDs but identical model input). Keying by HELM here
    # collapses those into one canonical row, which prevents (HELM, target)
    # leak across the train/test split downstream.
    #
    # Side effects worth knowing:
    #   * df_pos[COMPOUND_ID] now lists only the lex-first canonical id per
    #     HELM group; "ghost" ids never appear in df_pos and so are absent
    #     from the negative-sampling pool below.
    #   * pair_registry uses (canonical_compound_id, target) consistently,
    #     so negatives can never collide with a positive on (HELM, target).
    pair_key = [HELM_COL, TARGET_ID]
    pair_activity = df.groupby(pair_key)[activity_col].max().reset_index()
    pos_pairs = pair_activity.loc[pair_activity[activity_col] == 1, pair_key].copy()
    logger.info(f"Positive pairs (t_1u==1 after dedup): {len(pos_pairs)}")

    df_pos = df.merge(pos_pairs, on=pair_key, how="inner")
    df_pos = df_pos.drop_duplicates(subset=pair_key, keep="first").reset_index(drop=True)
    df_pos[LABEL_COL] = 1

    logger.info(f"Positive rows after dedup: {len(df_pos)}")
    logger.info(
        f"Unique compounds: {df_pos[COMPOUND_ID].nunique()}, "
        f"unique targets: {df_pos[TARGET_ID].nunique()}"
    )
    return df_pos


def build_record_maps(df_pos: pd.DataFrame) -> Tuple[Dict, Dict]:
    compound_df = (
        df_pos[COMPOUND_COLS]
        .groupby(COMPOUND_ID, sort=False, as_index=False)
        .first()
    )
    target_df = (
        df_pos[TARGET_COLS]
        .groupby(TARGET_ID, sort=False, as_index=False)
        .first()
    )
    compound_records = compound_df.set_index(COMPOUND_ID).to_dict("index")
    target_records = target_df.set_index(TARGET_ID).to_dict("index")
    return compound_records, target_records


def make_negative_df(
    pairs: List[Tuple[str, str]], compound_records: Dict, target_records: Dict
) -> pd.DataFrame:
    rows = []
    for cpd_id, tgt_id in pairs:
        cpd = compound_records[cpd_id]
        tgt = target_records[tgt_id]
        rows.append(
            {
                COMPOUND_ID: cpd_id,
                HELM_COL: cpd[HELM_COL],
                SMILES_COL: cpd[SMILES_COL],
                TARGET_ID: tgt_id,
                TARGET_NAME: tgt[TARGET_NAME],
                TARGET_ACCESSION: tgt[TARGET_ACCESSION],
                TARGET_SEQ: tgt[TARGET_SEQ],
                PROTEIN_CLASS: tgt[PROTEIN_CLASS],
                LABEL_COL: 0,
            }
        )
    return pd.DataFrame(rows)


def generate_negatives(
    n_negative: int,
    compound_pool: List[str],
    target_pool: List[str],
    excluded_pairs: Set[Tuple[str, str]],
    seed: int,
    compound_records: Dict,
    target_records: Dict,
) -> Tuple[pd.DataFrame, Set[Tuple[str, str]]]:
    cpd_arr = np.array(sorted(compound_pool))
    tgt_arr = np.array(sorted(target_pool))
    rng = np.random.default_rng(seed)

    result: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    batch = max(10_000, n_negative * 10)
    attempts = 0

    while len(result) < n_negative and attempts < 200:
        ci = rng.integers(0, len(cpd_arr), batch)
        ti = rng.integers(0, len(tgt_arr), batch)
        for c, t in zip(cpd_arr[ci], tgt_arr[ti]):
            pair = (c, t)
            if pair not in excluded_pairs and pair not in seen:
                result.append(pair)
                seen.add(pair)
                if len(result) >= n_negative:
                    break
        attempts += 1

    if len(result) < n_negative:
        logger.warning(f"Only generated {len(result)}/{n_negative} negatives")

    generated = set(result[:n_negative])
    return make_negative_df(result[:n_negative], compound_records, target_records), generated


def build_target_groups(df_pos: pd.DataFrame) -> List[Dict]:
    """Build one group per target with its positive-pair count."""
    groups: List[Dict] = []
    for target_id, target_df in df_pos.groupby(TARGET_ID):
        groups.append(
            {
                "id": str(target_id),
                "targets": {target_id},
                "pairs": len(target_df),
            }
        )
    logger.info(f"Built {len(groups)} target groups")
    return groups


def assign_targets_to_folds_balanced(
    groups: List[Dict],
    n_folds: int,
    seed: int,
    max_deviation: float = 0.15,
) -> Dict[str, int]:
    """Assign individual targets to folds using constrained balanced partitioning."""
    total_pairs = sum(g["pairs"] for g in groups)
    target = total_pairs / n_folds
    min_weight = target * (1 - max_deviation)
    max_weight = target * (1 + max_deviation)

    logger.info(
        f"Assigning {len(groups)} targets to {n_folds} folds "
        f"(max_deviation={max_deviation:.0%})"
    )
    logger.info(f"Total pairs: {total_pairs}, target per fold: {target:.0f}")
    logger.info(f"Allowed range: [{min_weight:.0f}, {max_weight:.0f}]")

    n_groups = len(groups)
    sorted_indices = sorted(range(n_groups), key=lambda i: groups[i]["pairs"], reverse=True)

    assignments = [0] * n_groups
    fold_weights = np.zeros(n_folds)
    for idx in sorted_indices:
        best_fold = int(np.argmin(fold_weights))
        assignments[idx] = best_fold
        fold_weights[best_fold] += groups[idx]["pairs"]

    rng = np.random.default_rng(seed)
    for iteration in range(100):
        sources = [fold for fold in range(n_folds) if fold_weights[fold] > max_weight]
        sinks = [fold for fold in range(n_folds) if fold_weights[fold] < min_weight]

        if not sources and not sinks:
            logger.info(f"Converged at iteration {iteration}: all folds within constraints")
            break

        if sources and not sinks:
            sinks = [fold for fold in range(n_folds) if fold not in sources]
        elif sinks and not sources:
            sources = [fold for fold in range(n_folds) if fold not in sinks]

        rng.shuffle(sources)
        rng.shuffle(sinks)

        best_move = None
        best_improvement = -float("inf")
        for source_fold in sources:
            candidates = [i for i in range(n_groups) if assignments[i] == source_fold]
            for idx in candidates:
                for sink_fold in sinks:
                    if fold_weights[sink_fold] + groups[idx]["pairs"] > max_weight * 1.05:
                        continue

                    old_max_dev = max(
                        abs(fold_weights[source_fold] - target),
                        abs(fold_weights[sink_fold] - target),
                    )
                    new_max_dev = max(
                        abs(fold_weights[source_fold] - groups[idx]["pairs"] - target),
                        abs(fold_weights[sink_fold] + groups[idx]["pairs"] - target),
                    )
                    improvement = old_max_dev - new_max_dev

                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_move = (idx, source_fold, sink_fold)

        if best_move and best_improvement > 0:
            idx, source_fold, sink_fold = best_move
            assignments[idx] = sink_fold
            fold_weights[source_fold] -= groups[idx]["pairs"]
            fold_weights[sink_fold] += groups[idx]["pairs"]
        else:
            logger.info(f"No improving moves at iteration {iteration}, stopping")
            break

    group_to_fold: Dict[str, int] = {}
    for i, group in enumerate(groups):
        group_to_fold[group["id"]] = assignments[i]

    logger.info("Final fold assignment:")
    for fold_idx in range(n_folds):
        fold_groups = [g for i, g in enumerate(groups) if assignments[i] == fold_idx]
        n_targets = len(fold_groups)
        deviation = abs(fold_weights[fold_idx] - target) / target if target else 0.0
        logger.info(
            f"  Fold {fold_idx}: {fold_weights[fold_idx]:.0f} pairs "
            f"({n_targets} targets, deviation: {deviation:.1%})"
        )

    return group_to_fold


def verify_and_log(result_df: pd.DataFrame, test_fold: int) -> None:
    logger.info(f"\n{'=' * 60}")
    logger.info("Verification — cold")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total rows: {len(result_df)}")

    pos_count = int((result_df[LABEL_COL] == 1).sum())
    neg_count = int((result_df[LABEL_COL] == 0).sum())
    logger.info(f"Positives: {pos_count}, Negatives: {neg_count}, Ratio: 1:{neg_count / max(1, pos_count):.1f}")

    for s1, s2 in [("train", "test")]:
        p1 = set(
            zip(
                result_df[result_df[SPLIT_COL] == s1][COMPOUND_ID],
                result_df[result_df[SPLIT_COL] == s1][TARGET_ID],
            )
        )
        p2 = set(
            zip(
                result_df[result_df[SPLIT_COL] == s2][COMPOUND_ID],
                result_df[result_df[SPLIT_COL] == s2][TARGET_ID],
            )
        )
        overlap = p1 & p2
        if overlap:
            raise AssertionError(f"{s1}-{s2} pair overlap: {len(overlap)}")

    pos_pairs = set(
        zip(
            result_df[result_df[LABEL_COL] == 1][COMPOUND_ID],
            result_df[result_df[LABEL_COL] == 1][TARGET_ID],
        )
    )
    neg_pairs = set(
        zip(
            result_df[result_df[LABEL_COL] == 0][COMPOUND_ID],
            result_df[result_df[LABEL_COL] == 0][TARGET_ID],
        )
    )
    if pos_pairs & neg_pairs:
        raise AssertionError("pos-neg collision")

    for split_name in ["train", "test"]:
        sd = result_df[result_df[SPLIT_COL] == split_name]
        n_pos = int((sd[LABEL_COL] == 1).sum())
        n_neg = int((sd[LABEL_COL] == 0).sum())
        logger.info(f"  {split_name}: {n_pos} pos + {n_neg} neg = {len(sd)}")

    train_pos = result_df[(result_df[SPLIT_COL] == "train") & (result_df[LABEL_COL] == 1)]
    test_pos = result_df[(result_df[SPLIT_COL] == "test") & (result_df[LABEL_COL] == 1)]

    target_overlap = set(train_pos[TARGET_ID]) & set(test_pos[TARGET_ID])
    if target_overlap:
        raise AssertionError(f"train-test target overlap: {len(target_overlap)}")
    logger.info("  train-test target overlap: 0 (enforced)")
    logger.info(f"  selected test fold: {test_fold}")

    logger.info("All verifications passed")


def main() -> None:
    global DEFAULT_LOG_DIR

    parser = argparse.ArgumentParser(description="ChEMBL PPI cold single split")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--negative-ratio", type=int, default=NEGATIVE_RATIO)
    parser.add_argument("--activity-col", type=str, default=ACTIVITY_COL)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--max-deviation", type=float, default=0.15)
    args = parser.parse_args()

    DEFAULT_LOG_DIR = Path(args.log_dir)
    L.seed_everything(args.seed, workers=True)
    setup_logging()

    logger.info("=" * 60)
    logger.info("ChEMBL PPI Cold Single Split")
    logger.info("=" * 60)
    logger.info(f"Source: {args.source}")
    logger.info(f"Seed: {args.seed}, Neg ratio: 1:{args.negative_ratio}")
    logger.info(
        f"Cold split: balanced target folds with one fold as test "
        f"(target test ratio ~{args.test_ratio:.2f})"
    )

    df_pos = load_positives(Path(args.source), args.activity_col)
    compound_records, target_records = build_record_maps(df_pos)
    global_positive_pairs = set(zip(df_pos[COMPOUND_ID], df_pos[TARGET_ID]))
    total_compound_count = int(df_pos[COMPOUND_ID].nunique())
    total_target_count = int(df_pos[TARGET_ID].nunique())

    logger.info("=" * 60)
    logger.info("Building target groups")
    logger.info("=" * 60)
    n_balance_folds = max(2, round(1 / args.test_ratio))
    if abs((1 / args.test_ratio) - n_balance_folds) > 0.05:
        logger.warning(
            f"test_ratio={args.test_ratio:.3f} does not map cleanly to an integer number of folds; "
            f"using {n_balance_folds} balance folds"
        )
    groups = build_target_groups(df_pos)

    logger.info("=" * 60)
    logger.info("Assigning targets to folds (constrained balanced)")
    logger.info("=" * 60)
    group_to_fold = assign_targets_to_folds_balanced(
        groups=groups,
        n_folds=n_balance_folds,
        seed=args.seed,
        max_deviation=args.max_deviation,
    )

    target_to_fold: Dict[str, int] = {}
    for group in groups:
        for target_id in group["targets"]:
            target_to_fold[target_id] = group_to_fold[group["id"]]

    total_pairs = sum(g["pairs"] for g in groups)
    fold_pair_counts = np.zeros(n_balance_folds)
    for g in groups:
        fold_pair_counts[group_to_fold[g["id"]]] += g["pairs"]
    fold_ratios = fold_pair_counts / total_pairs
    test_fold = int(np.argmin(np.abs(fold_ratios - args.test_ratio)))
    logger.info(
        f"Selected fold {test_fold} as test fold "
        f"(ratio={fold_ratios[test_fold]:.3f}, target={args.test_ratio:.3f})"
    )

    test_targets = {tid for tid, fold in target_to_fold.items() if fold == test_fold}
    train_targets = {tid for tid, fold in target_to_fold.items() if fold != test_fold}

    train_pos = df_pos[df_pos[TARGET_ID].isin(sorted(train_targets))].copy().reset_index(drop=True)
    test_pos = df_pos[df_pos[TARGET_ID].isin(sorted(test_targets))].copy().reset_index(drop=True)
    logger.info(
        f"Selected test set: {len(test_targets)} targets, {len(test_pos)} positive pairs "
        f"({len(test_pos) / len(df_pos):.3f} of dataset)"
    )
    logger.info(
        f"Targets: test={len(test_targets)}/{total_target_count}, "
        f"train={len(train_targets)}/{total_target_count}"
    )
    logger.info(f"Positives: train={len(train_pos)}, test={len(test_pos)}")

    pair_registry: Set[Tuple[str, str]] = set(global_positive_pairs)
    all_parts = []

    for split_name, pos_df, seed_offset in [("test", test_pos, 1), ("train", train_pos, 0)]:
        split_compounds = sorted(set(pos_df[COMPOUND_ID]))
        split_targets = sorted(set(pos_df[TARGET_ID]))
        n_neg = len(pos_df) * args.negative_ratio
        split_seed = get_fold_seed(args.seed, test_fold, seed_offset)

        neg_df, generated_pairs = generate_negatives(
            n_neg,
            split_compounds,
            split_targets,
            pair_registry,
            split_seed,
            compound_records,
            target_records,
        )
        pair_registry.update(generated_pairs)

        combined = pd.concat([pos_df, neg_df], ignore_index=True)
        combined = combined.sample(frac=1, random_state=split_seed).reset_index(drop=True)
        combined[SPLIT_COL] = split_name
        all_parts.append(combined)

        logger.info(f"  {split_name}: {len(pos_df)} pos + {len(neg_df)} neg = {len(combined)}")

    result_df = pd.concat(all_parts, ignore_index=True)

    verify_and_log(result_df, test_fold)

    for split_name in ["train", "test"]:
        split_df = result_df.loc[result_df[SPLIT_COL] == split_name, :].copy()
        log_mlm_coverage(split_df, HELM_COL, SMILES_COL, REPO_ROOT, logger, f"chembl_ppi_cold/{split_name}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_cols = [
        COMPOUND_ID,
        HELM_COL,
        SMILES_COL,
        TARGET_ID,
        TARGET_NAME,
        TARGET_ACCESSION,
        TARGET_SEQ,
        PROTEIN_CLASS,
        LABEL_COL,
    ]
    for split_name in ["train", "test"]:
        split_df = result_df.loc[result_df[SPLIT_COL] == split_name, :].copy()
        out_path = out_dir / f"chembl_ppi_cold_{split_name}.csv"
        split_df[out_cols].to_csv(out_path, index=False)
        logger.info(f"Saved: {out_path} ({len(split_df):,} rows)")

    meta = {
        "method": "cold single split (balanced target folds)",
        "source": str(args.source),
        "activity_column": args.activity_col,
        "seed": args.seed,
        "negative_ratio": args.negative_ratio,
        "test_ratio": args.test_ratio,
        "n_balance_folds": n_balance_folds,
        "selected_test_fold": test_fold,
        "max_deviation": args.max_deviation,
        "n_positive_pairs": len(df_pos),
        "n_unique_compounds": total_compound_count,
        "n_unique_targets": total_target_count,
        "n_train_targets": len(train_targets),
        "n_test_targets": len(test_targets),
        "actual_test_size": len(test_pos),
    }
    assert log_dir is not None
    meta_path = log_dir / "preparation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata: {meta_path}")
    logger.info("\nDone!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed: {e}", exc_info=True)
        sys.exit(1)
