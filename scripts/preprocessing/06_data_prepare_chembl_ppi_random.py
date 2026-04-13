"""Data preparation for ChEMBL peptide-protein binding — random split.

Single train/test split with random pair assignment.

Positives: compound-target pairs with binding activity <= 1 μM (t_1u == 1).
Negatives: all random (no measured negatives).

Source: local_data/intermediate_product/chembl_ppi_helm_normalized.csv
Output: data/downstream/chembl_ppi_random_{train,test}.csv
"""

import os
thread_count = str(min(8, os.cpu_count() or 8))
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key, thread_count)

import pandas as pd
import numpy as np
import logging
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Set, Tuple, List, Dict
import lightning as L

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.preprocessing.preprocessing_utils.downstream_utils import log_mlm_coverage
from scripts.preprocessing.preprocessing_utils.paths import (
    INTERMEDIATE_PRODUCT_DIR,
    PREPROCESSING_OUTPUT_DIR,
    REPO_ROOT,
)

DEFAULT_SOURCE = INTERMEDIATE_PRODUCT_DIR / 'chembl_ppi_helm_normalized.csv'
DEFAULT_OUTPUT_DIR = REPO_ROOT / 'data/downstream'
DEFAULT_LOG_DIR = PREPROCESSING_OUTPUT_DIR

SEED = 42
NEGATIVE_RATIO = 4
ACTIVITY_COL = 't_1u'
TEST_RATIO = 0.1

COMPOUND_ID = 'compound_chembl_id'
HELM_COL = 'helm_notation'
SMILES_COL = 'canonical_smiles'
TARGET_ID = 'target_chembl_id'
TARGET_NAME = 'target_name'
TARGET_ACCESSION = 'target_accession'
TARGET_SEQ = 'target_sequence'
PROTEIN_CLASS = 'protein_class_desc'

LABEL_COL = 'Label'
SPLIT_COL = 'split'

COMPOUND_COLS = [COMPOUND_ID, HELM_COL, SMILES_COL]
TARGET_COLS = [TARGET_ID, TARGET_NAME, TARGET_ACCESSION, TARGET_SEQ, PROTEIN_CLASS]

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
log_dir = None


def setup_logging() -> Path:
    global log_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(DEFAULT_LOG_DIR) / f"chembl_ppi_random_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.INFO)
    logger.handlers = []
    for handler in [logging.StreamHandler(sys.stdout),
                    logging.FileHandler(log_dir / 'prepare.log')]:
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    logger.info(f"Log directory: {log_dir}")
    return log_dir


def load_positives(source: Path, activity_col: str = ACTIVITY_COL) -> pd.DataFrame:
    df = pd.read_csv(source)
    logger.info(f"Loaded {len(df)} rows from {source}")

    pair_key = [COMPOUND_ID, TARGET_ID]
    pair_activity = df.groupby(pair_key)[activity_col].max().reset_index()
    pos_pairs = pair_activity[pair_activity[activity_col] == 1][pair_key]
    logger.info(f"Positive pairs (t_1u==1 after dedup): {len(pos_pairs)}")

    df_pos = df.merge(pos_pairs, on=pair_key, how='inner')
    df_pos = df_pos.drop_duplicates(subset=pair_key, keep='first').reset_index(drop=True)
    df_pos[LABEL_COL] = 1

    logger.info(f"Positive rows after dedup: {len(df_pos)}")
    logger.info(f"Unique compounds: {df_pos[COMPOUND_ID].nunique()}, "
                f"unique targets: {df_pos[TARGET_ID].nunique()}")
    return df_pos


def build_record_maps(df_pos: pd.DataFrame) -> Tuple[Dict, Dict]:
    compound_records = (
        df_pos[COMPOUND_COLS]
        .drop_duplicates(COMPOUND_ID)
        .set_index(COMPOUND_ID)
        .to_dict('index')
    )
    target_records = (
        df_pos[TARGET_COLS]
        .drop_duplicates(TARGET_ID)
        .set_index(TARGET_ID)
        .to_dict('index')
    )
    return compound_records, target_records


def make_negative_df(pairs: List[Tuple[str, str]],
                     compound_records: Dict, target_records: Dict) -> pd.DataFrame:
    rows = []
    for cpd_id, tgt_id in pairs:
        cpd = compound_records[cpd_id]
        tgt = target_records[tgt_id]
        rows.append({
            COMPOUND_ID: cpd_id,
            HELM_COL: cpd[HELM_COL],
            SMILES_COL: cpd[SMILES_COL],
            TARGET_ID: tgt_id,
            TARGET_NAME: tgt[TARGET_NAME],
            TARGET_ACCESSION: tgt[TARGET_ACCESSION],
            TARGET_SEQ: tgt[TARGET_SEQ],
            PROTEIN_CLASS: tgt[PROTEIN_CLASS],
            LABEL_COL: 0,
        })
    return pd.DataFrame(rows)


def generate_negatives(n_negative: int,
                       compound_pool: List[str],
                       target_pool: List[str],
                       excluded_pairs: Set[Tuple[str, str]],
                       seed: int,
                       compound_records: Dict,
                       target_records: Dict) -> pd.DataFrame:
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

    return make_negative_df(result[:n_negative], compound_records, target_records)


def verify_and_log(result_df: pd.DataFrame):
    logger.info(f"\n{'=' * 60}")
    logger.info("Verification")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total rows: {len(result_df)}")

    pos_count = int((result_df[LABEL_COL] == 1).sum())
    neg_count = int((result_df[LABEL_COL] == 0).sum())
    logger.info(f"Positives: {pos_count}, Negatives: {neg_count}, Ratio: 1:{neg_count/max(1,pos_count):.1f}")

    for s1, s2 in [('train', 'test')]:
        p1 = set(zip(result_df[result_df[SPLIT_COL] == s1][COMPOUND_ID],
                      result_df[result_df[SPLIT_COL] == s1][TARGET_ID]))
        p2 = set(zip(result_df[result_df[SPLIT_COL] == s2][COMPOUND_ID],
                      result_df[result_df[SPLIT_COL] == s2][TARGET_ID]))
        overlap = p1 & p2
        if overlap:
            raise AssertionError(f"{s1}-{s2} pair overlap: {len(overlap)}")

    pos_pairs = set(zip(
        result_df[result_df[LABEL_COL] == 1][COMPOUND_ID],
        result_df[result_df[LABEL_COL] == 1][TARGET_ID]))
    neg_pairs = set(zip(
        result_df[result_df[LABEL_COL] == 0][COMPOUND_ID],
        result_df[result_df[LABEL_COL] == 0][TARGET_ID]))
    if pos_pairs & neg_pairs:
        raise AssertionError("pos-neg collision")

    for split_name in ['train', 'test']:
        sd = result_df[result_df[SPLIT_COL] == split_name]
        n_pos = int((sd[LABEL_COL] == 1).sum())
        n_neg = int((sd[LABEL_COL] == 0).sum())
        logger.info(f"  {split_name}: {n_pos} pos + {n_neg} neg = {len(sd)}")

    logger.info("All verifications passed")


def main():
    global DEFAULT_LOG_DIR

    parser = argparse.ArgumentParser(description='ChEMBL PPI random single split')
    parser.add_argument('--source', type=str, default=str(DEFAULT_SOURCE))
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--log-dir', type=str, default=str(DEFAULT_LOG_DIR))
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--negative-ratio', type=int, default=NEGATIVE_RATIO)
    parser.add_argument('--activity-col', type=str, default=ACTIVITY_COL)
    parser.add_argument('--test-ratio', type=float, default=TEST_RATIO)
    args = parser.parse_args()

    DEFAULT_LOG_DIR = Path(args.log_dir)
    L.seed_everything(args.seed, workers=True)
    setup_logging()

    logger.info("=" * 60)
    logger.info("ChEMBL PPI Random Single Split")
    logger.info("=" * 60)
    logger.info(f"Source: {args.source}")
    logger.info(f"Seed: {args.seed}, Neg ratio: 1:{args.negative_ratio}")
    logger.info(f"Split: test={args.test_ratio}, train={1 - args.test_ratio:.2f}")

    df_pos = load_positives(Path(args.source), args.activity_col)
    compound_records, target_records = build_record_maps(df_pos)
    global_positive_pairs = set(zip(df_pos[COMPOUND_ID], df_pos[TARGET_ID]))

    # Shuffle and split positives
    df_pos = df_pos.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_total = len(df_pos)
    n_test = max(1, int(n_total * args.test_ratio))
    test_pos = df_pos.iloc[:n_test]
    train_pos = df_pos.iloc[n_test:]

    logger.info(f"Positives: test={len(test_pos)}, train={len(train_pos)}")

    # Generate negatives per split
    all_compounds = sorted(df_pos[COMPOUND_ID].unique())
    all_targets = sorted(df_pos[TARGET_ID].unique())

    pair_registry: Set[Tuple[str, str]] = set(global_positive_pairs)
    all_parts = []

    for split_name, pos_df, seed_offset in [
        ('test', test_pos, 1),
        ('train', train_pos, 0),
    ]:
        n_neg = len(pos_df) * args.negative_ratio
        split_seed = args.seed + seed_offset

        neg_df = generate_negatives(
            n_neg, all_compounds, all_targets,
            pair_registry, split_seed,
            compound_records, target_records
        )
        neg_pairs = set(zip(neg_df[COMPOUND_ID], neg_df[TARGET_ID]))
        pair_registry.update(neg_pairs)

        combined = pd.concat([pos_df, neg_df], ignore_index=True)
        combined = combined.sample(frac=1, random_state=split_seed).reset_index(drop=True)
        combined[SPLIT_COL] = split_name
        all_parts.append(combined)

        logger.info(f"  {split_name}: {len(pos_df)} pos + {len(neg_df)} neg = {len(combined)}")

    result_df = pd.concat(all_parts, ignore_index=True)

    verify_and_log(result_df)

    for split_name in ['train', 'test']:
        split_df = result_df[result_df[SPLIT_COL] == split_name].copy()
        log_mlm_coverage(split_df, HELM_COL, SMILES_COL, REPO_ROOT, logger, f"chembl_ppi_random/{split_name}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_cols = [COMPOUND_ID, HELM_COL, SMILES_COL,
                TARGET_ID, TARGET_NAME, TARGET_ACCESSION, TARGET_SEQ, PROTEIN_CLASS,
                LABEL_COL]
    for split_name in ['train', 'test']:
        split_df = result_df[result_df[SPLIT_COL] == split_name].copy()
        out_path = out_dir / f'chembl_ppi_random_{split_name}.csv'
        split_df[out_cols].to_csv(out_path, index=False)
        logger.info(f"Saved: {out_path} ({len(split_df):,} rows)")

    meta = {
        'method': 'random single split',
        'source': str(args.source),
        'activity_column': args.activity_col,
        'seed': args.seed,
        'negative_ratio': args.negative_ratio,
        'test_ratio': args.test_ratio,
        'n_positive_pairs': len(df_pos),
        'n_unique_compounds': df_pos[COMPOUND_ID].nunique(),
        'n_unique_targets': df_pos[TARGET_ID].nunique(),
    }
    meta_path = log_dir / 'preparation_metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata: {meta_path}")
    logger.info("\nDone!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed: {e}", exc_info=True)
        sys.exit(1)
