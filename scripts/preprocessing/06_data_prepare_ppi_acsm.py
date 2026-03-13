#!/usr/bin/env python
"""Prepare PPI data with aCSM clustering-based train/test split.

Uses aCSM-ALL molecular signatures to cluster peptide-protein complexes,
then splits by cluster to prevent structural data leakage between train/test.

Pipeline:
1. Load positive pairs with PDB IDs
2. Load pre-computed aCSM signatures (from 05_data_generate_acsm_signatures.py)
3. Cluster complexes via K-Means on PCA-reduced signatures
4. Split clusters into train/test groups
5. Generate negative pairs per split
6. Output: propedia_ppi_acsm_train.csv, propedia_ppi_acsm_test.csv
"""

import os
thread_count = str(min(8, os.cpu_count() or 8))
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key, thread_count)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
from typing import Set, Tuple, List, Dict

import numpy as np
import pandas as pd
import lightning as L
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "local_data/intermediate_product/Propedia_v2_unique_ppi_HELM_SMILES.csv"
DEFAULT_SIGNATURE_DIR = REPO_ROOT / "local_data/intermediate_product/signatures_acsm_all"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/downstream"

SEED = 42
TEST_RATIO = 0.2
NEGATIVE_RATIO = 4

# Clustering
N_CLUSTERS = 100
PCA_COMPONENTS = 50

# Column names (source)
PDB_COL = "PDB"
PEPTIDE_SEQ_COL = "Peptide_Sequence"
PROTEIN_SEQ_COL = "Receptor_Sequence"

# Column names (output — matches current repo format)
DRUG_COL = "Peptide_HELM"
TARGET_COL = "Receptor_Sequence"
LABEL_COL = "Label"

COMPLEX_SIGNATURE_FILE = "complex_signatures_acsm_all.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# aCSM signature loading & clustering
# ---------------------------------------------------------------------------

def load_signatures(signature_dir: Path) -> Tuple[Dict[str, np.ndarray], int]:
    """Load pre-computed aCSM-ALL complex signatures."""
    sig_path = signature_dir / COMPLEX_SIGNATURE_FILE
    if not sig_path.exists():
        raise FileNotFoundError(
            f"Signature file not found: {sig_path}. "
            "Run 05_data_generate_acsm_signatures.py first."
        )

    sig_df = pd.read_csv(sig_path, dtype={"pdb_id": "str"})
    sig_cols = [c for c in sig_df.columns if c.startswith("sig_")]
    sig_df = sig_df.sort_values("pdb_id").drop_duplicates("pdb_id")

    lookup = dict(zip(sig_df["pdb_id"].astype(str), sig_df[sig_cols].to_numpy(dtype=np.float32)))
    logger.info(f"Loaded {len(lookup)} complex signatures ({len(sig_cols)} dims)")
    return lookup, len(sig_cols)


def cluster_and_split(
    pdb_ids: List[str],
    sig_lookup: Dict[str, np.ndarray],
    n_features: int,
    n_clusters: int,
    test_ratio: float,
    seed: int,
) -> Tuple[Set[str], Set[str]]:
    """Cluster complexes and split clusters into train/test sets.

    Returns:
        (train_pdbs, test_pdbs) sets of PDB IDs
    """
    # Filter to PDBs with signatures
    available = [p for p in pdb_ids if p in sig_lookup]
    missing = len(pdb_ids) - len(available)
    if missing:
        logger.warning(f"{missing} PDBs missing signatures, excluded from clustering")
    if not available:
        raise ValueError("No PDBs with signatures")

    # Build feature matrix
    X = np.array([sig_lookup[p] for p in available])

    # Scale + PCA
    X_scaled = StandardScaler().fit_transform(X)
    n_components = min(PCA_COMPONENTS, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_components, random_state=seed)
    X_reduced = pca.fit_transform(X_scaled)
    logger.info(f"PCA: {X_scaled.shape[1]} -> {n_components} dims, "
                f"explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # K-Means
    actual_k = min(n_clusters, len(available))
    labels = KMeans(n_clusters=actual_k, random_state=seed, n_init=10).fit_predict(X_reduced)

    _, counts = np.unique(labels, return_counts=True)
    logger.info(f"Clustered {len(available)} complexes into {actual_k} clusters")
    logger.info(f"Cluster sizes: min={counts.min()}, max={counts.max()}, "
                f"mean={counts.mean():.1f}, std={counts.std():.1f}")

    pdb_to_cluster = {p: int(l) for p, l in zip(available, labels)}

    unique_clusters = sorted(set(pdb_to_cluster.values()))

    # Split clusters into test/train groups
    rng = np.random.default_rng(seed)
    cluster_order = list(unique_clusters)
    rng.shuffle(cluster_order)

    n_test_clusters = max(1, int(len(cluster_order) * test_ratio))
    test_clusters = set(cluster_order[:n_test_clusters])
    train_clusters = set(cluster_order[n_test_clusters:])

    train_pdbs = {p for p, c in pdb_to_cluster.items() if c in train_clusters}
    test_pdbs = {p for p, c in pdb_to_cluster.items() if c in test_clusters}

    logger.info(f"Split: {len(train_clusters)} train clusters ({len(train_pdbs)} PDBs), "
                f"{len(test_clusters)} test clusters ({len(test_pdbs)} PDBs)")

    return train_pdbs, test_pdbs


# ---------------------------------------------------------------------------
# Negative pair generation (same as random version)
# ---------------------------------------------------------------------------

def generate_negative_pairs(
    n_negative: int,
    peptides: List[str],
    proteins: List[str],
    positive_pairs: Set[Tuple[str, str]],
    existing_negatives: Set[Tuple[str, str]],
    seed: int,
) -> List[Tuple[str, str]]:
    """Generate negative pairs from peptide/protein pool."""
    peptides_array = np.array(sorted(peptides))
    proteins_array = np.array(sorted(proteins))

    excluded = positive_pairs | existing_negatives
    max_possible = len(peptides_array) * len(proteins_array) - len(excluded)
    if n_negative > max_possible:
        logger.warning(f"Requested {n_negative} negatives but only {max_possible} available")
        n_negative = max_possible

    rng = np.random.default_rng(seed)
    negative_pairs = []
    negative_set = set()
    batch_size = max(10000, n_negative * 10)

    for _ in range(100):
        pep_idx = rng.integers(0, len(peptides_array), batch_size)
        prot_idx = rng.integers(0, len(proteins_array), batch_size)
        for p, r in zip(peptides_array[pep_idx], proteins_array[prot_idx]):
            pair = (p, r)
            if pair not in excluded and pair not in negative_set:
                negative_pairs.append(pair)
                negative_set.add(pair)
                if len(negative_pairs) >= n_negative:
                    break
        if len(negative_pairs) >= n_negative:
            break

    if len(negative_pairs) < n_negative:
        logger.warning(f"Only generated {len(negative_pairs)}/{n_negative} negatives")

    return negative_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare PPI data (aCSM clustering split)")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE))
    parser.add_argument("--signature-dir", type=str, default=str(DEFAULT_SIGNATURE_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--negative-ratio", type=int, default=NEGATIVE_RATIO)
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    L.seed_everything(args.seed)

    source_file = Path(args.source)
    signature_dir = Path(args.signature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PPI Data Preparation (aCSM clustering split)")
    logger.info("=" * 60)
    logger.info(f"Source: {source_file}")
    logger.info(f"Signatures: {signature_dir}")
    logger.info(f"Test ratio: {args.test_ratio}")
    logger.info(f"Negative ratio: 1:{args.negative_ratio}")
    logger.info(f"Clusters: {args.n_clusters}")

    # Load data
    df = pd.read_csv(source_file)
    logger.info(f"Loaded {len(df)} rows")

    # Check required columns
    for col in [PDB_COL, DRUG_COL, TARGET_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Sort for deterministic dedup regardless of input order
    df = df.sort_values(by=[DRUG_COL, TARGET_COL, PDB_COL]).reset_index(drop=True)

    # Deduplicate
    original_len = len(df)
    df = df.drop_duplicates(subset=[DRUG_COL, TARGET_COL], keep="first")
    if len(df) < original_len:
        logger.info(f"Dropped {original_len - len(df)} duplicate pairs")
    logger.info(f"Unique positive pairs: {len(df)}")

    df[LABEL_COL] = 1
    global_positive_pairs = set(zip(df[DRUG_COL], df[TARGET_COL]))

    # Load signatures and cluster
    sig_lookup, n_features = load_signatures(signature_dir)
    unique_pdbs = sorted(df[PDB_COL].unique().tolist())

    train_pdbs, test_pdbs = cluster_and_split(
        unique_pdbs, sig_lookup, n_features, args.n_clusters, args.test_ratio, args.seed
    )

    # Split positive pairs by PDB cluster assignment
    train_pos = df[df[PDB_COL].isin(train_pdbs)].copy()
    test_pos = df[df[PDB_COL].isin(test_pdbs)].copy()

    # Handle pairs from PDBs without signatures (assign to train)
    unassigned = df[~df[PDB_COL].isin(train_pdbs | test_pdbs)]
    if len(unassigned) > 0:
        logger.info(f"Assigning {len(unassigned)} pairs from unmatched PDBs to train")
        train_pos = pd.concat([train_pos, unassigned], ignore_index=True)

    logger.info(f"Positive split: {len(train_pos)} train, {len(test_pos)} test")

    # Generate negatives
    train_peptides = list(train_pos[DRUG_COL].unique())
    train_proteins = list(train_pos[TARGET_COL].unique())
    test_peptides = list(test_pos[DRUG_COL].unique())
    test_proteins = list(test_pos[TARGET_COL].unique())

    logger.info(f"Train pool: {len(train_peptides)} peptides x {len(train_proteins)} proteins")
    logger.info(f"Test pool: {len(test_peptides)} peptides x {len(test_proteins)} proteins")

    logger.info("Generating train negatives...")
    train_neg_pairs = generate_negative_pairs(
        n_negative=len(train_pos) * args.negative_ratio,
        peptides=train_peptides, proteins=train_proteins,
        positive_pairs=global_positive_pairs, existing_negatives=set(),
        seed=args.seed,
    )
    logger.info(f"Generated {len(train_neg_pairs)} train negatives")

    all_negatives = set(train_neg_pairs)

    logger.info("Generating test negatives...")
    test_neg_pairs = generate_negative_pairs(
        n_negative=len(test_pos) * args.negative_ratio,
        peptides=test_peptides, proteins=test_proteins,
        positive_pairs=global_positive_pairs, existing_negatives=all_negatives,
        seed=args.seed + 1,
    )
    logger.info(f"Generated {len(test_neg_pairs)} test negatives")

    # Build output DataFrames
    output_cols = [DRUG_COL, TARGET_COL, LABEL_COL]

    train_neg_df = pd.DataFrame({
        DRUG_COL: [p[0] for p in train_neg_pairs],
        TARGET_COL: [p[1] for p in train_neg_pairs],
        LABEL_COL: 0,
    })
    test_neg_df = pd.DataFrame({
        DRUG_COL: [p[0] for p in test_neg_pairs],
        TARGET_COL: [p[1] for p in test_neg_pairs],
        LABEL_COL: 0,
    })

    train_df = pd.concat([train_pos[output_cols], train_neg_df], ignore_index=True)
    test_df = pd.concat([test_pos[output_cols], test_neg_df], ignore_index=True)

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
    train_file = output_dir / "propedia_ppi_acsm_train.csv"
    test_file = output_dir / "propedia_ppi_acsm_test.csv"

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
