#!/usr/bin/env python
"""Visualize SST2 train/val/test split distributions via t-SNE.

Uses Morgan fingerprints (ECFP4) from SMILES to embed molecules into 2D.

Usage:
    # Combined figure: Random + Scaffold side by side (default)
    python scripts/visualize_sst2_splits.py

    # Single split only
    python scripts/visualize_sst2_splits.py --split random
    python scripts/visualize_sst2_splits.py --split scaffold

    # Legend bar only
    python scripts/visualize_sst2_splits.py --legend

Output:
    results/visualization/tsne_sst2_splits_{timestamp}.pdf/.png
    results/visualization/tsne_sst2_{split}_{timestamp}.pdf/.png
    results/visualization/tsne_sst2_legend_{timestamp}.pdf/.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "visualization"
DEFAULT_LOG_DIR = REPO_ROOT / "outputs" / "visualization"

SEED = 42
VAL_RATIO = 0.1
PCA_COMPONENTS = 50
FP_RADIUS = 2
FP_NBITS = 2048
JITTER_SCALE = 0.3

SMILES_COL = "canonical_smiles"

ALL_SPLITS: List[Dict[str, str]] = [
    {
        "name": "Random Split",
        "tag": "random",
        "train_file": "data/downstream/sst2_random_train.csv",
        "test_file": "data/downstream/sst2_random_test.csv",
    },
    {
        "name": "Scaffold Split",
        "tag": "scaffold",
        "train_file": "data/downstream/sst2_scaffold_train.csv",
        "test_file": "data/downstream/sst2_scaffold_test.csv",
    },
]

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

SPLIT_COLORS = {"train": "#4C72B0", "val": "#55A868", "test": "#DD8452"}

LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"visualize_sst2_splits_{timestamp}.log"

    lg = logging.getLogger(__name__)
    lg.setLevel(LOG_LEVEL)
    lg.handlers = []

    for handler_cls, args in [
        (logging.StreamHandler, (sys.stdout,)),
        (logging.FileHandler, (log_file,)),
    ]:
        handler = handler_cls(*args)
        handler.setLevel(LOG_LEVEL)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        lg.addHandler(handler)

    lg.info(f"Log file: {log_file.absolute()}")
    return lg


def smiles_to_fingerprints(smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert SMILES to Morgan fingerprints. Returns (fps, valid_indices)."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)
    fingerprints, valid_idx = [], []
    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fingerprint = generator.GetFingerprintAsNumPy(mol)
        fingerprints.append(fingerprint.astype(np.float32))
        valid_idx.append(idx)

    logger.info(f"Fingerprints: {len(fingerprints)}/{len(smiles_list)} molecules valid")
    return np.array(fingerprints), np.array(valid_idx)


def run_tsne(X: np.ndarray) -> np.ndarray:
    """StandardScaler -> PCA(50) -> t-SNE(2)."""
    X = StandardScaler().fit_transform(X)
    if X.shape[1] > PCA_COMPONENTS:
        pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        X = pca.fit_transform(X)
        logger.info(
            f"PCA: {PCA_COMPONENTS} dims, explained variance: "
            f"{pca.explained_variance_ratio_.sum():.3f}"
        )

    logger.info(f"Running t-SNE on {X.shape}...")
    return TSNE(
        n_components=2,
        random_state=SEED,
        perplexity=30,
        init="pca",
        learning_rate="auto",
    ).fit_transform(X)


def get_split_data(split_info: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    """Load train/test CSVs, compute fingerprints, run t-SNE."""
    train_df = pd.read_csv(REPO_ROOT / split_info["train_file"])
    test_df = pd.read_csv(REPO_ROOT / split_info["test_file"])

    train_split, val_split = train_test_split(
        train_df,
        test_size=VAL_RATIO,
        random_state=SEED,
    )
    logger.info(
        f"  Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_df)}"
    )

    all_data = pd.concat([train_split, val_split, test_df], ignore_index=True)
    split_labels = np.array(
        ["train"] * len(train_split)
        + ["val"] * len(val_split)
        + ["test"] * len(test_df)
    )

    X, valid_idx = smiles_to_fingerprints(all_data[SMILES_COL].tolist())
    split_labels = split_labels[valid_idx]

    logger.info(f"  Embeddings: {X.shape}")
    Z = run_tsne(X)
    return Z, split_labels


def _scatter_splits(ax, Z: np.ndarray, split_labels: np.ndarray, jitter: float = 0.0) -> None:
    """Draw split-colored scatter on an axis."""
    rng = np.random.RandomState(SEED)
    permutation = rng.permutation(len(Z))
    Z = Z[permutation]
    split_labels = split_labels[permutation]

    if jitter > 0:
        Z = Z + rng.normal(0, jitter, Z.shape)

    for split in ["train", "val", "test"]:
        mask = split_labels == split
        if mask.any():
            ax.scatter(
                Z[mask, 0],
                Z[mask, 1],
                c=SPLIT_COLORS[split],
                alpha=0.7,
                s=40,
                edgecolors="none",
            )

    ax.set_xlabel("t-SNE 1", fontsize=13, fontweight="bold")
    ax.set_ylabel("t-SNE 2", fontsize=13, fontweight="bold")


def _save(fig, base: Path) -> None:
    """Save figure as PDF + PNG."""
    fig.savefig(f"{base}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", format="png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {base}.pdf")
    logger.info(f"Saved: {base}.png")


def _legend_handles():
    return [mpatches.Patch(color=SPLIT_COLORS[split], label=split) for split in ["train", "val", "test"]]


def run_legend(results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 0.6))
    ax.axis("off")
    ax.legend(
        handles=_legend_handles(),
        loc="center",
        ncol=3,
        fontsize=13,
        frameon=True,
        shadow=True,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=2.0,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save(fig, results_dir / f"tsne_sst2_legend_{timestamp}")


def run_single(split_tag: str, results_dir: Path, jitter: float) -> None:
    split_info = next(split for split in ALL_SPLITS if split["tag"] == split_tag)
    logger.info(f"Processing: {split_info['name']}")

    Z, split_labels = get_split_data(split_info)

    fig, ax = plt.subplots(figsize=(10, 10))
    _scatter_splits(ax, Z, split_labels, jitter=jitter)
    ax.set_title(split_info["name"], fontsize=15, fontweight="bold")
    fig.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save(fig, results_dir / f"tsne_sst2_{split_tag}_{timestamp}")


def run_combined(results_dir: Path, jitter: float) -> None:
    all_smiles: set[str] = set()
    split_dfs = []
    for split_info in ALL_SPLITS:
        train_df = pd.read_csv(REPO_ROOT / split_info["train_file"])
        test_df = pd.read_csv(REPO_ROOT / split_info["test_file"])
        all_smiles.update(train_df[SMILES_COL])
        all_smiles.update(test_df[SMILES_COL])
        split_dfs.append((split_info, train_df, test_df))

    canonical_list = sorted(all_smiles)
    smiles_to_idx = {s: i for i, s in enumerate(canonical_list)}

    logger.info(f"Computing shared t-SNE on {len(canonical_list)} unique molecules")
    X, valid_idx = smiles_to_fingerprints(canonical_list)
    Z = run_tsne(X)

    valid_smiles = [canonical_list[i] for i in valid_idx]
    smiles_to_z = {s: Z[i] for i, s in enumerate(valid_smiles)}

    panels = []
    for split_info, train_df, test_df in split_dfs:
        train_split, val_split = train_test_split(
            train_df, test_size=VAL_RATIO, random_state=SEED,
        )
        logger.info(
            f"  {split_info['name']}: Train={len(train_split)}, "
            f"Val={len(val_split)}, Test={len(test_df)}"
        )

        z_list, label_list = [], []
        for label, df_part in [("train", train_split), ("val", val_split), ("test", test_df)]:
            for s in df_part[SMILES_COL]:
                if s in smiles_to_z:
                    z_list.append(smiles_to_z[s])
                    label_list.append(label)

        panels.append((np.array(z_list), np.array(label_list), split_info["name"]))

    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 7))
    if len(panels) == 1:
        axes = [axes]

    for ax, (Z_panel, split_labels, subtitle) in zip(axes, panels):
        _scatter_splits(ax, Z_panel, split_labels, jitter=jitter)
        ax.set_title(subtitle, fontsize=15, fontweight="bold")

    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=3,
        fontsize=13,
        frameon=True,
        shadow=True,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=2.5,
    )
    fig.suptitle(
        "t-SNE of SST2 Splits (Morgan Fingerprints)",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save(fig, results_dir / f"tsne_sst2_splits_{timestamp}")


def main() -> None:
    global logger

    parser = argparse.ArgumentParser(
        description="Visualize SST2 train/val/test split distributions via t-SNE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/visualize_sst2_splits.py                     # combined\n"
            "  python scripts/visualize_sst2_splits.py --split random      # single\n"
            "  python scripts/visualize_sst2_splits.py --split scaffold    # single\n"
            "  python scripts/visualize_sst2_splits.py --legend            # legend only\n"
            "  python scripts/visualize_sst2_splits.py --jitter 0.5        # custom jitter\n"
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["random", "scaffold"],
        help="Single split to visualize. Omit for combined.",
    )
    parser.add_argument("--legend", action="store_true", help="Output legend bar only.")
    parser.add_argument(
        "--jitter",
        type=float,
        default=JITTER_SCALE,
        help=f"Jitter scale for scatter (0 = off). Default: {JITTER_SCALE}",
    )
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(Path(args.log_dir))

    logger.info("=" * 60)
    if args.legend:
        logger.info("SST2 Split Visualization: legend only")
    elif args.split:
        logger.info(f"SST2 Split Visualization: single ({args.split})")
    else:
        logger.info("SST2 Split Visualization: combined (all splits)")
    logger.info("=" * 60)
    logger.info(f"Results: {results_dir}")
    logger.info(f"Jitter: {args.jitter}")

    if args.legend:
        run_legend(results_dir)
    elif args.split:
        run_single(args.split, results_dir, jitter=args.jitter)
    else:
        run_combined(results_dir, jitter=args.jitter)

    logger.info("Done!")


if __name__ == "__main__":
    main()
