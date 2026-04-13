#!/usr/bin/env python
"""Visualize Propedia PPI train/val/test split distributions via t-SNE.

Uses aCSM complex signatures to show how splits are distributed.

Usage:
    # Combined figure with legend bar (default)
    python scripts/visualize_ppi_splits.py

    # Single split only (with title)
    python scripts/visualize_ppi_splits.py --config configs/ppi_acsm.yaml
    python scripts/visualize_ppi_splits.py --config configs/ppi_random.yaml

    # Legend bar only
    python scripts/visualize_ppi_splits.py --legend

Output:
    results/visualization/tsne_propedia_ppi_splits_{timestamp}.pdf/.png        (combined)
    results/visualization/tsne_propedia_ppi_{split}_acsm_{timestamp}.pdf/.png  (single)
    results/visualization/tsne_propedia_ppi_legend_{timestamp}.pdf/.png        (legend)
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.append(str(Path(__file__).parent.parent))

import argparse

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "visualization"
DEFAULT_LOG_DIR = REPO_ROOT / "outputs" / "visualization"

# aCSM signatures
SOURCE_FILE = REPO_ROOT / "local_data/intermediate_product/Propedia_v2_unique_ppi_HELM_SMILES.csv"
SIGNATURE_DIR = REPO_ROOT / "local_data/intermediate_product/signatures_acsm_all"
SIGNATURE_FILE = "complex_signatures_acsm_all.csv"

SEED = 42
PCA_COMPONENTS = 50

ALL_CONFIGS = [
    ("configs/ppi_random.yaml", "Random Split"),
    ("configs/ppi_acsm.yaml", "aCSM Split"),
]

# Plot style
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
    """Set up logging to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"visualize_propedia_ppi_splits_{timestamp}.log"

    lg = logging.getLogger(__name__)
    lg.setLevel(LOG_LEVEL)
    lg.handlers = []

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(LOG_LEVEL)
    ch.setFormatter(logging.Formatter(LOG_FORMAT))

    fh = logging.FileHandler(log_file)
    fh.setLevel(LOG_LEVEL)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))

    lg.addHandler(ch)
    lg.addHandler(fh)
    lg.info(f"Log file: {log_file.absolute()}")
    return lg


def load_config(config_path: str) -> OmegaConf:
    """Load config: default.yaml + ppi.yaml + override config."""
    configs = []
    default_path = CONFIG_DIR / "default.yaml"
    if default_path.exists():
        configs.append(OmegaConf.load(default_path))
    task_path = CONFIG_DIR / "ppi.yaml"
    if task_path.exists() and str(task_path) != str(Path(config_path).resolve()):
        configs.append(OmegaConf.load(task_path))
    configs.append(OmegaConf.load(config_path))
    return OmegaConf.merge(*configs)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_acsm_signatures(
    pairs_df: pd.DataFrame, drug_col: str, target_col: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load aCSM complex signatures. Returns (signatures, valid_indices)."""
    source_df = pd.read_csv(SOURCE_FILE)
    source_df = source_df.drop_duplicates(subset=["Peptide_HELM", "Receptor_Sequence"], keep="first")
    pair_to_pdb = dict(zip(
        zip(source_df["Peptide_HELM"], source_df["Receptor_Sequence"]),
        source_df["PDB"],
    ))

    sig_path = SIGNATURE_DIR / SIGNATURE_FILE
    sig_df = pd.read_csv(sig_path, dtype={"pdb_id": "str"})
    sig_cols = [c for c in sig_df.columns if c.startswith("sig_")]
    sig_df = sig_df.drop_duplicates("pdb_id")
    sig_lookup = dict(zip(sig_df["pdb_id"].astype(str), sig_df[sig_cols].to_numpy(dtype=np.float32)))

    logger.info(f"aCSM signatures: {len(sig_lookup)} PDBs, {len(sig_cols)} dims")

    sigs, valid_idx = [], []
    for i, (_, row) in enumerate(pairs_df.iterrows()):
        pdb = pair_to_pdb.get((row[drug_col], row[target_col]))
        if pdb and pdb in sig_lookup:
            sigs.append(sig_lookup[pdb])
            valid_idx.append(i)

    logger.info(f"Matched {len(sigs)}/{len(pairs_df)} pairs to signatures")
    return np.array(sigs, dtype=np.float32), np.array(valid_idx)


def run_tsne(X: np.ndarray) -> np.ndarray:
    """StandardScaler -> PCA(50) -> t-SNE(2)."""
    X = StandardScaler().fit_transform(X)
    if X.shape[1] > PCA_COMPONENTS:
        pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        X = pca.fit_transform(X)
        logger.info(f"PCA: {PCA_COMPONENTS} dims, explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    logger.info(f"Running t-SNE on {X.shape}...")
    return TSNE(n_components=2, random_state=SEED, perplexity=30, init="pca", learning_rate="auto").fit_transform(X)


def get_split_data(config: OmegaConf) -> Tuple[np.ndarray, np.ndarray]:
    """Load data, get aCSM signatures, run t-SNE for one config."""
    drug_col = config.data.drug_column
    target_col = config.data.target_column
    label_col = config.data.label_column
    val_ratio = config.data.val_ratio

    train_df = pd.read_csv(config.data.train_file)
    test_df = pd.read_csv(config.data.test_file)

    train_pos = train_df[train_df[label_col] == 1]
    test_pos = test_df[test_df[label_col] == 1]

    train_split, val_split = train_test_split(train_pos, test_size=val_ratio, random_state=SEED)
    logger.info(f"  Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_pos)}")

    all_data = pd.concat([train_split, val_split, test_pos], ignore_index=True)
    split_labels = np.array(
        ["train"] * len(train_split) + ["val"] * len(val_split) + ["test"] * len(test_pos)
    )

    X, valid_idx = load_acsm_signatures(all_data, drug_col, target_col)
    split_labels = split_labels[valid_idx]

    logger.info(f"  Embeddings: {X.shape}")
    Z = run_tsne(X)
    return Z, split_labels


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _scatter_splits(ax, Z, split_labels):
    """Draw split-colored scatter on an axis."""
    perm = np.random.RandomState(SEED).permutation(len(Z))
    Z = Z[perm]
    split_labels = split_labels[perm]
    for split in ["train", "val", "test"]:
        mask = split_labels == split
        if mask.any():
            ax.scatter(Z[mask, 0], Z[mask, 1], c=SPLIT_COLORS[split],
                       alpha=0.4, s=15, edgecolors="none")
    ax.set_xlabel("t-SNE 1", fontsize=13, fontweight="bold")
    ax.set_ylabel("t-SNE 2", fontsize=13, fontweight="bold")


def _save(fig, base: Path):
    """Save figure as PDF + PNG."""
    fig.savefig(f"{base}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", format="png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {base}.pdf")
    logger.info(f"Saved: {base}.png")


def _legend_handles():
    return [mpatches.Patch(color=SPLIT_COLORS[s], label=s) for s in ["train", "val", "test"]]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_legend(results_dir: Path):
    """Generate legend bar only."""
    fig, ax = plt.subplots(figsize=(10, 0.6))
    ax.axis("off")
    ax.legend(handles=_legend_handles(), loc="center", ncol=3,
              fontsize=13, frameon=True, shadow=True,
              handlelength=1.5, handletextpad=0.5, columnspacing=2.0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save(fig, results_dir / f"tsne_propedia_ppi_legend_{timestamp}")


def run_single(config_path: str, results_dir: Path):
    """Generate a single split figure with title."""
    config = load_config(config_path)
    split_name = config.data.split_name
    logger.info(f"Processing: {split_name} ({config_path})")

    Z, split_labels = get_split_data(config)

    fig, ax = plt.subplots(figsize=(10, 10))
    _scatter_splits(ax, Z, split_labels)
    ax.set_title(f"{split_name.upper()} Split", fontsize=15, fontweight="bold")
    fig.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save(fig, results_dir / f"tsne_propedia_ppi_{split_name}_acsm_{timestamp}")


def run_combined(results_dir: Path):
    """Generate combined figure with shared t-SNE embedding."""
    configs = []
    all_pos_pairs = []
    for config_path, subtitle in ALL_CONFIGS:
        config = load_config(config_path)
        drug_col = config.data.drug_column
        target_col = config.data.target_column
        label_col = config.data.label_column
        val_ratio = config.data.val_ratio

        train_df = pd.read_csv(config.data.train_file)
        test_df = pd.read_csv(config.data.test_file)
        train_pos = train_df[train_df[label_col] == 1]
        test_pos = test_df[test_df[label_col] == 1]

        all_pos_pairs.append(pd.concat([train_pos, test_pos], ignore_index=True))
        configs.append((config, subtitle, train_pos, test_pos, drug_col, target_col, val_ratio))

    merged = pd.concat(all_pos_pairs, ignore_index=True).drop_duplicates(
        subset=[configs[0][4], configs[0][5]], keep="first"
    )
    drug_col = configs[0][4]
    target_col = configs[0][5]

    logger.info(f"Computing shared t-SNE on {len(merged)} unique positive pairs")
    X, valid_idx = load_acsm_signatures(merged, drug_col, target_col)
    Z = run_tsne(X)

    valid_pairs = [
        (merged.iloc[i][drug_col], merged.iloc[i][target_col])
        for i in valid_idx
    ]
    pair_to_z = {pair: Z[j] for j, pair in enumerate(valid_pairs)}

    panels = []
    for config, subtitle, train_pos, test_pos, d_col, t_col, val_ratio in configs:
        logger.info(f"\n  {subtitle}")
        train_split, val_split = train_test_split(train_pos, test_size=val_ratio, random_state=SEED)
        logger.info(f"  Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_pos)}")

        z_list, label_list = [], []
        for label, df_part in [("train", train_split), ("val", val_split), ("test", test_pos)]:
            for _, row in df_part.iterrows():
                pair = (row[d_col], row[t_col])
                if pair in pair_to_z:
                    z_list.append(pair_to_z[pair])
                    label_list.append(label)

        panels.append((np.array(z_list), np.array(label_list), subtitle))

    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 7))
    if len(panels) == 1:
        axes = [axes]

    for ax, (Z_panel, split_labels, subtitle) in zip(axes, panels):
        _scatter_splits(ax, Z_panel, split_labels)
        ax.set_title(subtitle, fontsize=15, fontweight="bold")

    fig.legend(handles=_legend_handles(), loc="lower center", ncol=3,
               fontsize=13, frameon=True, shadow=True, bbox_to_anchor=(0.5, -0.02),
               handlelength=2.5)
    fig.suptitle("t-SNE of Propedia PPI Splits (aCSM Complex Signatures)",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save(fig, results_dir / f"tsne_propedia_ppi_splits_{timestamp}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global logger

    parser = argparse.ArgumentParser(
        description="Visualize Propedia PPI train/val/test split distributions via t-SNE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/visualize_ppi_splits.py                                 # combined + legend\n"
            "  python scripts/visualize_ppi_splits.py --config configs/ppi_acsm.yaml   # single with title\n"
            "  python scripts/visualize_ppi_splits.py --legend                         # legend bar only\n"
        ),
    )
    parser.add_argument("--config", type=str, default=None,
                        help="PPI config file for single split. Omit for combined.")
    parser.add_argument("--legend", action="store_true",
                        help="Output legend bar only.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(Path(args.log_dir))

    logger.info("=" * 60)
    if args.legend:
        logger.info("Propedia PPI Split Visualization: legend only")
    elif args.config:
        logger.info(f"Propedia PPI Split Visualization: single ({args.config})")
    else:
        logger.info("Propedia PPI Split Visualization: combined (all splits)")
    logger.info("=" * 60)
    logger.info(f"Results: {results_dir}")

    if args.legend:
        run_legend(results_dir)
    elif args.config:
        run_single(args.config, results_dir)
    else:
        run_combined(results_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
