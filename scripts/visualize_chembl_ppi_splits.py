#!/usr/bin/env python
"""Visualize ChEMBL PPI split distributions via t-SNE.

Uses concatenated compound Morgan fingerprints and target amino-acid
composition vectors to visualize positive pairs. Validation is derived
from the saved train split, matching the training pipeline.

Usage:
    python scripts/visualize_chembl_ppi_splits.py                                  # all comparisons
    python scripts/visualize_chembl_ppi_splits.py --compare random_vs_family       # one comparison
    python scripts/visualize_chembl_ppi_splits.py --compare random_vs_cold
    python scripts/visualize_chembl_ppi_splits.py --split random                   # single split
    python scripts/visualize_chembl_ppi_splits.py --split family
    python scripts/visualize_chembl_ppi_splits.py --split cold
    python scripts/visualize_chembl_ppi_splits.py --legend
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

from scripts.helpers.visualization import (
    apply_plot_style,
    legend_handles,
    run_tsne,
    save_figure,
    scatter_splits,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "visualization"
DEFAULT_LOG_DIR = REPO_ROOT / "outputs" / "visualization"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "downstream"

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
SEED = 42
FP_BITS = 256
PCA_COMPONENTS = 50
VAL_RATIO = 0.1

DATASETS = {
    "random": {
        "title": "Random Split",
        "files": {
            "train": DEFAULT_DATA_DIR / "chembl_ppi_random_train.csv",
            "test": DEFAULT_DATA_DIR / "chembl_ppi_random_test.csv",
        },
    },
    "family": {
        "title": "Family Split",
        "files": {
            "train": DEFAULT_DATA_DIR / "chembl_ppi_family_train.csv",
            "test": DEFAULT_DATA_DIR / "chembl_ppi_family_test.csv",
        },
    },
}

LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"visualize_chembl_ppi_splits_{timestamp}.log"

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


def smiles_to_fingerprint(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
    arr = np.zeros((FP_BITS,), dtype=np.float32)
    for idx in fp.GetOnBits():
        arr[idx] = 1.0
    return arr


def sequence_to_composition(seq: str) -> np.ndarray:
    arr = np.zeros((len(AA_ORDER) + 1,), dtype=np.float32)
    seq = str(seq or "").upper()
    if not seq:
        return arr
    total = len(seq)
    for i, aa in enumerate(AA_ORDER):
        arr[i] = seq.count(aa) / total
    arr[-1] = min(total, 3000) / 3000.0
    return arr


def pair_to_features(smiles: str, seq: str) -> np.ndarray | None:
    fp = smiles_to_fingerprint(smiles)
    if fp is None:
        return None
    comp = sequence_to_composition(seq)
    return np.concatenate([fp, comp]).astype(np.float32)


def load_split_dataset(split_name: str, val_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    spec = DATASETS[split_name]
    features = []
    labels = []
    total = 0

    train_path = spec["files"]["train"]
    test_path = spec["files"]["test"]
    for path in [train_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    train_pos = train_df[train_df["Label"] == 1].copy()
    test_pos = test_df[test_df["Label"] == 1].copy()
    train_split, val_split = train_test_split(train_pos, test_size=val_ratio, random_state=SEED)

    for part, df in [("train", train_split), ("val", val_split), ("test", test_pos)]:
        total += len(df)
        kept = 0
        for _, row in df.iterrows():
            feat = pair_to_features(row["canonical_smiles"], row["target_sequence"])
            if feat is not None:
                features.append(feat)
                labels.append(part)
                kept += 1
        logger.info(f"{split_name} {part}: {kept}/{len(df)} positive pairs embedded")

    logger.info(f"{split_name}: total positive pairs processed = {total:,}")
    return np.vstack(features), np.array(labels)


def load_split_frames(split_name: str, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = DATASETS[split_name]
    train_path = spec["files"]["train"]
    test_path = spec["files"]["test"]
    for path in [train_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    train_pos = train_df[train_df["Label"] == 1].copy()
    test_pos = test_df[test_df["Label"] == 1].copy()
    train_split, val_split = train_test_split(train_pos, test_size=val_ratio, random_state=SEED)
    return train_split.reset_index(drop=True), val_split.reset_index(drop=True), test_pos.reset_index(drop=True)


def run_legend(results_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 0.6))
    ax.axis("off")
    ax.legend(handles=legend_handles(), loc="center", ncol=3,
              fontsize=13, frameon=True, shadow=True,
              handlelength=1.5, handletextpad=0.5, columnspacing=2.0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_figure(fig, results_dir / f"tsne_chembl_ppi_legend_{timestamp}", logger)


def run_single(split_name: str, results_dir: Path):
    X, labels = load_split_dataset(split_name, VAL_RATIO)
    Z = run_tsne(X, logger, seed=SEED, pca_components=PCA_COMPONENTS)
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter_splits(ax, Z, labels, seed=SEED)
    ax.set_title(DATASETS[split_name]["title"], fontsize=15, fontweight="bold")
    fig.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_figure(fig, results_dir / f"tsne_chembl_ppi_{split_name}_{timestamp}", logger)


def run_comparison(split_names: list[str], title: str, output_stem: str, results_dir: Path):
    split_frames = []
    all_pairs = []
    for split_name in split_names:
        train_split, val_split, test_pos = load_split_frames(split_name, VAL_RATIO)
        logger.info(
            f"{split_name}: Train={len(train_split)}, Val={len(val_split)}, Test={len(test_pos)}"
        )
        split_frames.append((split_name, train_split, val_split, test_pos))
        all_pairs.append(pd.concat([train_split, val_split, test_pos], ignore_index=True))

    merged = pd.concat(all_pairs, ignore_index=True).drop_duplicates(
        subset=["canonical_smiles", "target_sequence"], keep="first"
    )

    features = []
    valid_keys = []
    for _, row in merged.iterrows():
        feat = pair_to_features(row["canonical_smiles"], row["target_sequence"])
        if feat is not None:
            features.append(feat)
            valid_keys.append((row["canonical_smiles"], row["target_sequence"]))

    logger.info(f"Computing shared t-SNE on {len(valid_keys):,} unique positive pairs")
    X = np.vstack(features)
    Z = run_tsne(X, logger, seed=SEED, pca_components=PCA_COMPONENTS)
    pair_to_z = {key: Z[i] for i, key in enumerate(valid_keys)}

    panels = []
    for split_name, train_split, val_split, test_pos in split_frames:
        z_list = []
        label_list = []
        for label, df in [("train", train_split), ("val", val_split), ("test", test_pos)]:
            kept = 0
            for _, row in df.iterrows():
                key = (row["canonical_smiles"], row["target_sequence"])
                if key in pair_to_z:
                    z_list.append(pair_to_z[key])
                    label_list.append(label)
                    kept += 1
            logger.info(f"{split_name} {label}: {kept}/{len(df)} positive pairs embedded")
        panels.append((split_name, np.array(z_list), np.array(label_list)))

    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 7))
    if len(panels) == 1:
        axes = [axes]

    for ax, (split_name, Z, labels) in zip(axes, panels):
        scatter_splits(ax, Z, labels, seed=SEED)
        ax.set_title(DATASETS[split_name]["title"], fontsize=15, fontweight="bold")

    fig.legend(handles=legend_handles(), loc="lower center", ncol=3,
               fontsize=13, frameon=True, shadow=True, bbox_to_anchor=(0.5, -0.02),
               handlelength=2.5)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_figure(fig, results_dir / f"{output_stem}_{timestamp}", logger)


def run_all_comparisons(comparison_keys: list[str], results_dir: Path):
    """Compute one shared t-SNE across all needed splits, then render each comparison figure."""
    all_split_names = sorted({s for k in comparison_keys for s in COMPARISONS[k]["splits"]})
    logger.info(f"Loading splits: {all_split_names}")

    split_data: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    all_pairs = []
    for split_name in all_split_names:
        train_split, val_split, test_pos = load_split_frames(split_name, VAL_RATIO)
        logger.info(f"{split_name}: Train={len(train_split)}, Val={len(val_split)}, Test={len(test_pos)}")
        split_data[split_name] = (train_split, val_split, test_pos)
        all_pairs.append(pd.concat([train_split, val_split, test_pos], ignore_index=True))

    merged = pd.concat(all_pairs, ignore_index=True).drop_duplicates(
        subset=["canonical_smiles", "target_sequence"], keep="first"
    )

    features = []
    valid_keys = []
    for _, row in merged.iterrows():
        feat = pair_to_features(row["canonical_smiles"], row["target_sequence"])
        if feat is not None:
            features.append(feat)
            valid_keys.append((row["canonical_smiles"], row["target_sequence"]))

    logger.info(f"Computing shared t-SNE on {len(valid_keys):,} unique positive pairs")
    X = np.vstack(features)
    Z = run_tsne(X, logger, seed=SEED, pca_components=PCA_COMPONENTS)
    pair_to_z = {key: Z[i] for i, key in enumerate(valid_keys)}

    for key in comparison_keys:
        spec = COMPARISONS[key]
        logger.info(f"\n--- {spec['title']} ---")

        panels = []
        for split_name in spec["splits"]:
            train_split, val_split, test_pos = split_data[split_name]
            z_list = []
            label_list = []
            for label, df in [("train", train_split), ("val", val_split), ("test", test_pos)]:
                kept = 0
                for _, row in df.iterrows():
                    k = (row["canonical_smiles"], row["target_sequence"])
                    if k in pair_to_z:
                        z_list.append(pair_to_z[k])
                        label_list.append(label)
                        kept += 1
                logger.info(f"{split_name} {label}: {kept}/{len(df)} positive pairs embedded")
            panels.append((split_name, np.array(z_list), np.array(label_list)))

        fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 7))
        if len(panels) == 1:
            axes = [axes]
        for ax, (split_name, Z_panel, labels) in zip(axes, panels):
            scatter_splits(ax, Z_panel, labels, seed=SEED)
            ax.set_title(DATASETS[split_name]["title"], fontsize=15, fontweight="bold")
        fig.legend(handles=legend_handles(), loc="lower center", ncol=3,
                   fontsize=13, frameon=True, shadow=True, bbox_to_anchor=(0.5, -0.02),
                   handlelength=2.5)
        fig.suptitle(spec["title"], fontsize=16, fontweight="bold", y=1.02)
        fig.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_figure(fig, results_dir / f"{spec['stem']}_{timestamp}", logger)


COMPARISONS = {
    "random_vs_family": {
        "splits": ["random", "family"],
        "title": "t-SNE of ChEMBL PPI Mix: Random vs Family",
        "stem": "tsne_chembl_ppi_mix_random_family",
    },
    "random_vs_cold": {
        "splits": ["random", "cold"],
        "title": "t-SNE of ChEMBL PPI Mix: Random vs Cold",
        "stem": "tsne_chembl_ppi_mix_random_cold",
    },
}


def main():
    global logger

    apply_plot_style(font_scale=1.4)

    parser = argparse.ArgumentParser(description="Visualize ChEMBL PPI split distributions via t-SNE")
    parser.add_argument("--split", choices=["random", "family", "cold"], default=None,
                        help="Visualize a single split. Omit for combined figure.")
    parser.add_argument("--compare", choices=list(COMPARISONS), default=None,
                        help="Generate one comparison. Omit to generate all comparisons.")
    parser.add_argument("--legend", action="store_true", help="Output legend only.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(Path(args.log_dir))

    targets = [args.compare] if args.compare else list(COMPARISONS)

    logger.info("=" * 60)
    if args.legend:
        logger.info("ChEMBL PPI Split Visualization: legend only")
    elif args.split:
        logger.info(f"ChEMBL PPI Split Visualization: single ({args.split})")
    else:
        logger.info(f"ChEMBL PPI Split Visualization: comparisons ({', '.join(targets)})")
    logger.info("=" * 60)
    logger.info(f"Results: {results_dir}")

    if args.legend:
        run_legend(results_dir)
    elif args.split:
        run_single(args.split, results_dir)
    elif len(targets) == 1:
        spec = COMPARISONS[targets[0]]
        run_comparison(
            split_names=spec["splits"],
            title=spec["title"],
            output_stem=spec["stem"],
            results_dir=results_dir,
        )
    else:
        for key in targets:
            spec = COMPARISONS[key]
            run_comparison(
                split_names=spec["splits"],
                title=spec["title"],
                output_stem=spec["stem"],
                results_dir=results_dir,
            )

    logger.info("Done!")


if __name__ == "__main__":
    main()
