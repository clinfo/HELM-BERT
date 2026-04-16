#!/usr/bin/env python
"""Visualize permeability train/val/test split distributions via t-SNE.

Uses Morgan fingerprints (ECFP4) from SMILES to embed molecules into 2D.

Usage:
    # Default: generate one figure per task, each with its own assay-specific space
    python scripts/visualize_permeability_splits.py

    # Single split only
    python scripts/visualize_permeability_splits.py --split random
    python scripts/visualize_permeability_splits.py --split scaffold

    # Assay-specific visualization (random/scaffold share one task-specific space)
    python scripts/visualize_permeability_splits.py --task pampa
    python scripts/visualize_permeability_splits.py --task caco2

    # Legend bar only
    python scripts/visualize_permeability_splits.py --legend

Output:
    results/visualization/tsne_permeability_mix_random_scaffold_{timestamp}.pdf/.png
    results/visualization/tsne_permeability_{split}_{timestamp}.pdf/.png
    results/visualization/tsne_permeability_legend_{timestamp}.pdf/.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
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

SEED = 42
VAL_RATIO = 0.1
PCA_COMPONENTS = 50
FP_RADIUS = 2
FP_NBITS = 2048
JITTER_SCALE = 0.3

SMILES_COL = "SMILES"

TASK_FILE_STEMS = {
    "permeability": "cycpeptmpdb_permeability",
    "pampa": "cycpeptmpdb_permeability_pampa",
    "caco2": "cycpeptmpdb_permeability_caco2",
}
ALL_TASKS = list(TASK_FILE_STEMS)

TASK_PLOT_META = {
    "permeability": {
        "title": "Permeability Mix",
        "output_stem": "permeability_mix",
    },
    "pampa": {
        "title": "Permeability PAMPA",
        "output_stem": "permeability_pampa",
    },
    "caco2": {
        "title": "Permeability CACO2",
        "output_stem": "permeability_caco2",
    },
}

apply_plot_style(font_scale=1.5)

LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"visualize_permeability_splits_{timestamp}.log"

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
    Z = run_tsne(X, logger, seed=SEED, pca_components=PCA_COMPONENTS)
    return Z, split_labels


def get_task_splits(task: str) -> List[Dict[str, str]]:
    """Return random/scaffold file paths for one task."""
    stem = TASK_FILE_STEMS[task]
    task_title = "Permeability" if task == "permeability" else task.upper()
    return [
        {
            "name": f"{task_title} Random Split",
            "tag": "random",
            "train_file": f"data/downstream/{stem}_random_train.csv",
            "test_file": f"data/downstream/{stem}_random_test.csv",
        },
        {
            "name": f"{task_title} Scaffold Split",
            "tag": "scaffold",
            "train_file": f"data/downstream/{stem}_scaffold_train.csv",
            "test_file": f"data/downstream/{stem}_scaffold_test.csv",
        },
    ]


def run_legend(results_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 0.6))
    ax.axis("off")
    ax.legend(
        handles=legend_handles(),
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
    save_figure(fig, results_dir / f"tsne_permeability_legend_{timestamp}", logger)


def run_single(split_tag: str, task: str, results_dir: Path, jitter: float) -> None:
    split_info = next(split for split in get_task_splits(task) if split["tag"] == split_tag)
    logger.info(f"Processing: {split_info['name']}")

    Z, split_labels = get_split_data(split_info)

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter_splits(ax, Z, split_labels, seed=SEED, jitter=jitter)
    ax.set_title(split_info["name"], fontsize=15, fontweight="bold")
    fig.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_figure(fig, results_dir / f"tsne_{task}_{split_tag}_{timestamp}", logger)


def run_combined(
    task: str,
    results_dir: Path,
    jitter: float,
) -> None:
    task_splits = get_task_splits(task)
    all_smiles: set[str] = set()
    for split_info in task_splits:
        train_df = pd.read_csv(REPO_ROOT / split_info["train_file"])
        test_df = pd.read_csv(REPO_ROOT / split_info["test_file"])
        all_smiles.update(train_df[SMILES_COL])
        all_smiles.update(test_df[SMILES_COL])

    canonical_list = sorted(all_smiles)
    logger.info(f"Computing assay-specific t-SNE on {len(canonical_list)} unique molecules for {task}")
    X, valid_idx = smiles_to_fingerprints(canonical_list)
    Z = run_tsne(X, logger, seed=SEED, pca_components=PCA_COMPONENTS)
    valid_smiles = [canonical_list[i] for i in valid_idx]
    smiles_to_z = {smiles: Z[i] for i, smiles in enumerate(valid_smiles)}

    split_dfs = []
    for split_info in task_splits:
        train_df = pd.read_csv(REPO_ROOT / split_info["train_file"])
        test_df = pd.read_csv(REPO_ROOT / split_info["test_file"])
        split_dfs.append((split_info, train_df, test_df))

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
        scatter_splits(ax, Z_panel, split_labels, seed=SEED, jitter=jitter)
        ax.set_title(subtitle, fontsize=15, fontweight="bold")

    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=3,
        fontsize=13,
        frameon=True,
        shadow=True,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=2.5,
    )
    task_meta = TASK_PLOT_META[task]
    fig.suptitle(
        f"t-SNE of {task_meta['title']}: Random vs Scaffold",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_figure(
        fig,
        results_dir / f"tsne_{task_meta['output_stem']}_random_scaffold_{timestamp}",
        logger,
    )


def run_all_tasks(results_dir: Path, jitter: float) -> None:
    """Generate one assay-specific comparison figure per task."""
    for task in ALL_TASKS:
        run_combined(task, results_dir, jitter=jitter)


def main() -> None:
    global logger

    parser = argparse.ArgumentParser(
        description="Visualize permeability train/val/test split distributions via t-SNE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/visualize_permeability_splits.py                     # all tasks, task-specific spaces\n"
            "  python scripts/visualize_permeability_splits.py --split random      # single\n"
            "  python scripts/visualize_permeability_splits.py --split scaffold    # single\n"
            "  python scripts/visualize_permeability_splits.py --task caco2        # assay-specific\n"
            "  python scripts/visualize_permeability_splits.py --legend            # legend only\n"
            "  python scripts/visualize_permeability_splits.py --jitter 0.5        # custom jitter\n"
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
        "--task",
        type=str,
        default="all",
        choices=["all"] + sorted(TASK_FILE_STEMS),
        help="Task to visualize.",
    )
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
        logger.info("Permeability Split Visualization: legend only")
    elif args.split:
        logger.info(f"Permeability Split Visualization: single ({args.task}, {args.split})")
    elif args.task == "all":
        logger.info("Permeability Split Visualization: all tasks in task-specific spaces")
    else:
        logger.info(f"Permeability Split Visualization: combined ({args.task})")
    logger.info("=" * 60)
    logger.info(f"Results: {results_dir}")
    logger.info(f"Jitter: {args.jitter}")

    if args.legend:
        run_legend(results_dir)
    elif args.split:
        if args.task == "all":
            raise ValueError("--split requires a specific --task, not --task all")
        run_single(args.split, args.task, results_dir, jitter=args.jitter)
    elif args.task == "all":
        run_all_tasks(results_dir, jitter=args.jitter)
    else:
        run_combined(args.task, results_dir, jitter=args.jitter)

    logger.info("Done!")


if __name__ == "__main__":
    main()
