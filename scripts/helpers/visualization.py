from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SPLIT_COLORS = {"train": "#4C72B0", "val": "#55A868", "test": "#DD8452"}
DEFAULT_SPLIT_ORDER = ("train", "val", "test")

PLOT_MARGIN_FRACTION = 0.05
NN_QUANTILE = 0.25
MARKER_RADIUS_SCALE = 0.85
MARKER_RADIUS_PX_MIN = 8.0
MARKER_RADIUS_PX_MAX = 9.5
TARGET_FILL_RATIO = 0.18
ALPHA_MIN = 0.16
ALPHA_MAX = 0.50


def apply_plot_style(font_scale: float = 1.5) -> None:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=font_scale)
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def run_tsne(
    X: np.ndarray,
    logger: logging.Logger,
    *,
    seed: int = 42,
    pca_components: int = 50,
    perplexity: float = 30,
) -> np.ndarray:
    X = StandardScaler().fit_transform(X)
    if X.shape[1] > pca_components:
        pca = PCA(n_components=pca_components, random_state=seed)
        X = pca.fit_transform(X)
        logger.info(
            f"PCA: {pca_components} dims, explained variance: "
            f"{pca.explained_variance_ratio_.sum():.3f}"
        )
    logger.info(f"Running t-SNE on {X.shape}...")
    return TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    ).fit_transform(X)


def scatter_splits(
    ax,
    Z: np.ndarray,
    split_labels: np.ndarray,
    *,
    seed: int = 42,
    jitter: float = 0.0,
    split_order: Iterable[str] = DEFAULT_SPLIT_ORDER,
) -> None:
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(len(Z))
    Z = Z[permutation]
    split_labels = split_labels[permutation]

    if jitter > 0:
        Z = Z + rng.normal(0, jitter, Z.shape)

    marker_size, split_alphas = _compute_scatter_style(ax, Z, split_labels, split_order=split_order)
    for split in split_order:
        mask = split_labels == split
        if mask.any():
            ax.scatter(
                Z[mask, 0],
                Z[mask, 1],
                c=SPLIT_COLORS[split],
                alpha=split_alphas[split],
                s=marker_size,
                edgecolors="none",
            )

    ax.set_xlabel("t-SNE 1", fontsize=13, fontweight="bold")
    ax.set_ylabel("t-SNE 2", fontsize=13, fontweight="bold")


def legend_handles(split_order: Iterable[str] = DEFAULT_SPLIT_ORDER):
    return [mpatches.Patch(color=SPLIT_COLORS[split], label=split) for split in split_order]


def save_figure(fig, base: Path, logger: logging.Logger | None = None) -> None:
    fig.savefig(f"{base}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", format="png", bbox_inches="tight")
    plt.close(fig)
    if logger is not None:
        logger.info(f"Saved: {base}.pdf")
        logger.info(f"Saved: {base}.png")


def _compute_scatter_style(
    ax,
    Z: np.ndarray,
    split_labels: np.ndarray,
    *,
    split_order: Iterable[str],
) -> Tuple[float, Dict[str, float]]:
    if len(Z) == 0:
        return 15.0, {split: 0.4 for split in split_order}

    panel_width_px = ax.get_position().width * ax.figure.get_figwidth() * ax.figure.dpi
    panel_height_px = ax.get_position().height * ax.figure.get_figheight() * ax.figure.dpi
    x_min, x_max = float(Z[:, 0].min()), float(Z[:, 0].max())
    y_min, y_max = float(Z[:, 1].min()), float(Z[:, 1].max())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    x_pad = x_span * PLOT_MARGIN_FRACTION
    y_pad = y_span * PLOT_MARGIN_FRACTION
    x_span += 2 * x_pad
    y_span += 2 * y_pad

    screen_xy = np.column_stack([
        (Z[:, 0] - (x_min - x_pad)) / x_span * panel_width_px,
        (Z[:, 1] - (y_min - y_pad)) / y_span * panel_height_px,
    ])
    if len(screen_xy) > 1:
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(screen_xy)
        distances = nn.kneighbors(screen_xy, return_distance=True)[0][:, 1]
        spacing_px = float(np.quantile(distances, NN_QUANTILE))
    else:
        spacing_px = min(panel_width_px, panel_height_px) / 10.0

    marker_radius_px = float(np.clip(
        spacing_px * MARKER_RADIUS_SCALE,
        MARKER_RADIUS_PX_MIN,
        MARKER_RADIUS_PX_MAX,
    ))
    marker_area_px = np.pi * marker_radius_px ** 2
    marker_radius_pt = marker_radius_px * 72.0 / ax.figure.dpi
    marker_area_pt2 = np.pi * marker_radius_pt ** 2
    axes_area_px = max(panel_width_px * panel_height_px, 1.0)

    split_alphas: Dict[str, float] = {}
    for split in split_order:
        count = int(np.sum(split_labels == split))
        if count == 0:
            split_alphas[split] = ALPHA_MIN
            continue
        fill_ratio = count * marker_area_px / axes_area_px
        split_alphas[split] = float(np.clip(
            TARGET_FILL_RATIO / max(fill_ratio, 1e-6),
            ALPHA_MIN,
            ALPHA_MAX,
        ))

    return marker_area_pt2, split_alphas
