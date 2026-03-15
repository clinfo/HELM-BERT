"""Metric computation utilities.

Single responsibility: Compute evaluation metrics for different task types.
"""

import logging
from typing import Dict

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
)
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def compute_regression_metrics(
    predictions: torch.Tensor, labels: torch.Tensor
) -> Dict[str, float]:
    """Compute regression metrics (R², RMSE, MAE, Pearson).

    Args:
        predictions: Model predictions [N] or [N, 1]
        labels: Ground truth labels [N] or [N, 1]

    Returns:
        Dictionary with mse, rmse, mae, r2, pearson
    """
    # Convert to numpy and squeeze to 1D
    preds_np = predictions.cpu().numpy().squeeze()
    labels_np = labels.cpu().numpy().squeeze()

    metrics = {
        "mse": float(mean_squared_error(labels_np, preds_np)),
        "rmse": float(root_mean_squared_error(labels_np, preds_np)),
        "mae": float(mean_absolute_error(labels_np, preds_np)),
        "r2": float(r2_score(labels_np, preds_np)),
    }

    # Pearson correlation
    if len(preds_np) > 1:
        try:
            corr, _ = pearsonr(preds_np, labels_np)
            metrics["pearson"] = float(corr)
        except (ValueError, RuntimeWarning) as e:
            logger.error(f"Pearson correlation failed: {e}")
            metrics["pearson"] = float("nan")
    else:
        metrics["pearson"] = float("nan")

    return metrics


def compute_classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics (Accuracy, F1, Precision, Recall, ROC-AUC).

    Args:
        predictions: Model logits [N, num_classes] or [N] for binary
        labels: Ground truth labels [N]
        num_classes: Number of classes (1 for binary, >1 for multi-class)
        threshold: Classification threshold for binary classification

    Returns:
        Dictionary with accuracy, f1, precision, recall, roc_auc, pr_auc
    """
    predictions = predictions.cpu()
    labels = labels.cpu().numpy()

    # Get probabilities and predictions
    if num_classes == 1:
        # Binary classification with single output
        probs = torch.sigmoid(predictions).numpy().squeeze()
        preds = (probs > threshold).astype(int)
    else:
        # Multi-class classification
        probs = F.softmax(predictions, dim=1).numpy()
        preds = probs.argmax(axis=1)

    # Ensure labels are 1D
    labels = labels.squeeze() if labels.ndim > 1 else labels

    # Initialize metrics dict with ordered keys
    # Order: roc_auc, pr_auc, balanced_accuracy, mcc, f1, accuracy, precision, recall
    metrics = {}

    # ROC-AUC and PR-AUC (requires at least 2 unique classes)
    if len(np.unique(labels)) >= 2:
        try:
            if num_classes == 1:
                # Binary with single output
                metrics["roc_auc"] = float(roc_auc_score(labels, probs))
                p_curve, r_curve, _ = precision_recall_curve(labels, probs)
                metrics["pr_auc"] = float(auc(r_curve, p_curve))
            elif num_classes == 2:
                # Binary with 2 outputs
                metrics["roc_auc"] = float(roc_auc_score(labels, probs[:, 1]))
                p_curve, r_curve, _ = precision_recall_curve(labels, probs[:, 1])
                metrics["pr_auc"] = float(auc(r_curve, p_curve))
            else:
                # Multi-class
                metrics["roc_auc"] = float(
                    roc_auc_score(labels, probs, multi_class="ovr")
                )
                metrics["pr_auc"] = float("nan")  # PR-AUC not defined for multi-class
        except ValueError as e:
            logger.error(f"Failed to compute AUC: {e}")
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    # Balanced accuracy
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, preds))

    # Matthews Correlation Coefficient
    metrics["mcc"] = float(matthews_corrcoef(labels, preds))

    # F1 score
    avg = "binary" if num_classes <= 2 else "weighted"
    metrics["f1"] = float(f1_score(labels, preds, average=avg, zero_division=0))

    # Additional metrics
    metrics["accuracy"] = float(accuracy_score(labels, preds))
    metrics["precision"] = float(
        precision_score(labels, preds, average=avg, zero_division=0)
    )
    metrics["recall"] = float(recall_score(labels, preds, average=avg, zero_division=0))

    return metrics


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply mean pooling with numerical stability.

    Args:
        hidden: Hidden states [batch_size, seq_len, hidden_dim]
        mask: Attention mask [batch_size, seq_len]

    Returns:
        Pooled output [batch_size, hidden_dim]
    """
    mask_expanded = mask.unsqueeze(-1).expand(hidden.size()).float()
    sum_embeddings = torch.sum(hidden * mask_expanded, dim=1)
    eps = torch.finfo(hidden.dtype).eps
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=eps)
    return sum_embeddings / sum_mask
