"""Evidential Deep Learning loss functions for HELM-BERT tasks."""

from .evidential import (
    dirichlet_loss,
    nig_loss,
)

__all__ = [
    "nig_loss",
    "dirichlet_loss",
]
