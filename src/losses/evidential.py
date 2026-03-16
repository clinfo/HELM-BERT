"""Evidential Deep Learning loss functions.

Regression: Normal-Inverse-Gamma (NIG) distribution.
    Amini et al. (2020) "Deep Evidential Regression" NeurIPS.

Classification: Dirichlet distribution.
    Sensoy et al. (2018) "Evidential Deep Learning to Quantify
    Classification Uncertainty" NeurIPS. arXiv:1806.01768

Molecular application reference:
    Soleimany et al. (2021) ACS Central Science.
    doi:10.1021/acscentsci.1c00546
"""

import math

import torch
from torch import Tensor


# =============================================================================
# Regression: Normal-Inverse-Gamma (NIG)
# =============================================================================


def nig_nll_loss(
    y: Tensor, gamma: Tensor, nu: Tensor, alpha: Tensor, beta: Tensor
) -> Tensor:
    """Negative log-likelihood under the Normal-Inverse-Gamma posterior.

    Computes the marginal NLL after integrating out (mu, sigma^2) from
    the NIG distribution, yielding a Student-t marginal.

    Args:
        y: Ground truth targets [batch_size]
        gamma: Predicted mean [batch_size]
        nu: Evidence for mean, > 0 [batch_size]
        alpha: Evidence for variance, > 1 [batch_size]
        beta: Scale parameter, > 0 [batch_size]

    Returns:
        Scalar mean NLL loss.
    """
    two_beta_nu_plus_one = 2.0 * beta * (1.0 + nu)

    nll = (
        0.5 * torch.log(torch.pi / nu)
        - alpha * torch.log(two_beta_nu_plus_one)
        + (alpha + 0.5) * torch.log(
            nu * (y - gamma).pow(2) + two_beta_nu_plus_one
        )
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return nll.mean()


def nig_reg_loss(
    y: Tensor, gamma: Tensor, nu: Tensor, alpha: Tensor
) -> Tensor:
    """Evidence regularization for NIG.

    Penalizes total evidence (2*nu + alpha) proportional to prediction error.

    Args:
        y: Ground truth targets [batch_size]
        gamma: Predicted mean [batch_size]
        nu: Evidence for mean, > 0 [batch_size]
        alpha: Evidence for variance, > 1 [batch_size]

    Returns:
        Scalar mean regularization loss.
    """
    return (torch.abs(y - gamma) * (2.0 * nu + alpha)).mean()


def nig_loss(
    y: Tensor,
    gamma: Tensor,
    nu: Tensor,
    alpha: Tensor,
    beta: Tensor,
    lambda_coeff: float,
) -> Tensor:
    """Combined NIG evidential regression loss.

    loss = NIG_NLL + lambda_coeff * NIG_Reg

    Args:
        y: Ground truth targets [batch_size]
        gamma: Predicted mean [batch_size]
        nu: Evidence for mean, > 0 [batch_size]
        alpha: Evidence for variance, > 1 [batch_size]
        beta: Scale parameter, > 0 [batch_size]
        lambda_coeff: Regularization coefficient.

    Returns:
        Scalar combined loss.
    """
    return nig_nll_loss(y, gamma, nu, alpha, beta) + lambda_coeff * nig_reg_loss(
        y, gamma, nu, alpha
    )


# =============================================================================
# Classification: Dirichlet
# =============================================================================


def dirichlet_digamma_loss(y_onehot: Tensor, alpha: Tensor) -> Tensor:
    """Expected log probability loss under Dirichlet prior.

    Eq. 4 from Sensoy et al. (2018):
        L = sum_k y_k * (digamma(S) - digamma(alpha_k))
    where S = sum(alpha).

    Args:
        y_onehot: One-hot encoded labels [batch_size, K]
        alpha: Dirichlet concentration parameters, > 1 [batch_size, K]

    Returns:
        Scalar mean digamma loss.
    """
    S = alpha.sum(dim=-1, keepdim=True)
    loss = (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)
    return loss.mean()


def dirichlet_kl_loss(y_onehot: Tensor, alpha: Tensor) -> Tensor:
    """KL divergence between modified Dirichlet and uniform Dirichlet.

    Removes non-misleading evidence (correct-class evidence) before
    computing KL to avoid penalizing confident correct predictions.

    Args:
        y_onehot: One-hot encoded labels [batch_size, K]
        alpha: Dirichlet concentration parameters [batch_size, K]

    Returns:
        Scalar mean KL divergence.
    """
    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
    S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)

    K = alpha.shape[-1]

    kl = (
        torch.lgamma(S_tilde.squeeze(-1))
        - math.lgamma(K)
        - torch.lgamma(alpha_tilde).sum(dim=-1)
        + (
            (alpha_tilde - 1.0)
            * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))
        ).sum(dim=-1)
    )

    return kl.mean()


def dirichlet_loss(
    y_onehot: Tensor,
    alpha: Tensor,
    lambda_coeff: float,
) -> Tensor:
    """Combined Dirichlet evidential classification loss.

    loss = Dirichlet_Digamma + lambda_coeff * KL_divergence

    Args:
        y_onehot: One-hot encoded labels [batch_size, K]
        alpha: Dirichlet concentration parameters [batch_size, K]
        lambda_coeff: KL regularization coefficient.

    Returns:
        Scalar combined loss.
    """
    return dirichlet_digamma_loss(y_onehot, alpha) + lambda_coeff * dirichlet_kl_loss(
        y_onehot, alpha
    )
