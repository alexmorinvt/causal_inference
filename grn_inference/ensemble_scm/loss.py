"""Discrepancy between simulated and real per-perturbation distributions.

Cells are unpaired and drawn from a distribution, so we compare
summary statistics rather than per-cell errors. The MVP uses
first-and-second moment matching; richer distributional metrics
(Sinkhorn, MMD) can slot in behind the same interface later.

All functions are differentiable in the SCM parameters, since the
simulated tensors carry gradient information through the batched
solve in ``simulator.py``.
"""

from __future__ import annotations

import torch


def moment_matching_discrepancy(
    sim_cells: torch.Tensor,
    real_cells: torch.Tensor,
    variance_weight: float = 1.0,
    unbiased_var: bool = False,
) -> torch.Tensor:
    """Per-candidate moment-matching discrepancy.

    Parameters
    ----------
    sim_cells
        ``(N, B_sim, G)`` — simulated cells, one batch per candidate.
    real_cells
        ``(B_real, G)`` — real cells for the same perturbation arm.
    variance_weight
        Multiplier on the variance term relative to the mean term.
    unbiased_var
        Whether to use the unbiased (n-1) estimator for sample variance.
        Default False to keep the discrepancy comparable across batch
        sizes.

    Returns
    -------
    ``(N,)`` tensor of per-candidate scalars. Each candidate's gradient
    flows only into its own slice of W.
    """
    sim_mean = sim_cells.mean(dim=1)  # (N, G)
    sim_var = sim_cells.var(dim=1, unbiased=unbiased_var)  # (N, G)

    real_mean = real_cells.mean(dim=0)  # (G,)
    real_var = real_cells.var(dim=0, unbiased=unbiased_var)  # (G,)

    mean_term = (sim_mean - real_mean).pow(2).sum(dim=-1)  # (N,)
    var_term = (sim_var - real_var).pow(2).sum(dim=-1)  # (N,)
    return mean_term + variance_weight * var_term


def l1_penalty(W: torch.Tensor, lam: float) -> torch.Tensor:
    """Per-candidate L1 penalty on W.

    W: ``(N, G, G)`` -> ``(N,)``. Scalar ``lam`` scales the penalty
    equally across candidates.
    """
    return lam * W.abs().sum(dim=(-2, -1))
