"""Batched linear cyclic SCM forward pass.

Each candidate SCM is parameterised by an adjacency matrix W of shape
``(n_genes, n_genes)``; we hold ``n_candidates`` of them stacked into a
single tensor of shape ``(N, G, G)`` so the whole ensemble advances
under one cascade sequence per simulation call.

The observational model matches ``make_synthetic_dataset``'s tree
cascade. Every cell is simulated as the leaf of a ``n_steps``-step
dynamical recursion::

    x_0 ~ N(0, I)
    x_{t+1} = W @ x_t + epsilon_{t+1},   epsilon_{t+1} ~ N(0, I)

Cells are independent trajectories (no shared-lineage correlations) —
that's fine for moment matching, which only looks at per-gene marginals.

An intervention on gene ``t`` zeros row ``t`` of W (``t`` no longer
responds to its regulators) and replaces ``t``'s own noise coordinate,
at every cascade step plus the initial state, with a knockdown-shifted
draw. This mirrors the data generator's "cells are born perturbed"
convention.

PyTorch is used here purely as an autodiff library. No ``nn.Module``,
no optimiser — just ``torch.Tensor(..., requires_grad=True)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LinearSCM:
    """Holds the learnable adjacency tensor for an ensemble of linear SCMs.

    Attributes
    ----------
    W
        Tensor of shape ``(N, G, G)`` with ``requires_grad=True``.
        ``W[k, i, j]`` is the weight of edge ``gene_j -> gene_i`` in
        candidate ``k``.
    n_genes
    n_candidates
    """

    W: torch.Tensor
    n_genes: int
    n_candidates: int

    @classmethod
    def random_init(
        cls,
        n_genes: int,
        n_candidates: int,
        weight_scale: float = 0.01,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> "LinearSCM":
        """Sample ``W ~ N(0, weight_scale^2)`` per candidate, zero the diagonal."""
        gen = torch.Generator().manual_seed(seed)
        W = torch.randn(
            n_candidates, n_genes, n_genes, generator=gen, dtype=dtype
        ) * weight_scale
        # Self-loops are excluded by the SCM definition.
        idx = torch.arange(n_genes)
        W[:, idx, idx] = 0.0
        W.requires_grad_(True)
        return cls(W=W, n_genes=n_genes, n_candidates=n_candidates)


def _cascade(
    W: torch.Tensor,
    x: torch.Tensor,
    n_steps: int,
    *,
    target_idx: int | None = None,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Run an ``n_steps``-step linear recursion ``x <- x @ W^T + eps``.

    ``W`` is ``(N, G, G)``, ``x`` is ``(N, n_cells, G)``. If
    ``target_idx`` is given, that coordinate's noise (and not the other
    coordinates) is drawn from ``N(target_mean, target_std^2)`` each step.
    """
    N, n_cells, G = x.shape
    Wt = W.transpose(-2, -1)  # (N, G, G); matmul on the right operand
    for _ in range(n_steps):
        eps = torch.randn(
            N, n_cells, G,
            dtype=x.dtype, device=x.device, generator=generator,
        )
        if target_idx is not None:
            kn = torch.randn(
                N, n_cells,
                dtype=x.dtype, device=x.device, generator=generator,
            )
            eps[:, :, target_idx] = target_std * kn + target_mean
        x = torch.matmul(x, Wt) + eps
    return x


def simulate_control(
    scm: LinearSCM,
    n_cells: int,
    generator: torch.Generator | None = None,
    n_steps: int = 5,
) -> torch.Tensor:
    """Sample control-arm cells from every candidate SCM as depth-T cascade leaves.

    Pass a ``torch.Generator`` for deterministic noise; otherwise the
    global torch RNG is used.

    Returns tensor of shape ``(N, n_cells, G)``.
    """
    N, G = scm.n_candidates, scm.n_genes
    x0 = torch.randn(
        N, n_cells, G,
        dtype=scm.W.dtype, device=scm.W.device, generator=generator,
    )
    return _cascade(scm.W, x0, n_steps, generator=generator)


def simulate_intervention(
    scm: LinearSCM,
    target_idx: int,
    n_cells: int,
    knockdown_factor: float = 0.3,
    generator: torch.Generator | None = None,
    n_steps: int = 5,
) -> torch.Tensor:
    """Sample cells under a soft knockdown of gene ``target_idx`` via cascade.

    Intervention semantics (matching ``make_synthetic_dataset`` at
    ``alpha = 1``, i.e. a full knockdown — the fitter doesn't know the
    true per-gene strength and uses the strongest-case approximation):

    - row ``target_idx`` of W is zeroed (target no longer responds to
      its regulators),
    - the target's own emission (initial state and every cascade-step
      noise draw) is replaced with
      ``knockdown_factor * N(0, 1) - (1 - knockdown_factor) * 2``.

    Pass a ``torch.Generator`` for deterministic noise; otherwise the
    global torch RNG is used.

    Returns tensor of shape ``(N, n_cells, G)``.
    """
    N, G = scm.n_candidates, scm.n_genes

    # Zero row target_idx in each candidate's W without mutating scm.W.
    mask = torch.ones(G, dtype=scm.W.dtype, device=scm.W.device)
    mask[target_idx] = 0.0
    W_t = scm.W * mask.view(1, G, 1)

    target_mean = -(1.0 - knockdown_factor) * 2.0
    target_std = knockdown_factor

    x0 = torch.randn(
        N, n_cells, G,
        dtype=scm.W.dtype, device=scm.W.device, generator=generator,
    )
    kn0 = torch.randn(
        N, n_cells,
        dtype=scm.W.dtype, device=scm.W.device, generator=generator,
    )
    x0[:, :, target_idx] = target_std * kn0 + target_mean

    return _cascade(
        W_t, x0, n_steps,
        target_idx=target_idx,
        target_mean=target_mean,
        target_std=target_std,
        generator=generator,
    )
