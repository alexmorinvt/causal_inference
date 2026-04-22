"""Batched linear cyclic SCM forward pass.

Each candidate SCM is parameterised by an adjacency matrix W of shape
``(n_genes, n_genes)``; we hold ``n_candidates`` of them stacked into a
single tensor of shape ``(N, G, G)`` so the whole ensemble advances
under one matrix solve per simulation call.

The observational model matches ``make_synthetic_dataset``::

    x = (I - W)^{-1} epsilon,   epsilon ~ N(0, I)

An intervention on gene ``t`` zeros row ``t`` of W (``t`` no longer
responds to its regulators) and replaces ``t``'s own noise coordinate
with a knockdown-shifted draw — same semantics as the synthetic
generator, so the fitted SCM is comparable against the ground-truth W.

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


def _solve_batch(A: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Batched solve: returns X with A @ X = eps.

    A: ``(N, G, G)``, eps: ``(N, G, B)`` -> ``(N, G, B)``.
    """
    return torch.linalg.solve(A, eps)


def simulate_control(
    scm: LinearSCM,
    n_cells: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample control-arm cells from every candidate SCM.

    Pass a ``torch.Generator`` for deterministic noise; otherwise the
    global torch RNG is used.

    Returns tensor of shape ``(N, n_cells, G)``.
    """
    N, G = scm.n_candidates, scm.n_genes
    I = torch.eye(G, dtype=scm.W.dtype, device=scm.W.device).expand(N, G, G)
    eps = torch.randn(
        N, G, n_cells,
        dtype=scm.W.dtype, device=scm.W.device, generator=generator,
    )
    x = _solve_batch(I - scm.W, eps)  # (N, G, n_cells)
    return x.transpose(-1, -2).contiguous()  # (N, n_cells, G)


def simulate_intervention(
    scm: LinearSCM,
    target_idx: int,
    n_cells: int,
    knockdown_factor: float = 0.3,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample cells under a soft knockdown of gene ``target_idx``.

    Matches ``make_synthetic_dataset``'s intervention semantics:

    - row ``target_idx`` of W is zeroed (the target no longer responds
      to its regulators),
    - the target's own noise coordinate is replaced with
      ``knockdown_factor * N(0, 1) - (1 - knockdown_factor) * 2``.

    Pass a ``torch.Generator`` for deterministic noise; otherwise the
    global torch RNG is used.

    Returns tensor of shape ``(N, n_cells, G)``.
    """
    N, G = scm.n_candidates, scm.n_genes

    # Zero row target_idx in each candidate's W without mutating scm.W.
    mask = torch.ones(G, dtype=scm.W.dtype, device=scm.W.device)
    mask[target_idx] = 0.0
    W_t = scm.W * mask.view(1, G, 1)  # zeroes row `target_idx`

    I = torch.eye(G, dtype=scm.W.dtype, device=scm.W.device).expand(N, G, G)

    eps = torch.randn(
        N, G, n_cells,
        dtype=scm.W.dtype, device=scm.W.device, generator=generator,
    )
    knockdown_noise = torch.randn(
        N, n_cells,
        dtype=scm.W.dtype, device=scm.W.device, generator=generator,
    )
    eps[:, target_idx, :] = (
        knockdown_factor * knockdown_noise - (1.0 - knockdown_factor) * 2.0
    )

    x = _solve_batch(I - W_t, eps)
    return x.transpose(-1, -2).contiguous()
