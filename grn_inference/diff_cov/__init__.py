"""Differential-covariance GRN inference.

For each perturbed gene ``G``, the within-``do(G)`` covariance
``Σ_G = Cov(x | do(G))`` has structure
``Σ_G = (I - W_G)^{-1} Σ_ε (I - W_G)^{-T}`` where ``W_G`` zeroes out
row ``G`` of ``W``. Subtracting ``Σ_ctrl - Σ_G`` highlights the
entries of ``Σ`` that depend on the interventions — i.e. the edges
in the regulatory subnetwork involving ``G`` and its descendants.

For a candidate edge ``(j, i)``, aggregating
``|Σ_ctrl[i, j] - Σ_G[i, j]|`` across perturbed ``G`` gives a graph-
theoretic "intervention sensitivity" score: edges that participate
in real regulatory cascades see their covariance change under many
interventions; spurious correlations from confounders or sampling
noise are invariant.

This is a **coverage-based** graph-theoretic method: we're not
inverting anything, just measuring how "connected" each gene pair is
to the perturbation regime via covariance shifts. Directionality is
imposed afterwards using the one-sided shift asymmetry of the
interventional means (``|shift[j, i]| > |shift[i, j]|`` → direction
``j → i``) when both genes are perturbed, and the precision-matrix-
diagonal asymmetry otherwise — a minimal reuse of the
``NeighborhoodRegressionModel`` direction ideas but operating on a
fundamentally different score.

Public entry point: :class:`DiffCovModel`.
"""

from .model import DiffCovModel

__all__ = ["DiffCovModel"]
