"""Path-inversion GRN inference.

Infer the direct-effect matrix ``W`` from the total-effect matrix ``T``
via the algebraic identity ``W = T (I + T)^{-1}``. In a linear cyclic
SCM ``x = Wx + eps`` the observed total effect from ``j`` onto ``i`` is
``T[i, j] = ((I - W)^{-1} W)[i, j]``, the sum over all paths from ``j``
to ``i`` weighted by path products. The Neumann-series expansion of
``(I + T)^{-1}`` gives ``W = T - T^2 + T^3 - T^4 + ...`` — an
inclusion-exclusion over path lengths in the shift graph.

Graph-theoretic interpretation: the shift matrix is a weighted directed
graph where edge weight ``T[i, j]`` is the cumulative influence of ``j``
on ``i`` through all paths. Subtracting ``T^2`` removes the two-step
paths that were double-counted; adding back ``T^3`` re-accounts for
three-step paths that were triple-subtracted, and so on. The limit is
the **direct edges only**.

For perturbed source genes, ``T[:, j]`` is directly observed as the
intervention shift column. For unperturbed source genes, we impute
``T[:, j]`` from the control-cell correlation matrix — a crude proxy
for the missing total-effect columns.

Public entry point: :class:`PathInversionModel`.
"""

from .model import PathInversionModel

__all__ = ["PathInversionModel"]
