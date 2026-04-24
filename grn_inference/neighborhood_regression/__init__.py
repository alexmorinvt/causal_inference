"""Hybrid shift + neighborhood-regression edge ranker.

Combines two signals:

- ``|shift(A, B)| = |mean(B | do(A)) - mean(B | control)|`` — for edges
  whose source ``A`` was intervened on. Direction is certified because
  the intervention severs back-door paths into ``A``.
- ``|beta(A, B)|`` = magnitude of the coefficient on gene ``A`` when
  regressing gene ``B`` on all other genes using control cells only.
  Equivalent to ``|Theta[B, A] / Theta[B, B]|`` where ``Theta`` is the
  control-cell precision matrix. Used only when the source is
  unperturbed; direction is unidentifiable from observational data but
  this is the only available signal for such sources.

The ranker buckets edges by whether their source is perturbed, ranks
within each bucket by its own score, and concatenates with a fixed
per-bucket quota of ``top_k``. This is the first method on the
autostrategy/apr24 branch that produces any non-zero score for
unperturbed-source edges.

Public entry point: :class:`NeighborhoodRegressionModel`.
"""

from .model import NeighborhoodRegressionModel

__all__ = ["NeighborhoodRegressionModel"]
