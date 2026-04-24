"""Invariant Causal Prediction GRN inference.

Identifiability argument (Peters, Bühlmann, Meinshausen 2016):
if ``S`` is a true direct parent of ``T``, the regression of ``x_T``
on ``x_S`` (or more generally on the parent set) is **invariant across
environments**. The regression coefficient ``β[T, S]`` and the residual
variance ``var(x_T − β x_S)`` should be statistically indistinguishable
across interventional arms. Confounded pairs or edges that pass
through a third variable that is itself intervened on break this
invariance.

On interventional gene expression data, each ``do(G)`` arm is a
different "environment." For each candidate edge ``(S, T)``, we
regress ``x_T`` on ``x_S`` inside each arm where neither ``S`` nor
``T`` is the intervention target, and score the candidate by the
inverse of the across-arm variance of the per-arm β estimate. Low
across-arm variance = invariant = strong evidence for ``S → T``.

Public entry point: :class:`InvarianceICPModel`.
"""

from .model import InvarianceICPModel

__all__ = ["InvarianceICPModel"]
