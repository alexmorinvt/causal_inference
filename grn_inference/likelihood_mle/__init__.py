"""Likelihood-based SCM fitter.

Maximum-likelihood estimation of the linear cyclic SCM weight matrix
``W`` from the joint observational + interventional distribution.
Under iid Gaussian noise with isotropic variance:

- Control distribution: ``x ~ N(0, Σ_W)`` with
  ``Σ_W = (I - W)^{-1} σ² I (I - W)^{-T}``.
- ``do(G)`` arm: gene ``G`` pinned (modified mean + row-``G``-zeroed
  ``W̃_G``), with ``Σ_{W, G} = (I - W̃_G)^{-1} σ² I (I - W̃_G)^{-T}``.

Gaussian log-likelihood on sufficient statistics (per-arm mean + cov)
is used as the fitting objective. Gradient descent on ``W`` with
spectral-radius projection keeps ``(I - W)`` invertible. Distinct from
:class:`EnsembleSCMFitter` — no Monte Carlo simulation, closed-form
per-arm likelihood; distinct from :class:`NeighborhoodRegressionModel`
— estimates ``W`` directly instead of the precision matrix ``Θ``.

Public entry point: :class:`LikelihoodMLEModel`.
"""

from .model import LikelihoodMLEModel

__all__ = ["LikelihoodMLEModel"]
