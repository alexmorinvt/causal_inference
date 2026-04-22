"""Entry-point class for the ensemble-SCM edge ranker.

This class conforms to ``grn_inference.models.Model`` so it drops into
the existing evaluation harness. It is **not** an ML model — it owns
an ensemble of parameterised SCMs, fits them to the observed per-arm
distributions, and ranks edges by the magnitude of the fitted weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..dataset import Dataset
from .fit import FitHistory, fit_scm_ensemble
from .simulator import LinearSCM

Edge = tuple[str, str]


@dataclass
class EnsembleSCMFitter:
    """Ensemble SCM fitter + edge ranker.

    Hyperparameters
    ---------------
    top_k
        Number of ranked edges to return from ``fit_predict``.
    n_candidates
        Number of independent candidate SCMs in the ensemble.
    n_steps
        Fixed number of gradient-descent steps per fit.
    step_size
        Manual SGD step size (the user's "delta").
    batch_size
        Cells per arm per step (simulated and real).
    l1_lambda
        L1 sparsity coefficient on W.
    variance_weight
        Relative weight of the variance term in the discrepancy.
    knockdown_factor
        Soft-knockdown coefficient for the intervention simulator; the
        default 0.3 matches ``make_synthetic_dataset``.
    spectral_threshold
        After each gradient step, candidates with ``rho(W) >
        spectral_threshold`` are uniformly rescaled so ``rho(W) =
        spectral_threshold``. Keeps ``(I - W)^{-1}`` bounded and stops
        the fit from overshooting into the stiff region near
        ``rho = 1``. Set to ``None`` to disable (fits will diverge).

        The default ``0.80`` matches the synthetic generator's
        ``target_spectral_radius`` and keeps amplification
        (``~1/(1 - rho)``) at a moderate level (~5x). Too low (< 0.5)
        over-constrains expressive power; too high (> 0.9) lets the
        simulator over-amplify the noise and the moment-matching
        discrepancy floor explodes. ``0.7 - 0.8`` is the sweet spot
        on the synthetic data we have.
    aggregation_power
        Exponent ``p`` for the generalized-mean aggregation of
        ``|W_k|`` across candidates. ``p=1`` is the arithmetic mean,
        ``p → ∞`` is the max. ``p=3`` biases toward strong single
        candidates without collapsing to the max.
    weight_scale
        Std-dev of the Gaussian random initialisation for W.
    seed
        RNG seed for initialisation + arm sampling.
    """

    top_k: int = 1000
    n_candidates: int = 5
    n_steps: int = 1000
    step_size: float = 0.01
    batch_size: int = 200
    l1_lambda: float = 1e-4
    variance_weight: float = 1.0
    knockdown_factor: float = 0.3
    spectral_threshold: float | None = 0.80
    aggregation_power: float = 3.0
    weight_scale: float = 0.01
    seed: int = 0
    log_every: int | None = 100

    # Populated by fit_predict; exposed so callers can inspect the fit.
    last_scm: LinearSCM | None = field(default=None, init=False, repr=False)
    last_history: FitHistory | None = field(default=None, init=False, repr=False)

    def fit_predict(self, data: Dataset) -> list[Edge]:
        scm = LinearSCM.random_init(
            n_genes=data.n_genes,
            n_candidates=self.n_candidates,
            weight_scale=self.weight_scale,
            seed=self.seed,
        )
        history = fit_scm_ensemble(
            scm,
            data,
            n_steps=self.n_steps,
            step_size=self.step_size,
            batch_size=self.batch_size,
            l1_lambda=self.l1_lambda,
            variance_weight=self.variance_weight,
            knockdown_factor=self.knockdown_factor,
            spectral_threshold=self.spectral_threshold,
            seed=self.seed,
            log_every=self.log_every,
        )

        self.last_scm = scm
        self.last_history = history

        score = aggregate_scores(scm.W.detach(), power=self.aggregation_power)
        return rank_edges(score, data.gene_names, self.top_k)


def aggregate_scores(W_ensemble: torch.Tensor, power: float) -> np.ndarray:
    """Generalised-mean aggregation of ``|W|`` across candidates.

    ``score[j, i] = ( mean_k |W[k, i, j]|^p )^(1/p)``

    Note the transpose: ``W[k, i, j]`` is the weight of edge
    ``gene_j -> gene_i`` (parent j acts on child i), so the score
    indexed by (source=j, target=i) is obtained by transposing.
    Diagonal entries are forced to zero so self-loops never rank.
    """
    abs_w = W_ensemble.abs().float()  # (N, G, G)
    # Power mean with a guard against zero-to-negative-power blowups.
    eps = 1e-12
    pm = (abs_w.pow(power).mean(dim=0) + eps).pow(1.0 / power)  # (G, G)
    # Transpose so score[j, i] corresponds to (source=j -> target=i).
    score = pm.transpose(0, 1).cpu().numpy()
    np.fill_diagonal(score, 0.0)
    return score


def rank_edges(score: np.ndarray, gene_names: list[str], top_k: int) -> list[Edge]:
    """Return the top-``k`` ``(source, target)`` pairs by score.

    ``score[j, i]`` is the weight of ``gene_names[j] -> gene_names[i]``.
    """
    G = score.shape[0]
    flat = score.ravel()
    k = min(top_k, flat.size)
    if k <= 0:
        return []
    top_idx = np.argpartition(-flat, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-flat[top_idx])]
    edges: list[Edge] = []
    for idx in top_idx:
        j, i = divmod(int(idx), G)
        if i == j:
            continue
        if flat[idx] <= 0.0:
            continue
        edges.append((gene_names[j], gene_names[i]))
    return edges
