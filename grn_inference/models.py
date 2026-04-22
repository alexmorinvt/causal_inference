"""Baseline GRN inference models.

Every model conforms to the :class:`Model` protocol: given a
:class:`~grn_inference.dataset.Dataset`, return a ranked list of
directed ``(source, target)`` edges. The list is already truncated to
the model's chosen ``top_k``; the evaluator does not re-rank.

Two baselines live here:

- :class:`RandomBaseline` — obvious sanity floor.
- :class:`MeanDifferenceModel` — the current CausalBench SOTA-class
  baseline (Kowiel et al., 2023). Score each pair ``(A, B)`` by
  ``|mean(B | do(A)) - mean(B | control)|``, keep the top-k.

Anything your new method does should beat ``MeanDifferenceModel`` on
the statistical metric before you consider it working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .dataset import Dataset

Edge = tuple[str, str]


class Model(Protocol):
    """All models expose ``fit_predict(data) -> list[Edge]``.

    Implementations may be stateful (cache results on self) or stateless.
    """

    def fit_predict(self, data: Dataset) -> list[Edge]: ...


# ---------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------
@dataclass
class RandomBaseline:
    """Return ``top_k`` random directed edges over perturbed sources.

    The sanity floor. Any method that doesn't clearly beat this on the
    statistical metric is broken.
    """

    top_k: int = 1000
    seed: int = 0

    def fit_predict(self, data: Dataset) -> list[Edge]:
        rng = np.random.default_rng(self.seed)
        sources = data.perturbed_genes()
        targets = data.gene_names
        if not sources or not targets:
            return []
        # Build candidate pool; cap to keep memory sane at very high d.
        max_pool = max(self.top_k * 10, 20_000)
        pool: list[Edge] = []
        while len(pool) < max_pool:
            s = sources[rng.integers(len(sources))]
            t = targets[rng.integers(len(targets))]
            if s != t:
                pool.append((s, t))
        # Deduplicate while preserving order, then take top_k.
        seen: set[Edge] = set()
        out: list[Edge] = []
        for e in pool:
            if e in seen:
                continue
            seen.add(e)
            out.append(e)
            if len(out) >= self.top_k:
                break
        return out


# ---------------------------------------------------------------------
# Mean Difference
# ---------------------------------------------------------------------
@dataclass
class MeanDifferenceModel:
    """Kowiel, Kotlowski, Brzezinski 2023 — top-performing CausalBench baseline.

    For each perturbed gene ``A`` and each measured gene ``B != A``,
    compute ``score(A, B) = |mean(B | do(A)) - mean(B | control)|``.
    Return the top ``top_k`` pairs ranked by score.

    Implementation notes
    --------------------
    - The paper's "top 1k" and "top 5k" variants correspond to
      ``top_k=1000`` and ``top_k=5000`` here. Top-1k has higher
      Wasserstein (precision) but higher FOR; top-5k is the opposite.
    - We vectorise the mean computation over targets: one matrix op per
      perturbation. This makes the full computation linear in
      ``n_perturbed_genes`` rather than quadratic.
    - No hyperparameter tuning is needed. That is exactly what makes
      this baseline dangerous to beat — it has zero degrees of freedom
      for overfitting.
    """

    top_k: int = 1000

    def fit_predict(self, data: Dataset) -> list[Edge]:
        control_mask = data.control_mask()
        if not control_mask.any():
            raise ValueError("No control cells; cannot run Mean Difference.")
        control_means = data.expression[control_mask].mean(axis=0)  # (n_genes,)

        perturbed = data.perturbed_genes()
        if not perturbed:
            return []

        # Accumulate scores across all perturbed sources.
        # shape: (n_perturbed, n_genes)
        n_genes = data.n_genes
        all_scores = np.empty((len(perturbed), n_genes), dtype=np.float32)

        for i, src in enumerate(perturbed):
            mask = data.intervention_mask(src)
            intv_means = data.expression[mask].mean(axis=0)
            all_scores[i] = np.abs(intv_means - control_means)

        # Mask out self-edges by setting their score to -inf.
        for i, src in enumerate(perturbed):
            src_idx = data.gene_idx(src)
            all_scores[i, src_idx] = -np.inf

        # Flatten and take top_k.
        flat = all_scores.ravel()
        k = min(self.top_k, flat.size)
        # argpartition is O(n) vs argsort's O(n log n); important at d=5000.
        top_idx = np.argpartition(-flat, kth=k - 1)[:k]
        # Sort just the top k for stable ranking.
        top_idx = top_idx[np.argsort(-flat[top_idx])]

        edges: list[Edge] = []
        for idx in top_idx:
            row, col = divmod(int(idx), n_genes)
            if not np.isfinite(all_scores[row, col]):
                continue
            edges.append((perturbed[row], data.gene_names[col]))
        return edges
