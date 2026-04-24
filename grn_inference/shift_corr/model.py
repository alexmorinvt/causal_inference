"""Mean-shift ranker augmented with within-arm correlation.

Pipeline
--------
1. For every perturbed source ``A`` and every target ``B``::

       shift[A, B] = | mean(B | do(A)) - mean(B | control) |
       corr[A, B]  = Pearson(A, B) within cells from do(A)

2. Combine into a single score::

       score[A, B] = shift[A, B] * (1 + corr_weight * |corr[A, B]|)

   ``corr_weight = 0`` reduces exactly to :class:`MeanDifferenceModel`.

3. Rank pairs by ``score`` and return the top ``top_k``.

Intuition
---------
Mean shift captures population-level effect. Within-arm correlation
captures whether cells with stronger residual knockdown of ``A`` also
show a proportionally stronger shift in ``B``. A true direct edge
propagates per-cell variation; a cascade shortcut attenuates it because
the intermediate gene adds noise between the two ends of the path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dataset import Dataset

Edge = tuple[str, str]


@dataclass
class ShiftCorrModel:
    """Rank edges by ``|shift| * (1 + w * |within-arm corr|)``.

    Parameters
    ----------
    top_k
        Maximum number of edges to return.
    corr_weight
        Multiplicative weight on the correlation bonus. ``0.0`` is
        identical to :class:`MeanDifferenceModel`.
    """

    top_k: int = 1000
    corr_weight: float = 1.0

    def fit_predict(self, data: Dataset) -> list[Edge]:
        ctrl_mask = data.control_mask()
        if not ctrl_mask.any():
            raise ValueError("No control cells; cannot compute shifts.")
        ctrl_means = data.expression[ctrl_mask].mean(axis=0)

        G = data.n_genes
        score = np.zeros((G, G), dtype=np.float32)

        for src in data.perturbed_genes():
            mask = data.intervention_mask(src)
            if not mask.any():
                continue
            src_idx = data.gene_idx(src)
            expr = data.expression[mask]

            shift = np.abs(expr.mean(axis=0) - ctrl_means)

            a = expr[:, src_idx]
            ac = a - a.mean()
            a_sq = float((ac ** 2).sum())
            if a_sq <= 0.0:
                corr_abs = np.zeros(G, dtype=np.float32)
            else:
                bc = expr - expr.mean(axis=0)
                b_sq = (bc ** 2).sum(axis=0)
                num = (ac[:, None] * bc).sum(axis=0)
                den = np.sqrt(a_sq * b_sq + 1e-12)
                corr = np.where(den > 0, num / den, 0.0)
                corr_abs = np.abs(corr).astype(np.float32)

            score[src_idx, :] = shift * (1.0 + self.corr_weight * corr_abs)

        np.fill_diagonal(score, 0.0)

        flat_idx = np.argsort(-score.ravel())
        G_idx = score.shape[1]
        edges: list[Edge] = []
        for idx in flat_idx[: self.top_k]:
            i, j = divmod(int(idx), G_idx)
            if score[i, j] <= 0:
                break
            edges.append((data.gene_names[i], data.gene_names[j]))
        return edges
