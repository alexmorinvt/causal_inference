"""Differential-covariance edge ranker.

Given:

- ``Σ_ctrl`` — covariance of control cells.
- ``Σ_G`` for each perturbed ``G`` — covariance of cells in the
  ``do(G)`` arm.

For a candidate edge ``(j, i)``:

    diff_score[i, j] = Σ_G |Σ_ctrl[i, j] - Σ_G[i, j]|

Edges whose covariance is invariant across interventions are
confounder-driven or sampling noise; edges whose covariance shifts
under many interventions are part of the regulatory subnetwork.

Directionality
--------------
The raw ``diff_score`` is symmetric in ``(i, j)`` (covariance is
symmetric). We impose direction by combining with two interventional
signals:

- ``|shift[j, i]|`` when ``j`` is perturbed: direct effect of ``do(j)``
  on ``i``. If ``j`` is perturbed and ``shift[j, i]`` is large, then
  the ``j → i`` direction is supported.
- ``|shift[i, j]|`` conversely supports ``i → j``.

For pairs where only one endpoint is perturbed, direction is
unambiguous. For pairs where both are perturbed, the larger-shift
direction wins (classical do-calculus). For pairs where neither is
perturbed, we fall back to the diagonal-precision-ratio asymmetry
of observational β regression (an unavoidable observational-only
identifiability tool).

Final score for edge ``(source=j, target=i)``:

    score[j, i] = diff_score[i, j] * direction_weight[j, i]

where ``direction_weight[j, i] ∈ [0, 1]`` is the normalised fraction
of direction evidence pointing ``j → i``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dataset import Dataset

Edge = tuple[str, str]


@dataclass
class DiffCovModel:
    """Rank edges by intervention-sensitivity of pairwise covariance.

    Parameters
    ----------
    top_k
        Number of ranked edges to return.
    min_cells_per_arm
        Skip arms with fewer cells than this threshold (the empirical
        covariance becomes too noisy below ~20 cells per arm).
    observational_ridge
        Ridge regulariser on the precision-matrix diagonal, used only
        for the unperturbed-unperturbed direction fallback.
    diff_power
        Element-wise exponent applied to ``|Σ_ctrl − Σ_G|`` before
        averaging across arms; after the mean, the reciprocal exponent
        un-does the scaling. ``1.0`` = plain mean of absolute diffs.
        ``< 1`` (default ``0.5``, i.e. mean of sqrt) downweights
        single-arm outliers and rewards consistency across arms — an
        edge with moderate diff across many arms beats an edge with
        one arm's outlier-large diff. ``> 1`` amplifies single-arm
        peaks. Sweep on train picked 0.5.
    """

    top_k: int = 1000
    min_cells_per_arm: int = 20
    observational_ridge: float = 1e-4
    diff_power: float = 0.5

    def fit_predict(self, data: Dataset) -> list[Edge]:
        ctrl_mask = data.control_mask()
        if not ctrl_mask.any():
            raise ValueError(
                "No control cells; cannot compute covariance."
            )
        G = data.n_genes
        perturbed_genes_sorted = sorted(data.perturbed_genes())
        pert_mask = np.zeros(G, dtype=bool)
        for g in perturbed_genes_sorted:
            pert_mask[data.gene_idx(g)] = True

        # ---- Control covariance ----------------------------------------
        ctrl_expr = data.expression[ctrl_mask].astype(np.float64)
        ctrl_mean = ctrl_expr.mean(axis=0)
        ctrl_centered = ctrl_expr - ctrl_mean
        Sigma_ctrl = (ctrl_centered.T @ ctrl_centered) / max(
            ctrl_centered.shape[0], 1
        )
        np.fill_diagonal(Sigma_ctrl, 0.0)

        # ---- Per-arm covariances + diff-cov aggregation ---------------
        diff_score = np.zeros((G, G), dtype=np.float64)
        n_valid_arms = 0
        p = float(self.diff_power)
        for g in perturbed_genes_sorted:
            mask = data.intervention_mask(g)
            if mask.sum() < self.min_cells_per_arm:
                continue
            arm = data.expression[mask].astype(np.float64)
            arm_mean = arm.mean(axis=0)
            arm_centered = arm - arm_mean
            Sigma_g = (arm_centered.T @ arm_centered) / max(
                arm_centered.shape[0], 1
            )
            np.fill_diagonal(Sigma_g, 0.0)
            diff_score += np.abs(Sigma_ctrl - Sigma_g) ** p
            n_valid_arms += 1
        if n_valid_arms > 0:
            diff_score /= n_valid_arms
        if p != 1.0:
            diff_score = diff_score ** (1.0 / p)

        # ---- Direction weights ----------------------------------------
        # shifts[j, i] = mean(x_i | do(j)) - mean(x_i | control).
        shifts = np.zeros((G, G), dtype=np.float64)
        for g in perturbed_genes_sorted:
            mask = data.intervention_mask(g)
            if not mask.any():
                continue
            s = data.gene_idx(g)
            shifts[s, :] = (
                data.expression[mask].mean(axis=0) - ctrl_mean
            ).astype(np.float64)

        # Observational β from ridge-regularised precision, for the
        # unperturbed-unperturbed direction fallback.
        cov = ctrl_centered.T @ ctrl_centered / max(ctrl_centered.shape[0], 1)
        cov = cov + self.observational_ridge * np.eye(G, dtype=np.float64)
        prec = np.linalg.inv(cov)
        diag_prec = np.diag(prec).copy()
        diag_prec = np.where(diag_prec > 0.0, diag_prec, 1.0)
        beta_obs = -prec / diag_prec[:, None]
        np.fill_diagonal(beta_obs, 0.0)

        # For candidate (j, i) i.e. j -> i, compute direction_weight[j, i].
        direction_weight = np.zeros((G, G), dtype=np.float64)
        for j in range(G):
            for i in range(G):
                if i == j:
                    continue
                j_pert = pert_mask[j]
                i_pert = pert_mask[i]
                if j_pert and i_pert:
                    a = abs(shifts[j, i])
                    b = abs(shifts[i, j])
                    total = a + b
                    direction_weight[j, i] = a / total if total > 0.0 else 0.5
                elif j_pert and not i_pert:
                    direction_weight[j, i] = 1.0
                elif (not j_pert) and i_pert:
                    a = abs(shifts[i, j])
                    if a > 0.0:
                        direction_weight[j, i] = 0.2
                    else:
                        direction_weight[j, i] = 0.8
                else:
                    a = abs(beta_obs[i, j])
                    b = abs(beta_obs[j, i])
                    total = a + b
                    direction_weight[j, i] = a / total if total > 0.0 else 0.5

        # diff_score is symmetric ([i, j] = [j, i]); direction_weight
        # breaks the tie. score[j, i] is the edge (source=j, target=i).
        score = diff_score * direction_weight.T  # diff indexed by (i,j), already symmetric
        np.fill_diagonal(score, 0.0)

        # Rank top_k with score indexed as [source=j, target=i].
        # Currently score[row=i, col=j] has (i,j) entry; we want (j,i) indexing.
        score_src_tgt = score.T.astype(np.float32)  # score_src_tgt[j, i]
        np.fill_diagonal(score_src_tgt, 0.0)
        flat = score_src_tgt.ravel()
        k = min(self.top_k, flat.size)
        if k <= 0:
            return []
        top_idx = np.argpartition(-flat, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-flat[top_idx])]

        names = data.gene_names
        edges: list[Edge] = []
        for idx in top_idx:
            j, i = divmod(int(idx), G)
            if j == i or flat[idx] <= 0.0:
                continue
            edges.append((names[j], names[i]))
        return edges[: self.top_k]
