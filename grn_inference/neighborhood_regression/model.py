"""Hybrid shift + neighborhood-regression ranker.

Scoring
-------
For a candidate edge ``(source=S, target=T)``:

- If ``S`` is a perturbed gene (has intervention cells):
    score = |mean(T | do(S)) - mean(T | control)|
  This is identical to :class:`MeanDifferenceModel`'s score, which is
  near-oracle on Wasserstein for this bucket of edges.

- If ``S`` is an unperturbed gene (no intervention cells for ``S``):
    score = |beta[T, S]|
  where ``beta[T, S]`` is the coefficient on gene ``S`` in the OLS
  regression of gene ``T`` on all other genes, fit on control cells.
  Equivalent to ``|-Theta[T, S] / Theta[T, T]|`` with ``Theta`` the
  precision matrix of the (ridge-regularised) control covariance.

The two buckets live on different scales (a mean-shift in log1p-CPM
space vs a unitless regression coefficient), so a direct numeric
comparison is not meaningful. We rank within each bucket and
concatenate with a fixed per-bucket quota:

- ``k_pert = round((1 - unperturbed_fraction) * top_k)`` from the
  perturbed-source bucket,
- ``k_unpert = top_k - k_pert`` from the unperturbed-source bucket.

``unperturbed_fraction = 0.5`` is the natural prior when the generator
perturbs half the genes uniformly at random: under a uniform edge
model, roughly half of all true edges have perturbed sources and the
other half unperturbed sources. This is a principled default, not a
tuned value.

Identifiability caveat
----------------------
Observational regression recovers an **undirected** signal: ``beta[T, S]``
is non-zero whenever ``S`` and ``T`` are conditionally dependent given
the remaining genes. That includes true ``S -> T``, true ``T -> S``, and
longer paths via common causes. For the unperturbed-source bucket this
limitation is irreducible without additional assumptions — interventions
on ``S`` are the only way to certify direction, and we don't have them
here. We accept the false-positive rate on direction and rank by
magnitude anyway, because the alternative (predict nothing for
unperturbed sources) gives the floor that the baselines already occupy.

Linear-SCM justification
------------------------
For a linear cyclic SCM ``x = W x + eps`` with ``eps`` iid Gaussian and
stable ``rho(W) < 1``, the observational distribution is multivariate
Gaussian with precision matrix

    Theta = (I - W)^T Sigma_eps^{-1} (I - W).

Non-zero entries of ``Theta`` correspond to non-zero entries of ``W`` or
``W^T`` (up to cancellations that are non-generic in the weights). The
regression coefficient ``-Theta[T, S] / Theta[T, T]`` is the
best-linear-predictor coefficient of gene ``S`` in forecasting gene
``T`` from all other genes, so |beta[T, S]| is a principled magnitude
score for the directed candidate ``(S, T)`` when no intervention on
``S`` is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dataset import Dataset

Edge = tuple[str, str]


@dataclass
class NeighborhoodRegressionModel:
    """Rank edges by shift (perturbed sources) + regression coefficient
    (unperturbed sources), concatenating fixed quotas from each bucket.

    Parameters
    ----------
    top_k
        Total number of ranked edges to return.
    unperturbed_fraction
        Fraction of ``top_k`` allocated to the unperturbed-source bucket.
        The default of ``0.5`` matches the uniform-source prior on a
        generator that perturbs half the genes.
    ridge_lambda
        Ridge regulariser added to the control-covariance diagonal
        before inversion. Keeps the precision estimate well-posed when
        the control sample covariance is near-singular (e.g. when
        ``n_control_cells`` is small relative to ``n_genes``, or when
        genes are highly collinear). ``1e-4`` is small enough that the
        regression coefficients of well-observed gene pairs are
        essentially unchanged, but large enough that pathological
        near-rank-deficient directions don't blow up.
    reverse_shift_damping
        If ``True``, down-weight unperturbed-source candidates
        ``(S, T)`` by ``exp(-shift[T, S] / scale)`` when ``T`` is
        perturbed, where ``scale`` is the mean of the non-zero entries
        of the shift matrix (data-adaptive, no free hyperparameter).
        A large ``shift[T, S]`` is evidence that ``T`` is a causal
        ancestor of ``S``, i.e. the reverse direction ``T -> S`` is
        plausible; under a generic no-two-cycle assumption (which the
        synthetic generator enforces but which is also a defensible
        sparsity prior on real GRNs), the candidate ``S -> T`` is
        then unlikely. When ``T`` is unperturbed we have no shift
        information and leave the score unchanged.

        Set to ``False`` to recover iteration-1 scoring exactly.
    """

    top_k: int = 1000
    unperturbed_fraction: float = 0.5
    ridge_lambda: float = 1e-4
    reverse_shift_damping: bool = True

    def fit_predict(self, data: Dataset) -> list[Edge]:
        ctrl_mask = data.control_mask()
        if not ctrl_mask.any():
            raise ValueError(
                "No control cells; cannot compute control-cell precision."
            )
        ctrl_expr = data.expression[ctrl_mask]  # (n_ctrl, G)
        ctrl_means = ctrl_expr.mean(axis=0)

        G = data.n_genes
        perturbed_set = set(data.perturbed_genes())

        # ---- Perturbed-source bucket: MD-style shift -------------------
        shift = np.zeros((G, G), dtype=np.float32)
        for src in perturbed_set:
            mask = data.intervention_mask(src)
            if not mask.any():
                continue
            s = data.gene_idx(src)
            intv_means = data.expression[mask].mean(axis=0)
            shift[s, :] = np.abs(intv_means - ctrl_means)
        np.fill_diagonal(shift, 0.0)

        # ---- Unperturbed-source bucket: neighborhood regression --------
        # Center control expression, compute ridge-regularised
        # precision, extract regression coefficients.
        Xc = ctrl_expr - ctrl_means
        n_ctrl = Xc.shape[0]
        cov = (Xc.T @ Xc) / max(n_ctrl, 1)
        cov = cov.astype(np.float64) + self.ridge_lambda * np.eye(G, dtype=np.float64)
        prec = np.linalg.inv(cov)
        diag = np.diag(prec).copy()
        # Guard against any degenerate Theta[i, i] <= 0 (shouldn't happen
        # with a ridge, but be safe).
        diag = np.where(diag > 0.0, diag, 1.0)
        # beta[i, j] = -Theta[i, j] / Theta[i, i], the coefficient on
        # gene j in the regression of gene i on the rest.
        beta = -prec / diag[:, None]
        np.fill_diagonal(beta, 0.0)
        # We want a score indexed by (source, target) = (j, i), so take
        # the transpose of |beta|.
        beta_score = np.abs(beta).T.astype(np.float32)
        np.fill_diagonal(beta_score, 0.0)

        # ---- Bucket edges by source perturbation status ---------------
        pert_mask = np.zeros(G, dtype=bool)
        for g in perturbed_set:
            pert_mask[data.gene_idx(g)] = True

        # Optional direction-disambiguation multiplier: for candidate
        # (S unperturbed, T perturbed), multiply |beta[T, S]| by
        # exp(-shift[T, S] / scale). Scale is the mean of non-zero
        # shift entries (data-adaptive). See the class docstring.
        if self.reverse_shift_damping:
            nonzero = shift[shift > 0.0]
            if nonzero.size > 0:
                scale = float(nonzero.mean())
                if scale <= 0.0:
                    scale = 1.0
            else:
                scale = 1.0
            # damping[s, t] used for (src=s, tgt=t); only matters when
            # s unperturbed and t perturbed — indexing on shift[t, s]
            # (reverse-direction shift) is what we need.
            damping = np.exp(-shift.T / scale).astype(np.float32)
        else:
            damping = None

        # Collect (score, src_idx, tgt_idx) per bucket.
        pert_scores = []
        unpert_scores = []
        for s in range(G):
            for t in range(G):
                if s == t:
                    continue
                if pert_mask[s]:
                    sc = float(shift[s, t])
                    if sc > 0.0:
                        pert_scores.append((sc, s, t))
                else:
                    sc = float(beta_score[s, t])
                    # Apply reverse-shift damping only when the target
                    # is perturbed (we have shift[t, s] to evaluate);
                    # when t is unperturbed, shift[t, :] is all zeros
                    # and the damping factor is 1 anyway.
                    if sc > 0.0 and damping is not None and pert_mask[t]:
                        sc *= float(damping[s, t])
                    if sc > 0.0:
                        unpert_scores.append((sc, s, t))

        pert_scores.sort(key=lambda e: -e[0])
        unpert_scores.sort(key=lambda e: -e[0])

        # ---- Concatenate fixed quotas ---------------------------------
        k_unpert = int(round(self.unperturbed_fraction * self.top_k))
        k_unpert = min(k_unpert, len(unpert_scores))
        k_pert = self.top_k - k_unpert
        k_pert = min(k_pert, len(pert_scores))
        # If the unperturbed bucket is smaller than its quota, top up
        # the perturbed bucket so we still return top_k edges when
        # possible.
        leftover = self.top_k - k_pert - k_unpert
        if leftover > 0:
            extra_unpert = min(leftover, len(unpert_scores) - k_unpert)
            k_unpert += extra_unpert
            leftover -= extra_unpert
        if leftover > 0:
            extra_pert = min(leftover, len(pert_scores) - k_pert)
            k_pert += extra_pert

        names = data.gene_names
        out: list[Edge] = []
        # Emit perturbed-source edges first (higher-confidence direction),
        # then unperturbed. Evaluator doesn't re-rank, but this ordering
        # makes matched-W1 prefix plots read more naturally.
        for _, s, t in pert_scores[:k_pert]:
            out.append((names[s], names[t]))
        for _, s, t in unpert_scores[:k_unpert]:
            out.append((names[s], names[t]))
        return out[: self.top_k]
