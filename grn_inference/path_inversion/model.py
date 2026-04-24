"""Path-inversion edge ranker.

Derivation
----------
Linear cyclic SCM: ``x = W x + eps`` with iid noise. Under an
intervention ``do(j)`` that pins ``x_j``, the downstream equilibrium
shift on gene ``i`` is

    shift[j, i] = E[x_i | do(j)] - E[x_i | ctrl]
                ≈ [(I - W)^{-1} W][i, j]    (for sparse W)

Define the total-effect matrix ``T`` with ``T[i, j] = shift[j, i]``.
Then ``T ≈ (I - W)^{-1} W``, equivalently

    (I - W) T = W  =>  T - W T = W  =>  T = W (I + T)  =>  W = T (I + T)^{-1}.

Neumann expansion (for ``rho(T) < 1``):

    W = T (I + T)^{-1} = T - T^2 + T^3 - T^4 + ...

Graph-theoretic reading
-----------------------
``T^k[i, j]`` counts length-``k`` weighted walks from ``j`` to ``i`` in
the shift graph. The alternating sum performs **inclusion-exclusion
over path lengths**: starting from all-paths (``T``), subtract the
two-step paths that the all-paths sum double-counted, add back the
three-step paths that the previous subtraction over-removed, and so
on. The limit isolates the direct edges.

This is the dual of the precision-matrix-based neighborhood regression
used by ``NeighborhoodRegressionModel``: instead of ``Σ^{-1}`` on
observational covariance, we invert ``(I + T)`` on interventional
shifts. No per-target regression, no bootstrap — a single matrix
inversion on the combined shift / observational-correlation matrix.

Imputation for unperturbed sources
----------------------------------
The total-effect column ``T[:, j]`` for perturbed ``j`` is directly
observed as the shift column. For unperturbed ``j`` the column is
missing; we impute it from the symmetric observational correlation
matrix, rescaled to match the magnitude of the observed shift columns
so the two halves of ``T`` live on compatible scales. The imputation is
crude — observational correlation is undirected and picks up both
``j -> i`` and ``i -> j`` as well as confounded pairs — so the
unperturbed half of ``T`` carries a direction-ambiguity cost that the
matrix inversion does not fully clean up. This is the known
observational-only identifiability gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dataset import Dataset

Edge = tuple[str, str]


@dataclass
class PathInversionModel:
    """Rank edges by ``|W|`` where ``W = T (I + T)^{-1}``.

    Parameters
    ----------
    top_k
        Number of ranked edges to return.
    spectral_target
        After assembling ``T``, rescale it uniformly so that
        ``rho(T) = spectral_target`` (spectral radius). Ensures
        ``(I + T)`` is well-conditioned and the Neumann series for
        ``W`` converges. Must be in ``(0, 1)``. Default ``0.7``
        selected by a train-seed sweep over {0.3, 0.5, 0.7, 0.8, 0.9,
        0.95, 0.99}; smaller values under-use the inversion and
        regress to MD, larger values over-expand and the Neumann
        alternating sum amplifies noise.
    ridge
        Small diagonal ridge added to ``(I + T)`` before inversion. A
        safety net for near-singular cases. Default ``1e-3`` is small
        enough not to materially change ``W`` on well-posed inputs.
    obs_correlation_weight
        Multiplier on the observational correlation columns used to
        impute ``T[:, j]`` for unperturbed ``j`` when
        ``imputation_mode="correlation"``. ``0.0`` leaves those columns
        at zero (method returns perturbed-source edges only). ``1.0``
        uses correlations at the same scale as the perturbed shift
        columns after magnitude matching.
    clip_obs_to_pos
        If ``True``, zero out any negative entries in the imputed
        observational columns (correlation mode only). Default
        ``False`` preserves signs.
    imputation_mode
        How to fill in ``T[:, j]`` for unperturbed source genes ``j``.

        - ``"correlation"`` (iter-14 original): use the rescaled
          control-cell correlation matrix. Undirected signal, carries
          observational-only direction ambiguity.
        - ``"iv_shift_regression"`` (iter-15 default): estimate the
          missing total-effect column via an IV regression of shift
          rows. For any perturbed ``G`` upstream of ``j``, a cascade
          ``G -> j -> i`` gives ``shift[G, i] ≈ shift[G, j] · T[i, j]``.
          Regressing ``shift[:, i]`` on ``shift[:, j]`` across perturbed
          ``G`` (i.e. ``β_iv[j, i] = ⟨s_j, s_i⟩ / ⟨s_j, s_j⟩`` with
          ``s_X = shift[pert, X]``) estimates ``T[i, j]`` and preserves
          direction: swapping ``i`` and ``j`` gives a different
          estimate. This is the direction-aware interventional-data
          imputation the correlation fallback lacks.
    n_bootstrap
        Number of bootstrap resamples of per-arm cells (control and
        every intervention arm resampled with replacement). The full
        pipeline — shifts, IV imputation, spectral projection, matrix
        inversion — is repeated per bootstrap and the resulting
        ``|W_est|`` matrices are averaged. At ``n=1`` a single
        resample just injects noise; ``n >= 10`` reliably reduces it.
        ``n=50`` is the default — a train-seed sweep over {1, 5, 10,
        20, 30, 50} showed the precision / hidden-recall frontier
        still improving slightly at 50 with sub-second runtime.
    bootstrap_seed
        Seed for the per-arm resampling RNG. Fixed default ``0`` so
        the ranker is deterministic.
    """

    top_k: int = 1000
    spectral_target: float = 0.7
    ridge: float = 1e-3
    obs_correlation_weight: float = 1.0
    clip_obs_to_pos: bool = False
    imputation_mode: str = "iv_shift_regression"
    n_bootstrap: int = 50
    bootstrap_seed: int = 0

    def fit_predict(self, data: Dataset) -> list[Edge]:
        ctrl_mask = data.control_mask()
        if not ctrl_mask.any():
            raise ValueError(
                "No control cells; cannot compute shifts or covariance."
            )
        ctrl_expr = data.expression[ctrl_mask]
        G = data.n_genes
        # Sort perturbed genes for deterministic iteration across Python
        # invocations (string hashes are randomized per-process; iterating
        # a set would change the order of bootstrap RNG draws).
        perturbed_genes_sorted = sorted(data.perturbed_genes())

        pert_mask = np.zeros(G, dtype=bool)
        pert_idx_list: list[int] = []
        for g in perturbed_genes_sorted:
            i = data.gene_idx(g)
            pert_mask[i] = True
            pert_idx_list.append(i)

        # Precompute per-arm cell indices for bootstrap.
        arm_indices: dict[str, np.ndarray] = {
            "__ctrl__": np.flatnonzero(ctrl_mask),
        }
        for g in perturbed_genes_sorted:
            arm_indices[g] = np.flatnonzero(data.intervention_mask(g))

        n_boot = max(1, int(self.n_bootstrap))
        rng = np.random.default_rng(self.bootstrap_seed)
        I_G = np.eye(G, dtype=np.float64)
        W_abs_agg = np.zeros((G, G), dtype=np.float64)

        for _ in range(n_boot):
            if n_boot > 1:
                ci = rng.choice(
                    arm_indices["__ctrl__"],
                    size=len(arm_indices["__ctrl__"]),
                    replace=True,
                )
                ctrl_expr_b = data.expression[ci]
                ctrl_means_b = ctrl_expr_b.mean(axis=0)
            else:
                ctrl_expr_b = ctrl_expr
                ctrl_means_b = ctrl_expr.mean(axis=0)

            shifts = np.zeros((G, G), dtype=np.float64)
            for src in perturbed_genes_sorted:
                idx_pool = arm_indices[src]
                if idx_pool.size == 0:
                    continue
                s = data.gene_idx(src)
                if n_boot > 1:
                    ii = rng.choice(idx_pool, size=idx_pool.size, replace=True)
                else:
                    ii = idx_pool
                shifts[s, :] = (
                    data.expression[ii].mean(axis=0) - ctrl_means_b
                ).astype(np.float64)

            T = shifts.T.copy()
            np.fill_diagonal(T, 0.0)

            if (~pert_mask).any():
                if self.imputation_mode == "iv_shift_regression" and pert_idx_list:
                    pert_idx = np.asarray(pert_idx_list)
                    S_pert = shifts[pert_idx, :]
                    cross = S_pert.T @ S_pert
                    diag_denom = np.diag(cross).copy()
                    diag_denom = np.where(diag_denom > 1e-12, diag_denom, 1.0)
                    beta_iv = cross / diag_denom[:, None]
                    np.fill_diagonal(beta_iv, 0.0)
                    unpert_cols = np.where(~pert_mask)[0]
                    for j in unpert_cols:
                        T[:, j] = beta_iv[j, :]
                    np.fill_diagonal(T, 0.0)
                elif (
                    self.imputation_mode == "correlation"
                    and self.obs_correlation_weight > 0.0
                ):
                    Xc = ctrl_expr_b - ctrl_means_b
                    Sigma = (Xc.T @ Xc) / max(Xc.shape[0], 1)
                    Sigma = Sigma.astype(np.float64)
                    std = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
                    Rho = Sigma / (std[:, None] * std[None, :])
                    np.fill_diagonal(Rho, 0.0)
                    shift_abs = np.abs(T[:, pert_mask])
                    shift_abs = shift_abs[shift_abs > 0.0]
                    rho_abs = np.abs(Rho)
                    rho_abs = rho_abs[rho_abs > 0.0]
                    if shift_abs.size > 0 and rho_abs.size > 0:
                        scale = shift_abs.mean() / rho_abs.mean()
                    else:
                        scale = 1.0
                    imputed = self.obs_correlation_weight * scale * Rho
                    if self.clip_obs_to_pos:
                        imputed = np.clip(imputed, 0.0, None)
                    unpert_cols = np.where(~pert_mask)[0]
                    T[:, unpert_cols] = imputed[:, unpert_cols]
                    np.fill_diagonal(T, 0.0)

            eigs = np.linalg.eigvals(T)
            rho_T = float(np.max(np.abs(eigs))) if eigs.size > 0 else 0.0
            if rho_T > self.spectral_target and rho_T > 0:
                T = T * (self.spectral_target / rho_T)

            M = I_G + T + self.ridge * I_G
            try:
                M_inv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                M_inv = np.linalg.pinv(M)
            W_est = T @ M_inv
            np.fill_diagonal(W_est, 0.0)
            W_abs_agg += np.abs(W_est)

        W_abs_agg /= n_boot

        # ---- Rank edges by averaged |W_est[i, j]|, edge (source=j, target=i)
        score = W_abs_agg.T.astype(np.float32)
        np.fill_diagonal(score, 0.0)

        flat = score.ravel()
        k = min(self.top_k, flat.size)
        if k <= 0:
            return []
        top_idx = np.argpartition(-flat, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-flat[top_idx])]

        names = data.gene_names
        edges: list[Edge] = []
        for idx in top_idx:
            j, i = divmod(int(idx), G)
            if j == i:
                continue
            if flat[idx] <= 0.0:
                continue
            edges.append((names[j], names[i]))
        return edges[: self.top_k]
