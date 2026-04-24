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
        ``W`` converges. Must be in ``(0, 1)``.
    ridge
        Small diagonal ridge added to ``(I + T)`` before inversion. A
        safety net for near-singular cases. Default ``1e-3`` is small
        enough not to materially change ``W`` on well-posed inputs.
    obs_correlation_weight
        Multiplier on the observational correlation columns used to
        impute ``T[:, j]`` for unperturbed ``j``. ``0.0`` leaves those
        columns at zero (method returns perturbed-source edges only).
        ``1.0`` uses correlations at the same scale as the perturbed
        shift columns after magnitude matching.
    clip_obs_to_pos
        If ``True``, zero out any negative entries in the imputed
        observational columns. Rationale: observational correlation
        is noisy and we're only using magnitude information downstream;
        keeping signs that reflect bivariate correlation (not causal
        direction) can interfere with the inversion. Default ``False``
        preserves signs (lets the inversion decompose signed paths
        correctly).
    """

    top_k: int = 1000
    spectral_target: float = 0.8
    ridge: float = 1e-3
    obs_correlation_weight: float = 1.0
    clip_obs_to_pos: bool = False

    def fit_predict(self, data: Dataset) -> list[Edge]:
        ctrl_mask = data.control_mask()
        if not ctrl_mask.any():
            raise ValueError(
                "No control cells; cannot compute shifts or covariance."
            )
        ctrl_expr = data.expression[ctrl_mask]
        ctrl_means = ctrl_expr.mean(axis=0)

        G = data.n_genes
        perturbed_set = set(data.perturbed_genes())

        # ---- Signed shift matrix (perturbed sources only) -------------
        shifts = np.zeros((G, G), dtype=np.float64)
        for src in perturbed_set:
            mask = data.intervention_mask(src)
            if not mask.any():
                continue
            s = data.gene_idx(src)
            shifts[s, :] = (
                data.expression[mask].mean(axis=0) - ctrl_means
            ).astype(np.float64)
        # shifts[s, t] = signed shift of gene t under do(s).

        # ---- Assemble T: T[i, j] = shifts[j, i] (perturbed j) ---------
        T = shifts.T.copy()  # T[i, j] = shift of i under do(j)
        np.fill_diagonal(T, 0.0)

        # ---- Impute unperturbed columns of T via observational corr ---
        pert_mask = np.zeros(G, dtype=bool)
        for g in perturbed_set:
            pert_mask[data.gene_idx(g)] = True

        if self.obs_correlation_weight > 0.0 and (~pert_mask).any():
            Xc = ctrl_expr - ctrl_means
            Sigma = (Xc.T @ Xc) / max(Xc.shape[0], 1)
            Sigma = Sigma.astype(np.float64)
            std = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
            Rho = Sigma / (std[:, None] * std[None, :])
            np.fill_diagonal(Rho, 0.0)

            # Rescale obs correlations to match the magnitude of the
            # observed shift columns: match the mean |nonzero entry|.
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
            # Only overwrite unperturbed columns.
            unpert_cols = np.where(~pert_mask)[0]
            T[:, unpert_cols] = imputed[:, unpert_cols]
            np.fill_diagonal(T, 0.0)

        # ---- Spectral projection to keep (I + T)^{-1} well-defined ----
        eigs = np.linalg.eigvals(T)
        rho_T = float(np.max(np.abs(eigs))) if eigs.size > 0 else 0.0
        if rho_T > self.spectral_target and rho_T > 0:
            T = T * (self.spectral_target / rho_T)

        # ---- W = T (I + T)^{-1} ---------------------------------------
        I_G = np.eye(G, dtype=np.float64)
        M = I_G + T + self.ridge * I_G
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            # Fallback: if somehow singular, use pseudo-inverse.
            M_inv = np.linalg.pinv(M)
        W_est = T @ M_inv
        np.fill_diagonal(W_est, 0.0)

        # ---- Rank edges by |W_est[i, j]|, edge is (source=j, target=i)
        # score[j, i] = |W_est[i, j]|
        score = np.abs(W_est).T.astype(np.float32)
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
