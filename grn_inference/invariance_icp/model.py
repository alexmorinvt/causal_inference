"""Invariant-across-environments edge ranker.

Algorithm
---------
For each candidate edge ``(S, T)``:

1. Identify usable arms: ``{control} ∪ {do(G) : G ≠ S, G ≠ T}``.
   Arms where ``T`` is perturbed don't satisfy ``T``'s structural
   equation; arms where ``S`` is perturbed pin ``x_S`` and make the
   univariate regression of ``x_T`` on ``x_S`` unidentifiable.
2. For each usable arm, compute the simple OLS coefficient
   ``β_arm[T, S] = Cov(x_S, x_T) / Var(x_S)`` within-arm.
3. Score the candidate:

       score(S, T) = mean_arm |β_arm| / (std_arm(β_arm) + ε)

   High score = strong, stable coefficient across arms (signal-to-
   noise ratio of an invariant coefficient). Confounded pairs have
   β that shifts per arm; noise-pairs have small mean; both get low
   score.

Pareto position
---------------
ICP's identifiability argument is orthogonal to the precision-matrix
one used by NR: ICP needs multiple environments (interventions) and
fails without them, while NR needs the noise to be close to Gaussian
and the SCM to be near-linear. On our partial-perturbation synthetic
data with 25 perturbation arms, ICP has many usable environments per
edge pair (~22-23 non-pinned arms). The within-arm samples are small
(200 cells) so per-arm β estimates are noisy; aggregating with the
SNR ratio leverages the noise-reduction across arms.

Unperturbed-source edges get IV-style boost: for ``(S unpert, T any)``,
the per-arm regression of ``x_T`` on ``x_S`` is well-defined in all
arms except ``do(T)``, so ICP already covers unperturbed sources
without a special imputation step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dataset import CONTROL_LABEL, Dataset

Edge = tuple[str, str]


@dataclass
class InvarianceICPModel:
    """Rank edges by cross-arm stability of pairwise regression.

    Parameters
    ----------
    top_k
        Number of ranked edges to return.
    min_cells_per_arm
        Skip arms with fewer cells (β estimate unreliable).
    score_mode
        How to aggregate per-arm ``|β_arm|`` into a single score.

        - ``"snr"`` (iter-34 exploration): ``mean|β| / (std β + eps)``,
          signal-to-noise. Theoretical match to ICP but empirically
          too conservative.
        - ``"mean_abs"`` (iter-35 default): plain ``mean|β|``. Rewards
          edges where the pairwise regression coefficient is
          consistently non-zero across arms, regardless of its
          cross-arm variance. Train sweep picked this.
        - ``"median_abs"``: ``median|β|``. Robust to outlier arms.
    shift_boost_power
        Multiply the per-edge score by ``|shift[s, t]|^shift_boost_power``
        for perturbed-source edges ``(s, t)``. Combines ICP's
        cross-arm invariance (which already covers unperturbed
        sources naturally, since ``β_arm[T, S]`` is defined in every
        non-pinning arm) with the direct-intervention signal for
        perturbed sources. ``0.5`` selected by train sweep — larger
        powers trade precision for hidden recall.
    var_epsilon
        Added to the std of per-arm β values before dividing — guards
        against division by near-zero in ``"snr"`` mode.
    regression_mode
        - ``"bivariate"``: per arm, regress each ``x_T`` on each
          ``x_S`` separately (pairwise OLS).
        - ``"multivariate"`` (default): per arm, regress each ``x_T``
          on all other genes jointly (via ridge-regularised precision),
          then read off ``β_arm[T, S] = -Θ[T, S]/Θ[T, T]``. Closer to
          the original ICP formulation; conditions on confounders
          available in the arm and gives cleaner direct-parent
          coefficients.
    ridge
        Ridge regulariser on the per-arm covariance diagonal before
        inversion (multivariate mode only). Keeps the per-arm
        precision well-posed when ``n_cells_per_arm`` is close to
        ``n_genes``.
    """

    top_k: int = 1000
    min_cells_per_arm: int = 50
    score_mode: str = "mean_abs"
    shift_boost_power: float = 0.5
    var_epsilon: float = 1e-3
    regression_mode: str = "multivariate"
    ridge: float = 1e-3

    def fit_predict(self, data: Dataset) -> list[Edge]:
        G = data.n_genes
        names = data.gene_names
        perturbed_genes_sorted = sorted(data.perturbed_genes())

        # ---- Collect per-arm cell indices -----------------------------
        # Control arm labeled by CONTROL_LABEL; perturbed arms by gene name.
        interventions = np.asarray(data.interventions, dtype=object)
        arm_indices: dict[str, np.ndarray] = {
            CONTROL_LABEL: np.flatnonzero(interventions == CONTROL_LABEL),
        }
        for g in perturbed_genes_sorted:
            arm_indices[g] = np.flatnonzero(interventions == g)

        # Filter arms that are too small.
        usable_arms = {
            name: idx
            for name, idx in arm_indices.items()
            if idx.size >= self.min_cells_per_arm
        }

        if len(usable_arms) < 2:
            # Need at least 2 arms for cross-arm invariance.
            return []

        # ---- Per-arm β estimation -------------------------------------
        arm_names = list(usable_arms.keys())
        A = len(arm_names)

        if self.regression_mode == "bivariate":
            # Bivariate: β_arm[t, s] = Cov(s, t) / Var(s) per arm.
            arm_cov: dict[str, np.ndarray] = {}
            arm_var: dict[str, np.ndarray] = {}
            for name, idx in usable_arms.items():
                x = data.expression[idx].astype(np.float64)
                xc = x - x.mean(axis=0)
                n = xc.shape[0]
                cov = (xc.T @ xc) / max(n, 1)
                arm_cov[name] = cov
                arm_var[name] = np.diag(cov).copy()
            cov_stack = np.stack([arm_cov[n] for n in arm_names], axis=0)
            var_stack = np.stack([arm_var[n] for n in arm_names], axis=0)
            var_safe = np.where(var_stack > 1e-12, var_stack, 1.0)
            beta = cov_stack / var_safe[:, None, :]  # (A, G, G); [a, t, s]
        elif self.regression_mode == "multivariate":
            # Multivariate: β_arm[t, s] = -Θ_arm[t, s] / Θ_arm[t, t]
            # where Θ_arm is ridge-regularised per-arm precision.
            beta_list = []
            for name, idx in usable_arms.items():
                x = data.expression[idx].astype(np.float64)
                xc = x - x.mean(axis=0)
                n = xc.shape[0]
                cov = (xc.T @ xc) / max(n, 1) + self.ridge * np.eye(G)
                prec = np.linalg.inv(cov)
                diag = np.diag(prec).copy()
                diag = np.where(diag > 0, diag, 1.0)
                beta_a = -prec / diag[:, None]
                np.fill_diagonal(beta_a, 0.0)
                beta_list.append(beta_a)
            beta = np.stack(beta_list, axis=0)
        else:
            raise ValueError(
                f"Unknown regression_mode={self.regression_mode!r}; "
                "expected 'bivariate' or 'multivariate'."
            )

        # ---- Usability mask: for candidate (s, t), skip arms that ---
        # pin x_t (T's structural eq breaks). In multivariate mode we
        # don't need to skip arms pinning x_s, because conditioning on
        # x_s = const is fine for the regression of x_t on everyone.
        # In bivariate mode, pinning x_s makes Var(x_s)=0 so skip both.
        arm_matches_gene = np.zeros((A, G), dtype=bool)
        for a, aname in enumerate(arm_names):
            if aname == CONTROL_LABEL:
                continue
            arm_matches_gene[a, data.gene_idx(aname)] = True

        if self.regression_mode == "bivariate":
            usable_mask = ~(
                arm_matches_gene[:, :, None] | arm_matches_gene[:, None, :]
            )
        else:  # multivariate: only T-pinning matters
            # usable[a, t, s] = not arm_matches_gene[a, t]
            usable_mask = ~np.broadcast_to(
                arm_matches_gene[:, :, None], (A, G, G)
            ).copy()

        beta_masked = np.where(usable_mask, beta, np.nan)

        # Aggregate across arms (axis=0).
        abs_beta = np.abs(beta_masked)
        if self.score_mode == "snr":
            mean_abs = np.nanmean(abs_beta, axis=0)
            std_beta = np.nanstd(beta_masked, axis=0)
            score_ts = mean_abs / (std_beta + self.var_epsilon)
        elif self.score_mode == "mean_abs":
            score_ts = np.nanmean(abs_beta, axis=0)
        elif self.score_mode == "median_abs":
            score_ts = np.nanmedian(abs_beta, axis=0)
        else:
            raise ValueError(
                f"Unknown score_mode={self.score_mode!r}; expected "
                "'snr', 'mean_abs', or 'median_abs'."
            )
        score_ts = np.nan_to_num(score_ts, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(score_ts, 0.0)

        # score_ts[t, s] is indexed target,source. For edge (source=s, target=t),
        # we want score[s, t] = score_ts[t, s]. Take transpose.
        score = score_ts.T.astype(np.float64)
        np.fill_diagonal(score, 0.0)

        # ---- Shift boost on perturbed-source rows --------------------
        if self.shift_boost_power != 0.0:
            # Compute |shift| for perturbed sources.
            ctrl_mask = data.control_mask()
            ctrl_expr_full = data.expression[ctrl_mask]
            ctrl_mean_full = ctrl_expr_full.mean(axis=0)
            shift_abs = np.zeros((G, G), dtype=np.float64)
            for g in perturbed_genes_sorted:
                m = data.intervention_mask(g)
                if not m.any():
                    continue
                s_idx = data.gene_idx(g)
                shift_abs[s_idx, :] = np.abs(
                    data.expression[m].mean(axis=0) - ctrl_mean_full
                )
            # Multiply score[s, :] by |shift[s, :]|^power for perturbed s.
            pert_mask = np.zeros(G, dtype=bool)
            for g in perturbed_genes_sorted:
                pert_mask[data.gene_idx(g)] = True
            for s_idx in np.where(pert_mask)[0]:
                score[s_idx, :] = score[s_idx, :] * (
                    shift_abs[s_idx, :] ** self.shift_boost_power
                )
            np.fill_diagonal(score, 0.0)

        score = score.astype(np.float32)

        flat = score.ravel()
        k = min(self.top_k, flat.size)
        if k <= 0:
            return []
        top_idx = np.argpartition(-flat, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-flat[top_idx])]

        edges: list[Edge] = []
        for idx in top_idx:
            s, t = divmod(int(idx), G)
            if s == t:
                continue
            if flat[idx] <= 0.0:
                continue
            edges.append((names[s], names[t]))
        return edges[: self.top_k]
