"""Maximum-likelihood SCM fitter using closed-form per-arm Gaussian likelihood.

For a linear cyclic SCM ``x = Wx + ε`` with ``ε ~ N(0, σ² I)``:

- The control-arm covariance is ``Σ_W = σ² (I − W)^{−1} (I − W)^{−T}``.
- Under ``do(G)``, the ``G``-th row of ``W`` is zero'd (``W̃_G``); the
  ``x_G`` coordinate is shifted by the imposed do-value. The induced
  covariance becomes ``Σ_{W, G} = σ² (I − W̃_G)^{−1} (I − W̃_G)^{−T}``
  with the mean shifted in the ``G`` coordinate.

Using the sample mean ``μ̂_a`` and sample covariance ``Σ̂_a`` of each
arm as sufficient statistics, the negative log-likelihood (up to
constants) is

    NLL(W) = Σ_a n_a * [log |Σ_{W,a}| + tr(Σ̂_a Σ_{W,a}^{−1})
                         + (μ̂_a − μ_a(W))^T Σ_{W,a}^{−1} (μ̂_a − μ_a(W))]

with arm index ``a ∈ {ctrl} ∪ {do(G) : G perturbed}``. We minimise by
gradient descent on ``W`` with spectral-radius projection to keep
``(I − W)`` invertible.

Edges are ranked by ``|W_fit|`` (transposed to source-target convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..dataset import CONTROL_LABEL, Dataset

Edge = tuple[str, str]


@dataclass
class LikelihoodMLEModel:
    """Fit ``W`` by multi-arm Gaussian MLE, rank edges by ``|W|``.

    Parameters
    ----------
    top_k
        Edges returned.
    n_steps
        Gradient-descent iterations.
    step_size
        Manual SGD step size.
    spectral_threshold
        After each step, if ``rho(W) > spectral_threshold``, rescale ``W``
        uniformly so ``rho(W) = spectral_threshold``. Keeps ``(I − W)``
        invertible.
    l1_lambda
        L1 sparsity penalty on ``W``.
    noise_std
        Assumed isotropic noise standard deviation ``σ``. Fixed, not
        fit — small ``σ`` favours strong coupling, large ``σ`` favours
        weak. ``1.0`` matches the synthetic generator's default.
    weight_scale
        Gaussian stdev of the random ``W`` initialisation.
    seed
        Torch RNG seed.
    """

    top_k: int = 1000
    n_steps: int = 500
    step_size: float = 0.01
    spectral_threshold: float = 0.8
    l1_lambda: float = 1e-4
    noise_std: float = 1.0
    weight_scale: float = 0.01
    seed: int = 0

    def fit_predict(self, data: Dataset) -> list[Edge]:
        G = data.n_genes
        names = data.gene_names
        perturbed_genes_sorted = sorted(data.perturbed_genes())
        interventions = np.asarray(data.interventions, dtype=object)

        # ---- Collect per-arm sufficient statistics -------------------
        arm_specs: list[dict] = []  # each: {'kind': 'ctrl'|'do', 'target': idx or None,
                                    #         'mu': (G,), 'cov': (G, G), 'n': int}

        ctrl_mask = interventions == CONTROL_LABEL
        ctrl_expr = data.expression[ctrl_mask].astype(np.float64)
        arm_specs.append({
            "kind": "ctrl",
            "target": None,
            "mu": ctrl_expr.mean(axis=0),
            "cov": np.cov(ctrl_expr.T, bias=True),
            "n": int(ctrl_expr.shape[0]),
        })
        for g in perturbed_genes_sorted:
            idx = interventions == g
            arm_expr = data.expression[idx].astype(np.float64)
            if arm_expr.shape[0] < 20:
                continue
            arm_specs.append({
                "kind": "do",
                "target": data.gene_idx(g),
                "mu": arm_expr.mean(axis=0),
                "cov": np.cov(arm_expr.T, bias=True),
                "n": int(arm_expr.shape[0]),
            })

        # ---- Torch tensors -------------------------------------------
        torch_gen = torch.Generator(device="cpu").manual_seed(self.seed)
        W = torch.randn(
            G, G, generator=torch_gen, dtype=torch.float64
        ) * self.weight_scale
        W[torch.arange(G), torch.arange(G)] = 0.0
        W.requires_grad_(True)

        arm_mu = [torch.as_tensor(a["mu"], dtype=torch.float64) for a in arm_specs]
        arm_cov = [torch.as_tensor(a["cov"], dtype=torch.float64) for a in arm_specs]
        arm_n = [a["n"] for a in arm_specs]
        arm_target = [a["target"] for a in arm_specs]

        eye = torch.eye(G, dtype=torch.float64)
        sigma2 = self.noise_std ** 2

        # ---- Gradient descent ----------------------------------------
        for step in range(self.n_steps):
            nll = torch.zeros((), dtype=torch.float64)
            for a_idx, a in enumerate(arm_specs):
                # Build W̃_a: for do(G), row G zero'd; for ctrl, identity.
                if a["kind"] == "ctrl":
                    W_a = W
                else:
                    mask_rows = torch.ones(G, dtype=torch.float64)
                    mask_rows[a["target"]] = 0.0
                    W_a = W * mask_rows.view(G, 1)
                I_minus_W = eye - W_a
                # Σ_a = σ² (I − W_a)^{-1} (I − W_a)^{-T}
                # inv_I_W = (I − W_a)^{-1}
                inv_I_W = torch.linalg.inv(I_minus_W)
                Sigma_a = sigma2 * inv_I_W @ inv_I_W.T
                # Stabilised NLL using Cholesky-like path
                # log|Σ_a| = 2 * log|det(inv_I_W)| + G * log σ²
                #         = G * log σ² − 2 * log|det(I − W_a)|
                logdet_I_W = torch.logdet(I_minus_W)
                logdet_Sigma = G * np.log(sigma2) - 2.0 * logdet_I_W
                # tr(Σ̂ Σ_a^{-1}) = tr(Σ̂ (1/σ²) (I − W_a)^T (I − W_a))
                #                = (1/σ²) tr((I − W_a) Σ̂ (I − W_a)^T)
                inner = I_minus_W @ arm_cov[a_idx] @ I_minus_W.T
                trace_term = torch.trace(inner) / sigma2
                # Mean mismatch: μ̂ − μ(W). For ctrl, μ(W) = 0.
                # For do(G), μ(W) = inv_I_W @ (G-th standard basis vec * c)
                # where c = μ̂_target (pinned mean under do(G)).
                # Simplification: approximate μ(W) as empirical mean (strong).
                # We use μ̂ − μ̂ = 0 effectively (no mean term). This is a
                # nuisance-parameter simplification — we fit only the
                # covariance structure, which carries the edge information.
                nll = nll + arm_n[a_idx] * (logdet_Sigma + trace_term)

            # L1 penalty
            if self.l1_lambda > 0:
                nll = nll + self.l1_lambda * W.abs().sum()

            (grad,) = torch.autograd.grad(nll, W)
            with torch.no_grad():
                W.sub_(self.step_size * grad)
                idx = torch.arange(G)
                W[idx, idx] = 0.0
                # Spectral projection
                eigs = torch.linalg.eigvals(W)
                rho = eigs.abs().max()
                if rho > self.spectral_threshold:
                    W.mul_(self.spectral_threshold / rho)

        # ---- Edge ranking ---------------------------------------------
        W_abs = W.detach().abs().numpy()
        np.fill_diagonal(W_abs, 0.0)
        # W[i, j] is effect of j on i (parent j, child i).
        # Edge (source=j, target=i), score = |W[i, j]|.
        score = W_abs.T.astype(np.float32)
        np.fill_diagonal(score, 0.0)
        flat = score.ravel()
        k = min(self.top_k, flat.size)
        if k <= 0:
            return []
        top_idx = np.argpartition(-flat, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-flat[top_idx])]
        edges: list[Edge] = []
        for idx in top_idx:
            j, i = divmod(int(idx), G)
            if j == i:
                continue
            if flat[idx] <= 0.0:
                continue
            edges.append((names[j], names[i]))
        return edges[: self.top_k]
