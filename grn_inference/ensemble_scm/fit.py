"""Manual-SGD fitting loop for the ensemble SCM.

At each step we pick one perturbation arm (control or an intervention
on a perturbed gene), simulate ``batch_size`` cells from every
candidate under that arm, compare to a matching batch of real cells
via moment-matching, compute the gradient of the summed per-candidate
discrepancy w.r.t. W, and take a small manual step.

"Training" vocabulary is deliberately avoided: this is parameter
estimation for a mechanistic SCM, not machine learning. Torch is used
only for ``torch.autograd.grad``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..dataset import CONTROL_LABEL, Dataset
from .loss import l1_penalty, moment_matching_discrepancy
from .simulator import LinearSCM, simulate_control, simulate_intervention


@dataclass
class FitHistory:
    """Per-step diagnostics from ``fit_scm_ensemble``."""

    discrepancy: np.ndarray  # (n_steps, n_candidates) — per-candidate discrepancy
    target: list[str]  # (n_steps,) — which arm was sampled each step


def _real_cells_for_arm(
    data: Dataset,
    target: str,
    batch_size: int,
    rng: np.random.Generator,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Pull up to ``batch_size`` real cells matching ``target`` into a tensor.

    Returns None if there are too few cells in this arm to be useful.
    """
    if target == CONTROL_LABEL:
        mask = data.control_mask()
    else:
        mask = data.intervention_mask(target)
    idx_all = np.flatnonzero(mask)
    if idx_all.size < 2:
        return None
    take = min(batch_size, idx_all.size)
    idx = rng.choice(idx_all, size=take, replace=False)
    return torch.as_tensor(data.expression[idx], dtype=dtype)


def fit_scm_ensemble(
    scm: LinearSCM,
    data: Dataset,
    *,
    n_steps: int = 1000,
    step_size: float = 0.01,
    batch_size: int = 200,
    l1_lambda: float = 1e-4,
    variance_weight: float = 1.0,
    knockdown_factor: float = 0.3,
    spectral_threshold: float | None = 0.80,
    n_cascade_steps: int = 5,
    seed: int = 0,
    log_every: int | None = 100,
) -> FitHistory:
    """Fit the ensemble by gradient descent on W.

    Parameters mirror the hyperparameters on ``EnsembleSCMFitter``. The
    SCM's ``W`` tensor is updated in place.

    Each step:
      1. sample an arm uniformly from {control} ∪ {perturbed genes},
      2. simulate ``batch_size`` cells per candidate from that arm,
      3. pull ``batch_size`` real cells from the same arm,
      4. compute per-candidate moment-matching discrepancy + L1 penalty,
      5. sum to a scalar, take grad w.r.t. W, step,
      6. (if ``spectral_threshold`` is set) project each candidate's W
         back into the stable region ``rho(W) <= spectral_threshold``.

    The projection is what keeps ``(I - W)^{-1}`` bounded; without it
    the loss landscape becomes stiff near ``rho = 1`` and any fixed
    step size will eventually overshoot into ``rho >> 1``, after which
    the fit is dead.

    Per-candidate gradients are independent because the simulated cells
    for candidate k come from W[k] only; summing the per-candidate
    losses therefore doesn't cross-contaminate the updates.
    """
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator(device=scm.W.device).manual_seed(seed)
    dtype = scm.W.dtype

    arms = [CONTROL_LABEL] + list(data.perturbed_genes())
    if not arms:
        raise ValueError("Dataset has no control and no perturbed genes; cannot fit.")

    history_discrepancy = np.zeros((n_steps, scm.n_candidates), dtype=np.float32)
    history_target: list[str] = []

    for step in range(n_steps):
        target = arms[rng.integers(len(arms))]
        real = _real_cells_for_arm(data, target, batch_size, rng, dtype)
        if real is None:
            # Degenerate arm — skip this step.
            history_discrepancy[step] = history_discrepancy[step - 1] if step else 0.0
            history_target.append(target)
            continue

        if target == CONTROL_LABEL:
            sim = simulate_control(
                scm, batch_size, generator=torch_gen, n_steps=n_cascade_steps,
            )
        else:
            tgt_idx = data.gene_idx(target)
            sim = simulate_intervention(
                scm, tgt_idx, batch_size,
                knockdown_factor=knockdown_factor,
                generator=torch_gen,
                n_steps=n_cascade_steps,
            )

        per_cand = moment_matching_discrepancy(
            sim, real, variance_weight=variance_weight
        )
        if l1_lambda > 0.0:
            per_cand = per_cand + l1_penalty(scm.W, l1_lambda)

        loss = per_cand.sum()
        (grad,) = torch.autograd.grad(loss, scm.W)

        with torch.no_grad():
            scm.W.sub_(step_size * grad)
            # Keep diagonals at zero (no self-loops in the SCM definition).
            idx = torch.arange(scm.n_genes)
            scm.W[:, idx, idx] = 0.0

            if spectral_threshold is not None:
                # Project each candidate's W back into {rho(W) <= threshold}
                # by uniform rescaling. This is the bare-minimum fix to
                # keep (I - W)^{-1} bounded.
                eigs = torch.linalg.eigvals(scm.W)  # complex (N, G)
                rho = eigs.abs().max(dim=-1).values  # (N,) real
                scale = torch.where(
                    rho > spectral_threshold,
                    spectral_threshold / rho,
                    torch.ones_like(rho),
                )
                scm.W.mul_(scale.view(-1, 1, 1))

        history_discrepancy[step] = per_cand.detach().cpu().numpy()
        history_target.append(target)

        if log_every and (step + 1) % log_every == 0:
            mean_disc = history_discrepancy[step].mean()
            print(
                f"[fit_scm_ensemble] step {step + 1:>5d}/{n_steps}  "
                f"arm={target:<12s}  mean discrepancy across candidates={mean_disc:.4f}"
            )

    return FitHistory(discrepancy=history_discrepancy, target=history_target)
