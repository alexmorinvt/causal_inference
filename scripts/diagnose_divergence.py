"""Diagnostic: track what happens to W during the ensemble fit.

Runs a manual loop that mirrors fit_scm_ensemble but records, at fixed
checkpoints, per-candidate:

- spectral radius rho(W) — the quantity that actually controls whether
  (I - W)^{-1} is bounded,
- Frobenius norm ||W||_F — a cheaper upper bound on the operator norm,
- gradient Frobenius norm ||grad||_F — how aggressive the current step is,
- the moment-matching discrepancy at the current step.

Lets us see:
- when divergence starts (step index),
- which candidates diverge and how fast,
- whether gradients explode before W does, or after.
"""

from __future__ import annotations

import numpy as np
import torch

from grn_inference import make_synthetic_dataset
from grn_inference.dataset import CONTROL_LABEL
from grn_inference.ensemble_scm import (
    LinearSCM,
    moment_matching_discrepancy,
    simulate_control,
    simulate_intervention,
)
from grn_inference.ensemble_scm.fit import _real_cells_for_arm


def spectral_radii(W: torch.Tensor) -> np.ndarray:
    """Per-candidate spectral radius. W: (N, G, G) -> (N,)."""
    eigs = torch.linalg.eigvals(W)  # complex (N, G)
    return eigs.abs().max(dim=-1).values.cpu().numpy()


def frobenius_per_candidate(W: torch.Tensor) -> np.ndarray:
    return W.flatten(1).norm(dim=-1).cpu().numpy()


def main() -> None:
    data, _ = make_synthetic_dataset(
        n_genes=50,
        edge_density=0.12,
        n_control_cells=2000,
        n_cells_per_perturbation=200,
        seed=7,
    )

    n_candidates = 5
    n_steps = 1000
    step_size = 0.01
    batch_size = 200
    l1_lambda = 1e-4
    spectral_threshold: float | None = 0.95  # set to None to reproduce the divergence
    seed = 0

    scm = LinearSCM.random_init(
        n_genes=data.n_genes,
        n_candidates=n_candidates,
        weight_scale=0.01,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator(device=scm.W.device).manual_seed(seed)

    arms = [CONTROL_LABEL] + list(data.perturbed_genes())

    # Checkpoint every step for the first 30, then every 50 after that.
    # All the action is in the earliest steps.
    fine_window = 30
    checkpoint_every = 50
    trajectories: dict[str, list] = {
        "step": [],
        "rho": [],          # (T, N)
        "frob": [],         # (T, N)
        "grad_frob": [],    # (T, N)
        "disc": [],         # (T, N)
    }

    def record(step: int, disc: np.ndarray, grad: torch.Tensor | None) -> None:
        trajectories["step"].append(step)
        trajectories["rho"].append(spectral_radii(scm.W.detach()))
        trajectories["frob"].append(frobenius_per_candidate(scm.W.detach()))
        if grad is None:
            trajectories["grad_frob"].append(np.zeros(n_candidates))
        else:
            trajectories["grad_frob"].append(
                frobenius_per_candidate(grad.detach())
            )
        trajectories["disc"].append(disc.copy())

    # Log the starting state.
    record(0, np.zeros(n_candidates), None)

    last_disc = np.zeros(n_candidates, dtype=np.float32)

    for step in range(n_steps):
        target = arms[rng.integers(len(arms))]
        real = _real_cells_for_arm(data, target, batch_size, rng, scm.W.dtype)
        if real is None:
            continue
        if target == CONTROL_LABEL:
            sim = simulate_control(scm, batch_size, generator=torch_gen)
        else:
            sim = simulate_intervention(
                scm, data.gene_idx(target), batch_size,
                knockdown_factor=0.3, generator=torch_gen,
            )
        per_cand = moment_matching_discrepancy(sim, real, variance_weight=1.0)
        if l1_lambda > 0.0:
            per_cand = per_cand + l1_lambda * scm.W.abs().sum(dim=(-2, -1))
        loss = per_cand.sum()
        (grad,) = torch.autograd.grad(loss, scm.W)

        last_disc = per_cand.detach().cpu().numpy()

        with torch.no_grad():
            scm.W.sub_(step_size * grad)
            idx = torch.arange(scm.n_genes)
            scm.W[:, idx, idx] = 0.0
            if spectral_threshold is not None:
                eigs = torch.linalg.eigvals(scm.W)
                rho = eigs.abs().max(dim=-1).values
                scale = torch.where(
                    rho > spectral_threshold,
                    spectral_threshold / rho,
                    torch.ones_like(rho),
                )
                scm.W.mul_(scale.view(-1, 1, 1))

        if (
            (step + 1) <= fine_window
            or (step + 1) % checkpoint_every == 0
            or step == n_steps - 1
        ):
            record(step + 1, last_disc, grad)

    # Convert to arrays.
    steps = np.array(trajectories["step"])
    rho = np.stack(trajectories["rho"])
    frob = np.stack(trajectories["frob"])
    grad_frob = np.stack(trajectories["grad_frob"])
    disc = np.stack(trajectories["disc"])

    # --- Print per-candidate trajectories -----------------------------------
    def print_metric(name: str, arr: np.ndarray, fmt: str) -> None:
        print(f"\n{name} per candidate (rows = step, cols = candidate 0..{n_candidates-1}):")
        header = f"  {'step':>5}  " + " ".join(f"{'c' + str(k):>12}" for k in range(n_candidates))
        print(header)
        for i, s in enumerate(steps):
            row = f"  {s:>5d}  " + " ".join(fmt.format(arr[i, k]) for k in range(n_candidates))
            print(row)

    print("=" * 78)
    print("Divergence diagnostic (n_candidates=5, n_steps=1000, step_size=0.01)")
    print("=" * 78)
    print_metric("Spectral radius rho(W)", rho, "{:>12.4f}")
    print_metric("Frobenius norm ||W||_F", frob, "{:>12.3e}")
    print_metric("Gradient Frobenius ||grad||_F", grad_frob, "{:>12.3e}")
    print_metric("Discrepancy (current arm)", disc, "{:>12.3e}")

    # --- Crossing points: first step where rho >= 1.0 per candidate ---------
    print("\nFirst step each candidate crosses rho(W) >= 1.0:")
    for k in range(n_candidates):
        cross_idx = np.argmax(rho[:, k] >= 1.0)
        if rho[cross_idx, k] < 1.0:
            print(f"  candidate[{k}]: never crossed   (final rho = {rho[-1, k]:.3f})")
        else:
            print(
                f"  candidate[{k}]: step {steps[cross_idx]:>4d}   "
                f"(rho before = {rho[cross_idx - 1, k]:.3f} -> "
                f"rho at = {rho[cross_idx, k]:.3f})"
            )


if __name__ == "__main__":
    main()
