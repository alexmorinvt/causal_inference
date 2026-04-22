"""Diagnose whether 1000 iterations + step_size=0.01 is the right budget.

For each step_size, fit on the partial-perturbation synthetic for a
large fixed n_steps. At several checkpoints (250, 500, 1000, 2000,
4000), evaluate:

- discrepancy trajectory (smoothed) — has the fit plateaued?
- mean W1 at top_k=500,
- precision@top_k=500,
- true unperturbed-source recall @ top_k=500.

Answers:
- "Is 1000 iterations enough?" — look at the 1000 vs 4000 rows for the
  default step_size. If metrics materially improve at 4000, we're
  under-training.
- "Is step_size=0.01 too high?" — compare rows across step_sizes at
  the same n_steps. If a smaller step_size reaches lower discrepancy
  or better metrics, 0.01 is overshooting.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from grn_inference import (
    EnsembleSCMFitter,
    evaluate_statistical,
    make_synthetic_dataset,
)
from grn_inference.dataset import CONTROL_LABEL
from grn_inference.ensemble_scm import (
    LinearSCM,
    aggregate_scores,
    moment_matching_discrepancy,
    rank_edges,
    simulate_control,
    simulate_intervention,
)
from grn_inference.ensemble_scm.fit import _real_cells_for_arm


def fit_with_checkpoints(
    data,
    *,
    step_size: float,
    n_steps: int,
    checkpoints: list[int],
    n_candidates: int = 5,
    batch_size: int = 200,
    l1_lambda: float = 1e-4,
    spectral_threshold: float | None = 0.95,
    weight_scale: float = 0.01,
    seed: int = 0,
) -> dict:
    """Run one fit, snapshot W at the listed checkpoints.

    Returns a dict: step_idx -> (W_copy, discrepancy_so_far).
    """
    scm = LinearSCM.random_init(
        n_genes=data.n_genes,
        n_candidates=n_candidates,
        weight_scale=weight_scale,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator(device=scm.W.device).manual_seed(seed)
    arms = [CONTROL_LABEL] + list(data.perturbed_genes())
    disc_history = np.zeros((n_steps, n_candidates), dtype=np.float32)

    snapshots: dict[int, tuple[torch.Tensor, np.ndarray]] = {}
    checkpoint_set = set(checkpoints)

    for step in range(n_steps):
        target = arms[rng.integers(len(arms))]
        real = _real_cells_for_arm(data, target, batch_size, rng, scm.W.dtype)
        if real is None:
            disc_history[step] = disc_history[step - 1] if step else 0.0
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

        disc_history[step] = per_cand.detach().cpu().numpy()

        step_idx = step + 1
        if step_idx in checkpoint_set:
            snapshots[step_idx] = (
                scm.W.detach().clone(),
                disc_history[:step_idx].copy(),
            )

    return snapshots


def evaluate_snapshot(
    W_snap: torch.Tensor,
    data,
    truth,
    perturbed_set: set,
    top_k: int,
    aggregation_power: float = 3.0,
) -> dict:
    score = aggregate_scores(W_snap, power=aggregation_power)
    edges = rank_edges(score, data.gene_names, top_k)
    true_set = set(truth.true_edges)
    true_unpert = {e for e in true_set if e[0] not in perturbed_set}

    res = evaluate_statistical(
        edges, data, omission_sample_size=500,
        rng=np.random.default_rng(99),
    )
    hits = sum(1 for e in edges if e in true_set)
    hits_unpert = sum(1 for e in edges if e in true_unpert)
    n_unpert_picks = sum(1 for s, _ in edges if s not in perturbed_set)
    recall_unpert = hits_unpert / max(len(true_unpert), 1)

    return {
        "mean_w1": res.mean_wasserstein,
        "precision": hits / top_k,
        "hits": hits,
        "hits_unpert": hits_unpert,
        "n_unpert_picks": n_unpert_picks,
        "recall_unpert": recall_unpert,
    }


def main() -> None:
    data, truth = make_synthetic_dataset(
        n_genes=50,
        edge_density=0.12,
        n_control_cells=2000,
        n_cells_per_perturbation=200,
        n_perturbed_genes=25,
        seed=7,
    )
    perturbed_set = set(data.perturbed_genes())

    step_sizes = [0.003, 0.01, 0.03]
    n_steps = 4000
    checkpoints = [250, 500, 1000, 2000, 4000]
    top_k = 500

    print(
        f"n_genes=50, n_perturbed=25, true_edges={len(truth.true_edges)}, "
        f"top_k={top_k}"
    )
    print(f"spectral_threshold=0.95, n_candidates=5, batch_size=200, l1=1e-4\n")

    all_results = {}
    for step_size in step_sizes:
        t0 = time.time()
        snaps = fit_with_checkpoints(
            data, step_size=step_size, n_steps=n_steps,
            checkpoints=checkpoints,
        )
        all_results[step_size] = snaps
        print(f"  step_size={step_size}: fit {n_steps} steps in {time.time() - t0:.1f}s")

    print()
    for step_size in step_sizes:
        print(f"\n=== step_size = {step_size} ===")
        header = (
            f"{'n_steps':>8} {'disc (med)':>12} {'disc (std)':>12} "
            f"{'mean W1':>10} {'prec@k':>8} "
            f"{'unpert hits':>12} {'unpert recall':>14}"
        )
        print(header)
        print("-" * len(header))
        for step_idx in checkpoints:
            W_snap, disc_so_far = all_results[step_size][step_idx]
            # Smoothed recent discrepancy: median of last ~100 steps,
            # std over those same steps (across candidates).
            tail = disc_so_far[-100:]  # (T_tail, N)
            tail_flat = tail.flatten()
            disc_med = float(np.median(tail_flat))
            disc_std = float(np.std(tail_flat))

            metrics = evaluate_snapshot(
                W_snap, data, truth, perturbed_set, top_k=top_k
            )
            print(
                f"{step_idx:>8d} {disc_med:>12.3f} {disc_std:>12.3f} "
                f"{metrics['mean_w1']:>10.4f} {metrics['precision']:>8.3f} "
                f"{metrics['hits_unpert']:>12d} "
                f"{metrics['recall_unpert']:>14.3f}"
            )


if __name__ == "__main__":
    main()
