"""Sweep L1 sparsity strength and report what breaks the plateau.

Both earlier diagnostics — the divergence trajectory and the
step-size/iteration sweep — pointed at L1 strength as the remaining
lever. Our default ``l1_lambda = 1e-4`` is effectively zero (it
contributes < 1 to a discrepancy of 70,000). This script sweeps L1
across four orders of magnitude on the partial-perturbation setup
and reports both the statistical metrics and the structural sparsity
of the fitted W.

Reads for each L1 value:
- final discrepancy (moment-matching term + L1 term, and the L1 term
  broken out so we can see when L1 dominates the objective),
- mean W1 at top_k = 500,
- precision @ 500,
- true hits from unperturbed-source edges (the interesting signal),
- ||W||_F and fraction of |W| entries above 0.01 — tells us whether
  L1 is actually inducing the sparse structure of the ground truth
  (which has 279 edges out of 2450 possible = 11.4% density).
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
    true_set = set(truth.true_edges)
    true_unpert = {e for e in true_set if e[0] not in perturbed_set}
    true_density = len(true_set) / (50 * 49)

    print(f"{data.summary()}   true_edges={len(true_set)}   "
          f"(ground-truth density = {true_density:.3%})")
    print(f"perturbed genes: {len(perturbed_set)}   "
          f"unperturbed-source true edges: {len(true_unpert)}\n")

    l1_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    top_k = 500
    sparsity_threshold = 0.01  # |W_ij| > this counts as "present"

    header = (
        f"{'l1':>8} {'disc_fit':>10} {'L1_term':>10} "
        f"{'mean W1':>10} {'prec@k':>8} "
        f"{'hits':>6} {'unpert':>8} {'recall_u':>10} "
        f"{'||W||_F':>10} {'density>0.01':>14}"
    )
    print(header)
    print("-" * len(header))

    for l1 in l1_values:
        fitter = EnsembleSCMFitter(
            top_k=top_k,
            n_candidates=5,
            n_steps=1000,
            step_size=0.01,
            batch_size=200,
            l1_lambda=l1,
            spectral_threshold=0.80,
            seed=0,
            log_every=None,
        )
        t0 = time.time()
        edges = fitter.fit_predict(data)
        fit_time = time.time() - t0

        # Final discrepancy (moment-matching part only) vs L1 penalty contribution.
        tail = fitter.last_history.discrepancy[-100:]  # already includes L1
        with torch.no_grad():
            l1_term = float((l1 * fitter.last_scm.W.abs().sum(dim=(-2, -1))).mean())
        disc_total = float(np.median(tail))
        disc_fit = disc_total - l1_term  # back out the L1 contribution

        res = evaluate_statistical(
            edges, data, omission_sample_size=500,
            rng=np.random.default_rng(99),
        )
        hits = sum(1 for e in edges if e in true_set)
        hits_unpert = sum(1 for e in edges if e in true_unpert)
        recall_unpert = hits_unpert / max(len(true_unpert), 1)

        W_abs = fitter.last_scm.W.detach().abs()
        frob = float(W_abs.pow(2).sum(dim=(-2, -1)).sqrt().mean())
        above = (W_abs > sparsity_threshold).float()
        density = float(above.mean())

        print(
            f"{l1:>8.1e} {disc_fit:>10.2f} {l1_term:>10.2f} "
            f"{res.mean_wasserstein:>10.4f} {hits / top_k:>8.3f} "
            f"{hits:>6d} {hits_unpert:>8d} {recall_unpert:>10.3f} "
            f"{frob:>10.4f} {density:>13.2%}"
        )

    print(f"\n(fit times ~{fit_time:.1f}s each; top_k={top_k}, "
          f"sparsity threshold = |W_ij| > {sparsity_threshold})")
    print(f"Ground-truth density = {true_density:.2%}")


if __name__ == "__main__":
    main()
