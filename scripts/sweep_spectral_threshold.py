"""Sweep spectral_threshold and test the "we capped too high" theory.

Ground-truth synthetic data is generated at ``target_spectral_radius
= 0.8``. Our fit was capping at ``spectral_threshold = 0.95``, which
gives the simulator ~4x more per-gene variance amplification than the
data (amplification scales like ``1/(1 - rho)``). If that cap is the
reason the discrepancy is stuck at ~70k, lowering it should lower the
floor.

For each threshold, report:
- final discrepancy (moment-matching only),
- post-fit spectral radius per candidate (did the fit actually use the
  headroom, or get stuck at the cap?),
- ||W||_F,
- mean W1, precision@500, true unperturbed-source hits/recall.
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


def spectral_radii(W: torch.Tensor) -> np.ndarray:
    return torch.linalg.eigvals(W).abs().max(dim=-1).values.cpu().numpy()


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

    print(f"{data.summary()}   true_edges={len(true_set)}   "
          f"unperturbed-source true: {len(true_unpert)}")
    print("Ground-truth data was generated at target_spectral_radius = 0.8\n")

    thresholds = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]
    top_k = 500
    l1 = 1e-3  # a little stronger than default, in case the fit escapes the cap

    header = (
        f"{'thresh':>8} {'disc_fit':>10} "
        f"{'rho(min)':>9} {'rho(med)':>9} {'rho(max)':>9} "
        f"{'||W||_F':>9} "
        f"{'mean W1':>9} {'prec@k':>8} {'unpert':>8} {'recall_u':>10}"
    )
    print(header)
    print("-" * len(header))

    for thresh in thresholds:
        fitter = EnsembleSCMFitter(
            top_k=top_k,
            n_candidates=5,
            n_steps=1000,
            step_size=0.01,
            batch_size=200,
            l1_lambda=l1,
            spectral_threshold=thresh,
            seed=0,
            log_every=None,
        )
        t0 = time.time()
        edges = fitter.fit_predict(data)
        fit_time = time.time() - t0

        # Discrepancy (median of last 100 steps), with L1 backed out.
        tail = fitter.last_history.discrepancy[-100:]
        with torch.no_grad():
            l1_term = float((l1 * fitter.last_scm.W.abs().sum(dim=(-2, -1))).mean())
        disc_fit = float(np.median(tail)) - l1_term

        # Post-fit per-candidate spectral radius.
        rhos = spectral_radii(fitter.last_scm.W.detach())
        frob = float(
            fitter.last_scm.W.detach().pow(2).sum(dim=(-2, -1)).sqrt().mean()
        )

        res = evaluate_statistical(
            edges, data, omission_sample_size=500,
            rng=np.random.default_rng(99),
        )
        hits_unpert = sum(1 for e in edges if e in true_unpert)
        recall_unpert = hits_unpert / max(len(true_unpert), 1)
        prec = sum(1 for e in edges if e in true_set) / top_k

        print(
            f"{thresh:>8.2f} {disc_fit:>10.2f} "
            f"{rhos.min():>9.3f} {np.median(rhos):>9.3f} {rhos.max():>9.3f} "
            f"{frob:>9.4f} "
            f"{res.mean_wasserstein:>9.4f} {prec:>8.3f} "
            f"{hits_unpert:>8d} {recall_unpert:>10.3f}"
        )

    print(f"\n(fit time ~{fit_time:.1f}s each; l1={l1}, step_size=0.01, n_steps=1000)")


if __name__ == "__main__":
    main()
