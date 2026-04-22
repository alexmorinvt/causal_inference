"""Report the maximum possible mean W1 at each top_k (the W1 oracle).

The statistical metric is precision-like: for each predicted edge
(A, B) it measures W1( P(B | control), P(B | do(A)) ), averages over
predicted edges, higher is better. Nothing in the metric enforces that
the predicted edge is causal — any (A, B) pair with a strong
intervention effect scores well. So the **maximum possible mean W1 at
top_k** is simply the average of the top_k largest per-pair W1 values
in the dataset.

This is an oracle that knows the W1 of every pair up front and sorts
by it directly; no causal modelling involved. Mean Difference is a
near-oracle on this metric (ranking by |mean shift|, which tightly
tracks W1 for roughly-Gaussian single-gene distributions), so it
should come very close to this ceiling. Gives us a realistic upper
bound to target.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import wasserstein_distance

from grn_inference import (
    EnsembleSCMFitter,
    MeanDifferenceModel,
    RandomBaseline,
    evaluate_statistical,
    make_synthetic_dataset,
)


def oracle_sorted_edges(data) -> list[tuple[tuple[str, str], float]]:
    """Return every evaluable (source -> target) edge paired with its W1.

    Sorted descending by W1. "Evaluable" = source was perturbed and has
    at least 25 intervention cells; target is any other measured gene.
    Matches the gating in ``evaluator.evaluate_statistical``.
    """
    control_mask = data.control_mask()
    control_expr = data.expression[control_mask]
    min_cells = 25

    edges: list[tuple[tuple[str, str], float]] = []
    for src in data.perturbed_genes():
        mask = data.intervention_mask(src)
        if mask.sum() < min_cells:
            continue
        intv_expr = data.expression[mask]
        for i, tgt in enumerate(data.gene_names):
            if tgt == src:
                continue
            w1 = wasserstein_distance(control_expr[:, i], intv_expr[:, i])
            edges.append(((src, tgt), float(w1)))

    edges.sort(key=lambda x: -x[1])
    return edges


def main() -> None:
    data, truth = make_synthetic_dataset(
        n_genes=50,
        edge_density=0.12,
        n_control_cells=2000,
        n_cells_per_perturbation=200,
        seed=7,
    )
    print(f"{data.summary()}   true_edges={len(truth.true_edges)}\n")

    t0 = time.time()
    sorted_edges = oracle_sorted_edges(data)
    print(
        f"Computed W1 for all {len(sorted_edges)} evaluable pairs in "
        f"{time.time() - t0:.2f}s\n"
    )

    top_k_values = [100, 500, 1000]
    true_set = set(truth.true_edges)

    # Pre-compute the model rankings once.
    md_edges = MeanDifferenceModel(top_k=max(top_k_values)).fit_predict(data)
    rb_edges = RandomBaseline(top_k=max(top_k_values), seed=0).fit_predict(data)

    fitter = EnsembleSCMFitter(
        top_k=max(top_k_values),
        n_candidates=5,
        n_steps=1000,
        step_size=0.01,
        batch_size=200,
        l1_lambda=1e-4,
        seed=0,
        log_every=None,
    )
    fitter_edges = fitter.fit_predict(data)

    header = (
        f"{'top_k':>6} {'oracle W1':>12} "
        f"{'MeanDiff W1':>12} {'MD gap':>8} "
        f"{'Fitter W1':>12} {'Fitter gap':>12} "
        f"{'Random W1':>12}"
    )
    print(header)
    print("-" * len(header))

    for k in top_k_values:
        oracle_top = [w for _, w in sorted_edges[:k]]
        oracle_w1 = float(np.mean(oracle_top))

        md_top = md_edges[:k]
        rb_top = rb_edges[:k]
        fit_top = fitter_edges[:k]

        md_res = evaluate_statistical(
            md_top, data, omission_sample_size=500,
            rng=np.random.default_rng(99),
        )
        rb_res = evaluate_statistical(
            rb_top, data, omission_sample_size=500,
            rng=np.random.default_rng(99),
        )
        fit_res = evaluate_statistical(
            fit_top, data, omission_sample_size=500,
            rng=np.random.default_rng(99),
        )

        md_gap = oracle_w1 - md_res.mean_wasserstein
        fit_gap = oracle_w1 - fit_res.mean_wasserstein

        print(
            f"{k:>6d} {oracle_w1:>12.4f} "
            f"{md_res.mean_wasserstein:>12.4f} {md_gap:>+8.4f} "
            f"{fit_res.mean_wasserstein:>12.4f} {fit_gap:>+12.4f} "
            f"{rb_res.mean_wasserstein:>12.4f}"
        )

    # How much of the oracle top-k is actually causal?
    print("\nOracle top-k composition (what fraction are true edges?):")
    for k in top_k_values:
        oracle_edges = [e for e, _ in sorted_edges[:k]]
        hits = sum(1 for e in oracle_edges if e in true_set)
        print(
            f"  top_{k:<4d}: {hits}/{k} true edges "
            f"({100 * hits / k:.1f}%)   "
            f"lowest-W1-in-top-{k} = {sorted_edges[k - 1][1]:.4f}"
        )


if __name__ == "__main__":
    main()
