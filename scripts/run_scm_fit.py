"""Demo: fit the ensemble SCM on synthetic data and compare to baselines.

Usage:
    python scripts/run_scm_fit.py

Procedure:
    1. Generate a synthetic dataset from a known linear cyclic SCM.
    2. Fit EnsembleSCMFitter; also run MeanDifferenceModel and
       RandomBaseline for comparison.
    3. Evaluate all three via the statistical metric (Wasserstein + FOR).
    4. Print a comparison table at several top_k's.
    5. On a small n_genes=10 run, print the candidate-averaged |W|
       next to the ground-truth |W| as a structural sanity check.
"""

from __future__ import annotations

import time

import numpy as np

from grn_inference import (
    EnsembleSCMFitter,
    MeanDifferenceModel,
    RandomBaseline,
    evaluate_statistical,
    make_synthetic_dataset,
)
from grn_inference.ensemble_scm import aggregate_scores


def comparison_table() -> None:
    print("=" * 78)
    print("Ensemble-SCM fitter vs baselines on synthetic data (n_genes=50)")
    print("=" * 78)

    t0 = time.time()
    data, truth = make_synthetic_dataset(
        n_genes=50,
        edge_density=0.12,
        n_control_cells=2000,
        n_cells_per_perturbation=200,
        seed=7,
    )
    print(f"\n{data.summary()}")
    print(f"True edges: {len(truth.true_edges)}")
    print(f"Data generation: {time.time() - t0:.2f}s")

    for top_k in (100, 500, 1000):
        print(f"\n--- top_k = {top_k} ---")

        t0 = time.time()
        fitter = EnsembleSCMFitter(
            top_k=top_k,
            n_candidates=5,
            n_steps=1000,
            step_size=0.01,
            batch_size=200,
            l1_lambda=1e-4,
            seed=0,
            log_every=None,
        )
        fitter_edges = fitter.fit_predict(data)
        fitter_time = time.time() - t0

        t0 = time.time()
        md_edges = MeanDifferenceModel(top_k=top_k).fit_predict(data)
        md_time = time.time() - t0

        t0 = time.time()
        rb_edges = RandomBaseline(top_k=top_k, seed=0).fit_predict(data)
        rb_time = time.time() - t0

        fitter_res = evaluate_statistical(
            fitter_edges, data, omission_sample_size=500,
            rng=np.random.default_rng(99),
        )
        md_res = evaluate_statistical(
            md_edges, data, omission_sample_size=500,
            rng=np.random.default_rng(99),
        )
        rb_res = evaluate_statistical(
            rb_edges, data, omission_sample_size=500,
            rng=np.random.default_rng(99),
        )

        header = (
            f"{'method':<22} {'mean W1':>10} {'FOR':>10} "
            f"{'n_eval':>8} {'time':>10}"
        )
        print(header)
        print("-" * len(header))
        for name, res, t in [
            ("EnsembleSCMFitter", fitter_res, fitter_time),
            ("Mean Difference", md_res, md_time),
            ("Random", rb_res, rb_time),
        ]:
            print(
                f"{name:<22} {res.mean_wasserstein:>10.4f} "
                f"{res.false_omission_rate:>10.4f} "
                f"{res.n_evaluable_predicted:>8d} {t:>9.2f}s"
            )

        true_set = set(truth.true_edges)
        print("True-edge recovery:")
        for name, edges in [
            ("  EnsembleSCMFitter", fitter_edges),
            ("  Mean Difference  ", md_edges),
            ("  Random           ", rb_edges),
        ]:
            hit = sum(1 for e in edges if e in true_set)
            print(
                f"{name} {hit}/{len(edges)} "
                f"({100 * hit / max(len(edges), 1):.1f}%)"
            )


def structural_sanity_check() -> None:
    """On a small graph, print fitted |W| next to ground-truth |W|."""
    print()
    print("=" * 78)
    print("Structural sanity (n_genes=10): fitted |W| vs ground-truth |W|")
    print("=" * 78)

    data, truth = make_synthetic_dataset(
        n_genes=10,
        edge_density=0.25,
        n_control_cells=500,
        n_cells_per_perturbation=150,
        seed=1,
    )

    fitter = EnsembleSCMFitter(
        top_k=30,
        n_candidates=5,
        n_steps=1500,
        step_size=0.01,
        batch_size=150,
        l1_lambda=1e-4,
        seed=0,
        log_every=None,
    )
    fitter.fit_predict(data)
    fitted_score = aggregate_scores(
        fitter.last_scm.W.detach(), power=fitter.aggregation_power
    )

    true_abs = np.abs(truth.W.T)  # transpose so [j, i] indexes source j -> target i
    np.fill_diagonal(true_abs, 0.0)

    with np.printoptions(precision=2, suppress=True, linewidth=140):
        print("\nGround-truth |W| (source=row, target=col):")
        print(true_abs)
        print("\nFitted (power-mean aggregated) score (source=row, target=col):")
        print(fitted_score)


if __name__ == "__main__":
    comparison_table()
    structural_sanity_check()
