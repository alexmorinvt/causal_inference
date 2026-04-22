"""Stage 0 entrypoint: run baselines on synthetic data and print results.

Usage:
    python scripts/run_baseline.py

What it does:
    1. Generate a synthetic dataset from a known linear cyclic SCM.
    2. Run Mean Difference and Random baseline on it.
    3. Evaluate both via the statistical metric (Wasserstein + FOR).
    4. Print a comparison table.

If Mean Difference does not clearly beat Random, STOP and debug
before adding any new code. Stage 0 is the foundation; if the
foundation is wrong, everything built on it is wrong.
"""

from __future__ import annotations

import time

import numpy as np

from grn_inference import (
    MeanDifferenceModel,
    RandomBaseline,
    evaluate_statistical,
    make_synthetic_dataset,
)


def main() -> None:
    print("=" * 70)
    print("Stage 0 — baseline comparison on synthetic data")
    print("=" * 70)

    t0 = time.time()
    data, truth = make_synthetic_dataset(
        n_genes=50,
        edge_density=0.12,
        n_control_cells=2000,
        n_cells_per_perturbation=200,
        seed=7,
    )
    print(f"\n{data.summary()}")
    print(f"True edges in ground truth: {len(truth.true_edges)}")
    print(f"Data generation: {time.time() - t0:.2f}s")

    for top_k in (100, 500, 1000):
        print(f"\n--- top_k = {top_k} ---")

        t0 = time.time()
        md = MeanDifferenceModel(top_k=top_k)
        md_edges = md.fit_predict(data)
        md_time = time.time() - t0

        t0 = time.time()
        rb = RandomBaseline(top_k=top_k, seed=0)
        rb_edges = rb.fit_predict(data)
        rb_time = time.time() - t0

        md_result = evaluate_statistical(
            md_edges, data,
            omission_sample_size=500,
            rng=np.random.default_rng(99),
        )
        rb_result = evaluate_statistical(
            rb_edges, data,
            omission_sample_size=500,
            rng=np.random.default_rng(99),
        )

        print(
            f"{'method':<20} {'mean W1':>10} {'FOR':>10} "
            f"{'n_eval':>8} {'time':>8}"
        )
        print(
            f"{'Mean Difference':<20} {md_result.mean_wasserstein:>10.4f} "
            f"{md_result.false_omission_rate:>10.4f} "
            f"{md_result.n_evaluable_predicted:>8d} {md_time:>7.2f}s"
        )
        print(
            f"{'Random':<20} {rb_result.mean_wasserstein:>10.4f} "
            f"{rb_result.false_omission_rate:>10.4f} "
            f"{rb_result.n_evaluable_predicted:>8d} {rb_time:>7.2f}s"
        )

        # Also: what fraction of predicted edges are in the true edge set?
        true_set = set(truth.true_edges)
        md_recovery = sum(1 for e in md_edges if e in true_set)
        rb_recovery = sum(1 for e in rb_edges if e in true_set)
        print(
            f"True-edge recovery: Mean Diff {md_recovery}/{len(md_edges)} "
            f"({100*md_recovery/max(len(md_edges),1):.1f}%), "
            f"Random {rb_recovery}/{len(rb_edges)} "
            f"({100*rb_recovery/max(len(rb_edges),1):.1f}%)"
        )

    print("\n" + "=" * 70)
    print("Done. If Mean Difference is NOT dominantly better, investigate.")
    print("=" * 70)


if __name__ == "__main__":
    main()
