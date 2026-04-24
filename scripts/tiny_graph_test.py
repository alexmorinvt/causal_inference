"""Fit the ensemble SCM on a tiny 5-gene graph we can inspect by hand.

With n_genes=5, there are 20 off-diagonal pairs. edge_density=0.25 gives
~3-6 true edges on average. Small enough to print the full score matrix
and rank every pair — no top-k cutoff hides bad calls.

Compares EnsembleSCMFitter, MeanDifferenceModel, and RandomBaseline at
the oracle cutoff (k = n_true).

Usage:
    python scripts/tiny_graph_test.py
"""

from __future__ import annotations

import numpy as np

from grn_inference import (
    EnsembleSCMFitter,
    MeanDifferenceModel,
    RandomBaseline,
    make_synthetic_dataset,
)
from grn_inference.ensemble_scm import aggregate_scores


def run_one(seed: int) -> dict[str, float]:
    print("=" * 72)
    print(f"Tiny graph — seed={seed}, n_genes=5, edge_density=0.25")
    print("=" * 72)

    data, truth = make_synthetic_dataset(
        n_genes=5,
        edge_density=0.25,
        n_control_cells=1000,
        n_cells_per_perturbation=300,
        seed=seed,
    )
    n_true = len(truth.true_edges)
    true_set = set(truth.true_edges)
    print(f"True edges ({n_true}): {truth.true_edges}")

    true_abs = np.abs(truth.W.T)  # transpose so [j, i] is source j -> target i
    np.fill_diagonal(true_abs, 0.0)

    # ---- Fitter ----------------------------------------------------------
    fitter = EnsembleSCMFitter(
        top_k=n_true,
        n_candidates=5,
        n_steps=1500,
        step_size=0.01,
        batch_size=200,
        l1_lambda=1e-4,
        seed=0,
        log_every=None,
    )
    fitter_edges = fitter.fit_predict(data)
    fitted_score = aggregate_scores(
        fitter.last_scm.W.detach(), power=fitter.aggregation_power
    )

    # ---- Mean Difference ------------------------------------------------
    md_edges = MeanDifferenceModel(top_k=n_true).fit_predict(data)

    # ---- Random ---------------------------------------------------------
    rb_edges = RandomBaseline(top_k=n_true, seed=seed).fit_predict(data)

    with np.printoptions(precision=3, suppress=True, linewidth=120):
        print("\nGround-truth |W| (row = source, col = target):")
        print(true_abs)
        print("\nFitter power-mean score (row = source, col = target):")
        print(fitted_score)

    # ---- Comparison table ----------------------------------------------
    def score(edges: list[tuple[str, str]]) -> tuple[int, float]:
        tp = sum(1 for e in edges if e in true_set)
        return tp, tp / max(n_true, 1)

    f_tp, f_p = score(fitter_edges)
    m_tp, m_p = score(md_edges)
    r_tp, r_p = score(rb_edges)

    print(f"\nAt top-{n_true} (oracle cutoff), (precision == recall):")
    print(f"  EnsembleSCMFitter  TP={f_tp}/{n_true}  precision={f_p:.2f}")
    print(f"  Mean Difference    TP={m_tp}/{n_true}  precision={m_p:.2f}")
    print(f"  Random             TP={r_tp}/{n_true}  precision={r_p:.2f}")

    # Show Mean Difference's picks (so we can see where it goes wrong too).
    print(f"\n  Fitter picks:         {fitter_edges}")
    print(f"  Mean Difference picks: {md_edges}")

    return {"fitter": f_p, "md": m_p, "random": r_p}


if __name__ == "__main__":
    results = []
    for seed in (0, 1, 2, 3):
        results.append(run_one(seed))
        print()

    print("=" * 72)
    print("Summary across seeds")
    print("=" * 72)
    print(f"{'seed':>6}  {'Fitter':>8}  {'MeanDiff':>10}  {'Random':>8}")
    for seed, r in enumerate(results):
        print(
            f"{seed:>6d}  {r['fitter']:>8.2f}  {r['md']:>10.2f}  {r['random']:>8.2f}"
        )
    mean = {k: np.mean([r[k] for r in results]) for k in results[0]}
    print(
        f"{'mean':>6}  {mean['fitter']:>8.2f}  {mean['md']:>10.2f}  {mean['random']:>8.2f}"
    )
