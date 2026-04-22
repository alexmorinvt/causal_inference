"""Evaluation on partial-perturbation synthetic data.

Generates a dataset where only half the genes receive an intervention
arm. True edges whose *source* is unperturbed are invisible to the
W1 metric (Mean Difference literally cannot score them). Simulation-
based methods that use observational structure could, in principle,
recover them — that's the contrast this script is set up to probe.

Reports, per top_k:
- mean W1 (CausalBench's statistical metric),
- **precision@k** — what fraction of the method's top_k is a real edge,
  broken down by whether the source is perturbed,
- FOR (false-omission rate).

Also reports a **matched-W1** pivot: for a target mean-W1 threshold,
find the largest prefix of each method's ranking whose cumulative mean
W1 stays above that target, and compare precisions at that point.
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
    """Rank every evaluable (source -> target) pair by its real-data W1."""
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


def cumulative_mean_w1(
    ordered_edges,
    per_edge_w1: dict,
) -> np.ndarray:
    """Cumulative mean W1 over a method's ranked list, averaging over
    **evaluable** edges only (matches evaluate_statistical's semantics).

    Returns an array the same length as ``ordered_edges``.
    """
    cum = np.empty(len(ordered_edges), dtype=np.float64)
    w_sum = 0.0
    eval_count = 0
    for k, edge in enumerate(ordered_edges):
        if edge in per_edge_w1:
            w_sum += per_edge_w1[edge]
            eval_count += 1
        cum[k] = w_sum / max(eval_count, 1)
    return cum


def main() -> None:
    n_genes = 50
    n_perturbed_genes = n_genes // 2  # only half the genes get intervention arms
    data, truth = make_synthetic_dataset(
        n_genes=n_genes,
        edge_density=0.12,
        n_control_cells=2000,
        n_cells_per_perturbation=200,
        n_perturbed_genes=n_perturbed_genes,
        seed=7,
    )
    perturbed_set = set(data.perturbed_genes())
    true_set = set(truth.true_edges)
    true_edges_with_perturbed_src = {e for e in true_set if e[0] in perturbed_set}
    true_edges_with_unperturbed_src = true_set - true_edges_with_perturbed_src

    print(f"{data.summary()}")
    print(f"n_genes={n_genes}, n_perturbed={n_perturbed_genes}")
    print(f"True edges total:                   {len(true_set)}")
    print(f"  with perturbed source  (visible): {len(true_edges_with_perturbed_src)}")
    print(f"  with unperturbed source (hidden): {len(true_edges_with_unperturbed_src)}")

    # --------------------------------------------------------------------
    # Oracle ceiling
    # --------------------------------------------------------------------
    t0 = time.time()
    oracle = oracle_sorted_edges(data)
    print(
        f"\nComputed W1 for all {len(oracle)} evaluable pairs "
        f"in {time.time() - t0:.2f}s"
    )

    # --------------------------------------------------------------------
    # Run each method at a large top_k; slice down for finer comparisons.
    # --------------------------------------------------------------------
    max_k = 1000
    print(f"\nFitting models (top_k={max_k})...")
    t0 = time.time()
    fitter = EnsembleSCMFitter(
        top_k=max_k, n_candidates=5, n_steps=1000,
        step_size=0.01, batch_size=200, l1_lambda=1e-4,
        seed=0, log_every=None,
    )
    fit_edges = fitter.fit_predict(data)
    print(f"  EnsembleSCMFitter: {time.time() - t0:.2f}s")

    md_edges = MeanDifferenceModel(top_k=max_k).fit_predict(data)
    rb_edges = RandomBaseline(top_k=max_k, seed=0).fit_predict(data)
    oracle_edges = [e for e, _ in oracle[:max_k]]

    methods = {
        "Oracle (W1 sorted)": oracle_edges,
        "EnsembleSCMFitter": fit_edges,
        "Mean Difference":   md_edges,
        "Random":            rb_edges,
    }

    # Evaluate at large top_k once per method to get per_edge_w1 maps;
    # reuse for cumulative analysis.
    per_edge_maps: dict[str, dict] = {}
    for name, edges in methods.items():
        res = evaluate_statistical(
            edges, data, omission_sample_size=500,
            rng=np.random.default_rng(99),
        )
        per_edge_maps[name] = res.per_edge_wasserstein

    # --------------------------------------------------------------------
    # Per-top_k table: mean W1, precision@k, FOR, source breakdown.
    # --------------------------------------------------------------------
    for top_k in (50, 100, 500, 1000):
        print(f"\n--- top_k = {top_k} ---")
        header = (
            f"{'method':<22} {'mean W1':>10} "
            f"{'prec@k':>8} {'prec (pert)':>12} {'prec (unpert)':>14} "
            f"{'FOR':>8} {'# pert':>8} {'# unpert':>10}"
        )
        print(header)
        print("-" * len(header))
        for name, edges in methods.items():
            edges_k = edges[:top_k]
            res = evaluate_statistical(
                edges_k, data, omission_sample_size=500,
                rng=np.random.default_rng(99),
            )
            n_pert = sum(1 for s, _ in edges_k if s in perturbed_set)
            n_unpert = top_k - n_pert
            hits = sum(1 for e in edges_k if e in true_set)
            hits_pert = sum(
                1 for e in edges_k
                if e in true_set and e[0] in perturbed_set
            )
            hits_unpert = hits - hits_pert
            prec = hits / top_k
            prec_pert = hits_pert / max(n_pert, 1)
            prec_unpert = hits_unpert / max(n_unpert, 1)
            print(
                f"{name:<22} {res.mean_wasserstein:>10.4f} "
                f"{prec:>8.3f} {prec_pert:>12.3f} {prec_unpert:>14.3f} "
                f"{res.false_omission_rate:>8.3f} "
                f"{n_pert:>8d} {n_unpert:>10d}"
            )

    # --------------------------------------------------------------------
    # Matched-W1 pivot: longest prefix with cumulative mean W1 >= target.
    # --------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Matched mean-W1 comparison (largest prefix k where cum. mean W1 >= target)")
    print("=" * 80)
    targets = [0.7, 0.5, 0.3]
    header = (
        f"{'method':<22} {'target':>8} {'k @ target':>12} "
        f"{'mean W1':>10} {'precision':>10} "
        f"{'true hits':>12} {'true @ unpert':>14}"
    )
    print(header)
    print("-" * len(header))
    for name, edges in methods.items():
        cum = cumulative_mean_w1(edges, per_edge_maps[name])
        for target in targets:
            valid = cum >= target
            if not valid.any():
                print(f"{name:<22} {target:>8.2f}   never reaches target")
                continue
            k_star = int(np.max(np.where(valid)[0])) + 1
            prefix = edges[:k_star]
            hits = sum(1 for e in prefix if e in true_set)
            hits_unpert = sum(
                1 for e in prefix
                if e in true_set and e[0] not in perturbed_set
            )
            prec = hits / k_star
            print(
                f"{name:<22} {target:>8.2f} {k_star:>12d} "
                f"{cum[k_star - 1]:>10.4f} {prec:>10.3f} "
                f"{hits:>12d} {hits_unpert:>14d}"
            )


if __name__ == "__main__":
    main()
