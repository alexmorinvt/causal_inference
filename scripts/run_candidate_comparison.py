"""Diagnostic: compare individual candidate GRNs to the fused ensemble.

After fitting the ensemble, extract each candidate's W in isolation,
rank edges from it alone, and evaluate with the statistical metric.
Also report the fused (power-mean aggregated) result and the single
"best-loss" candidate (the one with the lowest final discrepancy).

The question this answers: is the ensemble aggregation helping, or is
one candidate carrying the signal while the others add noise?

Usage:
    python scripts/run_candidate_comparison.py
"""

from __future__ import annotations

import time

import numpy as np

from grn_inference import (
    EnsembleSCMFitter,
    evaluate_statistical,
    make_synthetic_dataset,
)
from grn_inference.ensemble_scm import aggregate_scores, rank_edges


def evaluate_edges(edges, data, seed=99):
    return evaluate_statistical(
        edges, data, omission_sample_size=500,
        rng=np.random.default_rng(seed),
    )


def single_candidate_edges(W_k: np.ndarray, gene_names: list[str], top_k: int):
    """Rank edges using one candidate's |W| (no aggregation)."""
    score = np.abs(W_k).T.copy()  # (G, G); transpose so [j, i] = source j -> target i
    np.fill_diagonal(score, 0.0)
    return rank_edges(score, gene_names, top_k)


def main() -> None:
    data, truth = make_synthetic_dataset(
        n_genes=50,
        edge_density=0.12,
        n_control_cells=2000,
        n_cells_per_perturbation=200,
        seed=7,
    )
    print(f"{data.summary()}   true_edges={len(truth.true_edges)}\n")

    n_candidates = 5
    top_k = 100
    fitter = EnsembleSCMFitter(
        top_k=top_k,
        n_candidates=n_candidates,
        n_steps=1000,
        step_size=0.01,
        batch_size=200,
        l1_lambda=1e-4,
        seed=0,
        log_every=None,
    )

    t0 = time.time()
    fused_edges = fitter.fit_predict(data)
    fit_time = time.time() - t0
    print(f"Fit {n_candidates} candidates x 1000 steps in {fit_time:.2f}s\n")

    W_ens = fitter.last_scm.W.detach().cpu().numpy()  # (N, G, G)

    # Final discrepancy = average of the last 100 steps per candidate.
    tail = fitter.last_history.discrepancy[-100:]
    final_disc = tail.mean(axis=0)  # (N,)
    best_k = int(np.argmin(final_disc))

    true_set = set(truth.true_edges)

    rows = []
    per_candidate_edges: list[set] = []
    per_candidate_true_hits: list[set] = []

    # Per-candidate rows
    for k in range(n_candidates):
        edges_k = single_candidate_edges(W_ens[k], data.gene_names, top_k)
        edge_set = set(edges_k)
        true_hits_k = edge_set & true_set
        per_candidate_edges.append(edge_set)
        per_candidate_true_hits.append(true_hits_k)
        res = evaluate_edges(edges_k, data)
        rows.append((
            f"candidate[{k}]",
            final_disc[k],
            res.mean_wasserstein,
            res.false_omission_rate,
            len(true_hits_k),
            len(edges_k),
        ))

    # Single best-loss candidate (picked by final discrepancy — not cheating
    # against ground truth, just against the fit's own objective)
    edges_best = single_candidate_edges(W_ens[best_k], data.gene_names, top_k)
    res_best = evaluate_edges(edges_best, data)
    hits_best = sum(1 for e in edges_best if e in true_set)
    rows.append((
        f"best-loss (k={best_k})",
        final_disc[best_k],
        res_best.mean_wasserstein,
        res_best.false_omission_rate,
        hits_best,
        len(edges_best),
    ))

    # Fused (power-mean) aggregate
    fused_score = aggregate_scores(
        fitter.last_scm.W.detach(), power=fitter.aggregation_power
    )
    res_fused = evaluate_edges(fused_edges, data)
    hits_fused = sum(1 for e in fused_edges if e in true_set)
    rows.append((
        f"fused (p={fitter.aggregation_power:g})",
        float("nan"),
        res_fused.mean_wasserstein,
        res_fused.false_omission_rate,
        hits_fused,
        len(fused_edges),
    ))

    # Print table
    print(f"top_k = {top_k}")
    header = (
        f"{'source':<22} {'final disc':>12} {'mean W1':>10} "
        f"{'FOR':>8} {'true hits':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, fd, mw, fo, hits, n_edges in rows:
        fd_str = "   —" if np.isnan(fd) else f"{fd:>12.2e}"
        print(
            f"{name:<22} {fd_str:>12} {mw:>10.4f} "
            f"{fo:>8.4f} {hits:>6}/{n_edges:<5}"
        )

    # Interpretation helpers
    best_candidate_w1 = max(r[2] for r in rows[:n_candidates])
    fused_w1 = rows[-1][2]
    print()
    print(
        f"Best single candidate W1: {best_candidate_w1:.4f}   "
        f"Fused W1: {fused_w1:.4f}   "
        f"Delta: {fused_w1 - best_candidate_w1:+.4f}"
    )

    # ----------------------------------------------------------------
    # Candidate overlap: are they finding the same edges, or different?
    # ----------------------------------------------------------------
    print()
    print("-" * 60)
    print(f"Candidate overlap analysis (top_k={top_k}, true edges={len(true_set)})")
    print("-" * 60)

    # Pairwise true-hit overlap matrix.
    print("\nPairwise true-hit intersection (diagonal = own true-hit count):")
    header = "          " + " ".join(f"{'c' + str(k):>5}" for k in range(n_candidates))
    print(header)
    for k in range(n_candidates):
        row = f"    c{k:<2d}  "
        for j in range(n_candidates):
            inter = len(per_candidate_true_hits[k] & per_candidate_true_hits[j])
            row += f"{inter:>5d} "
        print(row)

    # Aggregate true-hit stats across the ensemble.
    union_true = set().union(*per_candidate_true_hits)
    intersection_true = per_candidate_true_hits[0].copy()
    for s in per_candidate_true_hits[1:]:
        intersection_true &= s
    print(
        f"\nTrue hits anywhere in the ensemble (union):        {len(union_true):>3d}"
    )
    print(
        f"True hits found by every candidate (intersection): {len(intersection_true):>3d}"
    )
    print(
        f"Mean hits per candidate:                           "
        f"{np.mean([len(s) for s in per_candidate_true_hits]):>5.1f}"
    )
    print(
        f"Max hits by a single candidate:                    "
        f"{max(len(s) for s in per_candidate_true_hits):>3d}"
    )

    # Pairwise Jaccard similarity on predicted edges (not just true hits) —
    # tells us whether the candidates' rankings themselves agree.
    print("\nPairwise Jaccard similarity on predicted edge sets:")
    print(header)
    for k in range(n_candidates):
        row = f"    c{k:<2d}  "
        for j in range(n_candidates):
            a, b = per_candidate_edges[k], per_candidate_edges[j]
            jacc = len(a & b) / max(len(a | b), 1)
            row += f"{jacc:>5.2f} "
        print(row)


if __name__ == "__main__":
    main()
