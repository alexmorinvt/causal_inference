"""7-gene, 6-edge benchmark for iterating on EnsembleSCMFitter.

The target is 100% recovery: all 6 true edges in the top-6 ranked output.

Fixed dataset (n_genes=7, edge_density=0.2, seed=0) with uniform ±1
edge weights, so the graph is:

    G2 -> G0  (+1)    G4 -> G1  (-1)    G1 -> G2  (+1)
    G3 -> G0  (-1)    G6 -> G1  (+1)    G6 -> G2  (+1)

    G5 is isolated (no in- or out-edges).

The graph has chains (G4 -> G1 -> G2 -> G0), convergent paths (G6 hits
G2 both directly and via G1), and a collider (G0 with parents G2, G3).
Small enough to inspect every off-diagonal cell but structurally rich
enough to expose direct-vs-indirect confusion.

Writes its full report to ``seven_gene_benchmark_output.txt`` at the
repo root (same directory tree as ``run_scm_fit_output.txt`` etc.).

Usage:
    python scripts/seven_gene_benchmark.py
"""

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from grn_inference import (
    EnsembleSCMFitter,
    IndirectPruningModel,
    MeanDifferenceModel,
    RandomBaseline,
    ShiftCorrModel,
    ShiftPathsModel,
    make_synthetic_dataset,
)
from grn_inference.ensemble_scm import aggregate_scores


OUTPUT_FILE = Path(__file__).resolve().parent.parent / "seven_gene_benchmark_output.txt"


def _run() -> None:
    data, truth = make_synthetic_dataset(
        n_genes=7, edge_density=0.2,
        n_control_cells=3000, n_cells_per_perturbation=3000, seed=0,
    )
    G = data.n_genes
    gene_names = data.gene_names
    n_true = len(truth.true_edges)
    true_set = set(truth.true_edges)

    assert n_true == 6, (
        f"Benchmark expects 6 true edges at this (n_genes, density, seed); "
        f"got {n_true}. Did the generator change?"
    )

    print("=" * 78)
    print(f"7-gene benchmark — n_genes={G}, n_true_edges={n_true}")
    print("=" * 78)

    print(f"\nTrue edges ({n_true}): {truth.true_edges}")
    print("\nTrue W (row i, col j = strength of edge j -> i):")
    with np.printoptions(precision=2, suppress=True, linewidth=120):
        print(truth.W)

    # Shift matrix (mean under do(A) minus mean under control) — the
    # signal Mean Difference ranks on.
    ctrl_mask = data.control_mask()
    ctrl_means = data.expression[ctrl_mask].mean(axis=0)
    print("\nShift matrix (mean_do(A) - mean_control):")
    header = "  {:<14s}".format("do(A) \\ B") + "".join(
        f"{g:>10s}" for g in gene_names
    )
    print(header)
    for arm in data.perturbed_genes():
        mask = data.intervention_mask(arm)
        shifts = data.expression[mask].mean(axis=0) - ctrl_means
        row_s = "".join(f"{s:>+10.3f}" for s in shifts)
        print(f"  {f'do({arm})':<14s}{row_s}")

    # ---- Fit the three methods -------------------------------------------
    fitter = EnsembleSCMFitter(
        top_k=n_true,
        n_candidates=5,
        n_steps=1500,
        step_size=0.01,
        batch_size=300,
        l1_lambda=1e-4,
        seed=0,
        log_every=None,
    )
    fitter_edges = fitter.fit_predict(data)
    md_edges = MeanDifferenceModel(top_k=n_true).fit_predict(data)
    rb_edges = RandomBaseline(top_k=n_true, seed=0).fit_predict(data)
    sp_edges = ShiftPathsModel(top_k=n_true).fit_predict(data)
    ip_edges = IndirectPruningModel(top_k=n_true).fit_predict(data)
    sc_edges = ShiftCorrModel(top_k=n_true, corr_weight=1.0).fit_predict(data)

    def score(edges: list[tuple[str, str]]) -> int:
        return sum(1 for e in edges if e in true_set)

    # ---- Results at the oracle cutoff ------------------------------------
    print()
    print("=" * 78)
    print(f"Results at top-{n_true} (oracle cutoff; precision == recall)")
    print("=" * 78)
    for name, edges in [
        ("IndirectPruning  ", ip_edges),
        ("ShiftCorrModel   ", sc_edges),
        ("ShiftPathsModel  ", sp_edges),
        ("EnsembleSCMFitter", fitter_edges),
        ("Mean Difference  ", md_edges),
        ("Random           ", rb_edges),
    ]:
        tp = score(edges)
        print(f"\n  {name}: TP={tp}/{n_true}  precision={tp/max(n_true,1):.0%}")
        for e in edges:
            mark = "*" if e in true_set else " "
            print(f"    {mark} {e[0]} -> {e[1]}")

    # ---- Fitter's full score matrix + extended ranking -------------------
    print()
    print("=" * 78)
    print("Fitter's aggregated |W| score (row = source, col = target)")
    print("=" * 78)
    score_mat = aggregate_scores(
        fitter.last_scm.W.detach(), power=fitter.aggregation_power,
    )
    with np.printoptions(precision=3, suppress=True, linewidth=120):
        print(score_mat)

    rows: list[tuple[str, str, float]] = []
    for j in range(G):
        for i in range(G):
            if i == j:
                continue
            rows.append((gene_names[j], gene_names[i], float(score_mat[j, i])))
    rows.sort(key=lambda r: -r[2])

    n_show = min(15, len(rows))
    print(f"\nFitter's top-{n_show} ranked pairs (* = true edge):")
    for k, (src, tgt, s) in enumerate(rows[:n_show], 1):
        mark = "*" if (src, tgt) in true_set else " "
        print(f"  {k:>3d} {mark} {src} -> {tgt}: {s:.3f}")


def main() -> None:
    with OUTPUT_FILE.open("w") as f, redirect_stdout(f):
        _run()
    print(f"Wrote benchmark report to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
