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
- **hidden-source recall** — fraction of true edges whose source is
  unperturbed that the method recovered in its top_k,
- FOR (false-omission rate).

Also reports a **matched-W1** pivot: for a target mean-W1 threshold,
find the largest prefix of each method's ranking whose cumulative mean
W1 stays above that target, and compare precisions at that point.

At the end, emits a single-line ``JSON_SUMMARY = {...}`` block on stdout
with per-seed and seed-mean numbers for every in-tree method at
``top_k=1000``, so downstream tooling can ingest results without
re-parsing the human-readable tables.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import wasserstein_distance

from grn_inference import (
    DiffCovModel,
    DominatorTreeModel,
    MeanDifferenceModel,
    NeighborhoodRegressionModel,
    PathInversionModel,
    RandomBaseline,
    evaluate_statistical,
    make_synthetic_dataset,
)


# Methods included in the JSON summary block. The default set is the
# MD + Random baselines plus whatever methods iterations have added.
# Oracle is printed in the per-seed tables but excluded from the summary
# (it reads ground truth). Each entry is a zero-arg factory that returns
# a fresh instance bound to ``top_k``.
def build_methods(top_k: int, fit_seed: int = 0):
    return {
        "MeanDifferenceModel": lambda: MeanDifferenceModel(top_k=top_k),
        "RandomBaseline": lambda: RandomBaseline(top_k=top_k, seed=fit_seed),
        "NeighborhoodRegressionModel": lambda: NeighborhoodRegressionModel(
            top_k=top_k,
        ),
        "PathInversionModel": lambda: PathInversionModel(top_k=top_k),
        "DiffCovModel": lambda: DiffCovModel(top_k=top_k),
        "DominatorTreeModel": lambda: DominatorTreeModel(top_k=top_k),
    }


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


def run_one_seed(seed: int, *, headline_top_k: int = 1000) -> dict:
    """Run every method once on the synthetic data at ``seed``.

    Returns a dict keyed by method name with headline metrics plus the
    raw per-edge W1 map and ranked edges so the caller can reuse them
    for the human-readable tables without re-running the methods.
    """
    n_genes = 50
    n_perturbed_genes = n_genes // 2  # only half the genes get intervention arms
    data, truth = make_synthetic_dataset(
        n_genes=n_genes,
        edge_density=0.12,
        n_control_cells=2000,
        n_cells_per_perturbation=200,
        n_perturbed_genes=n_perturbed_genes,
        seed=seed,
    )
    perturbed_set = set(data.perturbed_genes())
    true_set = set(truth.true_edges)
    true_edges_with_perturbed_src = {e for e in true_set if e[0] in perturbed_set}
    true_edges_with_unperturbed_src = true_set - true_edges_with_perturbed_src

    print(f"\n{'#' * 78}")
    print(f"# SEED = {seed}")
    print(f"{'#' * 78}")
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
    # Run each in-tree method at ``headline_top_k`` once; slice down
    # later for finer comparisons.
    # --------------------------------------------------------------------
    max_k = headline_top_k
    print(f"\nFitting models (top_k={max_k})...")
    factories = build_methods(max_k)
    method_edges: dict[str, list] = {}
    method_runtime: dict[str, float] = {}
    for name, factory in factories.items():
        t0 = time.time()
        model = factory()
        edges = model.fit_predict(data)
        dt = time.time() - t0
        method_edges[name] = edges
        method_runtime[name] = dt
        print(f"  {name:<22} {dt:>8.2f}s")

    oracle_edges = [e for e, _ in oracle[:max_k]]

    # Display order: oracle diagnostic first, then in-tree methods.
    display_methods = {
        "Oracle (W1 sorted)": oracle_edges,
        **method_edges,
    }

    # Cache per_edge_w1 maps at max_k for cumulative analysis.
    per_edge_maps: dict[str, dict] = {}
    for name, edges in display_methods.items():
        res = evaluate_statistical(
            edges, data, omission_sample_size=500,
            rng=np.random.default_rng(99),
        )
        per_edge_maps[name] = res.per_edge_wasserstein

    # --------------------------------------------------------------------
    # Per-top_k table: mean W1, precision@k, FOR, source breakdown,
    # hidden-source recall.
    # --------------------------------------------------------------------
    headline: dict[str, dict] = {}
    for top_k in (50, 100, 500, 1000):
        print(f"\n--- top_k = {top_k} ---")
        header = (
            f"{'method':<22} {'mean W1':>10} "
            f"{'prec@k':>8} {'prec (pert)':>12} {'prec (unpert)':>14} "
            f"{'hidden rec':>11} "
            f"{'FOR':>8} {'# pert':>8} {'# unpert':>10}"
        )
        print(header)
        print("-" * len(header))
        for name, edges in display_methods.items():
            edges_k = edges[:top_k]
            res = evaluate_statistical(
                edges_k, data, omission_sample_size=500,
                rng=np.random.default_rng(99),
            )
            n_pert = sum(1 for s, _ in edges_k if s in perturbed_set)
            n_unpert = len(edges_k) - n_pert
            hits = sum(1 for e in edges_k if e in true_set)
            hits_pert = sum(
                1 for e in edges_k
                if e in true_set and e[0] in perturbed_set
            )
            hits_unpert = hits - hits_pert
            prec = hits / max(len(edges_k), 1)
            prec_pert = hits_pert / max(n_pert, 1)
            prec_unpert = hits_unpert / max(n_unpert, 1)
            hidden_recall = (
                hits_unpert / max(len(true_edges_with_unperturbed_src), 1)
            )
            print(
                f"{name:<22} {res.mean_wasserstein:>10.4f} "
                f"{prec:>8.3f} {prec_pert:>12.3f} {prec_unpert:>14.3f} "
                f"{hidden_recall:>11.3f} "
                f"{res.false_omission_rate:>8.3f} "
                f"{n_pert:>8d} {n_unpert:>10d}"
            )
            if top_k == headline_top_k and name in method_edges:
                headline[name] = {
                    "mean_w1": float(res.mean_wasserstein),
                    "for": float(res.false_omission_rate),
                    "precision_at_k": float(prec),
                    "hidden_source_recall": float(hidden_recall),
                    "runtime_s": float(method_runtime[name]),
                }

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
    for name, edges in display_methods.items():
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

    return headline


def emit_json_summary(per_seed: dict[int, dict]) -> None:
    """Print a single-line JSON block with per-seed and seed-mean metrics.

    The block is emitted on its own ``JSON_SUMMARY = {...}`` line so
    downstream tools can regex-match it out of mixed stdout.
    """
    # Collate into {method: {"per_seed": {seed: {...}}, "mean": {...}}}
    methods = set()
    for seed_result in per_seed.values():
        methods.update(seed_result.keys())

    summary = {}
    for m in sorted(methods):
        per_seed_m = {}
        keys = ("mean_w1", "for", "precision_at_k", "hidden_source_recall", "runtime_s")
        for seed, seed_result in per_seed.items():
            if m in seed_result:
                per_seed_m[str(seed)] = seed_result[m]
        mean_m = {}
        for k in keys:
            vals = [v[k] for v in per_seed_m.values() if k in v]
            mean_m[k] = float(np.mean(vals)) if vals else None
        summary[m] = {"per_seed": per_seed_m, "mean": mean_m}

    summary["_seeds"] = sorted(per_seed.keys())

    # Single-line, machine-parseable.
    print()
    print("JSON_SUMMARY = " + json.dumps(summary, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[7],
        help="One or more data seeds to run (default: 7).",
    )
    parser.add_argument(
        "--top-k", type=int, default=1000,
        help="Headline top_k for the JSON summary (default: 1000).",
    )
    args = parser.parse_args()

    per_seed: dict[int, dict] = {}
    for seed in args.seeds:
        per_seed[seed] = run_one_seed(seed, headline_top_k=args.top_k)

    # --------------------------------------------------------------------
    # Cross-seed aggregate: for each in-tree method, print seed-means at
    # top_k so the headline numbers are easy to eyeball.
    # --------------------------------------------------------------------
    if len(args.seeds) > 1:
        print("\n" + "#" * 78)
        print(f"# CROSS-SEED MEAN (top_k={args.top_k}, seeds={args.seeds})")
        print("#" * 78)
        header = (
            f"{'method':<22} {'mean W1':>10} {'FOR':>8} "
            f"{'prec@k':>8} {'hidden rec':>11} {'runtime':>10}"
        )
        print(header)
        print("-" * len(header))
        methods = set()
        for r in per_seed.values():
            methods.update(r.keys())
        for m in sorted(methods):
            vals = [per_seed[s].get(m) for s in args.seeds if m in per_seed[s]]
            if not vals:
                continue
            mw1 = float(np.mean([v["mean_w1"] for v in vals]))
            mfor = float(np.mean([v["for"] for v in vals]))
            mprec = float(np.mean([v["precision_at_k"] for v in vals]))
            mhr = float(np.mean([v["hidden_source_recall"] for v in vals]))
            mrt = float(np.mean([v["runtime_s"] for v in vals]))
            print(
                f"{m:<22} {mw1:>10.4f} {mfor:>8.3f} "
                f"{mprec:>8.3f} {mhr:>11.3f} {mrt:>10.2f}s"
            )

    emit_json_summary(per_seed)


if __name__ == "__main__":
    main()
