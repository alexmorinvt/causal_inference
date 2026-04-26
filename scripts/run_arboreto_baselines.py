"""Run GRNBoost2 / GENIE3 baselines on a CausalBench dataset.

Mirrors the structure of run_causalbench_eval.py but for the two
arboreto-based methods that CausalBench ships as `arboreto_baselines`.
Reports statistical (mean W1, FOR) and biological (LR pairs, STRING
network, STRING physical, ChIP-seq) precision at top_k 250 / 500 / 1000.

Usage
-----
    python scripts/run_arboreto_baselines.py --dataset k562 --method grnboost2
    python scripts/run_arboreto_baselines.py --dataset rpe1 --method genie3
    python scripts/run_arboreto_baselines.py --dataset k562 --method both
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from grn_inference import (
    evaluate_statistical,
    load_causalbench_dataset,
)
from grn_inference.dataset import Dataset

vendor_path = str(Path(__file__).parent.parent / "vendor" / "causalbench")
if vendor_path not in sys.path:
    sys.path.insert(0, vendor_path)

if "pkg_resources" not in sys.modules:
    import importlib, os, types
    _pr = types.ModuleType("pkg_resources")
    def _resource_string(package_name: str, resource_name: str) -> bytes:
        pkg = importlib.import_module(package_name)
        path = os.path.join(os.path.dirname(pkg.__file__), resource_name)
        with open(path, "rb") as f:
            return f.read()
    _pr.resource_string = _resource_string
    sys.modules["pkg_resources"] = _pr


def filter_rare_genes(
    expression: np.ndarray,
    gene_names: list[str],
    expression_threshold: float = 0.25,
) -> tuple[np.ndarray, list[str]]:
    """Drop genes with non-zero expression in fewer than threshold-fraction of cells.

    Mirrors CausalBench's `remove_lowly_expressed_genes` (which uses scprep)
    but without the scprep dependency.
    """
    n_cells = expression.shape[0]
    min_cells = int(n_cells * expression_threshold)
    nonzero = (expression > 0).sum(axis=0)
    keep = nonzero >= min_cells
    return expression[:, keep], [g for g, k in zip(gene_names, keep) if k]


def run_arboreto(
    method: str,
    expression: np.ndarray,
    gene_names: list[str],
    seed: int = 0,
    n_workers: int = 8,
    threads_per_worker: int = 2,
    rf_n_estimators: int = 200,
) -> pd.DataFrame:
    """Return the arboreto network DataFrame (importance-sorted).

    For ``genie3`` we override ``n_estimators`` (default 1000 in arboreto) to
    a smaller value to keep per-worker RAM under WSL's per-worker limits.
    """
    import distributed
    from arboreto import algo
    from arboreto.core import RF_KWARGS

    cluster = distributed.LocalCluster(
        n_workers=n_workers, threads_per_worker=threads_per_worker
    )
    client = distributed.Client(cluster)
    try:
        if method == "grnboost2":
            net = algo.grnboost2(
                expression_data=expression,
                gene_names=gene_names,
                client_or_address=client,
                seed=seed,
                early_stop_window_length=10,
                verbose=True,
            )
        elif method == "genie3":
            kwargs = dict(RF_KWARGS)
            kwargs["n_estimators"] = rf_n_estimators
            net = algo.diy(
                expression_data=expression,
                gene_names=gene_names,
                regressor_type="RF",
                regressor_kwargs=kwargs,
                client_or_address=client,
                seed=seed,
                verbose=True,
            )
        else:
            raise ValueError(f"unknown method: {method}")
    finally:
        client.close()
        cluster.close()
    # network is returned sorted by importance descending
    return net


def network_to_edges(net: pd.DataFrame, top_k: int) -> list[tuple[str, str]]:
    return [(row.TF, row.target) for row in net.head(top_k).itertuples(index=False)]


def load_bio_references(dataset_name: str, data_dir: str):
    from causalscbench.data_access.create_evaluation_datasets import CreateEvaluationDatasets

    print("Loading biological reference datasets...")
    ev = CreateEvaluationDatasets(data_directory=data_dir, dataset_name=dataset_name)
    refs = {}
    for label, m in [
        ("LR pairs", ev._load_lr_pairs),
        ("STRING net", lambda: ev._load_string_pairs()[0]),
        ("STRING phys", lambda: ev._load_string_pairs()[1]),
        ("ChIP-seq", ev._load_chipseq),
    ]:
        t0 = time.time()
        refs[label] = m()
        print(f"  {label}: {len(refs[label]):,}  ({time.time() - t0:.1f}s)")
    return refs


def bio_precision(edges: list, gt: set, directed: bool) -> float:
    if not edges:
        return 0.0
    if directed:
        return sum(1 for e in edges if e in gt) / len(edges)
    return sum(1 for e in edges if e in gt or (e[1], e[0]) in gt) / len(edges)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="k562", choices=["k562", "rpe1"])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--method", default="grnboost2",
                        choices=["grnboost2", "genie3", "both"])
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--threads_per_worker", type=int, default=2)
    parser.add_argument("--skip_bio", action="store_true")
    parser.add_argument(
        "--out_dir", default=None,
        help="Directory to save the importance-scored network DataFrame "
             "as {dataset}_{method}_network.parquet. If unset, no save.",
    )
    parser.add_argument(
        "--n_cells", type=int, default=None,
        help="If set, subsample to this many cells before fitting (preserves "
             "control/intervention proportions). Recommended for GENIE3 to "
             "stay under per-worker memory limits.",
    )
    parser.add_argument(
        "--rf_n_estimators", type=int, default=200,
        help="GENIE3 random-forest trees per target gene. Default 200 "
             "(arboreto default 1000) for memory.",
    )
    args = parser.parse_args()

    name = f"weissmann_{args.dataset}"
    print(f"Loading {name} (filtered)...")
    t0 = time.time()
    data = load_causalbench_dataset(name, data_directory=args.data_dir, filter=True)
    print(f"  {data.summary()}  ({time.time() - t0:.1f}s)")

    print("\nApplying remove_lowly_expressed_genes (threshold=0.25)...")
    expr_ar, genes_ar = filter_rare_genes(
        data.expression, data.gene_names, expression_threshold=0.25
    )
    print(f"  Kept {len(genes_ar):,} of {data.n_genes:,} genes")

    if args.n_cells is not None and args.n_cells < expr_ar.shape[0]:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(expr_ar.shape[0], size=args.n_cells, replace=False)
        idx.sort()
        expr_ar = expr_ar[idx]
        print(f"  Subsampled to {args.n_cells:,} cells (seed={args.seed})")

    methods_to_run = ["grnboost2", "genie3"] if args.method == "both" else [args.method]
    method_edges: dict[str, list] = {}

    for m in methods_to_run:
        print(f"\nFitting {m} (top_k={args.top_k}, "
              f"workers={args.n_workers}x{args.threads_per_worker})...")
        t0 = time.time()
        net = run_arboreto(
            m, expr_ar, genes_ar,
            seed=args.seed,
            n_workers=args.n_workers,
            threads_per_worker=args.threads_per_worker,
            rf_n_estimators=args.rf_n_estimators,
        )
        print(f"  {m}: {time.time() - t0:.1f}s, returned {len(net):,} edges total")
        method_edges[m.upper()] = network_to_edges(net, args.top_k)

        if args.out_dir is not None:
            out_path = Path(args.out_dir) / f"{args.dataset}_{m}_network.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            net.to_parquet(out_path, index=False)
            print(f"  saved scored network -> {out_path}")

    # ------------------------------------------------------------------
    # Statistical evaluation (uses original data, not filtered)
    # ------------------------------------------------------------------
    for top_k in (250, 500, 1000):
        if top_k > args.top_k:
            continue
        print(f"\n--- Statistical  top_k={top_k} ---")
        print(f"{'method':<14} {'mean W1':>10} {'FOR':>8} {'# edges':>9}")
        for n, edges in method_edges.items():
            edges_k = edges[:top_k]
            res = evaluate_statistical(
                edges_k, data, omission_sample_size=500,
                rng=np.random.default_rng(99),
            )
            print(f"{n:<14} {res.mean_wasserstein:>10.4f} "
                  f"{res.false_omission_rate:>8.3f} {len(edges_k):>9d}")

    if args.skip_bio:
        return

    print()
    refs = load_bio_references(name, args.data_dir)
    directed_flags = {"ChIP-seq": True}

    for top_k in (250, 500, 1000):
        if top_k > args.top_k:
            continue
        print(f"\n--- Biological precision  top_k={top_k} ---")
        cols = list(refs.keys())
        header = f"{'method':<14} " + " ".join(f"{c:>13}" for c in cols)
        print(header)
        print("-" * len(header))
        for n, edges in method_edges.items():
            edges_k = edges[:top_k]
            row = f"{n:<14} "
            for c in cols:
                p = bio_precision(edges_k, refs[c], directed=directed_flags.get(c, False))
                row += f" {p:>12.4f}"
            print(row)


if __name__ == "__main__":
    main()
