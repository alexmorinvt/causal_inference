"""Evaluate all methods on a real CausalBench dataset (K562 or RPE1).

Usage
-----
    python scripts/run_causalbench_eval.py --dataset k562 --data_dir data
    python scripts/run_causalbench_eval.py --dataset rpe1 --data_dir data

Reports mean W1 and FOR at top_k = 250, 500, 1000 for every method.
EnsembleSCMFitter runs with reduced settings (n_candidates=3, n_steps=200)
due to the large gene count (~600 genes vs ~50 in synthetic benchmarks).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from grn_inference import (
    DominatorTreeModel,
    EnsembleSCMFitter,
    FullyConnectedBaseline,
    IndirectPruningModel,
    MeanDifferenceModel,
    RandomBaseline,
    ShiftCorrModel,
    evaluate_statistical,
    load_causalbench_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="k562", choices=["k562", "rpe1"])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset_name = f"weissmann_{args.dataset}"
    print(f"Loading {dataset_name} (filtered)...")
    t0 = time.time()
    data = load_causalbench_dataset(dataset_name, data_directory=args.data_dir, filter=True)
    print(f"  {data.summary()}  ({time.time() - t0:.1f}s)\n")

    max_k = args.top_k

    print(f"Fitting models (top_k={max_k})...")

    methods: dict[str, list] = {}

    t0 = time.time()
    methods["Mean Difference"] = MeanDifferenceModel(top_k=max_k).fit_predict(data)
    print(f"  Mean Difference:   {time.time() - t0:.2f}s")

    t0 = time.time()
    methods["ShiftCorr"] = ShiftCorrModel(top_k=max_k).fit_predict(data)
    print(f"  ShiftCorr:         {time.time() - t0:.2f}s")

    t0 = time.time()
    methods["IndirectPruning"] = IndirectPruningModel(top_k=max_k).fit_predict(data)
    print(f"  IndirectPruning:   {time.time() - t0:.2f}s")

    t0 = time.time()
    methods["DominatorTree"] = DominatorTreeModel(top_k=max_k).fit_predict(data)
    print(f"  DominatorTree:     {time.time() - t0:.2f}s")

    t0 = time.time()
    methods["EnsembleSCM"] = EnsembleSCMFitter(
        top_k=max_k, n_candidates=3, n_steps=200,
        step_size=0.01, batch_size=200, l1_lambda=1e-4,
        seed=args.seed, log_every=50,
    ).fit_predict(data)
    print(f"  EnsembleSCM:       {time.time() - t0:.2f}s")

    t0 = time.time()
    methods["Random"] = RandomBaseline(top_k=max_k, seed=args.seed).fit_predict(data)
    print(f"  Random:            {time.time() - t0:.2f}s")

    t0 = time.time()
    # FullyConnected returns all n*(n-1) edges; slice to max_k for table rows.
    methods["FullyConnected"] = FullyConnectedBaseline(seed=args.seed).fit_predict(data)
    print(f"  FullyConnected:    {time.time() - t0:.2f}s  ({len(methods['FullyConnected']):,} edges total)")

    for top_k in (250, 500, 1000):
        if top_k > max_k:
            continue
        print(f"\n--- top_k = {top_k} ---")
        header = f"{'method':<20} {'mean W1':>10} {'FOR':>8} {'# edges':>9}"
        print(header)
        print("-" * len(header))
        for name, edges in methods.items():
            edges_k = edges[:top_k]
            res = evaluate_statistical(
                edges_k, data,
                omission_sample_size=500,
                rng=np.random.default_rng(99),
            )
            print(
                f"{name:<20} {res.mean_wasserstein:>10.4f} "
                f"{res.false_omission_rate:>8.3f} {len(edges_k):>9d}"
            )


if __name__ == "__main__":
    main()
