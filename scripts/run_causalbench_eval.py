"""Evaluate all methods on a real CausalBench dataset (K562 or RPE1).

Usage
-----
    python scripts/run_causalbench_eval.py --dataset k562 --data_dir data
    python scripts/run_causalbench_eval.py --dataset rpe1 --data_dir data

Reports:
  - Statistical: mean W1 and FOR at top_k = 250, 500, 1000
  - Biological: precision against STRING (network + physical), LR pairs,
    and ChIP-seq (undirected) at top_k = 250, 500, 1000

EnsembleSCMFitter runs with reduced settings (n_candidates=3, n_steps=200)
due to the large gene count (~600 genes vs ~50 in synthetic benchmarks).
CORUM is excluded (download blocked by WAF; least important reference).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

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

vendor_path = str(Path(__file__).parent.parent / "vendor" / "causalbench")
if vendor_path not in sys.path:
    sys.path.insert(0, vendor_path)

# pkg_resources was removed from setuptools 82+; shim just enough for CausalBench's
# resource_string() calls (used only to load bundled ChIP-seq CSV files).
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


def load_bio_references(dataset_name: str, data_dir: str):
    """Load biological ground-truth sets, skipping CORUM (file unavailable).

    Calls each loader method individually on CreateEvaluationDatasets so we
    can skip _load_corum() without touching the rest.

    Returns a dict mapping reference name → set of (gene_a, gene_b) Ensembl ID pairs.
    """
    from causalscbench.data_access.create_evaluation_datasets import CreateEvaluationDatasets

    print("Loading biological reference datasets...")
    ev = CreateEvaluationDatasets(data_directory=data_dir, dataset_name=dataset_name)

    refs = {}
    for label, method in [
        ("LR pairs",             ev._load_lr_pairs),
        ("STRING network",       lambda: ev._load_string_pairs()[0]),
        ("STRING physical",      lambda: ev._load_string_pairs()[1]),
        ("ChIP-seq (directed)",  ev._load_chipseq),
    ]:
        t0 = time.time()
        ds = method()
        refs[label] = ds
        print(f"  {label}: {len(ds):,} known edges  ({time.time() - t0:.1f}s)")

    return refs


def load_cached_bio_references(refs_dir: str, dataset_short: str, string_threshold: int):
    """Load pre-built bio reference pickles from refs_dir.

    Built by scripts/build_filtered_refs.py. Expects files:
      {refs_dir}/string_net_{string_threshold}.pkl
      {refs_dir}/string_phys_{string_threshold}.pkl
      {refs_dir}/chipseq_{dataset_short}.pkl
      {refs_dir}/lr_pairs.pkl
    """
    import pickle
    from pathlib import Path

    refs_dir = Path(refs_dir)
    print(f"Loading cached biological references from {refs_dir} "
          f"(STRING combined_score ≥ {string_threshold})...")
    refs: dict = {}
    for label, fname in [
        ("LR pairs",             "lr_pairs.pkl"),
        ("STRING network",       f"string_net_{string_threshold}.pkl"),
        ("STRING physical",      f"string_phys_{string_threshold}.pkl"),
        ("ChIP-seq (directed)",  f"chipseq_{dataset_short}.pkl"),
    ]:
        path = refs_dir / fname
        t0 = time.time()
        with open(path, "rb") as f:
            refs[label] = pickle.load(f)
        print(f"  {label}: {len(refs[label]):,} known edges  ({time.time() - t0:.2f}s)")
    return refs


def bio_precision(edges: list, ground_truth: set, directed: bool = False) -> float:
    if not edges:
        return 0.0
    if directed:
        hits = sum(1 for e in edges if e in ground_truth)
    else:
        hits = sum(1 for e in edges if e in ground_truth or (e[1], e[0]) in ground_truth)
    return hits / len(edges)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="k562", choices=["k562", "rpe1"])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_bio", action="store_true",
                        help="Skip biological evaluation (faster)")
    parser.add_argument("--refs_dir", default=None,
                        help="If set, load pre-built bio reference pickles "
                             "from this directory (built by build_filtered_refs.py). "
                             "Falls back to slow CausalBench loaders if unset.")
    parser.add_argument("--string_threshold", type=int, default=900,
                        help="STRING combined_score threshold (used only with "
                             "--refs_dir). Default 900.")
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
    methods["FullyConnected"] = FullyConnectedBaseline(seed=args.seed).fit_predict(data)
    print(f"  FullyConnected:    {time.time() - t0:.2f}s  ({len(methods['FullyConnected']):,} edges total)")

    # ------------------------------------------------------------------
    # Statistical evaluation
    # ------------------------------------------------------------------
    for top_k in (250, 500, 1000):
        if top_k > max_k:
            continue
        print(f"\n--- Statistical  top_k={top_k} ---")
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

    # ------------------------------------------------------------------
    # Biological evaluation
    # ------------------------------------------------------------------
    if args.skip_bio:
        return

    print()
    try:
        if args.refs_dir is not None:
            refs = load_cached_bio_references(
                args.refs_dir, args.dataset, args.string_threshold,
            )
        else:
            refs = load_bio_references(dataset_name, args.data_dir)
    except Exception as e:
        print(f"Biological evaluation skipped: {e}")
        return

    ref_names = list(refs.keys())
    # ChIP-seq is directed; others are undirected
    directed_flags = {"ChIP-seq (directed)": True}

    for top_k in (250, 500, 1000):
        if top_k > max_k:
            continue
        print(f"\n--- Biological precision  top_k={top_k} ---")
        col_w = 16
        header = f"{'method':<20} " + " ".join(f"{n[:col_w]:>{col_w}}" for n in ref_names)
        print(header)
        print("-" * len(header))
        for name, edges in methods.items():
            edges_k = edges[:top_k]
            scores = []
            for ref_name in ref_names:
                directed = directed_flags.get(ref_name, False)
                p = bio_precision(edges_k, refs[ref_name], directed=directed)
                scores.append(p)
            row = f"{name:<20} " + " ".join(f"{s:>{col_w}.4f}" for s in scores)
            print(row)


if __name__ == "__main__":
    main()
