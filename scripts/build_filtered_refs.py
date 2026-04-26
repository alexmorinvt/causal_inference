"""Build and pickle filtered bio reference sets.

CausalBench's `_load_string_pairs` ignores STRING's `combined_score` column
and treats every pair with any evidence as a true edge. This script
applies a confidence threshold (default 900 = "highest confidence")
and writes the resulting symmetrised `set[tuple[ensg, ensg]]` to disk
so subsequent eval runs load in <1 s instead of repeating ~5 min of
h5ad parsing + Python row iteration.

While we have the gene-name → ensembl map in memory we also pickle
ChIP-seq (per cell line) and the LR pairs, since both are gated by the
same slow GeneNameMapLoader call.

Outputs (defaults to data/refs/):
  string_net_{threshold}.pkl
  string_phys_{threshold}.pkl
  chipseq_k562.pkl
  chipseq_rpe1.pkl
  lr_pairs.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

vendor_path = str(Path(__file__).parent.parent / "vendor" / "causalbench")
if vendor_path not in sys.path:
    sys.path.insert(0, vendor_path)

if "pkg_resources" not in sys.modules:
    import importlib, os, types
    _pr = types.ModuleType("pkg_resources")
    def _resource_string(package_name: str, resource_name: str) -> bytes:
        pkg = importlib.import_module(package_name)
        with open(os.path.join(os.path.dirname(pkg.__file__), resource_name), "rb") as f:
            return f.read()
    _pr.resource_string = _resource_string
    sys.modules["pkg_resources"] = _pr


def build_string_set(
    path: str,
    threshold: int,
    prot_to_name: dict[str, str],
    name_to_ensg: dict[str, str],
) -> set[tuple[str, str]]:
    """Symmetrised set of (ensg_a, ensg_b) at combined_score ≥ threshold."""
    df = pd.read_csv(path, sep=" ", compression="gzip")
    df = df[df.combined_score >= threshold]
    s: set[tuple[str, str]] = set()
    for p1, p2 in zip(df.protein1.values, df.protein2.values):
        n1 = prot_to_name.get(p1)
        n2 = prot_to_name.get(p2)
        if n1 is None or n2 is None:
            continue
        e1 = name_to_ensg.get(n1)
        e2 = name_to_ensg.get(n2)
        if e1 is None or e2 is None:
            continue
        s.add((e1, e2))
        s.add((e2, e1))
    return s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=900,
                        help="STRING combined_score threshold (default 900).")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default="data/refs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # gene-name -> ensembl map (slow; reads both h5ad files)
    # ------------------------------------------------------------------
    print("Building name→ensembl map (reads K562 + RPE1 h5ad ~ 20 GB)...")
    t0 = time.time()
    from causalscbench.data_access.create_name_to_ensembl_map import GeneNameMapLoader
    name_to_ensg = GeneNameMapLoader(args.data_dir).load()
    print(f"  {len(name_to_ensg):,} mappings ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # STRING protein.info -> protein-id → gene name
    # ------------------------------------------------------------------
    print("\nLoading STRING protein.info...")
    t0 = time.time()
    info = pd.read_csv(
        f"{args.data_dir}/protein.info.txt.gz", sep="\t", compression="gzip"
    )
    prot_to_name = dict(
        zip(info["#string_protein_id"].values, info["preferred_name"].values)
    )
    print(f"  {len(prot_to_name):,} proteins ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # STRING network at threshold
    # ------------------------------------------------------------------
    print(f"\nFiltering STRING network at combined_score ≥ {args.threshold}...")
    t0 = time.time()
    string_net = build_string_set(
        f"{args.data_dir}/protein.links.txt.gz",
        args.threshold, prot_to_name, name_to_ensg,
    )
    print(f"  {len(string_net):,} symmetrised pairs ({time.time() - t0:.1f}s)")
    out = out_dir / f"string_net_{args.threshold}.pkl"
    with open(out, "wb") as f:
        pickle.dump(string_net, f)
    print(f"  saved -> {out}")

    # ------------------------------------------------------------------
    # STRING physical at threshold
    # ------------------------------------------------------------------
    print(f"\nFiltering STRING physical at combined_score ≥ {args.threshold}...")
    t0 = time.time()
    string_phys = build_string_set(
        f"{args.data_dir}/protein.physical.links.txt.gz",
        args.threshold, prot_to_name, name_to_ensg,
    )
    print(f"  {len(string_phys):,} symmetrised pairs ({time.time() - t0:.1f}s)")
    out = out_dir / f"string_phys_{args.threshold}.pkl"
    with open(out, "wb") as f:
        pickle.dump(string_phys, f)
    print(f"  saved -> {out}")

    # ------------------------------------------------------------------
    # ChIP-seq (per cell line) and LR pairs (global) — small but gated
    # by the same slow GeneNameMapLoader, so cache while we have it.
    # ------------------------------------------------------------------
    from causalscbench.data_access.create_evaluation_datasets import (
        CreateEvaluationDatasets,
    )

    for short, full in [("k562", "weissmann_k562"), ("rpe1", "weissmann_rpe1")]:
        print(f"\nBuilding ChIP-seq for {full}...")
        t0 = time.time()
        ev = CreateEvaluationDatasets(args.data_dir, full)
        cs = ev._load_chipseq()
        print(f"  {len(cs):,} pairs ({time.time() - t0:.1f}s)")
        out = out_dir / f"chipseq_{short}.pkl"
        with open(out, "wb") as f:
            pickle.dump(cs, f)
        print(f"  saved -> {out}")

    print("\nBuilding LR pairs...")
    t0 = time.time()
    ev = CreateEvaluationDatasets(args.data_dir, "weissmann_k562")
    lr = ev._load_lr_pairs()
    print(f"  {len(lr):,} pairs ({time.time() - t0:.1f}s)")
    out = out_dir / "lr_pairs.pkl"
    with open(out, "wb") as f:
        pickle.dump(lr, f)
    print(f"  saved -> {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
