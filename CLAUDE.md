# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Simulation-based gene regulatory network (GRN) inference research, targeting the [CausalBench](https://github.com/causalbench/causalbench) benchmark (Chevalley et al. 2025). The benchmark evaluates methods that infer causal gene–gene interaction networks from large-scale perturbational single-cell gene expression data (Perturb-seq).

**Current state**: Multiple GRN inference methods implemented and benchmarked on synthetic data. CausalBench real data (K562, RPE1) is loaded and preprocessed; real-data evaluation is in progress.

## Framing

This is **causal parameter estimation**, not machine learning. We fit the adjacency matrix W of a linear cyclic structural causal model; the output is a causal graph. PyTorch is used purely as an autodiff tool (`torch.linalg.solve`, `torch.autograd.grad`, manual SGD) — there is no `nn.Module`, no `torch.optim`, no train/test split, no predictor, no generalisation target. When describing the fit procedure, avoid ML vocabulary ("training", "learning rate", "model" in the predictive sense, "inference"). Prefer "fitting procedure", "step size", "SCM fitter / edge ranker", "parameter estimation".

## Environment Setup

```bash
conda env create -f environment.yml   # Python 3.11, PyTorch cpuonly, editable install
conda activate grn-inference
pip install scanpy openpyxl gdown     # required for CausalBench data loading
```

The `environment.yml` has a `pip: -e .` line, so `import grn_inference` works immediately. Update with `conda env update -f environment.yml --prune`.

CausalBench is vendored as a git submodule at `vendor/causalbench` — no separate install needed. The package is installed editable (`pip install -e vendor/causalbench --no-deps`) so `pkg_resources` resolves bundled data files.

## File Structure

```
grn_inference/
├── environment.yml
├── pyproject.toml
├── vendor/causalbench/              # CausalBench submodule (data access + evaluators)
├── data/                            # gitignored — h5ad + npz cache (multi-GB)
├── grn_inference/                   # The package
│   ├── __init__.py                  # Public API — re-exports all methods
│   ├── dataset.py                   # Dataset dataclass
│   ├── data_loaders.py              # make_synthetic_dataset + load_causalbench_dataset
│   ├── evaluator.py                 # evaluate_statistical: Wasserstein + FOR
│   ├── models.py                    # Baselines only: MeanDifference, Random, FullyConnected
│   ├── ensemble_scm/                # Gradient-descent SCM fitter — see README inside
│   ├── indirect_pruning/            # Cascade-shortcut pruning ranker — see README inside
│   ├── shift_corr/                  # Shift × within-arm correlation ranker — see README inside
│   ├── shift_paths/                 # Path-scoring ranker (excluded, issue under investigation)
│   └── dominator_tree/              # Dominator-tree edge ranker — see README inside
├── tests/
│   ├── test_stage0.py
│   ├── test_scm_fitter.py
│   └── test_dominator_tree.py
└── scripts/
    ├── run_baseline.py                  # Stage 0 baseline comparison
    ├── run_scm_fit.py                   # EnsembleSCM vs baselines, full perturbation
    ├── eval_partial_perturbation.py     # All methods, n_perturbed=25, precision@k + matched-W1
    ├── run_causalbench_eval.py          # All methods on real K562/RPE1 data
    ├── run_candidate_comparison.py      # Per-candidate vs fused aggregation diagnostic
    ├── diagnose_divergence.py           # Spectral-radius / Frobenius trajectory
    ├── max_w1_oracle.py                 # Oracle max W1 ceiling
    ├── sweep_spectral_threshold.py
    ├── sweep_l1.py
    └── sweep_optimization_hparams.py
```

**Layout rule**: each new GRN inference method goes in its own `grn_inference/<method>/` subdirectory with a `model.py`, `__init__.py`, and `README.md`. Top-level `models.py` stays reserved for baselines. The method's entry-point class is re-exported from `grn_inference/__init__.py` so `from grn_inference import <Method>` works.

## Running Things

```bash
# Tests (should see 18 passing)
pytest tests/ -v

# Synthetic benchmarks
python scripts/run_baseline.py                 # Stage 0 baselines sanity check
python scripts/eval_partial_perturbation.py    # All methods, partial-perturbation synthetic

# Real CausalBench data (requires data/k562.h5ad and data/rpe1.h5ad)
python scripts/run_causalbench_eval.py --dataset k562 --data_dir data
python scripts/run_causalbench_eval.py --dataset rpe1 --data_dir data

# Diagnostics
python scripts/max_w1_oracle.py                # W1 ceiling — MD is near it
python scripts/diagnose_divergence.py          # EnsembleSCM spectral-radius trajectory
python scripts/sweep_spectral_threshold.py
python scripts/sweep_l1.py
python scripts/sweep_optimization_hparams.py
```

## Architecture

### Core Abstractions

**`Dataset`** (`grn_inference/dataset.py`) — the single type that models and the evaluator consume:
- `expression`: `np.ndarray` of shape `(n_cells, n_genes)`, normalised (log1p(CPM/10k))
- `interventions`: list of length `n_cells`; `"non-targeting"` = control, otherwise the targeted gene symbol
- `gene_names`: list of length `n_genes`

**`Model` protocol** (`grn_inference/models.py`): `fit_predict(data: Dataset) -> list[tuple[str, str]]`
- Returns a ranked list of `(source, target)` directed edges, already cut to `top_k`. The evaluator does not re-rank.
- `FullyConnectedBaseline` is the exception — it returns all n*(n-1) edges unranked (no `top_k`).

**`evaluate_statistical`** (`grn_inference/evaluator.py`): takes predicted edges + a Dataset, returns a `StatisticalResult` with:
- `mean_wasserstein` — precision-like; higher is better. Mean 1-D Wasserstein distance between `P(B | control)` and `P(B | do(A))` for each predicted `A → B`.
- `false_omission_rate` — recall-like; lower is better. Fraction of sampled non-predicted edges that pass a Mann-Whitney U test (α=0.05).

The Wasserstein metric is near-saturated by Mean Difference (see `scripts/max_w1_oracle.py` — MD is within ~3% of the oracle ceiling). The interesting differentiator is **precision@k and per-subgroup recall** on partial-perturbation data, not raw W1.

### Baselines (`grn_inference/models.py`)

- **`MeanDifferenceModel`**: scores `(A, B)` by `|mean(B | do(A)) − mean(B | control)|`. The CausalBench SOTA-class baseline. Near-oracle on W1.
- **`RandomBaseline`**: uniform random edges over perturbed sources. Sanity floor.
- **`FullyConnectedBaseline`**: returns all n*(n-1) directed edges unranked (port of CausalBench's `FullyConnected`). Perfect recall by construction. Use to establish the recall ceiling.

### Methods (one README per method)

| Method | Module | Key idea |
|---|---|---|
| `EnsembleSCMFitter` | `ensemble_scm/` | Gradient-descent SCM fit; recovers hidden-source edges |
| `DominatorTreeModel` | `dominator_tree/` | Dominator-tree votes; best precision on hidden sources among shift methods |
| `IndirectPruningModel` | `indirect_pruning/` | Prune cascade shortcuts from shift graph |
| `ShiftCorrModel` | `shift_corr/` | Shift × within-arm correlation |
| `ShiftPathsModel` | `shift_paths/` | Path-scoring (excluded — issue under investigation) |

See each method's `README.md` for algorithm details, hyperparameter notes, and failure modes.

### Data

**Synthetic**: `make_synthetic_dataset(n_genes, n_perturbed_genes=None, ...)` generates cells from a known linear cyclic SCM via a branching lineage cascade. Returns `(Dataset, SyntheticTruth)` with ground-truth edges. Pass `n_perturbed_genes` to limit intervention arms — crucial for testing hidden-source recovery.

**CausalBench**: `load_causalbench_dataset(dataset_name, data_directory, filter=True)` loads and preprocesses the Replogle Perturb-seq data from cached `.npz` files. `filter=True` applies CausalBench's strong-perturbation filter (>50 DEGs, ≤−30% knockdown, >25 cells). Available: `"weissmann_k562"` (622 genes, 129k cells filtered) and `"weissmann_rpe1"` (383 genes, 92k cells filtered). The raw `.h5ad` files (~10 GB each) must be placed in `data/` manually (Figshare blocks automated downloads).

## Conventions

- Expression shape: `(n_cells, n_genes)`, normalised.
- Edge direction: `(source, target)` means `source → target`.
- Control label: `"non-targeting"` (imported as `dataset.CONTROL_LABEL`).
- Gene names: case-sensitive strings matching the data source.
- SCM indexing: `W[i, j]` means "parent j regulates child i", i.e. `j → i`. The ranker transposes so external callers only see `(source, target)` pairs.
- Reproducibility: `fit_scm_ensemble` creates a `torch.Generator` from `seed`, so fits are deterministic for a given seed.
