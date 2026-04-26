# grn_inference

Simulation-based gene regulatory network (GRN) inference, targeting the
[CausalBench](https://github.com/causalbench/causalbench) benchmark
(Chevalley et al. 2025). The benchmark evaluates methods that infer causal
gene–gene interaction networks from large-scale perturbational single-cell
gene expression data (Perturb-seq).

This is **causal parameter estimation**, not machine learning. The output is
a causal graph (ranked directed edges). PyTorch is used only as an autodiff
library — there is no train/test split, no predictor, no generalisation target.

## Setup

```bash
conda env create -f environment.yml      # Python 3.11, PyTorch cpuonly, editable install
conda activate grn-inference
pip install scanpy openpyxl gdown        # required for real data loading
```

CausalBench is vendored as a submodule — clone with `--recurse-submodules` or run
`git submodule update --init` after cloning.

## Verify

```bash
pytest tests/ -v                                        # 18 tests
python scripts/eval_partial_perturbation.py             # all methods on synthetic data
```

## Real data

Download the Replogle et al. Perturb-seq `.h5ad` files manually (Figshare blocks
automated downloads) and place them in `data/`:

- **K562**: https://plus.figshare.com/ndownloader/files/35773219 → `data/k562.h5ad`
- **RPE1**: https://plus.figshare.com/ndownloader/files/35775606 → `data/rpe1.h5ad`

Preprocessing and caching happen automatically on first load (~5 min each).

```bash
python scripts/run_causalbench_eval.py --dataset k562 --data_dir data
python scripts/run_causalbench_eval.py --dataset rpe1 --data_dir data
```

## Methods

| Method | Module | Key idea |
|---|---|---|
| `MeanDifferenceModel` | `models.py` | `\|mean(B\|do(A)) − mean(B\|ctrl)\|` — CausalBench SOTA baseline |
| `ShiftCorrModel` | `shift_corr/` | Shift × within-arm Pearson correlation |
| `IndirectPruningModel` | `indirect_pruning/` | Shift graph with cascade-shortcut pruning |
| `DominatorTreeModel` | `dominator_tree/` | Dominator-tree votes across perturbed roots |
| `EnsembleSCMFitter` | `ensemble_scm/` | Gradient-descent linear cyclic SCM fit |
| `FullyConnectedBaseline` | `models.py` | All n*(n-1) edges — recall ceiling |
| `RandomBaseline` | `models.py` | Sanity floor |

Each method has a `README.md` in its subdirectory with algorithm details
and hyperparameter notes.

## Benchmark results — synthetic (n_genes=50, n_perturbed=25, seed=7)

The partial-perturbation regime is the key differentiator: only half the genes
have intervention arms, so true edges from unperturbed sources are invisible to
any method that ranks purely by intervention shift.

**At top_k=100:**

| Method | mean W1 | prec@k | prec (perturbed src) | prec (hidden src) |
|---|---|---|---|---|
| ShiftCorr | 0.781 | 0.350 | 0.350 | 0.000 |
| Mean Difference | 0.788 | 0.300 | 0.300 | 0.000 |
| **DominatorTree** | **0.813** | 0.320 | 0.371 | **0.200** |
| IndirectPruning | 0.771 | 0.200 | 0.294 | 0.000 |
| EnsembleSCM | 0.355 | 0.170 | 0.276 | 0.127 |

DominatorTree is the only shift-based method recovering hidden-source edges.
EnsembleSCM does so too (by fitting a generative model), but at lower W1.

## Evaluator

`evaluate_statistical(predicted_edges, data)` returns:

- `mean_wasserstein` — mean 1-D W1 distance between `P(B | control)` and
  `P(B | do(A))` over predicted `A → B` edges. Higher is better (precision-like).
- `false_omission_rate` — Mann-Whitney-U-based recall estimate over sampled
  non-predicted edges. Lower is better.

On synthetic data, `SyntheticTruth.true_edges` lets you also compute
**precision@k** and **per-subgroup recall** (perturbed vs hidden sources) —
see `scripts/eval_partial_perturbation.py`.

## Conventions

- Expression: `(n_cells, n_genes)`, normalised (log1p(CPM/10k)).
- Edge direction: `(source, target)` means `source → target`.
- Control label: `"non-targeting"` (exported as `grn_inference.CONTROL_LABEL`).
- `W[i, j]` in the SCM: "gene j regulates gene i". The ranker handles the
  transpose — callers only see `(source, target)`.
- New methods go in `grn_inference/<method>/` with their own `README.md`.
  Top-level `models.py` is reserved for baselines.
