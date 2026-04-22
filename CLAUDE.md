# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Simulation-based gene regulatory network (GRN) inference research, targeting the [CausalBench](https://github.com/causalbench/causalbench) benchmark (Chevalley et al. 2025). The benchmark evaluates methods that infer causal gene–gene interaction networks from large-scale perturbational single-cell gene expression data (Perturb-seq).

**Current state**: Stage 0 scaffold (evaluator, baselines, synthetic data) plus a **Stage 1 MVP** — an ensemble linear cyclic SCM fit by moment-matching (`EnsembleSCMFitter` in `grn_inference/ensemble_scm/`). The CausalBench data adapter is still a stub.

## Framing

This is **causal parameter estimation**, not machine learning. We fit the adjacency matrix W of a linear cyclic structural causal model; the output is a causal graph. PyTorch is used purely as an autodiff tool (`torch.linalg.solve`, `torch.autograd.grad`, manual SGD) — there is no `nn.Module`, no `torch.optim`, no train/test split, no predictor, no generalisation target. When describing the fit procedure, avoid ML vocabulary ("training", "learning rate", "model" in the predictive sense, "inference"). Prefer "fitting procedure", "step size", "SCM fitter / edge ranker", "parameter estimation".

## Environment Setup

```bash
conda env create -f environment.yml   # Python 3.11, PyTorch cpuonly, editable install
conda activate grn-inference
```

The `environment.yml` has a `pip: -e .` line, so `import grn_inference` works immediately. Update with `conda env update -f environment.yml --prune`.

## File Structure

```
grn_inference/
├── environment.yml
├── pyproject.toml
├── grn_inference/               # The package
│   ├── __init__.py              # Public API — re-exports everything including EnsembleSCMFitter
│   ├── dataset.py               # Dataset dataclass
│   ├── data_loaders.py          # make_synthetic_dataset + load_causalbench_dataset stub
│   ├── evaluator.py             # evaluate_statistical: Wasserstein + FOR
│   ├── models.py                # MeanDifferenceModel + RandomBaseline (baselines only)
│   └── ensemble_scm/            # Stage 1 MVP, self-contained module
│       ├── __init__.py          # Exports EnsembleSCMFitter + helpers
│       ├── simulator.py         # LinearSCM, simulate_control, simulate_intervention
│       ├── loss.py              # moment_matching_discrepancy, l1_penalty
│       ├── fit.py               # fit_scm_ensemble (manual-SGD loop + spectral projection)
│       └── model.py             # EnsembleSCMFitter (Model-protocol entry point)
├── tests/
│   ├── test_stage0.py
│   └── test_scm_fitter.py
└── scripts/
    ├── run_baseline.py                # Stage 0 baseline comparison
    ├── run_scm_fit.py                 # Stage 1 fitter vs baselines (full perturbation)
    ├── eval_partial_perturbation.py   # n_perturbed=25 + precision@k + matched-W1
    ├── run_candidate_comparison.py    # Per-candidate vs fused aggregation diagnostic
    ├── diagnose_divergence.py         # Spectral-radius / Frobenius trajectory
    ├── max_w1_oracle.py               # Oracle max W1 ceiling
    ├── sweep_spectral_threshold.py
    ├── sweep_l1.py
    └── sweep_optimization_hparams.py
```

**Layout rule**: each new GRN inference method goes in its own `grn_inference/<method>/` subdirectory with `simulator.py` / `loss.py` / `fit.py` / `model.py` / `__init__.py`. Top-level `models.py` stays reserved for baselines. The method's entry-point class is re-exported from `grn_inference/__init__.py` so `from grn_inference import <Method>Fitter` works.

## Running Things

```bash
# Tests (should see 15 passing)
pytest tests/ -v

# Stage 0 baselines — sanity check
python scripts/run_baseline.py

# Stage 1 fitter vs baselines, full perturbation
python scripts/run_scm_fit.py

# The interesting benchmark — only half the genes perturbed.
# Mean Difference structurally can't score edges whose source
# isn't perturbed; EnsembleSCMFitter can (and does at ~60% recall).
python scripts/eval_partial_perturbation.py

# Diagnostics (ran during tuning; still useful for reproducing each finding)
python scripts/max_w1_oracle.py                # W1 ceiling — MD is near it
python scripts/diagnose_divergence.py          # Step-by-step ρ(W) trajectory
python scripts/sweep_spectral_threshold.py     # Shows 0.80 is the sweet spot
python scripts/sweep_l1.py                     # L1 only bites at threshold ≤ 0.9
python scripts/sweep_optimization_hparams.py   # step_size × n_steps — neither lever helps at threshold=0.95
```

## Architecture

### Core Abstractions

**`Dataset`** (`grn_inference/dataset.py`) — the single type that models and the evaluator consume:
- `expression`: `np.ndarray` of shape `(n_cells, n_genes)`, normalised (log1p(CPM/10k))
- `interventions`: list of length `n_cells`; `"non-targeting"` = control, otherwise the targeted gene symbol
- `gene_names`: list of length `n_genes`

**`Model` protocol** (`grn_inference/models.py`): `fit_predict(data: Dataset) -> list[tuple[str, str]]`
- Returns a ranked list of `(source, target)` directed edges, already cut to `top_k`. The evaluator does not re-rank.

**`evaluate_statistical`** (`grn_inference/evaluator.py`): takes predicted edges + a Dataset, returns a `StatisticalResult` with:
- `mean_wasserstein` — precision-like; higher is better. Mean 1-D Wasserstein distance between `P(B | control)` and `P(B | do(A))` for each predicted `A → B`.
- `false_omission_rate` — recall-like; lower is better. Fraction of sampled non-predicted edges that pass a Mann-Whitney U test (α=0.05).

The Wasserstein metric is near-saturated by Mean Difference (see `scripts/max_w1_oracle.py` — MD is within ~3% of the oracle ceiling). The interesting differentiator is **precision@k and per-subgroup recall** on partial-perturbation data, not raw W1.

### Baselines (`grn_inference/models.py`)

- **`MeanDifferenceModel`**: scores `(A, B)` by `|mean(B | do(A)) − mean(B | control)|`. The CausalBench SOTA-class baseline. Near-oracle on W1.
- **`RandomBaseline`**: uniform random edges over perturbed sources. Sanity floor.

### EnsembleSCMFitter (`grn_inference/ensemble_scm/`)

An ensemble of N parameterised linear cyclic SCMs, fit by gradient descent to match the observed per-perturbation distributions. Procedure per step:
1. Sample an arm from `{control} ∪ {perturbed genes}`.
2. Simulate `batch_size` cells per candidate from that arm via `x = (I − W)⁻¹ ε` (batched over candidates in one solve).
3. `moment_matching_discrepancy(sim, real)` per candidate; sum + L1 penalty.
4. `torch.autograd.grad` → manual SGD step on `W`.
5. **Spectral-radius projection**: if `ρ(W_k) > spectral_threshold`, rescale `W_k` uniformly so `ρ = spectral_threshold`. Critical for stability.

After `n_steps`, aggregate `|W|` across candidates with a **generalised mean** `(mean_k |W_k|ᵖ)^(1/p)` (default p=3 — biased toward max but not pure max), zero diagonals, return top-k edges.

### Key defaults and what they mean

- `spectral_threshold = 0.80` — matches the synthetic generator's `target_spectral_radius`. Do not raise above ~0.9; see the failure mode below.
- `n_steps = 1000`, `step_size = 0.01` — saturated; more iterations / smaller step don't help (spectral projection dominates). Lowering the threshold is what unlocks further tuning.
- `l1_lambda = 1e-4` — weak. Only bites once `spectral_threshold ≤ 0.9`; above that, the threshold clamp dominates and L1 does nothing. At threshold=0.80, the sweet spot is ~1e-4 to 1e-3; `l1 = 1.0` collapses structure.
- `aggregation_power = 3.0` — "close to a mean but biased toward max"; the user's requested behaviour.

### Failure modes to watch for

- **Divergence**: without spectral projection, every candidate's `ρ(W)` crosses 1 by step 4–5 and shoots to 10¹³ within a few more steps. The fit is unrecoverable afterwards. `scripts/diagnose_divergence.py` shows the trajectory.
- **Discrepancy plateau at 10⁴–10⁵**: symptom of `spectral_threshold` being too high (typically 0.95). The simulator over-amplifies by 4× relative to the data. Lower the threshold toward 0.8.
- **Dense fitted W**: small L1 + loose threshold → `|W|` has no natural zeros, ranker has to pick `top_k` from noise. Either lower threshold (main fix) or raise L1 modestly.

### Synthetic Data

`make_synthetic_dataset(n_genes, n_perturbed_genes=None, ...)` generates cells from a known linear cyclic SCM:

```
x = (I − W)^{-1} ε
```

Interventions are soft knockdowns (zero out target's row in W, scale its emission). Returns `(Dataset, SyntheticTruth)` where `SyntheticTruth.true_edges` is the ground-truth edge list and `SyntheticTruth.W` is the adjacency matrix (`W[i,j]` nonzero means `j → i`).

Pass `n_perturbed_genes` (e.g. `n_genes // 2`) to limit which genes get intervention arms. The rest become "hidden" sources — edges from them are invisible to any method that ranks by intervention effect (Mean Difference gets zero such hits; `EnsembleSCMFitter` gets ~60% recall on them because it fits a generative model that has to explain observational structure too).

### CausalBench Adapter

`load_causalbench_dataset()` in `data_loaders.py` is a **stub** — it raises `NotImplementedError`. To use real Perturb-seq data:

1. `pip install causalbench`
2. Inspect `causalscbench/data_access/create_dataset.py` in the installed package.
3. Fill in the TODO in `data_loaders.py`.

Available datasets: `"weissmann_k562"` (K562, day 6) and `"weissmann_rpe1"` (RPE1, day 7).

## Conventions

- Expression shape: `(n_cells, n_genes)`, normalised.
- Edge direction: `(source, target)` means `source → target`.
- Control label: `"non-targeting"` (imported as `dataset.CONTROL_LABEL`).
- Gene names: case-sensitive strings matching the data source.
- SCM indexing: `W[i, j]` means "parent j regulates child i", i.e. `j → i`. The ranker transposes so external callers only see `(source, target)` pairs.
- Reproducibility: `fit_scm_ensemble` creates a `torch.Generator` from `seed`, so fits are deterministic for a given seed.
