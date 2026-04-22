# grn_inference — Stage 1 MVP

Simulation-based gene regulatory network inference, targeting the
CausalBench benchmark (Chevalley et al. 2025).

**Current state.** The Stage 0 scaffold (Dataset, evaluator, Mean
Difference + Random baselines, synthetic data) is in place and
verified, and the Stage 1 MVP — an ensemble linear cyclic SCM fit by
moment-matching against per-perturbation distributions — lives in
`grn_inference/ensemble_scm/`. On the partial-perturbation synthetic
benchmark, the fitter recovers 60% of true edges whose source wasn't
perturbed, a regime Mean Difference structurally cannot touch.

This is **causal parameter estimation**, not machine learning. We fit
the adjacency matrix W of a linear cyclic structural causal model so
that simulated distributions match the observed data. PyTorch is used
only as an autodiff library (`torch.linalg.solve`, `torch.autograd.grad`,
manual SGD). There is no train/test split, no predictor, no
generalisation target — the output is a causal graph.

## Setup

```bash
conda env create -f environment.yml      # Python 3.11, PyTorch, editable install
conda activate grn-inference
```

The `environment.yml` has a `pip: -e .` line, so `import grn_inference`
works immediately. Update an existing env with
`conda env update -f environment.yml --prune`.

Adding CausalBench (real data) — when ready:
```bash
conda run -n grn-inference pip install causalbench
# then fill in the stub in grn_inference/data_loaders.py
```

## Verify everything works

```bash
pytest -v                                # 15 tests
python scripts/run_baseline.py           # Stage 0 baselines vs synthetic
python scripts/run_scm_fit.py            # Stage 1 fitter vs baselines
python scripts/eval_partial_perturbation.py  # the interesting benchmark
```

## What's here

```
grn_inference/
├── environment.yml            # Conda env + pytorch
├── pyproject.toml
├── grn_inference/
│   ├── __init__.py            # Public API — re-exports everything
│   ├── dataset.py             # Dataset dataclass
│   ├── data_loaders.py        # make_synthetic_dataset + CausalBench stub
│   ├── evaluator.py           # Wasserstein + FOR
│   ├── models.py              # MeanDifferenceModel + RandomBaseline
│   └── ensemble_scm/          # Stage 1 MVP — one module per method
│       ├── simulator.py       # Batched linear cyclic SCM (torch)
│       ├── loss.py            # moment_matching_discrepancy + L1
│       ├── fit.py             # Manual-SGD fitting loop with
│       │                      # spectral-radius projection
│       └── model.py           # EnsembleSCMFitter (Model-protocol entry point)
├── tests/
│   ├── test_stage0.py
│   └── test_scm_fitter.py
└── scripts/
    ├── run_baseline.py                # Stage 0 comparison
    ├── run_scm_fit.py                 # Stage 1 vs baselines, full perturbation
    ├── eval_partial_perturbation.py   # Stage 1 vs baselines, n_perturbed=25
    │                                  # with precision@k + matched-W1 pivots
    ├── run_candidate_comparison.py    # Per-candidate vs fused diagnostic
    ├── diagnose_divergence.py         # Spectral-radius trajectory tracker
    ├── max_w1_oracle.py               # Oracle ceiling on mean W1
    ├── sweep_spectral_threshold.py    # Threshold ablation
    ├── sweep_l1.py                    # L1 coefficient ablation
    └── sweep_optimization_hparams.py  # Step size × iteration ablation
```

## The fitter

### Public entry point

```python
from grn_inference import EnsembleSCMFitter

fitter = EnsembleSCMFitter(
    top_k=1000,
    n_candidates=5,         # parallel random-init SCMs
    n_steps=1000,           # gradient-descent iterations
    step_size=0.01,
    batch_size=200,         # cells per arm per step
    l1_lambda=1e-4,         # modest sparsity prior
    spectral_threshold=0.80, # hard projection; see below
    aggregation_power=3.0,  # power-mean fusion across candidates
    seed=0,
)
edges = fitter.fit_predict(data)   # list[(source, target)]
```

### How the fit works

1. Initialise an ensemble of N random-init adjacency matrices
   `W ∈ R^{N×G×G}` with small entries (`weight_scale=0.01`).
2. At each of `n_steps`:
   - Sample an arm uniformly from {control} ∪ {perturbed genes}.
   - Simulate `batch_size` cells per candidate from that arm
     via `x = (I − W)⁻¹ ε` (batched solve across candidates).
   - Compute per-candidate moment-matching discrepancy vs real cells.
   - `grad = ∂(Σ_k discrepancy_k + l1_lambda·||W||₁)/∂W` via autograd.
   - Step `W ← W − step_size · grad`.
   - **Project** each `W_k` so its spectral radius stays
     ≤ `spectral_threshold`; without this, `(I − W)⁻¹` blows up within
     a handful of steps and the fit is dead.
3. Aggregate across candidates with a generalised mean
   `score[j, i] = (mean_k |W_k[i, j]|ᵖ)^(1/p)` (biased toward max, not
   pure max), zero diagonal, return top-k edges.

### Why spectral projection matters

Early optimisation iterations push W toward higher spectral radius
(`ρ`) to match the data's variance structure. Near `ρ = 1`, the
amplification `1/(1 − ρ)` is stiff — one step of size 0.01 at
`ρ = 0.9` lands at `ρ ≈ 100`, after which the fit cannot recover. The
projection rescales W back into `ρ ≤ spectral_threshold` after each
step. Default 0.80 matches the synthetic generator's
`target_spectral_radius = 0.8`; the sweet spot on this data is
0.7–0.8. Anything above 0.9 leaves the simulator dramatically
over-amplifying and the moment-matching discrepancy plateaus at a
useless floor.

## Results

### Full-perturbation synthetic (n_genes=50, all perturbed, seed=7)

| top_k | method          | mean W1 | true-edge recovery |
|-------|-----------------|---------|--------------------|
| 100   | Mean Difference | 0.93    | 49%                |
| 100   | Fitter          | 0.60    | 22%                |
| 100   | Random          | 0.25    | 7%                 |
| 1000  | Mean Difference | 0.42    | 21%                |
| 1000  | Fitter          | 0.32    | 12%                |
| 1000  | Random          | 0.26    | 11%                |

**Mean Difference is within ~3% of the theoretical W1 ceiling** (see
`scripts/max_w1_oracle.py`). The metric rewards ranking
large-effect edges, and sorting by `|mean shift|` is a near-oracle.
So the right comparison isn't "beat Mean Difference on W1" — it's
"what does the simulation-based fit do that Mean Difference
structurally cannot?"

### Partial-perturbation synthetic (n_genes=50, n_perturbed=25)

This mirrors real Perturb-seq data: only a fraction of genes have
intervention arms, so true edges with unperturbed sources are
invisible to any method that ranks by intervention effect.

At matched mean W1 = 0.30:

| method          | k | total true hits | **hits from unperturbed source** |
|-----------------|---|-----------------|----------------------------------|
| Mean Difference | 948 | 117 | **0** |
| Fitter          | 1000 | 125 | **90** |

**The fitter recovers 60% (90 / 150) of true edges whose sources
weren't perturbed** at matched W1 quality. Mean Difference recovers
zero — the statistical metric can't score these edges at all.

## The evaluator (unchanged from Stage 0)

`evaluate_statistical(predicted_edges, data)` returns:

- `mean_wasserstein` — mean 1-D Wasserstein-1 distance between
  `P(B | control)` and `P(B | do(A))` over predicted edges
  `A → B`. Higher is better (precision-like). Matches CausalBench.
- `false_omission_rate` — Mann-Whitney-U–based recall estimate over
  sampled non-predicted edges. Lower is better.

On synthetic data, you also get `SyntheticTruth.true_edges`, so
**precision@k** and **per-subgroup recall** (perturbed-source vs
unperturbed-source) are directly computable; the partial-perturbation
script uses all of these.

## Synthetic data

`make_synthetic_dataset(n_genes, n_perturbed_genes=None, ...)`
generates cells from a known linear cyclic SCM:
`x = (I − W)⁻¹ ε`. Interventions are soft knockdowns (zero row t of
W, shift gene t's noise). Pass `n_perturbed_genes` to limit how many
genes get intervention arms — crucial for evaluating methods that
should leverage observational structure, not just intervention shifts.

Because the generative family matches what `EnsembleSCMFitter` fits,
"can the fitter recover the truth on this synthetic?" is a meaningful
end-to-end test — the diagnostic scripts above quantify how close
it gets.

## Plugging in real CausalBench data

`load_causalbench_dataset` in `data_loaders.py` is still a stub — the
docstring walks through the steps. Once it returns a proper `Dataset`,
every evaluator, baseline, and fitter in this repo runs on it
unchanged.

## What's next

With the MVP in place, the highest-leverage follow-ups are:

1. **Real CausalBench data.** Fill the stub; rerun
   `eval_partial_perturbation.py`-style evaluation against Mean
   Difference and other CausalBench entries.
2. **Richer discrepancy** (MMD, Sinkhorn via `geomloss`) — moment
   matching is cheap but capped; higher-order distributional
   structure is now the likely next bottleneck, not stability.
3. **Ensemble diversity.** With the threshold fix, candidates now
   converge to similar basins (Jaccard ≈ 0.92). The ensemble isn't
   buying much over a single fit. Diverse `weight_scale`
   initialisations, or MCMC-style sampling, would help.
4. **Learned-or-scheduled spectral threshold.** Right now 0.80 is a
   tuned constant matching our synthetic data; on real data we don't
   know the true `ρ`.

## Conventions

- Expression: `(n_cells, n_genes)`, normalised (not raw counts).
- Edge direction: `(source, target)` means `source → target`.
- Control label: `"non-targeting"` (exported as
  `grn_inference.CONTROL_LABEL`).
- `W[i, j]` in the SCM means "gene j regulates gene i"
  (parent j → child i). The edge ranker handles the transpose so
  external callers only see `(source, target)`.
- Method-specific code lives in `grn_inference/<method>/`; the
  top-level `models.py` stays reserved for baselines.
- Don't describe gradient-based SCM fitting as "training" / "learning"
  / "ML" — this is parameter estimation for a mechanistic model.
