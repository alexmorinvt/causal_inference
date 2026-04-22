# grn_inference — Stage 0

Scaffold for a simulation-based gene regulatory network inference
project targeting the CausalBench benchmark (Chevalley et al. 2025).

This is **Stage 0** — the foundation. Before modelling anything, we
need:

- A dataset type everything downstream consumes
- A statistical evaluator that matches CausalBench's
- Baselines (Mean Difference, Random) to validate the evaluator
- Synthetic data with a known causal graph so we can test without
  downloading anything

Stage 1 (the linear cyclic SEM + Sinkhorn model) will be added on top
of this without touching the existing code.

## Setup (conda — recommended)

```bash
# One-time: create the environment and install this package in it.
conda env create -f environment.yml
conda activate grn-inference
```

That's it. `environment.yml` has a `pip: -e .` line at the bottom so
this package itself is installed in editable mode automatically —
changes to `grn_inference/*.py` are picked up immediately.

If you prefer `mamba` or `micromamba` (faster solver), substitute:

```bash
mamba env create -f environment.yml
```

### Updating the environment after editing `environment.yml`

```bash
conda env update -f environment.yml --prune
```

### Adding Stage 1 deps (when ready)

Stage 0 is CPU-only and intentionally minimal. When you start on the
simulation-based model you'll need PyTorch and `geomloss`:

```bash
# CPU-only (matches your current hardware)
conda install -n grn-inference -c pytorch -c conda-forge pytorch cpuonly
conda run -n grn-inference pip install geomloss

# Later, on a GPU machine — use the generator at pytorch.org to get
# the exact line for your CUDA version. Something like:
# conda install -n grn-inference -c pytorch -c nvidia pytorch pytorch-cuda=12.1
```

`geomloss` is only on PyPI, so it goes through `pip` inside the conda
env.

### Adding CausalBench (when ready for real data)

```bash
conda run -n grn-inference pip install causalbench
```

Then populate the stub in `grn_inference/data_loaders.py` — the
docstring there walks you through it.

## Setup (pip — alternative)

If you don't want conda:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Verify Stage 0 works

```bash
pytest -v -s
```

The load-bearing test is
`test_mean_difference_beats_random_on_synthetic` — if that passes,
your evaluator is behaving and you're cleared to build Stage 1 on it.
If it fails, stop and debug; everything else depends on this.

Then run the comparison script:

```bash
python scripts/run_baseline.py
```

You should see Mean Difference clearly winning on mean Wasserstein at
every `top_k`, and its FOR should be no worse than Random's. Expected
numbers on the synthetic data (seed=7, n_genes=50):

| top_k | method          | mean W1 | FOR    |
|-------|-----------------|---------|--------|
| 100   | Mean Difference | ~0.93   | ~0.21  |
| 100   | Random          | ~0.25   | ~0.23  |
| 500   | Mean Difference | ~0.58   | ~0.09  |
| 500   | Random          | ~0.24   | ~0.29  |
| 1000  | Mean Difference | ~0.42   | ~0.004 |
| 1000  | Random          | ~0.26   | ~0.24  |

## What's here

```
grn_inference/
├── environment.yml      # Conda environment spec
├── pyproject.toml       # Package metadata (editable install target)
├── grn_inference/
│   ├── __init__.py      # Public API
│   ├── dataset.py       # Dataset class
│   ├── data_loaders.py  # make_synthetic_dataset + CausalBench stub
│   ├── evaluator.py     # Wasserstein + FOR (the key file)
│   └── models.py        # MeanDifferenceModel + RandomBaseline
├── tests/
│   └── test_stage0.py   # Everything above, tested end-to-end
└── scripts/
    └── run_baseline.py  # Reproducible comparison on synthetic data
```

### The evaluator

`evaluate_statistical(predicted_edges, data)` returns:

- `mean_wasserstein` — mean over predicted edges of the 1-D Wasserstein
  distance between `P(B | control)` and `P(B | do(A))` for each
  predicted `A -> B`. Higher is better (precision-like).
- `false_omission_rate` — estimated via Mann-Whitney U tests on
  `omission_sample_size` non-predicted edges. Fraction of sampled
  omissions that have a statistically significant effect = fraction
  of real interactions missed. Lower is better (recall-like).

Both metrics match the CausalBench paper's definitions. The paper uses
`omission_sample_size=500` by default.

### The synthetic dataset

`make_synthetic_dataset(n_genes, ...)` generates cells from a known
linear cyclic SCM: `x = (I - W)^{-1} epsilon`. Interventions are soft
knockdowns that zero out the target's row in `W` and replace its
emission with a knockdown-scaled noise — intentionally close to how
CRISPRi data behaves. Returns `(Dataset, SyntheticTruth)` where
`SyntheticTruth.true_edges` is the ground-truth edge list.

Crucially, the generative process is the same family your Stage 1
model will parameterise. So "can my Stage 1 model recover the truth
on this synthetic data?" is a meaningful yes/no sanity check, not a
toy comparison.

## What's next (Stage 1)

1. Add `grn_inference/simulators.py` — differentiable linear cyclic
   SEM (PyTorch).
2. Add `grn_inference/losses.py` — Sinkhorn divergence loss via
   `geomloss`, batched per perturbation target.
3. Add `grn_inference/train.py` — training loop, warm-start from
   Mean Difference scores, spectral-radius + L1 regularisation.
4. First experiment: can the trained model recover
   `SyntheticTruth.W` from `make_synthetic_dataset(n_genes=50)`?

## Plugging in real CausalBench data

`load_causalbench_dataset` is a stub with detailed TODOs in its
docstring. Once you've installed `causalbench` (see the Stage-1-deps
section above), fill in the import/adapter code by inspecting
`causalscbench/data_access/` in your installed version. Once the
adapter returns a proper `Dataset`, everything else in this repo
works on it unchanged — that's the point of keeping the evaluator
and models behind the `Dataset` interface.

## Conventions

- Expression: `(n_cells, n_genes)`, normalised (not raw counts).
- Edge direction: `(source, target)` means `source -> target`.
- Control label: `"non-targeting"` (see `dataset.CONTROL_LABEL`).
- Gene names: case-sensitive strings matching the data source.