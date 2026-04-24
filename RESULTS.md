# autostrategy/apr24 — iteration log

Partial-perturbation synthetic benchmark: `n_genes=50`, `n_perturbed=25`, `top_k=1000`.
Train seeds: `[0, 1, 2]`. Test seeds: `[100, 101, 102]`. Both headline metrics are seed-means at `top_k=1000`.

Benchmark harness: `scripts/eval_partial_perturbation.py` (commit `df7b1c5`).

## Headline metrics

- **hidden-source recall** (↑): fraction of true edges whose source is unperturbed that the method recovered in its top-`k`. Both baselines sit at 0 by construction — MD can't score such edges, and `RandomBaseline` draws sources only from the perturbed set.
- **precision@k** (↑): fraction of top-`k` that is a true edge.

Sanity floor: mean W1 on evaluable edges must not drop below `RandomBaseline`'s on the same split. FOR is reported for context.

## Pre-iteration-0 baselines

### Train (seeds 0, 1, 2)

| method              | mean W1 | FOR   | precision@k | hidden-source recall | runtime/seed |
|---------------------|---------|-------|-------------|----------------------|--------------|
| MeanDifferenceModel | 0.2741  | 0.000 | **0.128**   | 0.000                | 0.01s        |
| RandomBaseline      | 0.2457  | 0.353 | 0.116       | 0.000                | 0.08s        |

### Test (seeds 100, 101, 102)

| method              | mean W1 | FOR   | precision@k | hidden-source recall | runtime/seed |
|---------------------|---------|-------|-------------|----------------------|--------------|
| MeanDifferenceModel | 0.2571  | 0.000 | 0.131       | 0.000                | 0.01s        |
| RandomBaseline      | 0.2309  | 0.357 | 0.118       | 0.000                | 0.08s        |

### Current best (on train) — the bar iteration 2 must beat

- **precision@k**: `NeighborhoodRegressionModel` at **0.138** (was 0.128 from MD).
- **hidden-source recall**: `NeighborhoodRegressionModel` at **0.367** (was 0.000 tied).
- W1 sanity floor: must stay above `RandomBaseline`'s 0.2457.

## Iteration log

### Iteration 1 — NeighborhoodRegressionModel (hybrid shift + precision-matrix regression)

**Hypothesis**: the baselines hit the 0 floor on hidden-source recall because neither produces edges whose source is unperturbed — `MeanDifferenceModel` needs an intervention arm on the source, and `RandomBaseline`'s pool is restricted to perturbed sources. For a linear cyclic SCM `x = Wx + ε`, the control-cell precision matrix `Θ = Σ⁻¹` encodes conditional linear dependence; the OLS regression coefficient of gene `S` when predicting gene `T` is `β_{T,S} = -Θ_{T,S}/Θ_{T,T}`, a magnitude signal for a directed candidate `(S, T)` that does not require intervening on `S`.

**Change**: new method `grn_inference/neighborhood_regression/NeighborhoodRegressionModel`.
- Perturbed-source bucket: `|shift(A, B)|` (identical to MD).
- Unperturbed-source bucket: `|β_{B,A}|` from control-cell ridge-regularised precision.
- Two buckets ranked independently, concatenated with `unperturbed_fraction=0.5` quota split (matches the uniform-source prior at half the genes perturbed).
- Ridge `λ=1e-4` on the covariance diagonal for numerical stability.

**Numbers (top_k=1000)**:

| split | method | mean W1 | FOR | precision@k | hidden recall | runtime/seed |
|-------|--------|---------|-----|-------------|---------------|--------------|
| train (0,1,2) | NeighborhoodRegressionModel | 0.3841 | 0.028 | **0.138** | **0.367** | 0.01s |
| train (0,1,2) | MeanDifferenceModel (prev best) | 0.2741 | 0.000 | 0.128 | 0.000 | 0.01s |
| test (100,101,102) | NeighborhoodRegressionModel | 0.3561 | 0.015 | **0.155** | **0.415** | 0.01s |
| test (100,101,102) | MeanDifferenceModel | 0.2571 | 0.000 | 0.131 | 0.000 | 0.01s |

Per-seed precision@k (NR vs MD): train `0.134/0.139`, `0.127/0.123`, `0.152/0.121`; test `0.154/0.137`, `0.151/0.130`, `0.159/0.126`. NR loses marginally on one train seed (seed 0, −3.6%) and wins on every other split × seed. Mean W1 stays well above Random's 0.2457 / 0.2309 floor on both splits.

**Verdict**: **KEPT**. Beats MD on both headline metrics on both splits; test numbers run slightly above train (no overfitting signal).
