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

### Current best (on train) — the bar iteration 3 must beat

- **precision@k**: `NeighborhoodRegressionModel` w/ reverse-shift damping at **0.142** (was 0.138 plain NR, was 0.128 from MD).
- **hidden-source recall**: `NeighborhoodRegressionModel` w/ reverse-shift damping at **0.401** (was 0.367 plain NR).
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

### Iteration 2 — reverse-shift damping for the unperturbed-source bucket

**Hypothesis**: for a candidate edge `(S, T)` with `S` unperturbed and `T` perturbed, we directly observe `shift[T, S] = |E[S | do(T)] − E[S | control]|`. A large reverse shift is evidence that `T` is a causal ancestor of `S`. Under a generic no-two-cycle sparsity prior (enforced by the generator, and also the default assumption in most GRN modelling), if `T → S` is in the graph then `S → T` is not — so the candidate `S → T` score should be suppressed when the reverse shift is large. When `T` is unperturbed we have no shift information and the score is unchanged. This exploits interventional data to direction-disambiguate the observational regression signal — the regression alone cannot tell `S → T` from `T → S`.

**Change**: added `reverse_shift_damping: bool = True` to `NeighborhoodRegressionModel`. When enabled, the unperturbed-source score becomes
```
score(S, T) = |β[T, S]| · exp(-shift[T, S] / scale)   if T perturbed
            = |β[T, S]|                                if T unperturbed
```
with `scale` = mean of non-zero shift entries (data-adaptive, no free hyperparameter). Setting the flag to `False` recovers iteration-1 scoring bit-identically.

**Numbers (top_k=1000)**:

| split | method | mean W1 | FOR | precision@k | hidden recall | runtime/seed |
|-------|--------|---------|-----|-------------|---------------|--------------|
| train (0,1,2) | NR + reverse-shift damping | 0.3841 | 0.032 | **0.142** | **0.401** | 0.01s |
| train (0,1,2) | NR plain (iter 1, prev best) | 0.3841 | 0.028 | 0.138 | 0.367 | 0.01s |
| test (100,101,102) | NR + reverse-shift damping | 0.3561 | 0.019 | **0.157** | **0.432** | 0.01s |
| test (100,101,102) | NR plain (iter 1) | 0.3561 | 0.015 | 0.155 | 0.415 | 0.01s |

Both headline metrics improve on both splits (precision: +3.3% train, +1.4% test; hidden recall: +9.3% train, +4.1% test). Mean W1 is unchanged because the perturbed-source bucket and W1-denominators (evaluable perturbed-source edges) are untouched.

**Verdict**: **KEPT**.
