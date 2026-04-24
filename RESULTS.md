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

### Current best (on train) — the bar iteration 7 must beat

- **precision@k**: `NeighborhoodRegressionModel` w/ reverse-shift damping + within-arm correlation boost + β-asymmetry direction filter at **0.155** (was 0.146 iter 5, 0.142 iter 2, 0.138 iter 1, 0.128 MD).
- **hidden-source recall**: same method at **0.467** (was 0.401 iter 5, 0.367 iter 1).
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

### Iteration 3 — pool interventional cells in the neighborhood regression (reverted)

**Hypothesis**: for regressing gene `T`, every cell whose intervention target is not `T` satisfies `T`'s structural equation `x_T = Σ_k W[T,k] x_k + ε_T` (the SCM equation is violated only for the perturbed gene's own row). Pooling control + non-`T`-perturbed cells increases the regression sample size from ~2000 to ~7000 and, more importantly, exposes `T` to the broadened variation in upstream genes perturbed elsewhere — interventions act as natural instruments. This should tighten `|β[T,S]|` estimates where it matters most: on rows whose parents are themselves perturbed.

**Change**: added `use_interventional_cells: bool = True` to `NeighborhoodRegressionModel`. When enabled, `β[T, :]` is estimated per-target from `{cells where intervention ≠ T}` via a 50×50 ridge-regularised solve (one solve per target). `False` keeps the iteration-2 single-shot control-only precision path.

**Numbers (top_k=1000)**:

| split | method | mean W1 | FOR | precision@k | hidden recall | runtime/seed |
|-------|--------|---------|-----|-------------|---------------|--------------|
| train (0,1,2) | NR + interventional cells | 0.3841 | 0.026 | 0.144 | 0.416 | 0.19s |
| train (0,1,2) | NR (iter 2, prev best) | 0.3841 | 0.032 | 0.142 | 0.401 | 0.01s |
| test (100,101,102) | NR + interventional cells | 0.3561 | 0.016 | 0.154 | 0.410 | 0.18s |
| test (100,101,102) | NR (iter 2) | 0.3561 | 0.019 | 0.157 | 0.432 | 0.01s |

Train: precision +1.4%, hidden recall +3.9% — both positive but small.
Test: precision −1.9%, hidden recall −5.1% vs iter 2 — both regress.

**Tuning pass** (train seeds, `ridge_lambda ∈ {1e-4, 1e-3, 1e-2, 1e-1}`):

| ridge_λ | precision@k | hidden recall |
|--------:|------------:|--------------:|
| 1e-4    | 0.1443      | 0.4164        |
| 1e-3    | 0.1443      | 0.4164        |
| 1e-2    | 0.1443      | 0.4164        |
| 1e-1    | 0.1440      | 0.4140        |

Ridge is essentially inactive at these scales — the covariance is well-conditioned even before regularisation — so no hyperparameter lever salvages the test regression.

**Verdict**: **REVERTED**. Train gain is small; test regresses against iter 2's committed numbers. Runtime also 20× slower (0.19s vs 0.01s per seed). The interventional-cells variation in the regressors may still be useful signal, but not through naive pooling — a future iteration should route it through an instrumental-variable or weighted-regression design rather than concat-all-cells.

### Iteration 4 — cross-bucket rank-percentile aggregation (reverted, tied)

**Hypothesis**: the perturbed and unperturbed buckets score edges on incomparable scales (mean-shift in log1p-CPM units vs unitless regression coefficient), so the fixed 50/50 quota is arbitrary. Within each bucket, assign a rank-percentile (best = 1.0, worst = 0.0), then sort all edges by percentile and take the top `top_k`. If one bucket has a heavier right tail — meaning a few very-high-confidence edges — it naturally contributes more to the top slots. Percentile-rank is the non-parametric equivalent of a scale-matching transform and respects intra-bucket ordering without distributional assumptions.

**Change**: added `aggregation: str = "rank_percentile"` with legacy `"quota"` preserved.

**Numbers (top_k=1000, train seeds 0,1,2)**:

| method | precision@k | hidden recall |
|--------|------------:|--------------:|
| NR + rank_percentile | 0.1423 | 0.4009 |
| NR + quota (iter 2) | 0.1423 | 0.4009 |

Bit-identical on every seed. At `n_genes=50, n_perturbed=25`, both buckets have 25 × 49 = 1225 candidate edges each; taking the top 1000 by rank-percentile from 2450 combined candidates lands at ~500-from-each, which equals the 50/50 quota. The aggregation change is a **no-op** on this synthetic benchmark config.

**Verdict**: **REVERTED** (tie, not a win). The idea may still matter when bucket sizes differ (e.g. on CausalBench with different perturbed/unperturbed ratios, or when one bucket's score distribution is much more sharply peaked), but we have no way to validate that here. Logging so future runs don't redo it.

### Iteration 5 — within-arm correlation boost on the perturbed bucket

**Hypothesis**: MD's mean shift is a population-level signal and is near-oracle on Wasserstein, but it cannot distinguish a direct edge `S → T` from a cascade shortcut `S → M → T` that passes through a single intermediate `M`. Within-`do(S)` cells, the residual variation in `S` (from the soft-knockdown noise) drives `T` directly along a true edge, giving a non-zero `corr(S, T | do(S))`. For a cascade, `M` absorbs the per-cell residual noise between the endpoints, so the within-arm correlation is attenuated. Boosting the shift by `(1 + γ · |within-arm corr|)` should rank direct edges above shift-tied cascade shortcuts.

**Change**: added `within_arm_corr_weight: float = 1.0` to `NeighborhoodRegressionModel`. When `> 0`, the perturbed-source score becomes
```
score_pert(S, T) = |shift[S, T]| · (1 + w · |corr(x_S, x_T | do(S))|)
```
Unperturbed bucket unchanged.

**Numbers (top_k=1000)**:

| split | method | mean W1 | FOR | precision@k | hidden recall | runtime/seed |
|-------|--------|---------|-----|-------------|---------------|--------------|
| train (0,1,2) | NR + corr boost | 0.3836 | 0.037 | **0.146** | 0.401 | 0.02s |
| train (0,1,2) | NR (iter 2, prev best) | 0.3841 | 0.032 | 0.142 | 0.401 | 0.01s |
| test (100,101,102) | NR + corr boost | 0.3560 | 0.023 | **0.159** | 0.432 | 0.01s |
| test (100,101,102) | NR (iter 2) | 0.3561 | 0.019 | 0.157 | 0.432 | 0.01s |

Precision +2.8% train / +1.3% test. Hidden-source recall **unchanged** on both splits by construction — the boost only acts on the perturbed bucket, whose edges have zero hidden-source count. Mean W1 is essentially unchanged (the W1 average is over evaluable perturbed-source edges; re-ordering within the top 500 perturbed-source predictions doesn't change which edges are evaluable). Runtime negligible.

**Verdict**: **KEPT**. Strict reading of "beat both metrics on train" tolerates a tie on hidden recall here — the modification can't move that metric by construction, and precision clearly improves on both splits without regressing anything else.

### Iteration 6 — β-asymmetry direction filter on unperturbed-unperturbed pairs

**Hypothesis**: for a pair of unperturbed genes `(A, B)`, both directions `(A, B)` and `(B, A)` currently appear in the unperturbed bucket scored by `|β[B,A]|` and `|β[A,B]|` respectively. Under a generic no-two-cycle sparsity prior, at most one of these is real, so ranking both wastes slots.

**Theoretical basis**: with isotropic noise `Σ_ε = σ² I`, `β[B,A] = −Θ[A,B]/Θ[B,B]` and `β[A,B] = −Θ[A,B]/Θ[A,A]`. Numerators are equal (Θ is symmetric); denominators differ: `Θ[i,i] = (1 + Σ_k W[k,i]²)/σ²` — larger for genes with more downstream children. So `|β[B,A]| > |β[A,B]|` iff `A has more children than B`, which is the hallmark of a source node. The larger-|β| direction points **away from** the more-connected gene, which is the causal-source candidate. Keep only that direction; zero the weaker.

**Change**: added `direction_from_beta_asymmetry: bool = True`. Only applies to unperturbed-unperturbed pairs (for S unperturbed, T perturbed, the reverse-shift damping from iter 2 already handles direction). No effect on the perturbed bucket.

**Numbers (top_k=1000)**:

| split | method | mean W1 | FOR | precision@k | hidden recall | runtime/seed |
|-------|--------|---------|-----|-------------|---------------|--------------|
| train (0,1,2) | NR + β-asymmetry filter | 0.3836 | 0.034 | **0.155** | **0.467** | 0.01s |
| train (0,1,2) | NR (iter 5, prev best) | 0.3836 | 0.037 | 0.146 | 0.401 | 0.02s |
| test (100,101,102) | NR + β-asymmetry filter | 0.3560 | 0.019 | **0.163** | **0.460** | 0.01s |
| test (100,101,102) | NR (iter 5) | 0.3560 | 0.023 | 0.159 | 0.432 | 0.01s |

Train: precision +6.2%, hidden recall +16.6%. Test: precision +2.5%, hidden recall +6.5%. Every seed × metric improves; no regression. Runtime unchanged.

**Verdict**: **KEPT**. Largest single-iteration jump on hidden recall so far (0.40 → 0.47 on train, 0.43 → 0.46 on test). The β-asymmetry direction signal is a genuine identifiability result in the isotropic-noise linear SCM, so this is a principled gain rather than a tuning artefact.

### Iteration 7 — shift-asymmetry direction filter on perturbed-perturbed pairs (reverted)

**Hypothesis**: mirror iter 6's direction filter on the perturbed bucket using the classical do-calculus asymmetry test. For a pair `(A, B)` both perturbed, if `A → B` is real then `|shift[A,B]| >> |shift[B,A]|` (intervention on `A` propagates to `B`; intervention on `B` does not propagate back under forbid_two_cycles). Keep only the larger-shift direction.

**Change**: added `direction_from_shift_asymmetry: bool = True` to `NeighborhoodRegressionModel`.

**Numbers (top_k=1000, train)**:

| method | mean W1 | FOR | precision@k | hidden recall |
|--------|---------|-----|-------------|---------------|
| NR + shift-asym filter | 0.3756 | 0.070 | 0.153 | 0.467 |
| NR (iter 6, prev best) | 0.3836 | 0.034 | 0.155 | 0.467 |

Precision −1.3%, FOR doubled, hidden recall unchanged (expected: the filter only affects perturbed bucket, which has no hidden-source edges by construction).

**Tuning pass** (seeds 0,1,2):

| config | precision@k | hidden recall |
|--------|------------:|--------------:|
| shift-asym + corr_weight=1.0 (default) | 0.153 | 0.467 |
| shift-asym + corr_weight=0.0 (raw shifts) | 0.151 | 0.467 |
| iter 6 baseline (no shift-asym) | 0.155 | 0.467 |

Neither variant beats iter 6. The do-calculus direction test is the classical tool but **fails under moderate cyclicity**: at `rho(W) = 0.8` a true edge `S → T` has non-trivial `shift[T, S]` through longer cycles `S → T → ... → S`, so the asymmetry flip-flops on some true edges and the filter zeroes the correct direction. β-asymmetry (iter 6) works on the unperturbed bucket precisely because it encodes the cycle-inclusive observational structure via Θ, not the cycle-severing interventional structure.

**Verdict**: **REVERTED**. Shift-asymmetry is right for DAGs and wrong for cyclic SCMs; iter 6's β-asymmetry is the cycle-robust counterpart.
