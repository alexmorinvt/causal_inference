# Previously explored method families

Summary of families that have already been tried in prior `/autostrategy` runs. **Read this before proposing a hypothesis.** Do not re-implement any family below under a different name — the skill forbids that ("Each new subdirectory must be a genuinely distinct estimator class, not a rename"). Picking an identification argument that differs from all of these is the main guard against wasted iterations.

The best numbers below are on the canonical partial-perturbation benchmark: `n_genes=50, n_perturbed=25, n_control_cells=2000, n_cells_per_perturbation=200, top_k=1000`, seed-mean over train `[0, 1, 2]` and test `[100, 101, 102]`.

## Family summary

| family | identification argument | best numbers (train prec / hidden, test prec / hidden) |
|---|---|---|
| **NR** (neighborhood regression) | observational precision matrix `Θ = Σ⁻¹`; regression of gene T on rest in control cells; `β_{T,S} = −Θ[T,S]/Θ[T,T]` for unperturbed sources, MD shift for perturbed | 0.164 / 0.736, 0.163 / 0.675 |
| **PI** (path inversion) | algebraic inversion of total-effect matrix: `W = T(I+T)⁻¹`; `T[i,j] = shift[j,i]` for perturbed sources, IV-shift-regression imputation for unperturbed | 0.160 / 0.568, 0.165 / 0.582 |
| **DC** (differential covariance) | intervention sensitivity of pairwise covariance: `diff[i,j] = mean_G |Σ_ctrl[i,j] − Σ_G[i,j]|`, directed by shift/β asymmetry | 0.145 / 0.632, 0.142 / 0.611 |
| **DT** (dominator tree) | Lengauer-Tarjan immediate dominators on thresholded shift graph: rooted at each gene, each reachable node's idom is its candidate direct parent | 0.157 / 0.514, 0.164 / 0.515 |
| **RA** (rank aggregation) | ensemble of NR+PI+DC+DT rankings via Borda-style rank-percentile sum | 0.165 / 0.634, **0.172** / 0.650 ← best test precision |
| **ICP** (invariant causal prediction) | Peters-Bühlmann-Meinshausen invariance: true direct edge has invariant regression coefficient across interventional arms; multivariate per-arm β | 0.142 / **0.781**, 0.146 / **0.798** ← best hidden recall |
| **LMLE** (likelihood MLE — failed) | direct MLE of W via gradient descent on multi-arm Gaussian likelihood with sufficient-statistics per arm | 0.128 / 0.481, 0.129 / 0.472 — failed MD baseline on test precision; implementation omits do-arm mean term |

## Key observations (use these to guide apr25)

1. **Hidden-recall ceiling**: ICP iter 35 at 0.78 train / 0.80 test is the current ceiling. Bootstrap pushed it to 0.83 / 0.85 at the cost of 2-5% precision (Pareto trade, not a beat). Any new family that doesn't beat 0.78 train hidden is not competitive on this axis.

2. **Precision ceiling**: RA iter 31 at 0.172 test precision is the current ceiling. A precision-focused compositional idea (icp² × nr) reached train 0.167 (highest single-metric train precision on the branch). The raw precision wall seems close — MD's ceiling is ~0.13 and the best we've seen is ~0.19 at extreme sparsity.

3. **MD baseline is hard to beat on precision alone**: many new methods can't clear MD's test precision (0.131). LMLE is the canonical example — proper implementation would need a full mean-term likelihood.

4. **Saturation patterns within a family**: NR, DC, DT all saturated after 3-5 iterations of tuning. The big wins came from *conceptual* changes (new signal or new aggregation), not hyperparameter sweeps. Don't iterate 15× on one knob.

5. **Bootstrap helps hidden recall, hurts precision**: across NR (iter 11), DC (explored), ICP (iter 42 exploration) — bootstrap trades are consistent. Pareto-optimal for the high-hidden corner but not a clean win.

6. **Product aggregation is too aggressive at top_k=top_k**: RA iter 41 showed product ensemble with `base_top_k=top_k` gives train prec 0.196 (+19% over sum) but hidden collapses to 0.327. Useful as a precision knob, not a default.

## What apr25 should probably try next (unexplored)

Ideas that were *not* thoroughly explored on apr24:

- **Proper LMLE with mean-term likelihood**: the do-arm mean `μ_G(W) = (I − W̃_G)⁻¹ W̃_G[:, G] · c_G` was omitted, which threw away interventional identifiability info. A correct multi-arm Gaussian MLE should beat observational-only identifiability.
- **Non-linear / kernel extensions**: synthetic data is linear but CausalBench (future) won't be. Kernel ridge ICP, GAM-per-target with cross-environment invariance, etc.
- **Score matching with explicit non-Gaussianity**: needs non-Gaussian noise to add over NR; generator's Gaussian so gains limited on current substrate but may matter for CausalBench.
- **Conditional Independence-based pruning (PC/IC)**: classical causal discovery via CI tests. Not tried because of conditional-set-search complexity, but at G=50 a greedy forward-selection version is tractable.
- **Causal Additive Models (Bühlmann et al.)**: non-linear structural equation regression, with identifiability via additive noise model (ANM) theorems. Needs non-Gaussian noise to be unique though.
- **Latent-variable / confounder-aware identification**: assume latent confounders exist, use overcomplete-ICA-like methods. Probably overkill for synthetic, could matter for real data.
- **Interventional 2SLS / IV estimator with carefully chosen instruments**: NR iter 13 and PI iter 15 use IV shift regressions but only for imputing unperturbed columns. A full IV-based fit of W over the multi-arm joint distribution is a distinct estimator.
- **Random-walk / diffusion / PageRank on shift graph**: mentioned as promising but not implemented — ranking by stationary distribution of a Markov chain on shifts would be graph-theoretic but different from DT.

## What apr25 should NOT redo

- Don't re-invent NR, PI, DC, DT, RA, ICP, or LMLE under a different name. They are all in `autostrategy/apr24` (branch preserved for reference).
- Don't spend iterations sweeping `unperturbed_fraction`, `spectral_target`, `ridge`, `shift_boost_power`, `n_bootstrap` — all mapped on apr24, saturation point known.
- Don't combine existing families via rank-percentile aggregation — already done by RA.
- Don't attempt shift-asymmetry direction filters in cyclic graphs — iter 7 showed this fails at `rho=0.8` (cycles leak reverse-shift through longer paths).

## Cross-reference

The full apr24 `RESULTS.md` (42 iteration entries with numbers and verdicts) lives on branch `autostrategy/apr24`. Pull it when digging into a specific family's saturation pattern.
