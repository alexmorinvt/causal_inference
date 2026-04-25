---
name: autostrategy
description: Run the autonomous GRN-inference method-exploration loop on a dedicated branch. Propose a hypothesis, implement it under grn_inference/<method>/, benchmark on partial-perturbation synthetic data across train seeds, validate on test seeds, commit or revert. Loops until the human interrupts.
user-invocable: true
---

# /autostrategy

This skill runs an autonomous loop that iterates on GRN inference methods. The bar to clear each iteration is the **current best committed method on the branch**, measured on the partial-perturbation synthetic benchmark on two metrics: **hidden-source recall** and **precision@k**.

## Setup

Before proposing any hypothesis, **read `TRIED_PATHS.md`** in this skill's directory. It lists method families already explored on prior `autostrategy/*` branches and their saturation points — the skill forbids re-implementing a family under a different name, and the Pareto ceilings already attained are the real bar to clear, not MD alone.

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr25`). The branch `autostrategy/<tag>` must not already exist — this is a fresh run.
2. **Ensure the working tree is clean**: run `git status` and confirm no uncommitted changes or untracked files that belong elsewhere. Stash or commit outstanding work on a different branch first. If unsure, confirm with the user before branching.
3. **Create the branch**: `git checkout -b autostrategy/<tag>` from a clean `main`.
4. **Read the in-scope files** for full context before writing a single line of code:
   - `CLAUDE.md` — project framing, layout rule, vocabulary discipline.
   - `grn_inference/dataset.py` — the `Dataset` dataclass.
   - `grn_inference/data_loaders.py` — `make_synthetic_dataset` and `SyntheticTruth`.
   - `grn_inference/evaluator.py` — `evaluate_statistical`, Wasserstein + FOR.
   - `grn_inference/models.py` — `Model` protocol, `MeanDifferenceModel`, `RandomBaseline`.
   - `grn_inference/ensemble_scm/` — simulation-based MVP, full `simulator.py / loss.py / fit.py / model.py / __init__.py` layout.
   - `grn_inference/indirect_pruning/`, `grn_inference/shift_corr/`, `grn_inference/shift_paths/` — non-simulation heuristic methods, minimal `__init__.py + model.py` layout.
   - `scripts/eval_partial_perturbation.py` — the canonical benchmark script.
5. **Verify environment**: `conda activate grn-inference && pytest tests/ -v` — all tests pass.
6. **Extend the benchmark script**: `scripts/eval_partial_perturbation.py` is single-seed and does not directly report the headline metric we need. Before any iteration begins, modify it so that it:
   - Accepts a list of data seeds (e.g. `--seeds 0 1 2`) and loops over them, re-running each method per seed.
   - For each method, reports **hidden-source recall** = `hits_unpert / len(true_edges_with_unperturbed_src)` (this is NOT the same as the existing `precision (unpert)` = `hits_unpert / n_unpert` column — keep both).
   - At `top_k = 1000`, emits a JSON summary block at the end of stdout (one line, parseable) with per-seed values and seed-mean of: `mean_w1`, `for`, `precision_at_k`, `hidden_source_recall`, `runtime_s` per method. This is the machine-readable ground truth for `results.tsv`.
   - Includes only the baseline methods in the `methods` dict: `MeanDifferenceModel` and `RandomBaseline`. Other in-tree method directories (`ensemble_scm/`, `indirect_pruning/`, `shift_corr/`, `shift_paths/`) are historical experiments, not baselines, and are not auto-included. When a new method is implemented during an iteration, add it to the dict for comparison against the baselines.
   - Preserves all existing output — additive only.
   
   This extension is a one-time setup task, independent of any particular new method. Commit it before the first iteration.
7. **Record pre-iteration-0 baselines**: run the extended benchmark on train seeds `[0, 1, 2]` and then on test seeds `[100, 101, 102]` with the baseline methods (MD + Random) only. Append one row per (method × split) to `results.tsv` and `RESULTS.md`. Record in `RESULTS.md` which baseline currently holds the top number on each headline metric on the **train** split — that is the "current best" the first iteration must beat.
8. **Initialize `results.tsv` and `RESULTS.md`**: create `results.tsv` with the header row (schema below). Create `RESULTS.md` with a header section and the baseline rows from step 7. `RESULTS.md` is committed; `results.tsv` is not.
9. **Confirm and go**: confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

**What you CAN do:**
- Add new fitter modules under `grn_inference/<method>/`. For simulation-based methods use CLAUDE.md's `simulator.py / loss.py / fit.py / model.py / __init__.py` template (as in `ensemble_scm/`). Non-simulation methods (as in `indirect_pruning/`, `shift_corr/`, `shift_paths/`) may use the minimal `model.py + __init__.py` layout. `model.py` implementing the `Model` protocol and `__init__.py` re-exporting the entry-point class are always required.
- Add a `tests/test_<method>.py` file that exercises the new method at minimum — a smoke test that `fit_predict` returns a valid ranked edge list on a tiny synthetic dataset, and any fitter-specific invariants (e.g. spectral radius below threshold after fit).
- Re-export the new method's entry-point class from `grn_inference/__init__.py`.
- Modify internals of any in-tree method family (`ensemble_scm/`, `indirect_pruning/`, `shift_corr/`, `shift_paths/`) if iterating on it.
- Add new diagnostic scripts under `scripts/`.
- Add a new method to the `methods` dict in the extended `scripts/eval_partial_perturbation.py` — but do not change what it measures.

**What you CANNOT do:**
- Modify `grn_inference/evaluator.py` — this is the fixed scoring harness.
- Modify `grn_inference/dataset.py` (the `Dataset` interface) or `grn_inference/data_loaders.py` (`make_synthetic_dataset` / `SyntheticTruth`). The synthetic generator is the test substrate; methods must work against it without rewriting it.
- Add new methods to `grn_inference/models.py` — that file is reserved for baselines.
- Use `SyntheticTruth` (true_edges, W) inside any `fit_predict`. Ground truth is for the evaluator and diagnostics only.
- Run a method that takes longer than **10 minutes wall-clock per seed** on the canonical benchmark config (`n_genes=50, n_perturbed=25, top_k=1000`). Across three train seeds that's a ~30-minute budget per iteration. If a method exceeds the per-seed budget, either fix it or revert.
- Copy an existing method — neither an in-tree one (the `MeanDifferenceModel` / `RandomBaseline` baselines, nor the historical experimental families `EnsembleSCMFitter`, `IndirectPruningModel`, `ShiftCorrModel`, `ShiftPathsModel`) nor a published method reimplemented under a new name. Each new `grn_inference/<method>/` subdirectory must be a genuinely distinct estimator class (distinct loss / parameterisation / identification argument), not a rename, a light refactor, or a thin wrapper. Extending an existing method family in place is fine; forking it into a new subdirectory that differs only superficially is not.

**The goal**: beat the **current best committed method on the branch** on the partial-perturbation benchmark (`top_k = 1000`, averaged over the three train seeds) on **both** headline metrics:
- **hidden-source recall** — fraction of true edges whose source is *not* in the intervention set that the method recovers in its top_k. `MeanDifferenceModel` is at zero by construction. The pre-iteration-0 baseline row in `RESULTS.md` establishes the exact starting numbers for MD + Random.
- **precision@k** — fraction of the top_k that is a true edge.

W1 is near-saturated by MD (within ~3% of the oracle ceiling per `scripts/max_w1_oracle.py`) and is not the differentiator. Don't chase it; don't let a method drop below `RandomBaseline` on it either. Note that `evaluate_statistical` averages W1 only over *evaluable* (perturbed-source) edges, so methods that predict many unperturbed-source edges have their W1 computed over a smaller subset — treat the W1 floor as a sanity check, not a ranking signal.

## Running a method

Every run uses the extended partial-perturbation benchmark:

```bash
# Smoke test — must pass before any benchmark run
pytest tests/ -v

# Canonical benchmark on train seeds (the bar to clear)
python scripts/eval_partial_perturbation.py --seeds 0 1 2

# Only if train seed results beat the current best — validate on test seeds
python scripts/eval_partial_perturbation.py --seeds 100 101 102
```

Extract the per-method numbers from the JSON summary block the extended script emits.

**Train/test discipline**: tune on train seeds `[0, 1, 2]`; validate on test seeds `[100, 101, 102]`. Run each method on every seed in the active split and report per-seed numbers plus the mean. **Never tune to the test seeds.** A method that wins on train but loses on test is overfitting to the stochastic dynamics of the fit (or to the specific graphs at those seeds), not producing an estimator that survives redraws of the data-generating process.

CausalBench is currently a stub (`load_causalbench_dataset` raises `NotImplementedError`). It will become a second test substrate later; methods that overfit to the synthetic generator's idiosyncrasies will fail obviously at that point. Design accordingly.

## Sanity checks

Every method change must pass, on **one train seed**, before being benchmarked across all seeds:

1. `pytest tests/ -v` — no regressions. Any new method should have its own smoke test in `tests/test_<method>.py`.
2. The method runs on the canonical config (`n_genes=50, n_perturbed=25, top_k=1000`) in **under 10 minutes** wall-clock.
3. Mean W1 on evaluable edges ≥ `RandomBaseline`'s mean W1 on the same seed. A method that scores worse than random on W1 is broken. (Caveat: denominators differ across methods — see the note above — but a gross violation indicates the ranker is broken.)
4. For any iterative fitter, a stability check: numerical values stay finite, and (for linear-SCM-based methods) the spectral radius of the fitted W stays below 1. See `scripts/diagnose_divergence.py` for the pattern.

**Do not sacrifice stability for a single-metric win.** A method that produces NaNs on one of the test seeds is disqualified, regardless of train numbers.

## Integrity Constraints

### No ground-truth leakage
- Fitters receive only `Dataset` (`expression`, `interventions`, `gene_names`).
- `SyntheticTruth` and the synthetic generator's internal `W` are accessible only to the evaluator and diagnostic scripts. Any `fit_predict` that reads them — directly, via globals, via a side-channel file — is disqualified.
- Hyperparameter selection against test seeds counts as leakage. Tune on train seeds, evaluate on test seeds, and that's it.

### Causal-inference framing (not ML)
- This is parameter estimation for an SCM, not predictive modelling. Follow the vocabulary rule in `CLAUDE.md`: "fitting procedure" not "training", "step size" not "learning rate", "SCM fitter / edge ranker" not "model" in the predictive sense.
- The seed split exists to guard against overfitting to fit-procedure stochasticity and graph-specific quirks, not to estimate held-out predictive loss. Don't import ML concepts that don't apply.

### Theoretical basis required
- Every change must have a clear theoretical justification rooted in causal inference, statistics, optimisation theory, or the structure of the linear SCM data-generating process.
- "It improved the benchmark" is not sufficient. Explain *why* it should work.
- Trying a well-reasoned hypothesis and failing is better than trying an ad-hoc tweak and succeeding. Ad-hoc successes are likely overfitting.
- Examples of good changes: a regulariser derived from a known identifiability condition, a moment-matching term that targets a property the current loss ignores, a stability projection grounded in the spectral-radius condition, an aggregation rule motivated by stability selection.
- Examples of bad changes: arbitrary multipliers, magic constants tuned to a single seed, thresholds without a theoretical role, hardcoded edge counts.

### No overfitting to the synthetic generator
- `make_synthetic_dataset` has specific idiosyncrasies (soft-knockdown intervention model, `target_spectral_radius=0.8`, tree-cascade lineage structure, log1p(CPM/10k) normalisation, particular edge-density distribution). Methods must not exploit these.
- A change that helps under one specific generator setting (specific `n_genes`, density, or seed) and not others is a red flag — it won't survive the move to CausalBench.
- Distinction that matters: encoding a *general* stability or identifiability condition that happens to align with the generator is fine (e.g. spectral radius < 1 is a real SCM stability requirement; the MVP's spectral projection at 0.80 is on the good side of the line because the value is a hyperparameter, not a magic match). What's forbidden is hardcoding the generator's specific numeric value as a constant, or tuning to a seed-specific quantity.

### Edge-ranking discipline
- `fit_predict` returns a ranked list of `(source, target)` tuples cut to `top_k`. The evaluator does not re-rank.
- `(source, target)` means `source → target`. Inside fitters, `W[i, j]` means `j → i` — the ranker transposes so external callers only see source→target.

## Logging results

When an experiment is done, append rows to `results.tsv` (tab-separated, one row per split):

```
commit	method	config	split	seeds	top_k	mean_w1	for	precision_at_k	hidden_source_recall	runtime_s_per_seed	description
a1b2c3d	ensemble_scm	default	train	0,1,2	1000	0.18	0.42	0.35	0.60	80	baseline EnsembleSCMFitter
```

Do NOT commit `results.tsv` — leave it untracked.

After each experiment, append a row to `RESULTS.md` — a human-readable table that includes the hypothesis, the change, train + test numbers for both headline metrics, the per-seed runtime, and the verdict (kept / reverted / pivoted). Commit `RESULTS.md` with each experiment commit.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autostrategy/apr24`).

LOOP FOREVER:

1. **Check state**: read `results.tsv` and `RESULTS.md` to recall what has been tried and what the current best numbers are.
2. **Form a hypothesis**. Areas to explore:
   - **Fitting objective**: higher moments, cross-moments, quantile matching, score matching, MMD, identifiable likelihood-based losses.
   - **Regularisation / priors**: structured sparsity (group lasso), low-rank+sparse decomposition of W, stability-margin penalties. (DAG-promoting penalties like NOTEARS `h(W) = tr(e^{W∘W}) - d` are available but fight the domain — GRNs have real feedback loops — so use them only as an ablation to quantify cyclicity's role, not as the main estimator.)
   - **Parameterisation of W**: factorised via latent factors, constrained to motif vocabularies, sign-constrained.
   - **Intervention model**: better treatment of soft vs hard knockdowns, learning per-gene intervention strength jointly with W.
   - **Aggregation across an ensemble**: rank aggregation, stability selection, bootstrap-bagging, instead of (or in addition to) the generalised-mean rule.
   - **Use of observational structure**: methods that exploit observational covariance to score edges from unperturbed sources — this is the lever for hidden-source recall.
   - **Path-based reasoning**: extend the `shift_paths` / `indirect_pruning` ideas with better path scoring, cascade-aware pruning, or principled two-hop vs direct disambiguation.
   - **Hybrid approaches**: warm-start an SCM fit from MD scores; use MD as a prior on W's row-supports.
   - **Non-linear extensions**: kernelised or neural-residual SCMs, with identifiability constraints — only if runtime budget allows.
3. **Implement the idea** — typically a new `grn_inference/<method>/` subdirectory, plus a `tests/test_<method>.py`. Re-export from `grn_inference/__init__.py`. Keep changes small and testable.
4. **Integrity check**: does this fitter touch any data it shouldn't (`SyntheticTruth`, the generator's W, test seeds)? If yes, redesign before going further.
5. **Run sanity checks** on one train seed (tests pass, runtime under 10 min, W1 floor, stability). Fix any failure before proceeding. Nothing is committed yet.
6. **Run the full benchmark on train seeds** `[0, 1, 2]` via the extended script. Parse the JSON summary.
7. **Decide whether to validate on test**:
   - If train means beat the current best on **both** headline metrics → proceed to step 8.
   - Else → discard the working-tree changes (`git restore .` for modifications, `git clean -fd grn_inference/<method>/` for new dirs), log the attempt as "reverted" in `RESULTS.md` with the train numbers, and go to step 11.
8. **Run the benchmark on test seeds** `[100, 101, 102]`. Parse the JSON summary.
9. **Decide whether to keep**:
   - Beats current best on **train** AND holds on **test** for both headline metrics (test numbers within 10% of train on each metric, and still above the baseline) → keep.
   - Else → discard the working-tree changes as in step 7, log as "reverted (test regression)" in `RESULTS.md`, and go to step 11.
10. **Commit the kept change**: `git add` the new files + `RESULTS.md` update, commit with a message describing the hypothesis and the numbers. Append the row to `results.tsv`. Update the "current best" pointers in `RESULTS.md`.
11. Repeat.

**Give conceptual changes a tuning pass**: when a new conceptual change (a new loss term, a new parameterisation, a new aggregation rule — anything beyond a single-parameter tweak) is introduced, do not immediately discard on a small regression vs the committed baseline. A new piece of logic at arbitrary default hyperparameters rarely shows its best self. Before fully discarding at step 7 or 9, spend 2–3 follow-up attempts tuning its hyperparameters on train (regularisation weights, ensemble size, step size, projection thresholds). Only discard permanently if after tuning it still fails the train/test rule, or if the regression is large enough that it clearly isn't a tuning problem (e.g. >20% worse on either headline metric on either split). Log the conceptual change + its tuning sweep under one method-version in `RESULTS.md` so the rationale is preserved, even the reverted attempts.

**Pivot rule**: count iterations per method family. If **20** consecutive iterations on the current method family have not produced any new record on either headline metric vs the current best, pivot to an entirely different method family — not a tweak, a distinct estimator class (e.g. moment-matching → likelihood-based → score-matching → spectral / algebraic). Log the pivot in `RESULTS.md` with the best numbers attained so future runs know the ceiling that family hit.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human may be asleep. You are autonomous. If you run out of obvious ideas, re-read the in-tree method code, study the ranked output vs ground truth on a few seeds to find failure patterns, read the identifiability literature on linear cyclic SCMs from interventional data, or scrutinise the matched-W1 pivot block to understand where each method's edge ordering breaks down. The loop runs until the human interrupts you, period.
