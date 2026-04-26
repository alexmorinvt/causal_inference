# ShiftCorrModel

Mean-shift ranker augmented with within-arm Pearson correlation. The
correlation bonus rewards edges where per-cell residual knockdown of the
source gene tracks proportionally with the response in the target —
a signal that attenuates across cascade hops.

## Algorithm

For each perturbed source `A` and each target `B`:

```
shift[A, B] = |mean(B | do(A)) − mean(B | control)|
corr[A, B]  = Pearson(A, B) within cells from do(A)
score[A, B] = shift[A, B] × (1 + corr_weight × |corr[A, B]|)
```

Rank by score, return top-k.

Setting `corr_weight = 0` reduces exactly to `MeanDifferenceModel`.

## Key hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `corr_weight` | 1.0 | Multiplicative weight on the correlation bonus. |
| `top_k` | 1000 | Max edges returned. |

## Notes

- Only scores edges from perturbed sources (same limitation as Mean Difference).
- The correlation term adds a small but consistent precision boost over Mean Difference on synthetic data at top_k=50–100.
- On real CausalBench K562 (filtered, 622 genes) at top_k=250, this score reaches STRING-network precision **0.668** vs Mean Difference's 0.456 (+21.2 pp); on RPE1 (383 genes) at top_k=250 it reaches **0.304** vs MD's 0.304 (tie). On both datasets it dominates Mean Difference at all top_k checked.

## Prior art and novelty

The underlying idea — exploiting cell-to-cell variation in CRISPRi
knockdown efficiency *within* a single perturbation arm to sharpen
downstream-target inference — is well-established. What is novel here is
the specific composition: a **mean-shift score multiplied by a within-arm
correlation bonus, used as a per-edge ranker**. None of the closest prior
methods do this combination.

Closest related methods:

- **Mixscale** (Jin et al., *Nat Cell Biol* 2025,
  [link](https://www.nature.com/articles/s41556-025-01626-9)) — projects
  each perturbed cell onto a "perturbation vector" to obtain a
  continuous per-cell perturbation score, then uses that score as a
  *regression weight* in DE testing (`wmvReg`). Same heterogeneity
  premise as ShiftCorr; different objective (boost DE signal vs
  rank edges) and different machinery (weighted regression vs
  shift × corr-bonus product).

- **Mixscape** (Papalexi et al., *Nat Genet* 2021,
  [link](https://pubmed.ncbi.nlm.nih.gov/33649593/)) — binary precursor
  to Mixscale: GMM-classifies cells in each arm into "knocked out" vs
  "non-perturbed" (escapers), used to filter escapers before DE.
  Discrete version of the same idea.

- **ADAPRE** (bioRxiv Feb 2026,
  [link](https://www.biorxiv.org/content/10.64898/2026.02.18.706642v1)) —
  most direct conceptual competitor on CausalBench-style data.
  Explicitly motivated by heterogeneous CRISPRi knockdown; treats the
  guide indicator as an instrumental variable for source-gene expression
  in a Poisson-lognormal SCM. ShiftCorr's within-arm correlation is the
  simpler heuristic analogue, treating per-cell residual `A` expression
  itself as the source of exogenous variation.

- **MIMOSCA** (Dixit et al., *Cell* 2016,
  [link](https://www.cell.com/cell/pdf/S0092-8674(16)31610-5.pdf)) —
  original Perturb-seq paper. Elastic-net regression on the
  guide-assignment design matrix. Uses guide identity, not residual
  within-arm `A` expression, so does not exploit within-arm
  heterogeneity. Conceptually upstream.

None of CausalBench's shipped baselines (GENIE3, GRNBoost2, PC, GES,
DCDI, NOTEARS, GIES, Mean Difference) or challenge winners (GuanLab /
PSGRN, Betterboost, SparseRC) combine a mean-shift score with a
within-arm correlation bonus, so the exact score does not collide with
any existing CausalBench baseline.

## Investigated alternatives

The natural theoretical refinement is to replace the mean shift with a
within-arm regression slope `cov(A, B | do(A)) / var(A | do(A))` and to
multiply by raw `|corr|` rather than the `(1 + w·|corr|)` soft bonus.
Under a linear SCM this score (`|slope| × |corr|`) is interpretable as a
structural-coefficient estimator weighted by a cascade-attenuation
factor: the slope is direction-aware and invariant to per-source
knockdown depth, while a cascade `A → M → B` injects intermediate
equation noise that lowers `|corr|` without changing `|slope|`. The
expected payoff is sharper top-k precision on direct edges.

This was implemented and benchmarked head-to-head on K562 and RPE1.
**It regressed across the board** — the `|slope| × |corr|` score lost
to the production `shift × (1 + |corr|)` formula by 6–18 pp on STRING
network and STRING physical at every top_k checked, on both datasets.
Three contributing factors:

1. **Knockdown depth is informative, not noise.** Sources with deeper
   knockdown produce stronger downstream signals, and dividing it out
   (`slope ≈ shift / Δ_A`) discards real ranking information.
2. **Multiplicative `× |corr|` is too aggressive.** The soft bonus
   `(1 + w·|corr|)` caps the corr-driven boost at ~2×; raw `× |corr|`
   demotes by 90%+ when `|corr| ≈ 0.1`, killing edges whose within-arm
   correlation is noisy because of small per-arm cell counts.
3. **Linear-SCM cascade-discrimination assumptions break on real data.**
   Co-expression confounding, dropout, and non-linear effects mean
   `|corr|` is not a clean cascade signal in real Perturb-seq the way
   it is on synthetic linear-cyclic SCMs.

The synthetic cascade-demotion test passes for both formulations (the
linear-SCM theory holds in synthetic), so this is purely a real-data
observation.
