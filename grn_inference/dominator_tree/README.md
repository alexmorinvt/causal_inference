# DominatorTreeModel

Dominator-tree edge ranker. Uses the graph-theoretic concept of
immediate dominators to identify which edges are structurally
indispensable: if `u` is the immediate dominator of `v` in the
dominator tree rooted at a perturbed source G, then *every* path from
G to v passes through u — meaning `u → v` cannot be explained away by
any alternative route.

## Algorithm

1. Build a directed graph `G_shift` where each perturbed gene s emits edges `s → t` with weight `|shift[s, t]|` above the `shift_quantile` threshold. For unperturbed source rows, edges come from an IV-regression proxy: `β_iv[s, t] = ⟨s_s, s_t⟩ / ⟨s_s, s_s⟩` over perturbed rows of the shift matrix.
2. For each root (perturbed gene, or all genes if `use_all_genes_as_roots=True`), compute the Lengauer–Tarjan dominator tree via `networkx.immediate_dominators`.
3. For each reachable node v with immediate dominator u, cast a vote for edge `(u, v)` weighted by `|edge_weight[u, v]|` (if `weight_by_edge_magnitude=True`) or 1.
4. Rank edges by total vote count across all root trees.
5. If `fill_tail_with_shift=True`, fill remaining top_k slots with unselected edges ordered by raw shift magnitude.

## Key hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `shift_quantile` | 0.94 | Graph sparsity threshold. At q ≤ 0.88 the graph is too dense (dominators collapse to root); at q ≥ 0.94 dominator trees are meaningful. Selected by train sweep. |
| `use_all_genes_as_roots` | True | Compute dominator trees from every gene as root (unperturbed roots use IV-imputed edges). Improves hidden-source recall. |
| `score_mode` | `"root_shift"` | How votes aggregate. See "Score modes" below. |
| `fill_tail_with_shift` | True | Fill remaining top_k slots with shift-ranked edges not already selected. |
| `top_k` | 1000 | Max edges returned. |

## Score modes

`score_mode` selects how the per-root immediate-dominator votes are
aggregated into a per-edge score.

- `"root_shift"` (default): `score[u, v] += |shifts[root, v]|` for each
  root R where `idom_R(v) = u`. Rewards `(u, v)` edges that dominate
  strong downstream signals from many perturbed roots. Unperturbed
  roots contribute zero (no direct shift). Selected as default after
  beating `"edge_weight"` on both statistical W1 and biological
  precision on K562 and RPE1.
- `"edge_weight"` (original behaviour): `score[u, v] += |edge_weight[u, v]|`
  for each root that has `idom(v) = u`. Strong edges accumulate more
  weight. Hidden-source recovery preserved through IV-imputed
  `edge_weight` for unperturbed sources.
- `"shift_rerank"`: collect integer vote counts and multiply by the
  Mean-Difference shift magnitude — `score[u, v] = vote_count[u, v] × |shifts[u, v]|`.
  Dominator structure reranks Mean-Difference's magnitude. Restores
  high statistical W1 at the cost of hidden-source recovery, since
  `shifts[u, v] = 0` for unperturbed `u`.

`shift_rerank` and `root_shift` were added to address the low
statistical W1 of `edge_weight` (Problem 1 below) by injecting
shift-magnitude information into the vote aggregation.

**`shift_rerank` is identical to `edge_weight` when every gene is
perturbed.** Both reduce to `vote_count × |shifts[u, v]|` for perturbed
``u``; the only difference is that `edge_weight` also picks up votes
from IV-imputed unperturbed-source edges. On both K562 (622 / 622
perturbed) and RPE1 (383 / 383 perturbed) the two modes give identical
numbers — `shift_rerank` is only useful on partial-perturbation data.

## Per-root vote scaling (Mann-Whitney |z|)

Every vote a root `R` casts is scaled by `|z|` of a Mann-Whitney U test
comparing `R`'s own expression in `do(R)` cells vs control — i.e. how
strongly `R` was actually knocked down. Roots whose CRISPRi failed
(weak knockdown → low `|z|`) contribute proportionally less;
unperturbed roots get zero weight (knockdown not defined).

This is always applied — there's no toggle. Adds one Mann-Whitney U
test per perturbed gene (~3 s on K562, ~1 s on RPE1).

## Benchmark table (CausalBench, STRING ≥ 900)

| Method | K562 STRING net top_250 | K562 STRING phys top_500 | RPE1 STRING net top_500 |
|---|---|---|---|
| MeanDifference (anchor) | 0.068 | 0.088 | 0.012 |
| ShiftCorr (anchor)      | **0.268** | **0.252** | 0.012 |
| DT[edge_weight] (orig)  | 0.048 | 0.012 | 0.012 |
| DT[shift_rerank]        | 0.048 | 0.012 | 0.012 |
| DT[root_shift] (default) | **0.152** | **0.182** | **0.024** |

`root_shift` (the new default) with mandatory MW-`|z|` weighting gives
a 3–8× lift over the unweighted variants on K562 and the only DT
setting that beats both Mean Difference and ShiftCorr on RPE1 (though
all RPE1 numbers are near random — the dataset is hard). Still trails
ShiftCorr on K562.

## Benchmark results (synthetic, n_genes=50, n_perturbed=25)

At top_k=100:
- Precision (perturbed sources): 0.371
- Precision (hidden/unperturbed sources): **0.200**

Only method besides EnsembleSCMFitter that recovers hidden-source edges.
At matched W1=0.50: finds 134 true hits (66 hidden) vs 108/0 for Mean Difference.

## Future directions
 Looking at the benchmark results and the algorithm, there are several distinct failure modes to     
  address:                                                                                            
                                                                                                      
  Problem 1: Low statistical W1 on real data (0.332 vs 0.740 for MeanDiff)
  Dominator edges are structurally indispensable paths, not necessarily the strongest-shift edges. By
  design, the ranking decorrelates from raw effect size.

  - **Implemented as `score_mode="shift_rerank"`** — `score = vote_count × |shifts|`. Reranks MeanDiff
  magnitudes by dominator structure. Restores W1 at the cost of hidden-source recovery.
  - **Implemented as `score_mode="root_shift"`** — vote weight = `|shifts[root, v]|`. Edges that
  dominate AND propagate strong downstream signal from perturbed roots score highest.

  Problem 2: Graph construction is noisy on real data
  The shift_quantile=0.94 was tuned on synthetic data. Real K562 shift distributions are different,
  and edges passing the threshold may be noise.

  - Replace raw shift with statistical significance (Mann-Whitney p-value or effect size / within-arm
  std) as the edge weight — filters sampling noise rather than just magnitude.
  - Adaptive quantile: Pick the quantile that produces a target average out-degree (e.g., 3–5 edges
  per source), data-adaptively.

  Problem 3: All roots weighted equally
  A root gene with weak knockdown (low shift, escaped cells) produces an unreliable dominator tree —
  but contributes the same vote weight as a strongly knocked-down gene.

  - Weight roots by knockdown confidence: Scale each root's vote contribution by mean_shift(root) or
  its Mann-Whitney z-score against control. Noisy roots contribute less.

  Problem 4: IV imputation for unperturbed sources is a rough heuristic
  β_iv[s, t] = ⟨s_s, s_t⟩ / ⟨s_s, s_s⟩ is just a dot product ratio — very sensitive to noise. On real
  data with 622 noisy shift rows, the IV-imputed edges for unperturbed genes will often be
  meaningless.

  - Restrict to perturbed roots only on real data (set use_all_genes_as_roots=False) and skip IV
  imputation — fewer but cleaner votes.
  - Or use a proper 2SLS estimate with the guide assignment as instrument.

  Problem 5: fill_tail_with_shift dilutes the signal
  After dominator edges run out (~O(G) per root), the tail is filled with plain MeanDiff edges. At
  top_k=1000 on 622 genes this tail is large and just replicates MeanDiff.

  - Don't fill the tail — return fewer edges with higher dominator confidence rather than padding with
   MeanDiff. Evaluate at whatever k the dominator method naturally produces.

  ---
  Highest-leverage starting point: The reranking approach (Problem 1) — final_score = dominator_votes
  × shift — is a one-line change that could dramatically improve W1 while retaining the hidden-source
  recovery advantage. Everything else is about improving the graph construction quality, which matters
   more on real than synthetic data.