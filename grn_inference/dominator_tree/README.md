# DominatorTreeModel

Dominator-tree edge ranker. Uses the graph-theoretic concept of
immediate dominators to identify which edges are structurally
indispensable: if `u` is the immediate dominator of `v` in the
dominator tree rooted at a perturbed source G, then *every* path from
G to v passes through u — meaning `u → v` cannot be explained away by
any alternative route.

## Algorithm

1. Build a directed graph `G_shift` whose edge weights are
   noise-normalised effect sizes:
   `edge_weight[s, t] = |shift[s, t]| / pooled_std(s, t)`. Pooled std
   combines the do(s) and control variances of target gene t (Cohen's
   d denominator); it is per-arm, not global. For unperturbed source rows, edges come from an
   IV-regression proxy `β_iv[s, t] = ⟨s_s, s_t⟩ / ⟨s_s, s_s⟩`,
   rescaled to match the perturbed-row effect-size magnitude. Apply
   the `shift_quantile` threshold **per source row** (not globally),
   so every source contributes a balanced number of outgoing edges
   regardless of its overall shift magnitude.
2. For each **perturbed** gene R, compute the Lengauer–Tarjan
   dominator tree of `G_shift` rooted at R via
   `networkx.immediate_dominators`. Unperturbed roots are skipped:
   `MW_z(R)` is undefined for them (no do(R) cells), so their votes
   wouldn't contribute. The IV-imputed unperturbed source rows still
   participate as edges in the graph — they let cascade routes
   through unperturbed genes appear in perturbed roots' trees.
3. For each reachable node v with immediate dominator u, add
   `MW_z(R) × |edge_weight[u, v]|` to `score[u, v]`, where `MW_z(R)`
   is the Mann-Whitney |z| of R's own expression in do(R) cells vs
   control. Roots with weak knockdown contribute less.
4. Rank edges by aggregated `score`.

## Key hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `shift_quantile` | 0.94 | Per-source quantile threshold (applied to effect-size weights). At q ≤ 0.88 the graph is too dense; at q ≥ 0.94 dominator trees become meaningful. |
| `top_k` | 1000 | Max edges returned. |

The two main heuristics — effect-size edge weights and Mann-Whitney
|z| root weighting — are both always applied; there is no toggle. Each
was selected after head-to-head benchmarks (see history below) showed
a clear win across both K562 and RPE1.

## Benchmark table (CausalBench, STRING ≥ 900)

| Method | K562 STRING net top_250 | K562 STRING phys top_500 | RPE1 STRING net top_500 |
|---|---|---|---|
| MeanDifference (anchor)               | 0.068 | 0.088 | 0.012 |
| ShiftCorr (anchor)                    | 0.268 | 0.252 | 0.012 |
| DT (raw shift, no MW weighting)       | 0.048 | 0.012 | 0.012 |
| DT (raw shift + MW |z|)               | 0.152 | 0.182 | 0.024 |
| DT (effect-size + MW |z|, global thr) | 0.388 | 0.402 | 0.038 |
| **DT (current default: + per-source thr)** | **0.556** | **0.532** | 0.032 |

The current default (effect-size edge weights + MW |z| root weighting +
per-source quantile thresholding) beats ShiftCorr by **2.1× on K562
STRING net** and **2.1× on K562 STRING physical** at top_500. Per-source
thresholding alone added a further +14 pp on K562 over the global-quantile
variant. RPE1 STRING numbers are still small but DT remains the only
method consistently above MD/ShiftCorr.

A separate experiment that *also* down-weighted noisy sources' outgoing
edges (multiply each row by `MW_z(s) / median(MW_z)`) was tested and
rejected: it slightly compressed the K562 wins from per-source
thresholding (0.524 vs 0.556 at top_250) without compounding cleanly.

## Benchmark results (synthetic, n_genes=50, n_perturbed=25)

At top_k=100:
- Precision (perturbed sources): 0.371
- Precision (hidden/unperturbed sources): **0.200**

Only method besides EnsembleSCMFitter that recovers hidden-source edges.
At matched W1=0.50: finds 134 true hits (66 hidden) vs 108/0 for Mean Difference.

## Future Paths
High leverage (likely real wins)                                                              
                                                                                                
  ~~1. use_all_genes_as_roots=True is currently dead code.~~ — **resolved**.
  The parameter has been dropped; the iteration over roots now starts from
  `perturbed_genes_sorted` directly. The IV-imputation branch still
  contributes to graph topology (cascade routes through unperturbed
  genes appear in perturbed roots' trees) — we just stopped iterating
  unperturbed roots whose votes were silently skipped by MW-|z| = 0.
  Option B (give unperturbed roots a non-zero confidence) is still
  open if we revisit hidden-source recovery on synthetic.

  ~~2. shift_rerank mode is dead on full-perturbation data.~~ — **resolved**.
  The score_mode parameter is gone; only the noise-normalised
  effect-size + edge_weight + MW-|z| pipeline remains.

  ~~3. Vectorise graph construction.~~ — **resolved**. Graph construction now uses
  `np.nonzero(mask)` (lines 186–188); the O(G²) per-cell loop is gone.
  `add_weighted_edges_from` is a minor remaining speedup but NetworkX's
  `immediate_dominators` is the bottleneck anyway.

  Medium leverage (worth trying)

  ~~4. Per-source thresholding instead of global quantile.~~ — **resolved**. Per-source
  quantile thresholding is the current default (lines 165–180). The IV-rescaling
  hack was kept; per-source thresholding alone was sufficient to achieve the +14 pp
  K562 STRING net gain.

  ~~5. Down-weight noisy edges in graph construction, not just votes.~~ — **tested and
  rejected**. Multiplying each row of edge_weight by `MW_z(s) / median(MW_z)` was
  benchmarked and slightly compressed K562 wins from per-source thresholding (0.524
  vs 0.556 at top_250) without compounding cleanly (see benchmark section above).

  6. Adaptive quantile by target out-degree. Pick cutoff to hit a target mean out-degree (3–5)  
  per perturbed source — robust across datasets without re-tuning. Already in README's Problem 2
   list.                                                                                        
                  
  Low leverage (worth investigating, not obvious wins)                                          
  
  7. Compute MW |z| analytically. _compute_root_confidence calls scipy.stats.mannwhitneyu then  
  re-derives z from the U-stat. With ties and large N, a direct rank-sum + tie-correction is
  faster and avoids the try/except ValueError. Saves seconds; not the bottleneck.               
                  
  8. Asymmetric edge enforcement. If both (u,v) and (v,u) survive thresholding, keep only the   
  stronger direction before building the graph — biologically A → B → A cycles via shifts are
  usually noise/feedback artefacts and inflate dominator-tree thrash.                           
                  
  9. Weighted dominator alternative. immediate_dominators is purely topological. Try            
  shortest-path-tree parent as a weighted analogue (use 1/edge_weight as cost) — may rank edges
  by path strength rather than pure indispensability.                                           
                  
  Open questions (not improvements yet)

  - root_shift weights votes by |shifts[root, v]| — the root → leaf shift, not the (u,v) edge's 
  shift. This is what restores W1 (per README), but it's a strange object: an edge u→v scores
  high if many roots have strong shift to v, regardless of how u relates to v. May explain why  
  it lifts STRING precision but still trails ShiftCorr — the score is closer to "v is downstream
   of many strong perturbations" than "u→v is direct."