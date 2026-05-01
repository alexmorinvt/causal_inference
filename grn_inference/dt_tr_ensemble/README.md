# DTTREnsemble

Score-sum ensemble of DominatorTreeModel and TransitiveReductionModel.
Both methods share the same edge-weight construction, run on separate
graphs with their own tuned thresholds, and combine via percentile-rank
normalisation + intersection masking.

## Algorithm

1. Compute `edge_weight[s, t] = |shift[s, t]| / pooled_std(s, t)` and
   the IV-imputed unperturbed source rows — identical to both component
   methods (shared, computed once).
2. Compute `root_conf[R] = MW_z(R)` for each perturbed gene R (shared).
3. Build two separate graphs from `edge_weight`:
   - **DT graph**: per-source quantile 0.94 (matches DominatorTreeModel).
   - **TR graph**: global quantile 0.80 (matches TransitiveReductionModel).
4. Run DominatorTree voting on the DT graph → `score_dt[u, v]`.
5. Run TransitiveReduction voting on the TR graph → `score_tr[u, v]`.
6. Convert each score matrix to percentile ranks within its non-zero
   entries (`ranked[u,v] = rank / n_nonzero`).
7. Sum the rank matrices and zero out any entry where either raw score
   is zero (intersection mask). Edges voted for by only one method are
   excluded entirely.
8. Rank surviving edges by combined rank score; return `top_k`.

## Key hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `dt_shift_quantile` | 0.94 | Per-source quantile for the DT graph. Matches DominatorTreeModel default. |
| `tr_shift_quantile` | 0.80 | Global quantile for the TR graph. Matches TransitiveReductionModel default. |
| `top_k` | 1000 | Max edges returned. In practice the intersection often yields fewer than `top_k` non-zero entries. |

## Why this combination strategy

Three variants were benchmarked on seeds 7, 42, 123 (n_genes=50,
n_perturbed=25):

| Variant | k=50 | k=100 | k=500 | k=1000 |
|---|---|---|---|---|
| DominatorTree (anchor)     | 0.340 | 0.290 | 0.136 | 0.133 |
| TransitiveReduction (anchor) | 0.327 | 0.280 | 0.185 | 0.185 |
| A: raw score sum, union    | 0.367 | 0.290 | 0.175 | 0.137 |
| **B: rank sum, intersection (current)** | **0.373** | **0.303** | **0.264** | **0.264** |
| C: union graph, shared voting | 0.380 | 0.297 | 0.175 | 0.124 |

**A (raw score sum, union)**: edges from either method appear in the
output. Gains at small k but collapses at large k as DT-only tail
entries dilute precision.

**C (union graph)**: both methods run on one shared graph built from
the union of both thresholds. Nearly identical to A — the shared
topology doesn't materially change the voting patterns.

**B (current default)**: percentile-rank normalisation removes raw-score
scale differences between DT and TR. The intersection mask excludes
edges that only one method voted for, eliminating the tail noise that
hurts A and C. The gains at k=500/1000 (+7–13 pp over next-best)
are robust across all three seeds.

## Relationship to component methods

| | DominatorTreeModel | TransitiveReductionModel | DTTREnsemble |
|---|---|---|---|
| Graph | Per-source q=0.94 | Global q=0.80 | Both, separately |
| Structural test | Every path from R to v passes through u | No alternative forward path u→v in BFS-DAG | Both must vote |
| Score | MW_z(R) × edge_weight[u,v] | MW_z(R) × edge_weight[u,v] | Percentile rank sum, intersection-masked |
| Output | Union of all DT votes | Union of all TR votes | Intersection of both vote sets |

## Future Paths

1. **Threshold co-tuning.** `dt_shift_quantile` and `tr_shift_quantile`
   were inherited from the component methods and not jointly optimised.
   A grid search over (dt_q, tr_q) pairs may shift the intersection
   size to a better precision/recall operating point.

2. **Weighted rank sum.** The two rank matrices are currently summed
   with equal weight. If one method is consistently better on a dataset
   (e.g., DT on K562, TR on RPE1), a learned or heuristic weight
   `α × rank_dt + (1-α) × rank_tr` could improve performance.

3. **Extend to more methods.** The rank-sum + intersection-mask pattern
   generalises: add a third structural criterion (e.g., ShiftCorr) and
   require all three to vote. Each additional method tightens the
   intersection and raises precision at the cost of recall.
