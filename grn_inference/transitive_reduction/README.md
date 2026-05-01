# TransitiveReductionModel

Transitive-reduction edge ranker. For each perturbed root R, builds a
BFS-layered DAG from the reachable subgraph and computes its transitive
reduction. An edge `(u, v)` survives iff no alternative forward path
`u→w→…→v` exists in that DAG — meaning it cannot be explained as a
cascade shortcut. Votes are aggregated across all perturbed roots.

## Algorithm

1. Build a directed graph `G_shift` with noise-normalised effect-size
   edge weights `edge_weight[s, t] = |shift[s, t]| / pooled_std(s, t)`.
   Unperturbed source rows are filled via cross-arm IV regression
   `β_iv[s, t] = ⟨s_s, s_t⟩ / ⟨s_s, s_s⟩`, rescaled to match the
   perturbed-row magnitude. (Identical to DominatorTreeModel step 1.)
2. Apply a **global** quantile threshold at `shift_quantile` over all
   non-zero weights. Edges at or above the cutoff are kept. Global
   (not per-source) because every edge is a candidate forward-path
   segment in any root's BFS-DAG; edges from all sources compete
   symmetrically. The default (0.80) is lower than DominatorTreeModel's
   per-source 0.94 — TR needs alternative paths to detect shortcuts, and
   a sparser BFS-DAG makes TR a near-identity.
3. For each **perturbed** gene R:
   - BFS from R in `G_shift`; record distance `dist_R[v]` for each
     reachable `v`.
   - Build the **BFS-layered DAG**: keep only edges `(u, v)` from the
     reachable subgraph where `dist_R[v] > dist_R[u]`. Back and
     same-layer edges are dropped, guaranteeing acyclicity.
   - Compute the **transitive reduction** of this DAG via
     `networkx.transitive_reduction`. Edge `(u, v)` is removed if any
     path `u→w→…→v` exists — i.e., `(u, v)` is a shortcut.
   - For each surviving edge `(u, v)`, add
     `MW_z(R) × edge_weight[u, v]` to `score[u, v]`.
4. Rank edges by aggregated `score`; return top `top_k`.

## Key hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `shift_quantile` | 0.80 | Global quantile threshold. Higher → sparser graph → fewer alternative paths → TR less discriminating. Lower → denser graph → more shortcuts detected but more noise. |
| `top_k` | 1000 | Max edges returned. |

## Relationship to DominatorTreeModel

Both methods vote `MW_z(R) × edge_weight[u, v]` per root, and both use
the same edge-weight construction. They differ in the structural test:

| | DominatorTreeModel | TransitiveReductionModel |
|---|---|---|
| **Question** | Does every path from R to v pass through u? | Is there *no* alternative path from u to v? |
| **Graph scope** | Full `G_shift` (cycles allowed) | BFS-DAG reachable from R (acyclic) |
| **Cycle handling** | Lengauer–Tarjan (handles cycles) | BFS layering removes back/lateral edges |
| **Thresholding** | Per-source quantile (0.94) | Global quantile (0.80) |
| **Degenerate case** | Dense graph → many roots dominate v → noisy | Sparse graph → no alternatives → TR keeps all edges |

The two criteria are complementary: an edge that passes both is a strong
direct-regulation candidate. A future ensemble could intersect or union
the two score matrices.

## Limitations

- **BFS ties**: nodes at the same BFS distance are in the same layer;
  edges between them are dropped. If a true direct edge happens to be a
  same-layer edge from R's perspective, it loses votes from R.
- **Topological TR**: `networkx.transitive_reduction` is purely
  topological — it removes `(u, v)` if *any* alternative path exists,
  regardless of edge weights. A strong direct edge is pruned if a weak
  indirect route connects u to v.
- **Computational cost**: `nx.transitive_reduction` is O(V·E) per DAG.
  On K562 (622 genes, ~600 perturbed roots, BFS-DAGs potentially
  spanning hundreds of nodes) this may be slow — profile before tuning.

## Future Paths

High leverage

1. Weight-aware TR. Replace `nx.transitive_reduction` with a
   shortest-path-tree from R using `1/edge_weight` as cost (Dijkstra).
   Keep only SPT edges. This is a weighted analogue of TR: an edge is
   "necessary" if it's on the shortest path, not merely the only path.
   The unweighted TR may prune strong direct edges when weak indirect
   routes exist; shortest-path-tree only prunes when the detour is
   equally or more efficient.

2. Same-layer edges. BFS layering drops all edges between nodes at the
   same distance from R. These may include direct regulations between
   co-regulated genes. Consider running TR on a richer DAG that includes
   same-layer edges via a tiebreak (e.g., DFS finish time).

3. Global TR + per-root reweighting. Compute TR once on the global
   `G_shift` (after SCC condensation for cycle handling), then weight
   surviving edges by sum of MW_z(R) × edge_weight[u, v] over all R
   from which both u and v are reachable. Avoids re-running TR per root
   and handles cycles properly.

Medium leverage

4. Intersection with DominatorTreeModel. An edge that survives both TR
   and dominates from R should be a very high-precision prediction.
   Score by product or minimum of the two scores.

5. Adaptive global threshold. Target a mean out-degree of 5–10 per node
   rather than a fixed quantile — robust across datasets without
   re-tuning `shift_quantile`.
