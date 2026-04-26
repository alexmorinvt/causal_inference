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
| `weight_by_edge_magnitude` | True | Vote weight = edge weight magnitude rather than 1. |
| `fill_tail_with_shift` | True | Fill remaining top_k slots with shift-ranked edges not already selected. |
| `top_k` | 1000 | Max edges returned. |

## Benchmark results (synthetic, n_genes=50, n_perturbed=25)

At top_k=100:
- Precision (perturbed sources): 0.371
- Precision (hidden/unperturbed sources): **0.200**

Only method besides EnsembleSCMFitter that recovers hidden-source edges.
At matched W1=0.50: finds 134 true hits (66 hidden) vs 108/0 for Mean Difference.
