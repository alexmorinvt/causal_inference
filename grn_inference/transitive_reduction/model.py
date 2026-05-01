"""Transitive-reduction edge ranker.

Algorithm
---------
1. Build edge_weight matrix (noise-normalised effect sizes, identical to
   DominatorTreeModel): ``edge_weight[s, t] = |shift[s, t]| / pooled_std(s, t)``.
   Unperturbed source rows are filled via cross-arm IV regression, rescaled to
   match the perturbed-row magnitude.
2. Apply a **global** quantile threshold: keep edges whose weight is at or
   above the ``shift_quantile`` percentile of all non-zero weights. Global
   (not per-source) because every edge is a candidate forward-path segment
   in any root's BFS-DAG — edges from all sources compete symmetrically.
   The default (0.80) is lower than DominatorTree's per-source 0.94 so that
   each root's BFS-DAG retains enough forward paths for TR to detect shortcuts;
   at 0.94 global the BFS-DAGs are too sparse and TR is a near-identity.
3. Build a NetworkX DiGraph from surviving edges.
4. For each perturbed root R:
   a. BFS from R; record distance dist_R[v] for each reachable v.
   b. Build a DAG from the reachable subgraph: keep only edges (u, v) where
      dist_R[v] > dist_R[u] (forward edges in BFS layering). Back and lateral
      edges are dropped, guaranteeing acyclicity.
   c. Compute the transitive reduction of this DAG via nx.transitive_reduction.
      Edge (u, v) survives iff no alternative forward path u→w→...→v exists —
      i.e., (u, v) is not a cascade shortcut.
   d. For each surviving edge (u, v), add MW_z(R) × edge_weight[u, v] to
      score[u, v].
5. Rank edges by aggregated score; return top_k.

Graph-theoretic rationale
-------------------------
Transitive reduction removes edges that are redundant given alternative paths
in the DAG. An edge (u, v) that is pruned has a detour u→w→...→v, meaning
the *direct* u→v signal is explainable as a cascade effect routed through w.
Surviving edges are the ones for which no such detour exists — structurally
indispensable direct regulations from R's vantage point.

This is complementary to the dominator-tree criterion: dominators identify
edges that every path *from R* must cross, while TR identifies edges with no
parallel alternative path *regardless of root*. Both select for direct over
cascade, but from different angles.

Limitations
-----------
- BFS layering breaks ties arbitrarily when multiple nodes have the same BFS
  distance; edges between same-layer nodes are dropped, which may lose some
  direct edges. Longer cycles are entirely excluded.
- TR on the BFS-DAG is purely topological — edge weights do not influence
  which shortcut is kept vs pruned. A strong direct edge (u, v) is dropped
  if *any* alternative path exists, even a weak one.
- Computational cost is O(n_perturbed × |DAG| × |E_DAG|) — feasible on
  synthetic (50 genes) but may be slow on full K562 (622 genes, 600 roots).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..dataset import Dataset

Edge = tuple[str, str]


@dataclass
class TransitiveReductionModel:
    """Rank edges by vote-weighted transitive-reduction membership across source roots.

    For each perturbed gene R, builds a BFS-layered DAG from the thresholded
    shift graph reachable from R and computes its transitive reduction. Edges
    that survive TR are structurally indispensable — no detour path exists in
    the DAG. Votes are weighted by MW_z(R) × edge_weight[u, v], identical
    to DominatorTreeModel.

    Parameters
    ----------
    top_k
        Number of ranked edges to return.
    shift_quantile
        Global quantile threshold applied to all non-zero edge weights.
        Edges at or above this percentile are retained in the shared graph.
        Default 0.80 is lower than DominatorTreeModel's per-source 0.94:
        TR needs alternative paths to prune; a sparser BFS-DAG makes TR
        degenerate (nothing to remove, all edges survive).
    """

    top_k: int = 1000
    shift_quantile: float = 0.80

    def fit_predict(self, data: Dataset) -> list[Edge]:
        ctrl_mask = data.control_mask()
        if not ctrl_mask.any():
            raise ValueError("No control cells; cannot compute shifts.")
        G = data.n_genes
        perturbed_genes_sorted = sorted(data.perturbed_genes())
        pert_mask = np.zeros(G, dtype=bool)
        for g in perturbed_genes_sorted:
            pert_mask[data.gene_idx(g)] = True

        ctrl_expr = data.expression[ctrl_mask].astype(np.float64)
        ctrl_mean = ctrl_expr.mean(axis=0)
        n_ctrl = int(ctrl_mask.sum())
        ctrl_var = ctrl_expr.var(axis=0)

        # ---- Shifts + per-arm pooled std for effect-size denominator ----
        shifts = np.zeros((G, G), dtype=np.float64)
        pooled_std = np.zeros((G, G), dtype=np.float64)
        for g in perturbed_genes_sorted:
            m = data.intervention_mask(g)
            n_p = int(m.sum())
            if n_p == 0:
                continue
            s = data.gene_idx(g)
            pert_expr = data.expression[m].astype(np.float64)
            shifts[s, :] = pert_expr.mean(axis=0) - ctrl_mean
            pert_var = pert_expr.var(axis=0)
            pooled_var = (n_p * pert_var + n_ctrl * ctrl_var) / (n_p + n_ctrl)
            pooled_std[s, :] = np.sqrt(pooled_var + 1e-12)

        # ---- Edge weights: noise-normalised effect size ----------------
        denom = np.where(pooled_std > 0, pooled_std, 1.0)
        edge_weight = np.abs(shifts) / denom

        # ---- IV shift regression for unperturbed source rows -----------
        pert_idx = np.where(pert_mask)[0]
        if pert_idx.size > 0:
            S_pert = shifts[pert_idx, :]
            cross = S_pert.T @ S_pert
            diag_denom = np.diag(cross).copy()
            diag_denom = np.where(diag_denom > 1e-12, diag_denom, 1.0)
            beta_iv = cross / diag_denom[:, None]
            np.fill_diagonal(beta_iv, 0.0)
            unpert_idx = np.where(~pert_mask)[0]
            if unpert_idx.size > 0:
                target_nonzero = edge_weight[pert_idx, :]
                target_nonzero = target_nonzero[target_nonzero > 0.0]
                iv_nonzero = np.abs(beta_iv[unpert_idx, :])
                iv_nonzero_pos = iv_nonzero[iv_nonzero > 0.0]
                if target_nonzero.size > 0 and iv_nonzero_pos.size > 0:
                    scale = target_nonzero.mean() / iv_nonzero_pos.mean()
                else:
                    scale = 1.0
                for s in unpert_idx:
                    edge_weight[s, :] = np.abs(beta_iv[s, :]) * scale
        np.fill_diagonal(edge_weight, 0.0)

        # ---- Global quantile threshold ---------------------------------
        nz = edge_weight[edge_weight > 0.0]
        if nz.size == 0:
            return []
        cutoff = float(np.quantile(nz, self.shift_quantile))
        mask = (edge_weight >= cutoff) & (edge_weight > 0.0)
        np.fill_diagonal(mask, False)

        # ---- Build shared directed graph --------------------------------
        graph = nx.DiGraph()
        graph.add_nodes_from(range(G))
        src_idx, tgt_idx = np.nonzero(mask)
        for s, t in zip(src_idx.tolist(), tgt_idx.tolist()):
            graph.add_edge(s, t, weight=float(edge_weight[s, t]))

        # ---- Per-root Mann-Whitney |z| weighting -----------------------
        root_conf = self._compute_root_confidence(
            data, ctrl_mask, perturbed_genes_sorted, G,
        )

        # ---- TR votes (perturbed roots only) ---------------------------
        score = np.zeros((G, G), dtype=np.float64)
        for g in perturbed_genes_sorted:
            root = data.gene_idx(g)
            if root not in graph:
                continue
            rc = float(root_conf[root])
            if rc <= 0.0:
                continue

            # BFS distances from root in the shared graph
            dist = nx.single_source_shortest_path_length(graph, root)
            if len(dist) <= 1:
                continue

            # BFS-layered DAG: keep only forward edges (dist strictly increases)
            dag = nx.DiGraph()
            dag.add_nodes_from(dist.keys())
            for u in dist:
                for v in graph.successors(u):
                    if v in dist and dist[v] > dist[u]:
                        dag.add_edge(u, v)

            if dag.number_of_edges() == 0:
                continue

            # Transitive reduction: remove shortcut edges
            try:
                tr_dag = nx.transitive_reduction(dag)
            except Exception:
                continue

            # Vote for each structurally indispensable edge
            for u, v in tr_dag.edges():
                score[u, v] += rc * float(edge_weight[u, v])

        np.fill_diagonal(score, 0.0)
        flat = score.astype(np.float32).ravel()
        order = np.argsort(-flat)
        order = order[flat[order] > 0.0]
        names = data.gene_names
        result: list[Edge] = []
        for idx in order[: self.top_k]:
            u, v = divmod(int(idx), G)
            if u == v:
                continue
            result.append((names[u], names[v]))
        return result

    def _compute_root_confidence(
        self,
        data: Dataset,
        ctrl_mask: np.ndarray,
        perturbed_genes_sorted: list[str],
        G: int,
    ) -> np.ndarray:
        """Per-root vote-scaling: Mann-Whitney |z| of source gene's own
        expression in do(R) vs control. Unperturbed roots get 0.
        """
        from scipy.stats import mannwhitneyu

        conf = np.zeros(G, dtype=np.float64)
        ctrl_expr = data.expression[ctrl_mask]
        n_ctrl = int(ctrl_mask.sum())
        for g in perturbed_genes_sorted:
            m = data.intervention_mask(g)
            n_p = int(m.sum())
            if n_p == 0 or n_ctrl == 0:
                continue
            idx = data.gene_idx(g)
            x_pert = data.expression[m, idx]
            x_ctrl = ctrl_expr[:, idx]
            try:
                u_stat, _ = mannwhitneyu(
                    x_pert, x_ctrl, alternative="two-sided",
                )
            except ValueError:
                continue
            mu = n_p * n_ctrl / 2.0
            sigma = np.sqrt(n_p * n_ctrl * (n_p + n_ctrl + 1) / 12.0)
            if sigma > 0:
                conf[idx] = abs((u_stat - mu) / sigma)
        return conf
