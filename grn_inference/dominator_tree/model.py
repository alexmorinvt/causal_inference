"""Dominator-tree edge ranker.

Algorithm
---------
1. Build a directed graph ``G_shift`` where every gene ``s`` emits edges
   ``s -> t`` with weight equal to the **noise-normalised effect size**
   ``|shift[s, t]| / pooled_std(t)`` above a quantile threshold. For
   unperturbed source rows (no direct shift), edges come from the
   cross-arm IV shift regression
   ``β_iv[s, t] = ⟨s_s, s_t⟩ / ⟨s_s, s_s⟩`` (over perturbed rows of the
   shift matrix), rescaled to match the perturbed-row effect-size
   magnitude.
2. For each gene ``R``, compute the dominator tree of ``G_shift`` rooted
   at ``R`` (Lengauer-Tarjan, via ``networkx.immediate_dominators``).
3. For each immediate-dominator pair ``(u, v)`` with ``idom_R(v) = u``,
   add ``MW_z(R) * |edge_weight[u, v]|`` to ``score[u, v]``, where
   ``MW_z(R)`` is the Mann-Whitney |z| of ``R``'s own expression in
   ``do(R)`` vs control cells. Roots with weak knockdown contribute
   less; unperturbed roots (no do(R) cells) contribute zero.
4. Rank edges by aggregated score.

Graph-theoretic rationale
-------------------------
If ``u`` is the immediate dominator of ``v`` in the dominator tree
rooted at ``R``, then every path from ``R`` to ``v`` passes through
``u``. Removing the ``u -> v`` edge disconnects ``v`` from ``R``.
This is the strongest possible graph-theoretic evidence that ``u -> v``
is a direct causal edge; cascade alternatives cannot substitute.
Aggregating votes across many source roots amplifies edges that are
structurally essential from many vantage points.

Limitations:
- Dominator trees are defined only for reachable nodes; genes ``v``
  that aren't reachable from any perturbed ``G`` get no vote (rare on
  our synthetic data with dense enough shift graphs).
- Immediate dominators capture the *last* gene on every path, which
  for a direct edge is the correct parent, but for a cascade
  ``u -> w -> v`` where ``u`` is the only ``G``-ancestor also flags
  ``u`` (via dominator transitivity). We accept this over-flagging
  because the dominator tree itself is sparse (at most ``|V|-1``
  edges per root) and aggregating across roots disperses the overflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..dataset import Dataset

Edge = tuple[str, str]


@dataclass
class DominatorTreeModel:
    """Rank edges by aggregated dominator-tree membership across source roots.

    Edge weights are noise-normalised effect sizes
    ``|shift| / pooled_std``; per-vote weights are scaled by the
    Mann-Whitney |z| of the root's own knockdown. Both choices were
    selected empirically (effect-size + edge_weight scoring beats the
    raw-magnitude / root_shift / shift_rerank alternatives by 2-7x on
    K562 STRING precision and is the only DT setting that beats
    Mean-Difference and ShiftCorr on RPE1).

    Parameters
    ----------
    top_k
        Number of ranked edges to return.
    shift_quantile
        Keep only edges whose effect-size weight is at or above this
        quantile of the non-zero weights. Higher = sparser graph, so
        dominator trees bottleneck through fewer intermediate nodes
        and the resulting immediate-dominator pairs are more
        discriminating. ``0.94`` selected by train sweep on the
        magnitude variant; effect-size distributions are similar.

    Notes
    -----
    Only perturbed genes are used as dominator-tree roots: unperturbed
    roots have ``MW_z(R) = 0`` (no do(R) cells to test against), so
    their votes wouldn't count. The IV-imputed unperturbed source rows
    still participate as edges in the graph — they let cascade routes
    through unperturbed genes appear in perturbed roots' trees.

    Hidden-source recovery happens through this indirect mechanism:
    when a perturbed root R's dominator tree has an unperturbed gene
    U as ``idom(v)``, the ``(U, v)`` edge collects a vote even though
    U is never iterated as a root.

    A surrogate-confidence experiment that gave unperturbed roots
    their own dominator-tree votes
    (``median(perturbed |z|) × mean(|edge_weight[R, :]|)``) was tested
    and rejected on both synthetic (n_perturbed = n_genes / 2) and
    partial real K562 (50 % of arms kept). It hurt STRING precision by
    1–4 pp at every top_k and dropped synthetic hidden-src precision
    from 0.20 to 0.14 (top_k=250). The extra unperturbed-root votes
    flow into the same (u, v) cells as perturbed-root votes, diluting
    the cleaner indirect-mechanism contributions rather than surfacing
    new unperturbed-source edges.
    """

    top_k: int = 1000
    shift_quantile: float = 0.94

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

        # ---- Shifts + per-arm pooled std for effect-size denominator -
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

        # ---- Edge weights: noise-normalised effect size --------------
        denom = np.where(pooled_std > 0, pooled_std, 1.0)
        edge_weight = np.abs(shifts) / denom

        # ---- IV shift regression for unperturbed source rows ----------
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

        # ---- Per-source quantile threshold to sparse graph ----------
        # Each row gets its own quantile cutoff, equalising graph
        # contribution across sources. A global quantile let strong-
        # shift sources dominate; per-source thresholding gave +14 pp
        # K562 STRING net at top_500 with negligible RPE1 cost.
        if not (edge_weight > 0).any():
            return []
        mask = np.zeros_like(edge_weight, dtype=bool)
        for s in range(G):
            row = edge_weight[s, :]
            row_nz = row[row > 0.0]
            if row_nz.size == 0:
                continue
            row_cutoff = float(np.quantile(row_nz, self.shift_quantile))
            mask[s, :] = (row >= row_cutoff) & (row > 0.0)
        np.fill_diagonal(mask, False)

        # ---- Build graph --------------------------------------------
        names = data.gene_names
        graph = nx.DiGraph()
        graph.add_nodes_from(range(G))
        src_idx, tgt_idx = np.nonzero(mask)
        for s, t in zip(src_idx.tolist(), tgt_idx.tolist()):
            graph.add_edge(s, t, weight=float(edge_weight[s, t]))

        # ---- Per-root Mann-Whitney |z| weighting --------------------
        root_conf = self._compute_root_confidence(
            data, ctrl_mask, perturbed_genes_sorted, G,
        )

        # ---- Dominator-tree votes (perturbed roots only) ------------
        score = np.zeros((G, G), dtype=np.float64)
        for g in perturbed_genes_sorted:
            root = data.gene_idx(g)
            if root not in graph:
                continue
            rc = float(root_conf[root])
            if rc <= 0.0:
                continue
            try:
                idoms = nx.immediate_dominators(graph, root)
            except Exception:
                continue
            for v, u in idoms.items():
                if v == u or v == root:
                    continue
                score[u, v] += rc * float(edge_weight[u, v])

        np.fill_diagonal(score, 0.0)
        flat = score.astype(np.float32).ravel()

        nonzero_order = np.argsort(-flat)
        nonzero_order = nonzero_order[flat[nonzero_order] > 0.0]
        dom_edges: list[Edge] = []
        for idx in nonzero_order[: self.top_k]:
            u, v = divmod(int(idx), G)
            if u == v:
                continue
            dom_edges.append((names[u], names[v]))
        return dom_edges

    def _compute_root_confidence(
        self,
        data: Dataset,
        ctrl_mask: np.ndarray,
        perturbed_genes_sorted: list[str],
        G: int,
    ) -> np.ndarray:
        """Per-root vote-scaling factor: Mann-Whitney |z| of the source
        gene's own expression in do(R) cells vs control cells.

        Unperturbed roots get 0 (no do(R) cells to test against).
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
