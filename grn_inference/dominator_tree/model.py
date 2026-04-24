"""Dominator-tree edge ranker.

Algorithm
---------
1. Build a directed graph ``G_shift`` where nodes are genes and each
   perturbed gene ``s`` emits edges ``s -> t`` with weight
   ``|shift[s, t]|`` above a threshold. For unperturbed source rows
   (no direct shift), edges come from the cross-arm IV shift
   regression ``β_iv[s, t] = ⟨s_s, s_t⟩ / ⟨s_s, s_s⟩`` (over perturbed
   rows of the shift matrix).
2. For each perturbed source ``G``, compute the dominator tree of
   ``G_shift`` rooted at ``G`` (Lengauer-Tarjan, via
   ``networkx.immediate_dominators``). For each reachable node ``v ≠
   G`` with immediate dominator ``u = idom(v)``, the pair ``(u, v)``
   is a direct-edge candidate with weight
   ``min(edge_weight[source=u, target=v], path-length discount)``.
3. Aggregate across perturbed sources: ``score[u, v] += vote_weight``
   for every dominator-tree edge across all source roots.
4. Rank edges by aggregated score.

Graph-theoretic rationale
-------------------------
If ``u`` is the immediate dominator of ``v`` in the dominator tree
rooted at ``G``, then every path from ``G`` to ``v`` passes through
``u``. Removing the ``u -> v`` edge disconnects ``v`` from ``G``.
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

    Parameters
    ----------
    top_k
        Number of ranked edges to return.
    shift_quantile
        Keep only edges whose edge weight (``|shift|`` for perturbed
        sources, ``|β_iv|`` for unperturbed) is at or above this
        quantile of the non-zero weights. Higher = sparser graph, so
        dominator trees bottleneck through fewer intermediate nodes
        and the resulting immediate-dominator pairs are more
        discriminating. ``0.94`` selected by train sweep — at q <= 0.88
        the graph is too dense (dominators collapse to the root); at
        q >= 0.94 the graph is sparse enough that dominator trees are
        meaningful and together with shift-tail fill produce a good
        precision × hidden-recall tradeoff. Past 0.94 precision holds
        but hidden recall saturates.
    weight_by_edge_magnitude
        If ``True``, vote weight is ``|edge_weight|`` — the actual
        shift/β_iv magnitude — so that strong edges contribute more
        than weak ones to the vote count. If ``False``, each
        dominator-tree edge gets a vote of 1 (unit vote).
    fill_tail_with_shift
        If ``True``, after the dominator-tree edges are ranked, fill
        any remaining slots in ``top_k`` by falling back to the
        shift-magnitude ranking (``|shift|`` for perturbed sources,
        ``|β_iv|`` for unperturbed) over edges not already selected.
        Dominator trees produce only O(G) edges per source root, so
        this is usually necessary to reach ``top_k = 1000``.
    use_all_genes_as_roots
        If ``True``, compute dominator trees rooted at *every* gene,
        not just the perturbed ones. Unperturbed roots rely on the
        IV-imputed edge_weight rows; each extra root contributes more
        dominator-tree votes, mostly improving hidden-source recall.
        If ``False``, only perturbed genes are roots (iter-27 default).
    """

    top_k: int = 1000
    shift_quantile: float = 0.94
    weight_by_edge_magnitude: bool = True
    fill_tail_with_shift: bool = True
    use_all_genes_as_roots: bool = True

    def fit_predict(self, data: Dataset) -> list[Edge]:
        ctrl_mask = data.control_mask()
        if not ctrl_mask.any():
            raise ValueError(
                "No control cells; cannot compute shifts."
            )
        G = data.n_genes
        perturbed_genes_sorted = sorted(data.perturbed_genes())
        pert_mask = np.zeros(G, dtype=bool)
        for g in perturbed_genes_sorted:
            pert_mask[data.gene_idx(g)] = True

        ctrl_expr = data.expression[ctrl_mask].astype(np.float64)
        ctrl_mean = ctrl_expr.mean(axis=0)

        # ---- Shifts (perturbed source rows) --------------------------
        shifts = np.zeros((G, G), dtype=np.float64)
        for g in perturbed_genes_sorted:
            m = data.intervention_mask(g)
            if not m.any():
                continue
            s = data.gene_idx(g)
            shifts[s, :] = (
                data.expression[m].mean(axis=0) - ctrl_mean
            ).astype(np.float64)

        # ---- IV shift regression for unperturbed source rows ----------
        pert_idx = np.where(pert_mask)[0]
        edge_weight = np.abs(shifts)  # perturbed-row edges = |shift|
        if pert_idx.size > 0:
            S_pert = shifts[pert_idx, :]
            cross = S_pert.T @ S_pert
            diag_denom = np.diag(cross).copy()
            diag_denom = np.where(diag_denom > 1e-12, diag_denom, 1.0)
            beta_iv = cross / diag_denom[:, None]  # β_iv[s, t] = T[t, s] estimate
            np.fill_diagonal(beta_iv, 0.0)
            # For unperturbed sources, fill edge_weight rows with |β_iv|
            # RESCALED to the same magnitude as the perturbed-row
            # shifts, so the threshold doesn't silently exclude them.
            unpert_idx = np.where(~pert_mask)[0]
            if unpert_idx.size > 0:
                shift_nonzero = np.abs(shifts[pert_idx, :])
                shift_nonzero = shift_nonzero[shift_nonzero > 0.0]
                iv_nonzero = np.abs(beta_iv[unpert_idx, :])
                iv_nonzero_pos = iv_nonzero[iv_nonzero > 0.0]
                if shift_nonzero.size > 0 and iv_nonzero_pos.size > 0:
                    scale = shift_nonzero.mean() / iv_nonzero_pos.mean()
                else:
                    scale = 1.0
                for s in unpert_idx:
                    edge_weight[s, :] = np.abs(beta_iv[s, :]) * scale
        np.fill_diagonal(edge_weight, 0.0)

        # ---- Threshold to sparse graph -------------------------------
        nonzero = edge_weight[edge_weight > 0.0]
        if nonzero.size == 0:
            return []
        cutoff = float(np.quantile(nonzero, self.shift_quantile))

        # ---- Build graph --------------------------------------------
        names = data.gene_names
        graph = nx.DiGraph()
        graph.add_nodes_from(range(G))
        for s in range(G):
            for t in range(G):
                if s == t:
                    continue
                w = edge_weight[s, t]
                if w >= cutoff and w > 0.0:
                    graph.add_edge(s, t, weight=float(w))

        # ---- Per-source dominator tree votes -------------------------
        score = np.zeros((G, G), dtype=np.float64)
        if self.use_all_genes_as_roots:
            root_indices = list(range(G))
        else:
            root_indices = [data.gene_idx(g) for g in perturbed_genes_sorted]
        for root in root_indices:
            if root not in graph:
                continue
            try:
                idoms = nx.immediate_dominators(graph, root)
            except Exception:
                continue
            for v, u in idoms.items():
                if v == u:
                    continue
                if v == root:
                    continue
                if self.weight_by_edge_magnitude:
                    w = float(edge_weight[u, v])
                else:
                    w = 1.0
                score[u, v] += w

        # score[u, v] = total vote weight for direct edge u -> v.
        np.fill_diagonal(score, 0.0)
        score_src_tgt = score.astype(np.float32)
        flat = score_src_tgt.ravel()

        # Collect dominator-tree edges (score > 0), ranked by vote weight.
        nonzero_order = np.argsort(-flat)
        nonzero_order = nonzero_order[flat[nonzero_order] > 0.0]
        dom_edges: list[Edge] = []
        dom_set: set[Edge] = set()
        for idx in nonzero_order:
            u, v = divmod(int(idx), G)
            if u == v:
                continue
            e = (names[u], names[v])
            dom_edges.append(e)
            dom_set.add(e)

        if len(dom_edges) >= self.top_k or not self.fill_tail_with_shift:
            return dom_edges[: self.top_k]

        # ---- Fill tail with shift/β_iv ranking -----------------------
        # Use the same edge_weight matrix (shift for perturbed, β_iv
        # rescaled for unperturbed). Skip edges already selected.
        tail_flat = edge_weight.ravel()
        tail_order = np.argsort(-tail_flat)
        out = list(dom_edges)
        for idx in tail_order:
            if len(out) >= self.top_k:
                break
            if tail_flat[idx] <= 0.0:
                break
            s, t = divmod(int(idx), G)
            if s == t:
                continue
            e = (names[s], names[t])
            if e in dom_set:
                continue
            out.append(e)
            dom_set.add(e)
        return out[: self.top_k]
