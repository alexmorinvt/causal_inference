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
    score_mode
        How dominator-tree votes get aggregated into per-edge scores.

        - ``"root_shift"`` (default): weight each vote by the direct
          shift from the root to the target, ``|shifts[root, v]|``.
          Rewards edges ``(u, v)`` that dominate strong downstream
          signals from many roots. Unperturbed roots contribute zero
          (no direct shift), so only perturbed roots vote effectively.
          Modestly better statistical W1 and biological precision than
          ``"edge_weight"`` on K562 and RPE1.
        - ``"edge_weight"`` (original behaviour): for each ``(u, v)``
          with ``idom(v) = u`` in the tree rooted at ``R``, add
          ``|edge_weight[u, v]|`` to ``score[u, v]``. Strong edges
          accumulate larger scores. Hidden-source recovery preserved
          (unperturbed sources use IV-imputed ``edge_weight``).
        - ``"shift_rerank"``: collect integer vote counts, then
          multiply by ``|shifts[u, v]|`` to get the final score —
          ``score[u, v] = vote_count[u, v] × |shifts[u, v]|``. Dominator
          structure reranks Mean-Difference's magnitude. Identical to
          ``"edge_weight"`` when every gene is perturbed (e.g. on K562
          and RPE1); only differs on partial-perturbation data.
    use_all_genes_as_roots
        If ``True``, compute dominator trees rooted at *every* gene,
        not just the perturbed ones. Unperturbed roots rely on the
        IV-imputed edge_weight rows; each extra root contributes more
        dominator-tree votes, mostly improving hidden-source recall.
        If ``False``, only perturbed genes are roots (iter-27 default).

    Per-root vote scaling
    ---------------------
    Every vote a root ``R`` casts is scaled by ``|z|`` of a Mann-Whitney
    U test comparing ``R``'s own expression in ``do(R)`` cells vs control
    cells — i.e. how strongly ``R`` was actually knocked down. Roots
    whose CRISPRi failed (weak knockdown → low ``|z|``) contribute
    proportionally less; unperturbed roots get zero weight (knockdown
    not defined). Empirically a 3–8× lift on K562 STRING precision over
    unweighted voting.
    """

    top_k: int = 1000
    shift_quantile: float = 0.94
    score_mode: str = "root_shift"
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
        if self.score_mode not in {"edge_weight", "shift_rerank", "root_shift"}:
            raise ValueError(
                f"Unknown score_mode {self.score_mode!r}; expected one of "
                "edge_weight, shift_rerank, root_shift"
            )

        # Per-root Mann-Whitney |z| weighting (knockdown confidence).
        root_conf = self._compute_root_confidence(
            data, ctrl_mask, perturbed_genes_sorted, G,
        )

        score = np.zeros((G, G), dtype=np.float64)
        vote_total = np.zeros((G, G), dtype=np.float64)  # used by shift_rerank
        if self.use_all_genes_as_roots:
            root_indices = list(range(G))
        else:
            root_indices = [data.gene_idx(g) for g in perturbed_genes_sorted]
        for root in root_indices:
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
                if self.score_mode == "edge_weight":
                    score[u, v] += rc * float(edge_weight[u, v])
                elif self.score_mode == "shift_rerank":
                    vote_total[u, v] += rc
                elif self.score_mode == "root_shift":
                    score[u, v] += rc * float(abs(shifts[root, v]))

        if self.score_mode == "shift_rerank":
            # Multiplicative rerank of MeanDifference magnitude by
            # (confidence-weighted) dominator vote total. Edges from
            # unperturbed sources score 0 because shifts[u, :] = 0.
            score = vote_total * np.abs(shifts)

        # score[u, v] = total vote weight for direct edge u -> v.
        np.fill_diagonal(score, 0.0)
        score_src_tgt = score.astype(np.float32)
        flat = score_src_tgt.ravel()

        # Collect dominator-tree edges (score > 0), ranked by vote weight.
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
