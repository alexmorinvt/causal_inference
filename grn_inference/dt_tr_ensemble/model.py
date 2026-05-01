"""DominatorTree + TransitiveReduction score-sum ensemble.

Both methods share the same edge-weight construction (shifts, pooled_std,
IV imputation, root_conf) computed once. Each runs on its own graph with
its own tuned threshold:
  - DominatorTree graph: per-source quantile 0.94
  - TransitiveReduction graph: global quantile 0.80

Scores are converted to percentile ranks within their non-zero entries,
summed, then masked to the intersection: only edges that received votes
from *both* methods appear in the output. This removes union noise
(DT-only entries filling the tail at large k) while giving double weight
to edges both structural criteria agree on.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..dataset import Dataset

Edge = tuple[str, str]


def _build_edge_weight(
    data: Dataset,
) -> tuple[np.ndarray, list[str], np.ndarray, int]:
    G = data.n_genes
    ctrl_mask = data.control_mask()
    perturbed_genes_sorted = sorted(data.perturbed_genes())
    pert_mask = np.zeros(G, dtype=bool)
    for g in perturbed_genes_sorted:
        pert_mask[data.gene_idx(g)] = True

    ctrl_expr = data.expression[ctrl_mask].astype(np.float64)
    ctrl_mean = ctrl_expr.mean(axis=0)
    n_ctrl = int(ctrl_mask.sum())
    ctrl_var = ctrl_expr.var(axis=0)

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

    denom = np.where(pooled_std > 0, pooled_std, 1.0)
    edge_weight = np.abs(shifts) / denom

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
            target_nz = edge_weight[pert_idx, :]
            target_nz = target_nz[target_nz > 0.0]
            iv_nz = np.abs(beta_iv[unpert_idx, :])
            iv_nz_pos = iv_nz[iv_nz > 0.0]
            scale = (
                target_nz.mean() / iv_nz_pos.mean()
                if target_nz.size > 0 and iv_nz_pos.size > 0
                else 1.0
            )
            for s in unpert_idx:
                edge_weight[s, :] = np.abs(beta_iv[s, :]) * scale

    np.fill_diagonal(edge_weight, 0.0)
    return edge_weight, perturbed_genes_sorted, pert_mask, G


def _compute_root_confidence(
    data: Dataset,
    ctrl_mask: np.ndarray,
    perturbed_genes_sorted: list[str],
    G: int,
) -> np.ndarray:
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
            u_stat, _ = mannwhitneyu(x_pert, x_ctrl, alternative="two-sided")
        except ValueError:
            continue
        mu = n_p * n_ctrl / 2.0
        sigma = np.sqrt(n_p * n_ctrl * (n_p + n_ctrl + 1) / 12.0)
        if sigma > 0:
            conf[idx] = abs((u_stat - mu) / sigma)
    return conf


def _dt_mask(edge_weight: np.ndarray, quantile: float) -> np.ndarray:
    G = edge_weight.shape[0]
    mask = np.zeros_like(edge_weight, dtype=bool)
    for s in range(G):
        row = edge_weight[s, :]
        row_nz = row[row > 0.0]
        if row_nz.size == 0:
            continue
        cutoff = float(np.quantile(row_nz, quantile))
        mask[s, :] = (row >= cutoff) & (row > 0.0)
    np.fill_diagonal(mask, False)
    return mask


def _tr_mask(edge_weight: np.ndarray, quantile: float) -> np.ndarray:
    nz = edge_weight[edge_weight > 0.0]
    if nz.size == 0:
        return np.zeros_like(edge_weight, dtype=bool)
    cutoff = float(np.quantile(nz, quantile))
    mask = (edge_weight >= cutoff) & (edge_weight > 0.0)
    np.fill_diagonal(mask, False)
    return mask


def _build_graph(mask: np.ndarray, edge_weight: np.ndarray) -> nx.DiGraph:
    G = mask.shape[0]
    graph = nx.DiGraph()
    graph.add_nodes_from(range(G))
    for s, t in zip(*np.nonzero(mask)):
        graph.add_edge(int(s), int(t), weight=float(edge_weight[s, t]))
    return graph


def _dt_votes(
    graph: nx.DiGraph,
    edge_weight: np.ndarray,
    perturbed_genes_sorted: list[str],
    root_conf: np.ndarray,
    data: Dataset,
    G: int,
) -> np.ndarray:
    score = np.zeros((G, G), dtype=np.float64)
    for g in perturbed_genes_sorted:
        root = data.gene_idx(g)
        rc = float(root_conf[root])
        if rc <= 0.0 or root not in graph:
            continue
        try:
            idoms = nx.immediate_dominators(graph, root)
        except Exception:
            continue
        for v, u in idoms.items():
            if v == u or v == root:
                continue
            score[u, v] += rc * float(edge_weight[u, v])
    return score


def _tr_votes(
    graph: nx.DiGraph,
    edge_weight: np.ndarray,
    perturbed_genes_sorted: list[str],
    root_conf: np.ndarray,
    data: Dataset,
    G: int,
) -> np.ndarray:
    score = np.zeros((G, G), dtype=np.float64)
    for g in perturbed_genes_sorted:
        root = data.gene_idx(g)
        rc = float(root_conf[root])
        if rc <= 0.0 or root not in graph:
            continue
        dist = nx.single_source_shortest_path_length(graph, root)
        if len(dist) <= 1:
            continue
        dag = nx.DiGraph()
        dag.add_nodes_from(dist.keys())
        for u in dist:
            for v in graph.successors(u):
                if v in dist and dist[v] > dist[u]:
                    dag.add_edge(u, v)
        if dag.number_of_edges() == 0:
            continue
        try:
            tr_dag = nx.transitive_reduction(dag)
        except Exception:
            continue
        for u, v in tr_dag.edges():
            score[u, v] += rc * float(edge_weight[u, v])
    return score


def _percentile_rank(score: np.ndarray) -> np.ndarray:
    """Map non-zero entries to percentile ranks in (0, 1]; zeros stay zero."""
    ranked = np.zeros_like(score, dtype=np.float64)
    nz_mask = score > 0.0
    if not nz_mask.any():
        return ranked
    nz_vals = score[nz_mask]
    ranks = np.argsort(np.argsort(nz_vals))
    ranked[nz_mask] = (ranks + 1) / len(nz_vals)
    return ranked


@dataclass
class DTTREnsemble:
    """Score-sum ensemble of DominatorTreeModel and TransitiveReductionModel.

    Each method runs on its own graph with its own tuned threshold. Scores
    are percentile-rank normalised, summed, then masked to the intersection:
    only edges voted for by *both* methods appear in the output.

    Parameters
    ----------
    top_k
        Number of ranked edges to return.
    dt_shift_quantile
        Per-source quantile threshold for the DominatorTree graph.
        Default 0.94 matches DominatorTreeModel.
    tr_shift_quantile
        Global quantile threshold for the TransitiveReduction graph.
        Default 0.80 matches TransitiveReductionModel.

    Notes
    -----
    Benchmarked against score-sum-union and union-graph variants on
    seeds 7, 42, 123 (n_genes=50, n_perturbed=25). This variant wins
    at every top_k ≥ 100, with the largest gains at k=500/1000
    (+7–13 pp over the next-best method) due to the intersection mask
    eliminating low-quality DT-only tail entries.
    """

    top_k: int = 1000
    dt_shift_quantile: float = 0.94
    tr_shift_quantile: float = 0.80

    def fit_predict(self, data: Dataset) -> list[Edge]:
        ctrl_mask = data.control_mask()
        if not ctrl_mask.any():
            raise ValueError("No control cells; cannot compute shifts.")
        edge_weight, perturbed_genes_sorted, pert_mask, G = _build_edge_weight(data)
        if not (edge_weight > 0).any():
            return []
        root_conf = _compute_root_confidence(
            data, ctrl_mask, perturbed_genes_sorted, G,
        )
        dt_graph = _build_graph(_dt_mask(edge_weight, self.dt_shift_quantile), edge_weight)
        tr_graph = _build_graph(_tr_mask(edge_weight, self.tr_shift_quantile), edge_weight)
        score_dt = _dt_votes(dt_graph, edge_weight, perturbed_genes_sorted, root_conf, data, G)
        score_tr = _tr_votes(tr_graph, edge_weight, perturbed_genes_sorted, root_conf, data, G)
        score = _percentile_rank(score_dt) + _percentile_rank(score_tr)
        score[(score_dt == 0.0) | (score_tr == 0.0)] = 0.0
        np.fill_diagonal(score, 0.0)
        flat = score.astype(np.float32).ravel()
        order = np.argsort(-flat)
        order = order[flat[order] > 0.0]
        names = data.gene_names
        result: list[Edge] = []
        for idx in order[: self.top_k]:
            u, v = divmod(int(idx), G)
            if u != v:
                result.append((names[u], names[v]))
        return result
