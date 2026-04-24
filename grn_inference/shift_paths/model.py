"""Shift-paths ranker.

Pipeline
--------
1. Compute the shift matrix ``S[i, j] = |mean(gene_j | do(gene_i)) -
   mean(gene_j | control)|`` for every perturbed source ``i`` and every
   measured target ``j``.

2. Threshold: keep the top ``top_frac`` fraction of off-diagonal entries
   as candidate edges. The survivors form a directed graph ``H``.

3. Re-rank nodes by ``in_degree(H) - out_degree(H)`` (descending). The
   top node is the most "sink-like" gene ``N``.

4. For every other gene ``G``, enumerate simple paths ``G -> ... -> N``
   in ``H`` up to ``max_path_length`` hops. Score each path with an
   exponentially-decaying weight that is *largest on the final hop*::

       score(path) = sum_{i=0..L-1}  S[v_i, v_{i+1}] / 2^(L-1-i)

   So for a 3-hop path, the weights are (1/4, 1/2, 1) — the edge that
   actually terminates at ``N`` carries the dominant contribution, with
   upstream hops as successively smaller corrections. Keep only the
   highest-scoring path per source. Each retained path contributes its
   score to the tally of its *final* edge ``(v_{L-1}, v_L)``. Multiple
   sources can stack on the same final edge.

5. Pick the final edge with the highest cumulative tally. Record it,
   remove it from ``H``, and go back to step 3.

6. Stop once we have ``top_k`` edges, or no sink-like node has any
   incoming path left.

Incremental recomputation
-------------------------
The expensive step is path enumeration (step 4). We cache, per sink
``T``, the best-scoring path from each source ``G``. When edge
``(A, B)`` is removed at the end of step 5, only cached paths that
actually traverse ``(A, B)`` become invalid — every other cached path
still exists in the graph, still scores the same, and is still the
best option for its source. We evict only the invalidated entries;
the next time a sink is visited we only recompute the missing sources.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import networkx as nx
import numpy as np

from ..dataset import Dataset

Edge = tuple[str, str]


@dataclass
class ShiftPathsModel:
    """Rank edges by discounted path score with iterative re-ranking.

    Parameters
    ----------
    top_k
        Maximum number of edges to return. Iteration also stops early
        if the graph runs out of paths to sinks.
    top_frac
        Fraction of off-diagonal (source, target) pairs to keep when
        seeding the graph from the shift matrix.
    max_path_length
        Upper bound on the number of edges (hops) in a scored path.
    """

    top_k: int = 1000
    top_frac: float = 0.30
    max_path_length: int = 5
    verbose: bool = False
    debug: bool = False

    def fit_predict(self, data: Dataset) -> list[Edge]:
        # ---- 1. Shift matrix (G, G), zeros outside the perturbed rows --
        ctrl_mask = data.control_mask()
        if not ctrl_mask.any():
            raise ValueError("No control cells; cannot compute shifts.")
        ctrl_means = data.expression[ctrl_mask].mean(axis=0)

        G = data.n_genes
        shift = np.zeros((G, G), dtype=np.float32)
        for src in data.perturbed_genes():
            mask = data.intervention_mask(src)
            if not mask.any():
                continue
            src_idx = data.gene_idx(src)
            intv_means = data.expression[mask].mean(axis=0)
            shift[src_idx, :] = np.abs(intv_means - ctrl_means)
        np.fill_diagonal(shift, 0.0)

        name_to_idx = {n: i for i, n in enumerate(data.gene_names)}

        # ---- 2. Threshold to the top fraction of directed pairs -------
        total_pairs = G * (G - 1)
        k_keep = max(1, int(round(self.top_frac * total_pairs)))
        flat = shift.ravel()
        if k_keep >= flat.size:
            cutoff = 0.0
        else:
            cutoff = float(-np.partition(-flat, k_keep - 1)[k_keep - 1])

        graph = nx.DiGraph()
        for name in data.gene_names:
            graph.add_node(name)
        for i in range(G):
            for j in range(G):
                if i == j:
                    continue
                if shift[i, j] >= cutoff and shift[i, j] > 0.0:
                    graph.add_edge(data.gene_names[i], data.gene_names[j])

        # ---- 3-5. Iterate with per-sink best-path cache ---------------
        max_hops = self.max_path_length
        inv_powers = [1.0 / (2 ** i) for i in range(max_hops + 1)]

        def path_score(path: list[str]) -> float:
            L = len(path) - 1
            s = 0.0
            for hop_idx in range(L):
                a = name_to_idx[path[hop_idx]]
                b = name_to_idx[path[hop_idx + 1]]
                s += float(shift[a, b]) * inv_powers[L - 1 - hop_idx]
            return s

        def find_best_path(
            source: str, target: str
        ) -> tuple[float, list[str]] | None:
            best: tuple[float, list[str]] | None = None
            for path in nx.all_simple_paths(
                graph, source, target, cutoff=max_hops
            ):
                if len(path) < 2:
                    continue
                s = path_score(path)
                if best is None or s > best[0]:
                    best = (s, path)
            return best

        def path_uses_edge(path: list[str], edge: Edge) -> bool:
            a, b = edge
            for i in range(len(path) - 1):
                if path[i] == a and path[i + 1] == b:
                    return True
            return False

        # sink -> {source: (score, path) or None}. ``None`` means we
        # already checked and no simple path within ``max_hops`` exists;
        # that can't become wrong because removing edges only removes
        # paths. Tuple entries are invalidated only when the winner
        # edge lies on the cached path.
        Entry = tuple[float, list[str]] | None
        best_paths_cache: dict[str, dict[str, Entry]] = {}

        predicted: list[Edge] = []
        t_start = time.time()
        while len(predicted) < self.top_k and graph.number_of_edges() > 0:
            if self.verbose:
                print(
                    f"[shift_paths] iter={len(predicted):4d} "
                    f"edges_left={graph.number_of_edges():5d} "
                    f"elapsed={time.time() - t_start:6.1f}s",
                    file=sys.stderr, flush=True,
                )
            sinks = sorted(
                (n for n in graph.nodes if graph.in_degree(n) > 0),
                key=lambda n: -(graph.in_degree(n) - graph.out_degree(n)),
            )
            winner: Edge | None = None
            for target in sinks:
                target_cache = best_paths_cache.setdefault(target, {})
                for source in graph.nodes:
                    if source == target or source in target_cache:
                        continue
                    target_cache[source] = find_best_path(source, target)

                edge_scores: dict[Edge, float] = {}
                edge_contribs: dict[Edge, list[tuple[str, float, list[str]]]] = {}
                for source, entry in target_cache.items():
                    if entry is None:
                        continue
                    s, path = entry
                    edge = (path[-2], path[-1])
                    edge_scores[edge] = edge_scores.get(edge, 0.0) + s
                    edge_contribs.setdefault(edge, []).append((source, s, path))

                if edge_scores:
                    winner = max(edge_scores, key=edge_scores.get)
                    if self.debug:
                        print(
                            f"\n=== iter {len(predicted)}  sink={target} "
                            f"(in={graph.in_degree(target)}, "
                            f"out={graph.out_degree(target)}) ===",
                            flush=True,
                        )
                        ranked = sorted(
                            edge_scores.items(), key=lambda kv: -kv[1]
                        )
                        for edge, total in ranked:
                            contribs = edge_contribs[edge]
                            mark = " <-- WINNER" if edge == winner else ""
                            print(
                                f"  {edge[0]} -> {edge[1]}: "
                                f"total={total:.4f}  "
                                f"n_paths={len(contribs)}{mark}",
                                flush=True,
                            )
                            for source, s, path in sorted(
                                contribs, key=lambda t: -t[1]
                            ):
                                path_str = " -> ".join(path)
                                print(
                                    f"      {source}:  {path_str}  "
                                    f"score={s:.4f}",
                                    flush=True,
                                )
                    break

            if winner is None:
                break
            predicted.append(winner)
            graph.remove_edge(*winner)

            # Invalidate only the cached tuples whose path traversed
            # the winner edge; ``None`` entries stay — unreachable
            # stays unreachable when you remove an edge.
            for tgt_cache in best_paths_cache.values():
                to_evict = [
                    src
                    for src, entry in tgt_cache.items()
                    if entry is not None and path_uses_edge(entry[1], winner)
                ]
                for src in to_evict:
                    del tgt_cache[src]

        return predicted[: self.top_k]
