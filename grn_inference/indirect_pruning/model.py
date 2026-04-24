"""Indirect-path pruning ranker.

Pipeline
--------
1. Compute the shift matrix ``S[i, j] = |mean(gene_j | do(gene_i)) -
   mean(gene_j | control)|`` for every perturbed source ``i`` and every
   measured target ``j``.

2. Threshold: keep the top ``top_frac`` fraction of off-diagonal entries
   as candidate edges. The survivors form a directed graph ``H``.

3. For every candidate edge ``A -> B`` in ``H``, compute its best
   two-hop alternative explanation::

       indirect(A, B) = max over intermediates C in H:
                       (S(A -> C) + S(C -> B)) / 2

   i.e. the strongest intermediate path through some ``C`` where both
   ``A -> C`` and ``C -> B`` are themselves candidate edges.

4. Remove ``A -> B`` if ``indirect(A, B) / S(A -> B) > prune_ratio``.
   The intuition: if a two-hop alternative has nearly the same average
   edge strength as the direct shift, the "direct" edge is likely a
   cascade shortcut, not a true regulatory link. True direct edges
   either have no two-hop alternative at all (source has no out-edges
   to another parent of the target) or have a much weaker alternative
   dragged down by a noise-level edge.

5. Rank the surviving edges by ``S(A -> B)`` and return the top ``top_k``.

The threshold ``prune_ratio`` has a sweet spot: too low (e.g. 0.5)
removes real edges that happen to share paths; too high (e.g. 0.9) lets
obvious cascade shortcuts through. 0.7 has worked well on the 7-gene
benchmark but should be swept as more test graphs appear.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dataset import Dataset

Edge = tuple[str, str]


@dataclass
class IndirectPruningModel:
    """Rank edges by shift, after pruning cascade shortcuts.

    Parameters
    ----------
    top_k
        Upper bound on the number of edges returned.
    top_frac
        Fraction of off-diagonal (source, target) pairs to keep when
        building the candidate graph from the shift matrix.
    prune_ratio
        Prune an edge if its best two-hop alternative averages to more
        than this fraction of the edge's own shift. ``0.7`` is the
        working default.
    """

    top_k: int = 1000
    top_frac: float = 0.30
    prune_ratio: float = 0.90

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

        # ---- 2. Threshold to the top fraction of directed pairs -------
        total_pairs = G * (G - 1)
        k_keep = max(1, int(round(self.top_frac * total_pairs)))
        flat = shift.ravel()
        if k_keep >= flat.size:
            cutoff = 0.0
        else:
            cutoff = float(-np.partition(-flat, k_keep - 1)[k_keep - 1])

        # Boolean adjacency of the thresholded graph.
        in_graph = (shift >= cutoff) & (shift > 0.0)
        np.fill_diagonal(in_graph, False)

        # ---- 3. Two-hop alternative score per edge -------------------
        # For each edge (A, B) in the graph, find the best intermediate
        # C such that (A, C) and (C, B) are also in the graph; score is
        # the average of the two hop shifts.
        edges = list(zip(*np.where(in_graph)))
        indirect: dict[tuple[int, int], float] = {}
        for a, b in edges:
            best = 0.0
            for c in range(G):
                if c == a or c == b:
                    continue
                if in_graph[a, c] and in_graph[c, b]:
                    avg = 0.5 * (shift[a, c] + shift[c, b])
                    if avg > best:
                        best = avg
            indirect[(a, b)] = best

        # ---- 4. Prune edges well-explained by a two-hop path ---------
        survivors: list[tuple[int, int, float]] = []
        for (a, b), ind in indirect.items():
            direct = float(shift[a, b])
            if direct == 0.0:
                continue
            if ind / direct > self.prune_ratio:
                continue
            survivors.append((a, b, direct))

        # ---- 5. Rank by shift, take top_k -----------------------------
        survivors.sort(key=lambda e: -e[2])
        names = data.gene_names
        return [
            (names[a], names[b]) for a, b, _ in survivors[: self.top_k]
        ]
