"""Dominator-tree GRN inference.

For a directed graph ``G_shift`` built from the shift matrix and any
source node ``S``, the **dominator tree** rooted at ``S`` encodes which
nodes lie on every path from ``S``: each node ``v`` reachable from
``S`` has a unique immediate dominator ``idom(v)`` such that removing
``idom(v)`` from the graph disconnects ``v`` from ``S``. Immediate
dominator pairs are exactly the graph-theoretic candidates for direct
causal parents — if any path from ``S`` to ``v`` must pass through
``u``, then ``u`` is uniquely positioned to be ``v``'s proximal
regulator.

For each perturbed gene ``G`` we build a thresholded shift graph,
compute the dominator tree rooted at ``G``, and emit the
immediate-dominator edges with their shift-magnitude weights as
candidate direct edges. Aggregating across perturbed sources gives a
vote-weighted ranking of directed edges.

For unperturbed sources, we use an analogous construction with the
cross-arm IV shift regression in place of direct shifts.

Public entry point: :class:`DominatorTreeModel`.
"""

from .model import DominatorTreeModel

__all__ = ["DominatorTreeModel"]
