"""Transitive-reduction GRN inference.

For each perturbed root R, builds a BFS-layered DAG from the global
thresholded shift graph and computes its transitive reduction. An edge
(u, v) survives iff no alternative forward path u→w→...→v exists from
R's vantage point — making it structurally indispensable, not a cascade
shortcut. Votes are aggregated across all perturbed roots, weighted by
MW_z(R) × edge_weight[u, v].

Complements :class:`~grn_inference.DominatorTreeModel`: dominators
identify edges every path *from R must cross*; TR identifies edges with
*no parallel detour*, regardless of root position.

Public entry point: :class:`TransitiveReductionModel`.
"""

from .model import TransitiveReductionModel

__all__ = ["TransitiveReductionModel"]
