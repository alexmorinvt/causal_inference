"""DominatorTree + TransitiveReduction score-sum ensemble.

Runs both structural voting methods on a shared edge-weight matrix but
separate graphs, then sums their score matrices before ranking. Edges
that satisfy both structural criteria receive double weight.

Public entry point: :class:`DTTREnsemble`.
"""

from .model import DTTREnsemble

__all__ = ["DTTREnsemble"]
