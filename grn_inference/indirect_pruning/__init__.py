"""Indirect-path pruning GRN inference.

Threshold the shift matrix, then prune any candidate edge whose
"direct" shift is well-explained by a two-hop alternative through some
intermediate gene. The edges that *can't* be explained away are the
survivors — ranked by shift magnitude.

Public entry point: :class:`IndirectPruningModel`.
"""

from .model import IndirectPruningModel

__all__ = ["IndirectPruningModel"]
