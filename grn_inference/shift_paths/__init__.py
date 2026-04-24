"""Shift-paths GRN inference.

Simple heuristic method: read the interventional shift matrix, threshold
to a sparse directed graph, then iteratively pick one edge at a time by
scoring paths with an exponentially-decaying per-hop weight and tallying
the final edge of each source's best path into the current top sink.

Public entry point: :class:`ShiftPathsModel`.
"""

from .model import ShiftPathsModel

__all__ = ["ShiftPathsModel"]
