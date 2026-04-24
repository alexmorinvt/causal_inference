"""Shift + within-arm correlation ranker.

Extends :class:`MeanDifferenceModel`'s |shift| score with a multiplicative
boost from the Pearson correlation between source and target *within*
the source's perturbation arm. The hypothesis is that for a direct
``A -> B`` edge, cells receiving stronger knockdown of ``A`` also show a
proportionally stronger shift in ``B`` — a per-cell signal that mean
shift alone can't see.

Public entry point: :class:`ShiftCorrModel`.
"""

from .model import ShiftCorrModel

__all__ = ["ShiftCorrModel"]
