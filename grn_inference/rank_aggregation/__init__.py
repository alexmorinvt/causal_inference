"""Rank-aggregation ensemble GRN inference.

For an ensemble of base edge rankers (here: the four in-tree estimator
families — NR, PI, DC, DT — each with its own inductive bias), the
rank-aggregation estimator combines their rankings via rank-percentile
averaging. An edge with consistently high rank across diverse
estimators is more likely a true edge than one ranked high by any
single estimator, which may be biased by family-specific artefacts.

The identification argument is ensemble-style rather than
identifiability-style: different families (observational precision,
total-effect matrix inversion, differential covariance, dominator
tree) fail in different ways. Their agreement is evidence of a true
edge; their disagreement is evidence of a family-specific quirk.

Public entry point: :class:`RankAggregationModel`.
"""

from .model import RankAggregationModel

__all__ = ["RankAggregationModel"]
