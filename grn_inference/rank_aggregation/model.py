"""Rank-aggregation edge ranker.

Run each base estimator at a shared ``top_k``, extract the ranked edge
lists, and convert each into a rank-percentile score per edge. Edges
present in the top-`k` of one base estimator get a percentile in
``(0, 1]``; absent edges get 0. Aggregate across base estimators by
summing percentiles, then take the ``top_k`` by aggregated score.

This is classical rank aggregation (Borda-count style). Distinct
estimators with different inductive biases (observational regression,
path-inversion, differential covariance, dominator tree) fail in
different ways — their agreement is evidence of a true edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..dataset import Dataset

Edge = tuple[str, str]


@dataclass
class RankAggregationModel:
    """Aggregate rankings from multiple base estimators via rank-percentile sum.

    Parameters
    ----------
    top_k
        Number of ranked edges to return.
    base_top_k
        top_k passed to each base estimator. Usually equal to
        ``top_k`` but can be larger to get richer rank information.
    include_nr, include_pi, include_dc, include_dt, include_icp
        Toggle which base estimator contributes to the ensemble.
    aggregation
        - ``"sum"`` (iter-31 default): sum of weighted rank-percentiles.
          Equal weights, four estimators (NR, PI, DC, DT). ICP cannot
          be added cleanly with sum because its precision dilutes the
          ensemble.
        - ``"product"`` (iter-41 default): geometric ensemble. Product
          of ``percentile^weight`` per edge. Penalises edges absent
          from any single ranker (it gets pct≈0). With ICP weighted
          higher (weight 3) the ensemble inherits ICP's hidden-recall
          advantage without letting its lower precision bog down the
          rest.
    weights
        Optional per-estimator weights (must match the number of
        enabled estimators in the order NR, PI, DC, DT, ICP). ``None``
        means equal weighting.
    """

    top_k: int = 1000
    base_top_k: int = 1000
    include_nr: bool = True
    include_pi: bool = True
    include_dc: bool = True
    include_dt: bool = True
    include_icp: bool = False
    aggregation: str = "sum"
    weights: tuple[float, ...] | None = None

    def fit_predict(self, data: Dataset) -> list[Edge]:
        # Import inside fit_predict to avoid import-time cycles across
        # grn_inference package init.
        from ..diff_cov import DiffCovModel
        from ..dominator_tree import DominatorTreeModel
        from ..invariance_icp import InvarianceICPModel
        from ..neighborhood_regression import NeighborhoodRegressionModel
        from ..path_inversion import PathInversionModel

        estimators = []
        if self.include_nr:
            estimators.append(NeighborhoodRegressionModel(top_k=self.base_top_k))
        if self.include_pi:
            estimators.append(PathInversionModel(top_k=self.base_top_k))
        if self.include_dc:
            estimators.append(DiffCovModel(top_k=self.base_top_k))
        if self.include_dt:
            estimators.append(DominatorTreeModel(top_k=self.base_top_k))
        if self.include_icp:
            estimators.append(InvarianceICPModel(top_k=self.base_top_k))

        if not estimators:
            return []

        if self.weights is None:
            weights = [1.0] * len(estimators)
        else:
            weights = list(self.weights)
            if len(weights) != len(estimators):
                raise ValueError(
                    f"weights length {len(weights)} != number of active "
                    f"estimators {len(estimators)}"
                )

        if self.aggregation == "sum":
            agg_score: dict[Edge, float] = {}
            for est, w in zip(estimators, weights):
                edges = est.fit_predict(data)
                n = len(edges)
                if n == 0 or w == 0.0:
                    continue
                for rank, edge in enumerate(edges):
                    pct = (n - rank) / n
                    agg_score[edge] = agg_score.get(edge, 0.0) + w * pct
            ordered = sorted(agg_score.items(), key=lambda kv: -kv[1])
            return [edge for edge, _ in ordered[: self.top_k]]

        elif self.aggregation == "product":
            # Product of pct^weight per estimator; missing edges contribute 0.
            edge_lists = []
            edge_pcts = []
            for est, w in zip(estimators, weights):
                edges = est.fit_predict(data)
                n = len(edges)
                pcts = {}
                if n > 0 and w != 0.0:
                    for rank, edge in enumerate(edges):
                        pcts[edge] = (n - rank) / n
                edge_lists.append(edges)
                edge_pcts.append(pcts)
            all_edges = set()
            for el in edge_lists:
                all_edges.update(el)
            agg: dict[Edge, float] = {}
            for edge in all_edges:
                p = 1.0
                for w, pcts in zip(weights, edge_pcts):
                    if w == 0.0:
                        continue
                    pct = pcts.get(edge, 0.0)
                    if pct == 0.0:
                        p = 0.0
                        break
                    p *= pct ** w
                if p > 0.0:
                    agg[edge] = p
            ordered = sorted(agg.items(), key=lambda kv: -kv[1])
            return [edge for edge, _ in ordered[: self.top_k]]
        else:
            raise ValueError(
                f"Unknown aggregation={self.aggregation!r}; "
                "expected 'sum' or 'product'."
            )
