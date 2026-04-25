"""Tests for grn_inference.rank_aggregation."""

from __future__ import annotations

from grn_inference import (
    RandomBaseline,
    RankAggregationModel,
    make_synthetic_dataset,
)


def test_fit_predict_returns_valid_edges():
    data, _ = make_synthetic_dataset(
        n_genes=15,
        edge_density=0.15,
        n_control_cells=500,
        n_cells_per_perturbation=80,
        n_perturbed_genes=8,
        seed=0,
    )
    edges = RankAggregationModel(top_k=30).fit_predict(data)
    assert len(edges) <= 30
    gene_set = set(data.gene_names)
    for s, t in edges:
        assert s in gene_set
        assert t in gene_set
        assert s != t
    assert len(set(edges)) == len(edges)


def test_beats_random_on_precision():
    data, truth = make_synthetic_dataset(
        n_genes=25,
        edge_density=0.15,
        n_control_cells=800,
        n_cells_per_perturbation=120,
        n_perturbed_genes=12,
        seed=2,
    )
    true_set = set(truth.true_edges)
    ra_edges = RankAggregationModel(top_k=50).fit_predict(data)
    rb_edges = RandomBaseline(top_k=50, seed=0).fit_predict(data)
    ra_prec = sum(1 for e in ra_edges if e in true_set) / max(len(ra_edges), 1)
    rb_prec = sum(1 for e in rb_edges if e in true_set) / max(len(rb_edges), 1)
    assert ra_prec >= rb_prec, f"RA precision {ra_prec} < Random {rb_prec}"


def test_returns_edges_even_with_subset_of_estimators():
    data, _ = make_synthetic_dataset(
        n_genes=15,
        edge_density=0.15,
        n_control_cells=500,
        n_cells_per_perturbation=80,
        n_perturbed_genes=8,
        seed=3,
    )
    # Just NR + PI; ICP also off; use sum aggregation since product
    # with weights matching estimator count is fragile.
    edges = RankAggregationModel(
        top_k=30,
        include_dc=False,
        include_dt=False,
        include_icp=False,
        aggregation="sum",
        weights=None,
    ).fit_predict(data)
    assert len(edges) > 0
