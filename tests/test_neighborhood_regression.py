"""Tests for grn_inference.neighborhood_regression."""

from __future__ import annotations

import numpy as np

from grn_inference import (
    MeanDifferenceModel,
    NeighborhoodRegressionModel,
    RandomBaseline,
    evaluate_statistical,
    make_synthetic_dataset,
)


def test_fit_predict_returns_valid_edge_list():
    data, _ = make_synthetic_dataset(
        n_genes=15,
        edge_density=0.15,
        n_control_cells=500,
        n_cells_per_perturbation=80,
        n_perturbed_genes=8,
        seed=0,
    )
    model = NeighborhoodRegressionModel(top_k=30)
    edges = model.fit_predict(data)

    assert len(edges) <= 30
    gene_set = set(data.gene_names)
    for s, t in edges:
        assert s in gene_set
        assert t in gene_set
        assert s != t
    # No duplicates.
    assert len(set(edges)) == len(edges)


def test_produces_unperturbed_source_edges():
    """The method's whole point: source distribution must include
    genes that have no intervention arm."""
    data, _ = make_synthetic_dataset(
        n_genes=20,
        edge_density=0.15,
        n_control_cells=500,
        n_cells_per_perturbation=100,
        n_perturbed_genes=10,
        seed=1,
    )
    perturbed = set(data.perturbed_genes())
    model = NeighborhoodRegressionModel(top_k=100, unperturbed_fraction=0.5)
    edges = model.fit_predict(data)

    n_unpert_sources = sum(1 for s, _ in edges if s not in perturbed)
    assert n_unpert_sources > 0, (
        "NeighborhoodRegressionModel should produce edges whose source "
        "is unperturbed; that is the entire point of the method."
    )


def test_beats_random_on_w1():
    """Sanity floor: must not drop below RandomBaseline on mean W1 of
    evaluable edges."""
    data, _ = make_synthetic_dataset(
        n_genes=25,
        edge_density=0.15,
        n_control_cells=800,
        n_cells_per_perturbation=120,
        n_perturbed_genes=12,
        seed=2,
    )
    nr = NeighborhoodRegressionModel(top_k=50)
    rb = RandomBaseline(top_k=50, seed=0)

    nr_edges = nr.fit_predict(data)
    rb_edges = rb.fit_predict(data)

    nr_res = evaluate_statistical(
        nr_edges, data, omission_sample_size=200,
        rng=np.random.default_rng(99),
    )
    rb_res = evaluate_statistical(
        rb_edges, data, omission_sample_size=200,
        rng=np.random.default_rng(99),
    )
    assert nr_res.mean_wasserstein >= rb_res.mean_wasserstein


def test_equals_mean_difference_when_unperturbed_fraction_is_zero():
    """With unperturbed_fraction=0, the method collapses to MD on the
    perturbed-source bucket. The top-k rankings should match MD's
    exactly (both sort the same shift matrix)."""
    data, _ = make_synthetic_dataset(
        n_genes=20,
        edge_density=0.15,
        n_control_cells=500,
        n_cells_per_perturbation=100,
        n_perturbed_genes=10,
        seed=3,
    )
    # within_arm_corr_weight=0 disables the per-cell coupling boost
    # so the perturbed bucket scores edges identically to MD.
    nr = NeighborhoodRegressionModel(
        top_k=50, unperturbed_fraction=0.0, within_arm_corr_weight=0.0,
    )
    md = MeanDifferenceModel(top_k=50)

    nr_edges = nr.fit_predict(data)
    md_edges = md.fit_predict(data)

    # NeighborhoodRegressionModel at unpert=0 returns only the
    # perturbed-source bucket, same score as MD. Order may differ by
    # ties, but the sets should match at any stable prefix.
    assert set(nr_edges) == set(md_edges)
