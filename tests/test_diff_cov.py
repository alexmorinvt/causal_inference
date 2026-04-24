"""Tests for grn_inference.diff_cov."""

from __future__ import annotations

import numpy as np

from grn_inference import (
    DiffCovModel,
    RandomBaseline,
    evaluate_statistical,
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
    edges = DiffCovModel(top_k=30).fit_predict(data)
    assert len(edges) <= 30
    gene_set = set(data.gene_names)
    for s, t in edges:
        assert s in gene_set
        assert t in gene_set
        assert s != t
    assert len(set(edges)) == len(edges)


def test_returns_unperturbed_source_edges():
    data, _ = make_synthetic_dataset(
        n_genes=20,
        edge_density=0.15,
        n_control_cells=500,
        n_cells_per_perturbation=100,
        n_perturbed_genes=10,
        seed=1,
    )
    perturbed = set(data.perturbed_genes())
    edges = DiffCovModel(top_k=100).fit_predict(data)
    n_unpert = sum(1 for s, _ in edges if s not in perturbed)
    assert n_unpert > 0, (
        "DiffCovModel should produce edges with unperturbed sources — "
        "the diff-cov score is symmetric so both directions of each "
        "sensitive pair appear."
    )


def test_beats_random_on_w1():
    data, _ = make_synthetic_dataset(
        n_genes=25,
        edge_density=0.15,
        n_control_cells=800,
        n_cells_per_perturbation=120,
        n_perturbed_genes=12,
        seed=2,
    )
    dc = DiffCovModel(top_k=50)
    rb = RandomBaseline(top_k=50, seed=0)
    dc_edges = dc.fit_predict(data)
    rb_edges = rb.fit_predict(data)
    dc_res = evaluate_statistical(
        dc_edges, data, omission_sample_size=200,
        rng=np.random.default_rng(99),
    )
    rb_res = evaluate_statistical(
        rb_edges, data, omission_sample_size=200,
        rng=np.random.default_rng(99),
    )
    assert dc_res.mean_wasserstein >= rb_res.mean_wasserstein
