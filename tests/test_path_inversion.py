"""Tests for grn_inference.path_inversion."""

from __future__ import annotations

import numpy as np

from grn_inference import (
    PathInversionModel,
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
    edges = PathInversionModel(top_k=30).fit_predict(data)
    assert len(edges) <= 30
    gene_set = set(data.gene_names)
    for s, t in edges:
        assert s in gene_set
        assert t in gene_set
        assert s != t
    assert len(set(edges)) == len(edges)


def test_returns_unperturbed_source_edges_when_obs_weight_positive():
    data, _ = make_synthetic_dataset(
        n_genes=20,
        edge_density=0.15,
        n_control_cells=500,
        n_cells_per_perturbation=100,
        n_perturbed_genes=10,
        seed=1,
    )
    perturbed = set(data.perturbed_genes())
    edges = PathInversionModel(top_k=100, obs_correlation_weight=1.0).fit_predict(data)
    n_unpert = sum(1 for s, _ in edges if s not in perturbed)
    assert n_unpert > 0, (
        "PathInversionModel with obs_correlation_weight>0 should return "
        "some edges whose source is unperturbed."
    )


def test_no_unperturbed_edges_when_correlation_imputation_is_zero():
    data, _ = make_synthetic_dataset(
        n_genes=20,
        edge_density=0.15,
        n_control_cells=500,
        n_cells_per_perturbation=100,
        n_perturbed_genes=10,
        seed=2,
    )
    perturbed = set(data.perturbed_genes())
    edges = PathInversionModel(
        top_k=100,
        imputation_mode="correlation",
        obs_correlation_weight=0.0,
    ).fit_predict(data)
    # With correlation mode and zero obs weight, unperturbed T columns
    # stay at zero, so no edges with unperturbed source.
    for s, _ in edges:
        assert s in perturbed, (
            "With correlation imputation + obs_correlation_weight=0 "
            "no unperturbed-source edges should appear."
        )


def test_beats_random_on_w1():
    data, _ = make_synthetic_dataset(
        n_genes=25,
        edge_density=0.15,
        n_control_cells=800,
        n_cells_per_perturbation=120,
        n_perturbed_genes=12,
        seed=3,
    )
    pi = PathInversionModel(top_k=50)
    rb = RandomBaseline(top_k=50, seed=0)
    pi_edges = pi.fit_predict(data)
    rb_edges = rb.fit_predict(data)

    pi_res = evaluate_statistical(
        pi_edges, data, omission_sample_size=200,
        rng=np.random.default_rng(99),
    )
    rb_res = evaluate_statistical(
        rb_edges, data, omission_sample_size=200,
        rng=np.random.default_rng(99),
    )
    assert pi_res.mean_wasserstein >= rb_res.mean_wasserstein
