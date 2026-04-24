"""Tests for grn_inference.invariance_icp."""

from __future__ import annotations

from grn_inference import (
    InvarianceICPModel,
    RandomBaseline,
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
    edges = InvarianceICPModel(top_k=30).fit_predict(data)
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
    icp_edges = InvarianceICPModel(top_k=50).fit_predict(data)
    rb_edges = RandomBaseline(top_k=50, seed=0).fit_predict(data)
    icp_prec = sum(1 for e in icp_edges if e in true_set) / max(len(icp_edges), 1)
    rb_prec = sum(1 for e in rb_edges if e in true_set) / max(len(rb_edges), 1)
    assert icp_prec >= rb_prec, f"ICP precision {icp_prec} < Random {rb_prec}"


def test_emits_unperturbed_source_edges():
    data, _ = make_synthetic_dataset(
        n_genes=20,
        edge_density=0.15,
        n_control_cells=500,
        n_cells_per_perturbation=100,
        n_perturbed_genes=10,
        seed=1,
    )
    perturbed = set(data.perturbed_genes())
    edges = InvarianceICPModel(top_k=100).fit_predict(data)
    n_unpert = sum(1 for s, _ in edges if s not in perturbed)
    assert n_unpert > 0
