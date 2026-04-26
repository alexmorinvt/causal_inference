"""Tests for grn_inference.dominator_tree."""

from __future__ import annotations

import numpy as np
import pytest

from grn_inference import (
    DominatorTreeModel,
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
    edges = DominatorTreeModel(top_k=30).fit_predict(data)
    assert len(edges) <= 30
    gene_set = set(data.gene_names)
    for s, t in edges:
        assert s in gene_set
        assert t in gene_set
        assert s != t
    assert len(set(edges)) == len(edges)


def test_returns_unperturbed_source_edges():
    """edge_weight mode keeps the IV-imputed branch alive for unperturbed
    sources; this test guards that behaviour. The default root_shift mode
    intentionally drops it (vote weight = 0 when shifts[root, v] = 0).
    """
    data, _ = make_synthetic_dataset(
        n_genes=20,
        edge_density=0.15,
        n_control_cells=500,
        n_cells_per_perturbation=100,
        n_perturbed_genes=10,
        seed=1,
    )
    edges = DominatorTreeModel(
        top_k=100, score_mode="edge_weight"
    ).fit_predict(data)
    assert len(edges) > 0


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
    dt_edges = DominatorTreeModel(top_k=50).fit_predict(data)
    rb_edges = RandomBaseline(top_k=50, seed=0).fit_predict(data)
    dt_prec = sum(1 for e in dt_edges if e in true_set) / max(len(dt_edges), 1)
    rb_prec = sum(1 for e in rb_edges if e in true_set) / max(len(rb_edges), 1)
    assert dt_prec >= rb_prec, f"DT precision {dt_prec} < Random {rb_prec}"


@pytest.mark.parametrize("score_mode", ["edge_weight", "shift_rerank", "root_shift"])
def test_score_mode_returns_valid_edges(score_mode):
    data, _ = make_synthetic_dataset(
        n_genes=20,
        edge_density=0.15,
        n_control_cells=500,
        n_cells_per_perturbation=100,
        n_perturbed_genes=10,
        seed=3,
    )
    edges = DominatorTreeModel(top_k=30, score_mode=score_mode).fit_predict(data)
    assert 0 < len(edges) <= 30
    gene_set = set(data.gene_names)
    for s, t in edges:
        assert s in gene_set and t in gene_set and s != t
    assert len(set(edges)) == len(edges)


@pytest.mark.parametrize("score_mode", ["edge_weight", "shift_rerank", "root_shift"])
def test_score_mode_beats_random_on_precision(score_mode):
    data, truth = make_synthetic_dataset(
        n_genes=25,
        edge_density=0.15,
        n_control_cells=800,
        n_cells_per_perturbation=120,
        n_perturbed_genes=12,
        seed=4,
    )
    true_set = set(truth.true_edges)
    dt_edges = DominatorTreeModel(top_k=50, score_mode=score_mode).fit_predict(data)
    rb_edges = RandomBaseline(top_k=50, seed=0).fit_predict(data)
    dt_prec = sum(1 for e in dt_edges if e in true_set) / max(len(dt_edges), 1)
    rb_prec = sum(1 for e in rb_edges if e in true_set) / max(len(rb_edges), 1)
    assert dt_prec >= rb_prec, (
        f"DT[{score_mode}] precision {dt_prec} < Random {rb_prec}"
    )


def test_unknown_score_mode_raises():
    data, _ = make_synthetic_dataset(
        n_genes=10, n_control_cells=100, n_cells_per_perturbation=50,
        n_perturbed_genes=5, seed=0,
    )
    with pytest.raises(ValueError, match="Unknown score_mode"):
        DominatorTreeModel(score_mode="bogus").fit_predict(data)
