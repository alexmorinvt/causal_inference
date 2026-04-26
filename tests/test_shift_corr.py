"""Tests for grn_inference.shift_corr."""

from __future__ import annotations

import numpy as np

from grn_inference import (
    RandomBaseline,
    ShiftCorrModel,
    make_synthetic_dataset,
)
from grn_inference.dataset import Dataset


def test_fit_predict_returns_valid_edges():
    data, _ = make_synthetic_dataset(
        n_genes=15,
        edge_density=0.15,
        n_control_cells=500,
        n_cells_per_perturbation=80,
        n_perturbed_genes=8,
        seed=0,
    )
    edges = ShiftCorrModel(top_k=30).fit_predict(data)
    assert len(edges) <= 30
    gene_set = set(data.gene_names)
    for s, t in edges:
        assert s in gene_set
        assert t in gene_set
        assert s != t
    assert len(set(edges)) == len(edges)


def test_demotes_cascade_relative_to_direct():
    """A direct edge should outrank a cascade edge with the same total effect.

    Construct a 4-gene system where A is perturbed with knockdown
    heterogeneity, A -> D is direct with coefficient 0.7, and
    A -> M -> C is a cascade with coefficients (1.0, 0.7) so the total
    effect on C also equals 0.7. The mean-shift on D and C is identical;
    only the within-arm correlation distinguishes them. ShiftCorr should
    rank (A, D) above (A, C).
    """
    rng = np.random.default_rng(0)
    n_ctrl = 500
    n_pert = 500

    A_ctrl = rng.normal(0.0, 1.0, n_ctrl)
    A_pert = rng.normal(-2.0, 0.5, n_pert)
    A = np.concatenate([A_ctrl, A_pert])

    eps_D = rng.normal(0.0, 0.5, n_ctrl + n_pert)
    eps_M = rng.normal(0.0, 1.0, n_ctrl + n_pert)
    eps_C = rng.normal(0.0, 0.5, n_ctrl + n_pert)

    D = 0.7 * A + eps_D
    M = 1.0 * A + eps_M
    C = 0.7 * M + eps_C

    expression = np.column_stack([A, D, M, C]).astype(np.float32)
    interventions = ["non-targeting"] * n_ctrl + ["A"] * n_pert

    data = Dataset(
        expression=expression,
        interventions=interventions,
        gene_names=["A", "D", "M", "C"],
    )

    edges = ShiftCorrModel(top_k=10).fit_predict(data)
    rank = {e: i for i, e in enumerate(edges)}

    assert ("A", "D") in rank, f"direct edge missing from output: {edges}"
    assert ("A", "C") in rank, f"cascade edge missing from output: {edges}"
    assert rank[("A", "D")] < rank[("A", "C")], (
        f"Cascade not demoted: D rank {rank[('A', 'D')]}, "
        f"C rank {rank[('A', 'C')]}; full edges: {edges}"
    )


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
    sc_edges = ShiftCorrModel(top_k=50).fit_predict(data)
    rb_edges = RandomBaseline(top_k=50, seed=0).fit_predict(data)
    sc_prec = sum(1 for e in sc_edges if e in true_set) / max(len(sc_edges), 1)
    rb_prec = sum(1 for e in rb_edges if e in true_set) / max(len(rb_edges), 1)
    assert sc_prec >= rb_prec, f"ShiftCorr precision {sc_prec} < Random {rb_prec}"
