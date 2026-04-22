"""Tests for Stage 0.

The key assertion: on a synthetic linear-cyclic-SEM dataset where
ground truth is known, :class:`MeanDifferenceModel` achieves higher
mean Wasserstein and lower FOR than :class:`RandomBaseline`. If this
ever fails, either the evaluator or the data generator is wrong — do
not touch any new code until you've fixed it.
"""

from __future__ import annotations

import numpy as np

from grn_inference import (
    Dataset,
    MeanDifferenceModel,
    RandomBaseline,
    evaluate_statistical,
    make_synthetic_dataset,
)
from grn_inference.dataset import CONTROL_LABEL


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
def test_dataset_construction_valid():
    expr = np.zeros((10, 3), dtype=np.float32)
    iv = [CONTROL_LABEL] * 5 + ["G0001", "G0001", "G0002", "G0002", "G0002"]
    names = ["G0000", "G0001", "G0002"]
    d = Dataset(expression=expr, interventions=iv, gene_names=names)
    assert d.n_cells == 10
    assert d.n_genes == 3
    assert d.control_mask().sum() == 5
    assert set(d.perturbed_genes()) == {"G0001", "G0002"}
    assert d.gene_idx("G0002") == 2


def test_dataset_rejects_mismatched_shapes():
    expr = np.zeros((10, 3))
    names = ["a", "b", "c"]
    try:
        Dataset(expression=expr, interventions=["x"] * 9, gene_names=names)
    except ValueError:
        pass
    else:
        raise AssertionError("should have raised on cell count mismatch")


def test_dataset_subset_genes():
    expr = np.arange(12, dtype=np.float32).reshape(4, 3)
    iv = [CONTROL_LABEL] * 4
    names = ["A", "B", "C"]
    d = Dataset(expression=expr, interventions=iv, gene_names=names)
    sub = d.subset_genes(["A", "C"])
    assert sub.gene_names == ["A", "C"]
    assert sub.expression.shape == (4, 2)
    np.testing.assert_array_equal(sub.expression[:, 0], expr[:, 0])
    np.testing.assert_array_equal(sub.expression[:, 1], expr[:, 2])


# ---------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------
def test_synthetic_dataset_well_formed():
    data, truth = make_synthetic_dataset(
        n_genes=20,
        n_control_cells=500,
        n_cells_per_perturbation=100,
        seed=42,
    )
    assert data.n_genes == 20
    assert data.n_cells == 500 + 20 * 100
    assert data.control_mask().sum() == 500
    assert len(data.perturbed_genes()) == 20
    # Spectral radius should be strictly < 1 for stability.
    assert np.max(np.abs(np.linalg.eigvals(truth.W))) < 1.0


def test_synthetic_dataset_reproducible():
    d1, t1 = make_synthetic_dataset(n_genes=15, seed=7)
    d2, t2 = make_synthetic_dataset(n_genes=15, seed=7)
    np.testing.assert_array_equal(t1.W, t2.W)
    np.testing.assert_array_equal(d1.expression, d2.expression)


# ---------------------------------------------------------------------
# Evaluator smoke test
# ---------------------------------------------------------------------
def test_evaluator_runs_on_synthetic():
    data, truth = make_synthetic_dataset(n_genes=20, seed=1)
    # Evaluate ground-truth edges; they should have non-trivial Wasserstein.
    result = evaluate_statistical(truth.true_edges, data, omission_sample_size=100)
    assert result.n_evaluable_predicted > 0
    assert result.mean_wasserstein > 0.0
    assert 0.0 <= result.false_omission_rate <= 1.0


def test_evaluator_handles_empty_predictions():
    data, _ = make_synthetic_dataset(n_genes=15, seed=2)
    result = evaluate_statistical([], data, omission_sample_size=50)
    assert result.n_predicted_edges == 0
    assert result.mean_wasserstein == 0.0
    # With no predictions, every real effect is an omission — FOR should
    # be substantial on this dense synthetic graph.
    assert result.false_omission_rate > 0.0


# ---------------------------------------------------------------------
# The load-bearing test: Mean Difference > Random
# ---------------------------------------------------------------------
def test_mean_difference_beats_random_on_synthetic():
    data, truth = make_synthetic_dataset(
        n_genes=30,
        edge_density=0.15,
        n_control_cells=1000,
        n_cells_per_perturbation=150,
        seed=123,
    )

    md = MeanDifferenceModel(top_k=200)
    rb = RandomBaseline(top_k=200, seed=0)

    md_edges = md.fit_predict(data)
    rb_edges = rb.fit_predict(data)

    # Use a shared RNG for FOR so the comparison is on equal footing.
    rng = np.random.default_rng(99)

    md_result = evaluate_statistical(
        md_edges, data, omission_sample_size=300,
        rng=np.random.default_rng(99),
    )
    rb_result = evaluate_statistical(
        rb_edges, data, omission_sample_size=300,
        rng=np.random.default_rng(99),
    )

    print()
    print(f"Mean Difference: {md_result.summary()}")
    print(f"Random:          {rb_result.summary()}")

    assert md_result.mean_wasserstein > rb_result.mean_wasserstein, (
        "Mean Difference must achieve higher Wasserstein than Random. "
        "If this fails, something is wrong with the evaluator or data."
    )
    assert md_result.false_omission_rate <= rb_result.false_omission_rate, (
        "Mean Difference should have no worse FOR than Random."
    )
