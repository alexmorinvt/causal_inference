"""Tests for grn_inference.ensemble_scm."""

from __future__ import annotations

import numpy as np
import torch

from grn_inference import (
    EnsembleSCMFitter,
    RandomBaseline,
    evaluate_statistical,
    make_synthetic_dataset,
)
from grn_inference.ensemble_scm import (
    LinearSCM,
    moment_matching_discrepancy,
    simulate_control,
    simulate_intervention,
)


# ---------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------
def test_simulator_with_zero_W_is_pure_noise():
    """W = 0 → x = epsilon, which has mean ≈ 0 and var ≈ 1 per gene."""
    scm = LinearSCM.random_init(n_genes=30, n_candidates=2, weight_scale=0.0, seed=0)
    assert torch.all(scm.W == 0)

    torch.manual_seed(42)
    x = simulate_control(scm, n_cells=5000).detach()
    assert x.shape == (2, 5000, 30)
    # Allow a little slack — 5k samples per candidate per gene.
    assert x.mean(dim=1).abs().max().item() < 0.1
    assert (x.var(dim=1) - 1.0).abs().max().item() < 0.15


def test_intervention_shifts_target_gene_mean_down():
    """Soft-knockdown puts the target gene's mean well below zero."""
    scm = LinearSCM.random_init(n_genes=20, n_candidates=1, weight_scale=0.0, seed=0)
    torch.manual_seed(0)
    x_ctrl = simulate_control(scm, n_cells=2000).detach()
    x_int = simulate_intervention(
        scm, target_idx=7, n_cells=2000, knockdown_factor=0.3
    ).detach()

    # Target gene (index 7) should have a markedly negative mean under
    # the knockdown; the mean of control at that gene should be ~0.
    ctrl_mean_7 = x_ctrl[0, :, 7].mean().item()
    int_mean_7 = x_int[0, :, 7].mean().item()
    assert int_mean_7 < ctrl_mean_7 - 0.5


def test_W_requires_grad():
    scm = LinearSCM.random_init(n_genes=10, n_candidates=3, weight_scale=0.01)
    assert scm.W.requires_grad is True


# ---------------------------------------------------------------------
# Discrepancy
# ---------------------------------------------------------------------
def test_discrepancy_is_near_zero_on_same_distribution():
    """If sim and real are drawn from the same distribution with large N,
    the moment-matching discrepancy should be close to zero."""
    torch.manual_seed(1)
    real = torch.randn(5000, 20)
    sim = torch.randn(3, 5000, 20)
    d = moment_matching_discrepancy(sim, real)
    assert d.shape == (3,)
    assert d.max().item() < 0.1


def test_discrepancy_is_positive_on_shifted_distribution():
    torch.manual_seed(2)
    real = torch.randn(2000, 10)
    sim = torch.randn(2, 2000, 10) + 2.0  # shifted mean
    d = moment_matching_discrepancy(sim, real)
    # Mean shift of 2 over 10 genes → mean term ≈ 4 * 10 = 40
    assert d.min().item() > 20.0


def test_discrepancy_is_differentiable():
    scm = LinearSCM.random_init(n_genes=8, n_candidates=2, weight_scale=0.01, seed=3)
    torch.manual_seed(0)
    sim = simulate_control(scm, n_cells=200)
    real = torch.randn(200, 8)
    d = moment_matching_discrepancy(sim, real).sum()
    (grad,) = torch.autograd.grad(d, scm.W)
    assert grad.shape == scm.W.shape
    assert torch.isfinite(grad).all()


# ---------------------------------------------------------------------
# Smoke test: EnsembleSCMFitter beats Random on a small synthetic graph
# ---------------------------------------------------------------------
def test_fitter_beats_random_baseline_on_small_synthetic():
    # n_genes=15 instead of 10 so that top_k=30 isn't "one third of the
    # universe" (random easily wins there by just filling the bag).
    # 15 * 14 = 210 possible off-diagonal edges; top_k=30 is ~14% of those.
    data, truth = make_synthetic_dataset(
        n_genes=15,
        edge_density=0.2,
        n_control_cells=500,
        n_cells_per_perturbation=120,
        seed=0,
    )

    fitter = EnsembleSCMFitter(
        top_k=30,
        n_candidates=3,
        n_steps=500,
        step_size=0.01,
        batch_size=120,
        l1_lambda=1e-4,
        seed=0,
        log_every=None,
    )
    rb = RandomBaseline(top_k=30, seed=0)

    fitter_edges = fitter.fit_predict(data)
    rb_edges = rb.fit_predict(data)

    fitter_res = evaluate_statistical(
        fitter_edges, data, omission_sample_size=200,
        rng=np.random.default_rng(99),
    )
    rb_res = evaluate_statistical(
        rb_edges, data, omission_sample_size=200,
        rng=np.random.default_rng(99),
    )

    print()
    print(f"EnsembleSCMFitter: {fitter_res.summary()}")
    print(f"RandomBaseline:    {rb_res.summary()}")

    assert fitter_res.mean_wasserstein > rb_res.mean_wasserstein, (
        "EnsembleSCMFitter must achieve higher Wasserstein than Random."
    )
