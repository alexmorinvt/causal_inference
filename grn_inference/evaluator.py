"""Statistical evaluation of predicted gene regulatory networks.

Implements the two primary metrics from Chevalley et al. 2025 (CausalBench):

- **Mean Wasserstein distance**: a precision-like metric. For each
  predicted edge ``A -> B`` where ``A`` was perturbed in the data,
  compute the 1-D Wasserstein-1 distance between the distribution of
  ``B`` under control cells and the distribution of ``B`` under cells
  where ``A`` was perturbed. Average over edges. Higher = better.

- **False Omission Rate (FOR)**: a recall-like metric. Among edges the
  model did *not* predict, estimate the fraction that correspond to a
  real interventional effect using a Mann-Whitney U test. Lower = better.

These two metrics trade off and are reported jointly. The Wasserstein
metric is what the CausalBench challenge evaluated on.

This module is deliberately free of the ``causalbench`` dependency so
you can run the full evaluation on any ``Dataset`` — synthetic or real.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .dataset import Dataset

logger = logging.getLogger(__name__)

Edge = tuple[str, str]  # (source, target), i.e. source -> target


@dataclass
class StatisticalResult:
    """Output of :func:`evaluate_statistical`."""

    mean_wasserstein: float
    per_edge_wasserstein: dict[Edge, float]
    false_omission_rate: float
    n_predicted_edges: int
    n_evaluable_predicted: int
    n_omission_candidates: int
    n_omission_sampled: int

    def summary(self) -> str:
        return (
            f"StatisticalResult("
            f"mean_W1={self.mean_wasserstein:.4f}, "
            f"FOR={self.false_omission_rate:.4f}, "
            f"predicted={self.n_predicted_edges}, "
            f"evaluable={self.n_evaluable_predicted}, "
            f"omission_sampled={self.n_omission_sampled}/{self.n_omission_candidates})"
        )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Empirical 1-D Wasserstein-1 distance. Wraps scipy for clarity."""
    return float(stats.wasserstein_distance(a, b))


def _mannwhitney_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Mann-Whitney U p-value. Returns 1.0 if undefined."""
    if len(a) < 2 or len(b) < 2:
        return 1.0
    # With ties, scipy uses asymptotic by default which is fine at our sizes.
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        # e.g. all values identical in both arrays
        return 1.0


def _canonicalize_edges(edges: list[Edge]) -> list[Edge]:
    """Deduplicate, preserve order."""
    seen: set[Edge] = set()
    out: list[Edge] = []
    for e in edges:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


# ---------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------
def evaluate_statistical(
    predicted_edges: list[Edge],
    data: Dataset,
    *,
    omission_sample_size: int = 500,
    significance_level: float = 0.05,
    min_cells_per_group: int = 25,
    rng: np.random.Generator | None = None,
) -> StatisticalResult:
    """Compute mean Wasserstein and FOR for a predicted edge list.

    Parameters
    ----------
    predicted_edges
        List of ``(source, target)`` tuples. Gene symbols must match
        ``data.gene_names``. Duplicates are removed silently.
    data
        The perturbation dataset. Must contain control cells
        (``interventions == CONTROL_LABEL``) and at least one
        intervention.
    omission_sample_size
        How many non-predicted edges to sample when estimating FOR. The
        default of 500 mirrors CausalBench's default CLI. Larger =
        tighter estimate of FOR but slower; ~O(N) Mann-Whitney tests.
    significance_level
        Threshold for "this is really an edge" in the FOR computation.
        The paper uses 0.05.
    min_cells_per_group
        Skip edges where fewer cells are available in either the
        control group or the intervention group. CausalBench's
        preprocessing enforces 25 perturbed cells per strong
        perturbation, so this is a sane floor.
    rng
        Source of randomness for omission sampling. Pass a seeded
        ``np.random.default_rng(seed)`` for reproducibility.

    Returns
    -------
    StatisticalResult
    """
    if rng is None:
        rng = np.random.default_rng()

    predicted_edges = _canonicalize_edges(predicted_edges)
    predicted_set = set(predicted_edges)

    control_mask = data.control_mask()
    n_control = int(control_mask.sum())
    if n_control < min_cells_per_group:
        raise ValueError(
            f"Only {n_control} control cells; need >= {min_cells_per_group}. "
            "The statistical metric is not well-defined without a robust control."
        )

    perturbed_genes = set(data.perturbed_genes())
    gene_set = set(data.gene_names)

    # Pre-compute per-perturbation masks and expression slices once.
    # This is the main CPU-time win vs. recomputing inside the loop.
    perturbation_cells: dict[str, np.ndarray] = {}
    for g in perturbed_genes:
        m = data.intervention_mask(g)
        if m.sum() >= min_cells_per_group:
            perturbation_cells[g] = m
    control_expr = data.expression[control_mask]  # (n_control, n_genes)

    # -----------------------------------------------------------------
    # (1) Wasserstein for predicted edges
    # -----------------------------------------------------------------
    per_edge: dict[Edge, float] = {}
    for src, tgt in predicted_edges:
        if src not in perturbation_cells:
            # Source wasn't perturbed; cannot evaluate this edge
            # statistically. This is neither a positive nor a penalty —
            # the paper simply excludes these from the precision-like
            # metric.
            continue
        if tgt not in gene_set:
            continue
        if src == tgt:
            continue
        mask = perturbation_cells[src]
        tgt_idx = data.gene_idx(tgt)
        intv = data.expression[mask, tgt_idx]
        ctrl = control_expr[:, tgt_idx]
        per_edge[(src, tgt)] = _wasserstein_1d(ctrl, intv)

    mean_w = float(np.mean(list(per_edge.values()))) if per_edge else 0.0

    # -----------------------------------------------------------------
    # (2) FOR via sampled omissions
    # -----------------------------------------------------------------
    candidates: list[Edge] = []
    for src in perturbation_cells:
        for tgt in data.gene_names:
            if tgt == src:
                continue
            edge = (src, tgt)
            if edge in predicted_set:
                continue
            candidates.append(edge)
    n_candidates = len(candidates)

    if n_candidates == 0:
        return StatisticalResult(
            mean_wasserstein=mean_w,
            per_edge_wasserstein=per_edge,
            false_omission_rate=0.0,
            n_predicted_edges=len(predicted_edges),
            n_evaluable_predicted=len(per_edge),
            n_omission_candidates=0,
            n_omission_sampled=0,
        )

    sample_size = min(omission_sample_size, n_candidates)
    sampled_idx = rng.choice(n_candidates, size=sample_size, replace=False)

    n_significant = 0
    for i in sampled_idx:
        src, tgt = candidates[int(i)]
        mask = perturbation_cells[src]
        tgt_idx = data.gene_idx(tgt)
        intv = data.expression[mask, tgt_idx]
        ctrl = control_expr[:, tgt_idx]
        if _mannwhitney_p(ctrl, intv) < significance_level:
            n_significant += 1
    false_omission_rate = n_significant / sample_size

    return StatisticalResult(
        mean_wasserstein=mean_w,
        per_edge_wasserstein=per_edge,
        false_omission_rate=false_omission_rate,
        n_predicted_edges=len(predicted_edges),
        n_evaluable_predicted=len(per_edge),
        n_omission_candidates=n_candidates,
        n_omission_sampled=sample_size,
    )
