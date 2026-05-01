"""Data sources for the pipeline.

Two things live here:

1. :func:`make_synthetic_dataset` — generate data from a known linear
   cyclic SCM. Lets you unit-test the evaluator and any new model
   without downloading anything. The generative process matches the
   family of models we plan to fit in Stage 1, so "can my Stage 1
   model recover the truth on this synthetic data?" is a meaningful
   sanity check.

2. :func:`load_causalbench_dataset` — adapter from the ``causalbench``
   package to our :class:`Dataset`. This is intentionally a stub: the
   exact call depends on the installed ``causalbench`` version, and I'd
   rather you read their ``data_access`` module once and adapt the
   TODOs than have a silently-wrong conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import CONTROL_LABEL, Dataset


# ---------------------------------------------------------------------
# Synthetic data — linear cyclic SCM
# ---------------------------------------------------------------------
@dataclass
class SyntheticTruth:
    """Ground-truth information returned alongside a synthetic dataset.

    Attributes
    ----------
    W
        The true adjacency matrix used to generate the data. ``W[i, j]``
        is the weight of edge ``gene_j -> gene_i`` (i.e. parent j acts
        on child i).
    true_edges
        List of directed edges derived from non-zero entries of ``W``,
        in ``(source, target)`` order — so ``(gene_j, gene_i)`` if
        ``W[i, j] != 0``.
    gene_names
        Gene symbol list corresponding to rows/columns of ``W``.
    """

    W: np.ndarray
    true_edges: list[tuple[str, str]]
    gene_names: list[str]


def _simulate_lineage_cascade(
    W: np.ndarray,
    n_cells: int,
    n_genes: int,
    depth: int,
    branching: int,
    noise_std: float,
    rng: np.random.Generator,
    *,
    target_idx: int | None = None,
    target_mean: float = 0.0,
    target_std: float = 1.0,
) -> np.ndarray:
    """Grow a branching lineage tree and return (a sample of) the leaves.

    At every generation each cell begets ``branching`` children whose
    state is ``W @ parent + eps``. Enough founders are seeded so that
    the number of leaves is at least ``n_cells``; the leaves are then
    subsampled (without replacement) to exactly ``n_cells``.

    If ``target_idx`` is given, gene ``target_idx`` is treated as
    perturbed: each founder draws a single target value from
    ``N(target_mean, target_std^2)`` that is held constant through
    every cascade step and inherited by all of the founder's
    descendants. Downstream genes see the same target value at every
    cascade step, preserving per-cell A-B coupling that a fresh noise
    draw each step would otherwise destroy.
    """
    n_founders = int(np.ceil(n_cells / (branching ** depth)))
    x = rng.normal(0.0, noise_std, size=(n_founders, n_genes))
    hold_target = None
    if target_idx is not None:
        hold_target = rng.normal(target_mean, target_std, size=n_founders)
        x[:, target_idx] = hold_target

    for _ in range(depth):
        x = np.repeat(x, branching, axis=0)
        if hold_target is not None:
            hold_target = np.repeat(hold_target, branching)
        eps = rng.normal(0.0, noise_std, size=x.shape)
        x = x @ W.T + eps
        if hold_target is not None:
            x[:, target_idx] = hold_target

    if x.shape[0] > n_cells:
        idx = rng.choice(x.shape[0], size=n_cells, replace=False)
        x = x[idx]
    return x


def _simulate_intervention_per_cell_alpha(
    W: np.ndarray,
    n_cells: int,
    n_genes: int,
    depth: int,
    noise_std: float,
    rng: np.random.Generator,
    *,
    target_idx: int,
    alpha_per_cell: np.ndarray,
    knockdown_factor: float,
) -> np.ndarray:
    """Intervention cascade with a per-cell knockdown strength.

    Each cell gets its own ``alpha`` drawn from the intervention-strength
    range, so within a single ``do(target)`` population there is
    cell-to-cell variation in how hard the knockdown bites. No
    branching — each of the ``n_cells`` trajectories is independent, so
    every cell's alpha is truly private (not shared with lineage
    siblings).
    """
    assert alpha_per_cell.shape == (n_cells,)
    knock_mean = -alpha_per_cell * (1.0 - knockdown_factor) * 2.0
    # Preserve baseline variance on the perturbed gene (shrinking it
    # collapses the within-arm coupling signal).
    hold_target = rng.standard_normal(n_cells) * noise_std + knock_mean

    x = rng.normal(0.0, noise_std, size=(n_cells, n_genes))
    x[:, target_idx] = hold_target

    for _ in range(depth):
        eps = rng.normal(0.0, noise_std, size=x.shape)
        x_next = x @ W.T + eps
        # Perfect-intervention semantic: the perturbed gene is pinned
        # to its per-cell value at every cascade step, so the signal
        # propagates to downstream genes instead of being erased by a
        # fresh noise draw each step.
        x_next[:, target_idx] = hold_target
        x = x_next

    return x


def make_synthetic_dataset(
    n_genes: int = 50,
    edge_density: float = 0.1,
    weight_scale: float = 0.4,
    target_spectral_radius: float = 0.8,
    n_control_cells: int = 1000,
    n_cells_per_perturbation: int = 200,
    knockdown_factor: float = 0.3,
    noise_std: float = 1.0,
    n_perturbed_genes: int | None = None,
    forbid_two_cycles: bool = True,
    lineage_depth: int = 5,
    branching_factor: int = 2,
    perturbation_strength_range: tuple[float, float] = (0.3, 0.9),
    per_cell_alpha_spread: float = 0.0,
    tf_fraction: float = 0.0,
    tf_out_degree_alpha: float = 2.0,
    seed: int = 0,
) -> tuple[Dataset, SyntheticTruth]:
    """Generate a Dataset from a known linear cyclic SCM via a tree cascade.

    Observed cells are the leaves of a depth-``lineage_depth`` branching
    tree. A small number of founders are seeded with random states
    ``x_0 ~ N(0, noise_std^2 I)``. At each generation, every cell begets
    ``branching_factor`` children whose state is::

        x_child = W @ x_parent + epsilon,   epsilon ~ N(0, noise_std^2 I)

    After ``lineage_depth`` generations, the leaves are returned (shuffled
    by :func:`numpy.random.Generator.choice` and truncated to the requested
    cell count). The number of founders per arm is picked automatically so
    that ``n_founders * branching_factor ** lineage_depth >= n_cells``.

    This replaces the old closed-form equilibrium ``x = (I - W)^{-1} eps``
    with a transient-dynamics process. At ``lineage_depth -> infinity`` the
    marginal distribution of each leaf converges back to the equilibrium
    (assuming ``rho(W) < 1``); a finite depth like 5 gives cascades that
    carry directional information the equilibrium smears out.

    Interventions are modelled as *partial soft knockdowns*. For each gene
    ``t``, a per-gene strength ``alpha_t ~ U(*perturbation_strength_range)``
    is drawn once and applied uniformly across the arm's cells:

    * ``W_t[t, :] = (1 - alpha_t) * W[t, :]`` — gene ``t`` only partially
      listens to its regulators.
    * Gene ``t``'s emission at every cascade step is drawn from
      ``N(-alpha_t * (1 - knockdown_factor) * 2, noise_std^2)`` — the
      mean shifts with ``alpha_t`` but the variance is held at baseline.
      We keep the variance to preserve per-cell coupling: within
      ``do(t)`` cells gene ``t`` retains genuine cell-to-cell variance,
      so downstream genes on a direct edge from ``t`` still have a
      detectable ``corr(t, child | do(t))`` signal. Shrinking the noise
      under strong ``alpha`` collapsed that signal to noise.
    * The founder's own gene-``t`` state is initialised from the same
      intervention distribution, so cells are "born perturbed".

    Parameters
    ----------
    n_genes
        Number of genes. Runtime scales with ``n_genes^3`` for the
        matrix inverse, so keep this <= 100 for tests.
    edge_density
        Probability that each off-diagonal entry of ``W`` is non-zero.
    weight_scale
        Initial scale of non-zero weights, sampled uniformly from
        ``[-weight_scale, weight_scale]``.
    target_spectral_radius
        After sampling, ``W`` is rescaled so its spectral radius equals
        this value. Must be < 1 for ``(I - W)`` to be invertible with
        bounded solution.
    n_control_cells
        Number of non-targeting cells.
    n_cells_per_perturbation
        Per-gene count for the interventional arm.
    knockdown_factor
        Fraction of the baseline expression retained under perturbation.
        0.3 means "expression of target drops to ~30% of baseline."
    noise_std
        Gaussian noise standard deviation. Same across genes for
        simplicity.
    n_perturbed_genes
        If ``None`` (default), every gene gets an intervention arm —
        matching the original behaviour. Otherwise, a random subset of
        this many genes is selected; only those genes have intervention
        arms, and edges whose source is unperturbed will be invisible
        to the statistical metric. Useful for testing methods that are
        supposed to leverage observational data to pick up
        unperturbed-source edges.
    forbid_two_cycles
        If True (default), for every pair ``(i, j)`` the mask keeps at
        most one direction: either ``i -> j`` or ``j -> i``, never
        both. Longer cycles (``A -> B -> C -> A``) are still allowed.
        This removes the direction-confusion failure mode that hits
        any method that ranks edges by correlated activity, since for
        a two-cycle the observational covariance is symmetric in
        ``(i, j)``.
    lineage_depth
        Number of cascade steps (tree depth) between founders and leaves.
    branching_factor
        Each parent begets this many children per generation.
    perturbation_strength_range
        ``(lo, hi)`` interval for the per-gene intervention strength
        ``alpha``. At 0 the gene behaves normally; at 1 it is fully
        knocked out. Different genes are hit with different ``alpha``.
    per_cell_alpha_spread
        If non-zero, each cell in a given arm draws its own
        ``alpha_cell ~ U(alpha_gene - spread/2, alpha_gene + spread/2)``
        (clipped to [0, 1]), so knockdown strength varies cell-to-cell
        within a single intervention population. ``0.0`` (default)
        reproduces the existing per-gene-only behaviour bit-identically.
        Enabling this switches the intervention simulator from the
        branching tree to independent per-cell cascades.
    tf_fraction
        Fraction of genes designated as transcription factors (TFs).
        ``0.0`` (default) uses the original Erdős–Rényi construction.
        When positive, only TF genes emit outgoing edges; non-TFs
        have out-degree zero. TF out-degrees follow a Pareto
        distribution (shape ``tf_out_degree_alpha``) with mean
        ``edge_density * n_genes`` targets per TF. Note: the total
        edge count is lower than the ER regime — scale-free graphs
        are sparser overall, with edges concentrated in hubs.
    tf_out_degree_alpha
        Pareto shape parameter for TF out-degrees (used only when
        ``tf_fraction > 0``). Lower values give a heavier tail — a
        few TFs dominate with very high out-degree. Default 2.0 gives
        moderate skew (finite variance); 1.5 gives infinite variance /
        heavier hubs.
    seed
        RNG seed.

    Returns
    -------
    dataset, truth
    """
    rng = np.random.default_rng(seed)

    # ---- Build W --------------------------------------------------------
    if tf_fraction > 0.0:
        # Scale-free: a small fraction of genes are TFs with Pareto-distributed
        # out-degree; non-TFs emit no edges. Total edge count is matched to the
        # ER regime via normalisation, so edge_density remains meaningful.
        n_tf = max(1, round(tf_fraction * n_genes))
        tf_idx = rng.choice(n_genes, size=n_tf, replace=False)
        # edge_density sets mean TF out-degree as a fraction of n_genes;
        # the Pareto gives the shape. This is independent of ER total-edge
        # count — a scale-free graph is naturally sparser overall.
        mean_out = max(1.0, edge_density * n_genes)
        total_edges = max(n_tf, round(mean_out * n_tf))
        raw = rng.pareto(tf_out_degree_alpha, size=n_tf) + 1.0
        degrees = np.round(raw / raw.sum() * total_edges).astype(int)
        degrees = np.clip(degrees, 1, n_genes - 1)
        # Correct integer rounding so degrees sum exactly to total_edges.
        diff = total_edges - int(degrees.sum())
        if diff != 0:
            hub = int(np.argmax(degrees))
            degrees[hub] = int(np.clip(degrees[hub] + diff, 1, n_genes - 1))
        mask = np.zeros((n_genes, n_genes), dtype=float)
        all_genes = np.arange(n_genes)
        for i, s in enumerate(tf_idx):
            d = int(degrees[i])
            if d == 0:
                continue
            candidates = all_genes[all_genes != s]
            # mask[t, s] = 1 encodes s → t in the W[child, parent] convention
            targets = rng.choice(candidates, size=min(d, len(candidates)), replace=False)
            mask[targets, s] = 1.0
    else:
        mask = (rng.uniform(size=(n_genes, n_genes)) < edge_density).astype(float)
        np.fill_diagonal(mask, 0.0)

    if forbid_two_cycles:
        # Where both (i,j) and (j,i) are 1, drop one at random.
        both = np.logical_and(mask, mask.T)
        i_idx, j_idx = np.where(np.triu(both, k=1))
        drop_ij = rng.random(len(i_idx)) < 0.5
        mask[i_idx[drop_ij], j_idx[drop_ij]] = 0.0
        mask[j_idx[~drop_ij], i_idx[~drop_ij]] = 0.0
    # Every edge has magnitude 1 with random sign; the spectral rescaling
    # below uniformly scales the whole matrix, so all non-zero entries
    # end up with the same magnitude. `weight_scale` is unused in this
    # uniform-sign regime and is kept only for signature compatibility.
    _ = weight_scale  # intentionally unused
    signs = np.where(rng.random(size=(n_genes, n_genes)) < 0.5, -1.0, 1.0)
    W = signs * mask

    # Rescale to hit target spectral radius
    eigs = np.linalg.eigvals(W)
    current_radius = float(np.max(np.abs(eigs)))
    if current_radius > 0:
        W = W * (target_spectral_radius / current_radius)

    gene_names = [f"G{i:04d}" for i in range(n_genes)]

    # ---- Observational cells (tree cascade) ----------------------------
    x_ctrl = _simulate_lineage_cascade(
        W, n_control_cells, n_genes,
        lineage_depth, branching_factor, noise_std, rng,
    )

    interventions: list[str] = [CONTROL_LABEL] * n_control_cells
    all_expr = [x_ctrl]

    # ---- Decide which genes get intervention arms ----------------------
    if n_perturbed_genes is None:
        perturbed_indices = list(range(n_genes))
    else:
        if not 0 < n_perturbed_genes <= n_genes:
            raise ValueError(
                f"n_perturbed_genes must be in (0, {n_genes}]; got {n_perturbed_genes}"
            )
        perturbed_indices = sorted(
            rng.choice(n_genes, size=n_perturbed_genes, replace=False).tolist()
        )

    # ---- Per-gene partial-knockdown strength ---------------------------
    alpha_lo, alpha_hi = perturbation_strength_range
    if not 0.0 <= alpha_lo <= alpha_hi <= 1.0:
        raise ValueError(
            f"perturbation_strength_range must be in [0, 1]; got {perturbation_strength_range}"
        )
    if per_cell_alpha_spread < 0.0:
        raise ValueError(
            f"per_cell_alpha_spread must be >= 0; got {per_cell_alpha_spread}"
        )
    alphas = rng.uniform(alpha_lo, alpha_hi, size=n_genes)

    # ---- Interventional cells (tree cascade + partial knockdown) -------
    for t in perturbed_indices:
        alpha = float(alphas[t])

        if per_cell_alpha_spread > 0.0:
            half = per_cell_alpha_spread / 2.0
            alpha_per_cell = np.clip(
                rng.uniform(
                    alpha - half, alpha + half, size=n_cells_per_perturbation,
                ),
                0.0, 1.0,
            )
            x_t = _simulate_intervention_per_cell_alpha(
                W, n_cells_per_perturbation, n_genes,
                lineage_depth, noise_std, rng,
                target_idx=t,
                alpha_per_cell=alpha_per_cell,
                knockdown_factor=knockdown_factor,
            )
        else:
            W_t = W.copy()
            W_t[t, :] = (1.0 - alpha) * W[t, :]
            knock_mean = -alpha * (1.0 - knockdown_factor) * 2.0
            # Keep the perturbed gene's noise variance at baseline so that
            # cells within an arm still have meaningful A-variance. Without
            # this, strong knockdown shrinks Var(A | do(A)) so small that
            # within-arm correlations collapse to noise.
            knock_std = noise_std
            x_t = _simulate_lineage_cascade(
                W_t, n_cells_per_perturbation, n_genes,
                lineage_depth, branching_factor, noise_std, rng,
                target_idx=t, target_mean=knock_mean, target_std=knock_std,
            )
        all_expr.append(x_t)
        interventions.extend([gene_names[t]] * n_cells_per_perturbation)

    expression = np.vstack(all_expr).astype(np.float32)

    true_edges: list[tuple[str, str]] = []
    for i in range(n_genes):
        for j in range(n_genes):
            if i != j and W[i, j] != 0.0:
                # W[i, j] non-zero means j regulates i, i.e. j -> i
                true_edges.append((gene_names[j], gene_names[i]))

    dataset = Dataset(
        expression=expression,
        interventions=interventions,
        gene_names=gene_names,
    )
    truth = SyntheticTruth(W=W, true_edges=true_edges, gene_names=gene_names)
    return dataset, truth


# ---------------------------------------------------------------------
# CausalBench adapter — stub
# ---------------------------------------------------------------------
def load_causalbench_dataset(
    dataset_name: str,
    data_directory: str,
    filter: bool = True,
    subset_genes: list[str] | None = None,
    subset_cells: int | None = None,
    seed: int = 0,
) -> Dataset:
    """Load a CausalBench dataset and convert to our ``Dataset``.

    Parameters
    ----------
    dataset_name
        Either ``"weissmann_k562"`` or ``"weissmann_rpe1"`` per the
        CausalBench CLI.
    data_directory
        Path CausalBench uses for caching its preprocessed data. The
        first time this runs it will download and preprocess the
        Replogle Perturb-seq data (multi-GB, takes a while). After
        that it's fast.
    filter
        If ``True`` (default), apply CausalBench's strong-perturbation
        filter: keeps only genes with >50 DEGs, ≤-30% knockdown, and
        >25 cells (per the Replogle et al. summary stats), and filters
        cells to those where the target gene is below the 10th percentile
        of control. This matches the CausalBench benchmark regime.
        Set ``False`` to load the full unfiltered dataset.
    subset_genes
        Optional gene list to restrict to. Useful for the mini-dataset
        regime — pass ~300 genes for fast iteration.
    subset_cells
        If provided, subsample to at most this many cells while
        preserving stratification by intervention target.
    seed
        For cell subsampling.
    """
    import sys
    from pathlib import Path

    vendor_path = str(Path(__file__).parent.parent / "vendor" / "causalbench")
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

    from causalscbench.data_access.create_dataset import CreateDataset

    name = dataset_name.lower()
    if "k562" in name:
        key = "k562"
    elif "rpe1" in name:
        key = "rpe1"
    else:
        raise ValueError(
            f"Unknown dataset_name {dataset_name!r}. "
            "Use 'weissmann_k562' or 'weissmann_rpe1'."
        )

    loader = CreateDataset(data_directory=data_directory, filter=filter)
    path_k562, path_rpe1 = loader.load()
    npz_path = path_k562 if key == "k562" else path_rpe1

    raw = np.load(npz_path, allow_pickle=True)
    expression = np.asarray(raw["expression_matrix"], dtype=np.float32)
    gene_names = list(raw["var_names"])
    interventions_raw = list(raw["interventions"])

    # Drop cells whose perturbation target had <100 cells (marked "excluded")
    # and keep only control + valid intervention cells.
    keep_mask = np.array([iv != "excluded" for iv in interventions_raw], dtype=bool)
    expression = expression[keep_mask]
    interventions = [iv for iv, k in zip(interventions_raw, keep_mask) if k]

    dataset = Dataset(
        expression=expression,
        interventions=interventions,
        gene_names=gene_names,
    )

    if subset_genes is not None:
        dataset = dataset.subset_genes(subset_genes)

    if subset_cells is not None and dataset.n_cells > subset_cells:
        rng = np.random.default_rng(seed)
        # Stratified subsample: preserve proportional counts per intervention.
        unique_ivs = list(dict.fromkeys(dataset.interventions))
        indices: list[int] = []
        for iv in unique_ivs:
            iv_idx = [i for i, v in enumerate(dataset.interventions) if v == iv]
            n_keep = max(1, round(subset_cells * len(iv_idx) / dataset.n_cells))
            chosen = rng.choice(iv_idx, size=min(n_keep, len(iv_idx)), replace=False)
            indices.extend(chosen.tolist())
        mask = np.zeros(dataset.n_cells, dtype=bool)
        mask[indices] = True
        dataset = dataset.subset_cells(mask)

    return dataset
