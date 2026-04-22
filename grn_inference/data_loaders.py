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


def make_synthetic_dataset(
    n_genes: int = 50,
    edge_density: float = 0.1,
    weight_scale: float = 0.4,
    target_spectral_radius: float = 0.8,
    n_control_cells: int = 1000,
    n_cells_per_perturbation: int = 200,
    knockdown_factor: float = 0.3,
    noise_std: float = 1.0,
    seed: int = 0,
) -> tuple[Dataset, SyntheticTruth]:
    """Generate a Dataset from a known linear cyclic SCM.

    The generative model is::

        x = (I - W)^{-1} (epsilon + c)

    where ``W`` is a sparse adjacency with spectral radius < 1 (for
    stability), ``c`` is a per-gene baseline offset (zero here, but
    ``knockdown_factor`` shifts it for intervened genes), and
    ``epsilon ~ N(0, noise_std^2 I)``.

    Interventions are modelled as *soft knockdowns*: when gene ``t``
    is targeted, we zero out row ``t`` of ``W`` (``t`` no longer
    responds to its regulators) and set its emission to
    ``knockdown_factor * epsilon_t``. This mimics CRISPRi's incomplete
    knockdown with residual noise, which is how the Replogle data
    looks in practice.

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
    seed
        RNG seed.

    Returns
    -------
    dataset, truth
    """
    rng = np.random.default_rng(seed)

    # ---- Build W --------------------------------------------------------
    mask = (rng.uniform(size=(n_genes, n_genes)) < edge_density).astype(float)
    np.fill_diagonal(mask, 0.0)
    weights = rng.uniform(-weight_scale, weight_scale, size=(n_genes, n_genes))
    W = weights * mask

    # Rescale to hit target spectral radius
    eigs = np.linalg.eigvals(W)
    current_radius = float(np.max(np.abs(eigs)))
    if current_radius > 0:
        W = W * (target_spectral_radius / current_radius)

    gene_names = [f"G{i:04d}" for i in range(n_genes)]

    # ---- Observational cells -------------------------------------------
    # x = (I - W)^{-1} epsilon
    I = np.eye(n_genes)
    inv = np.linalg.inv(I - W)
    eps_ctrl = rng.normal(0.0, noise_std, size=(n_control_cells, n_genes))
    x_ctrl = eps_ctrl @ inv.T  # (n_ctrl, n_genes)

    interventions: list[str] = [CONTROL_LABEL] * n_control_cells
    all_expr = [x_ctrl]

    # ---- Interventional cells ------------------------------------------
    # For gene t: zero row t of W (t no longer regulated by its parents),
    # replace epsilon_t with a knockdown-specific noise.
    for t in range(n_genes):
        W_t = W.copy()
        W_t[t, :] = 0.0
        inv_t = np.linalg.inv(I - W_t)
        eps_t = rng.normal(0.0, noise_std, size=(n_cells_per_perturbation, n_genes))
        # Knockdown: force gene t's own emission toward lower values.
        # Using scaled-down noise shifts the mean/variance of x_t downward.
        eps_t[:, t] = knockdown_factor * rng.normal(
            0.0, noise_std, size=n_cells_per_perturbation
        ) - (1.0 - knockdown_factor) * 2.0  # offset for visibility
        x_t = eps_t @ inv_t.T
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
    subset_genes
        Optional gene list to restrict to. Useful for the mini-dataset
        regime — pass ~300 genes for fast iteration.
    subset_cells
        If provided, subsample to at most this many cells while
        preserving stratification by intervention target.
    seed
        For cell subsampling.

    Notes
    -----
    This is a **stub**. The ``causalbench`` package's internal
    ``data_access`` module has shifted across versions. Steps:

    1. ``pip install causalbench``
    2. Inspect ``causalscbench/data_access/create_dataset.py`` (or the
       equivalent in your version) to find the function that returns
       ``(expression_matrix, interventions, gene_names)`` from the
       cached data.
    3. Fill in the TODO below.

    When in doubt, run the CLI once:

        causalbench_run --dataset_name weissmann_k562 \\
            --output_directory /tmp/out \\
            --data_directory {data_directory} \\
            --training_regime observational \\
            --model_name pc --subset_data 0.01

    That forces CausalBench to download and preprocess; afterwards the
    ``data_directory`` will contain the arrays this function needs.
    """
    raise NotImplementedError(
        "Fill in this stub by importing from causalscbench.data_access. "
        "Read the docstring for the steps. Until you do, use "
        "make_synthetic_dataset() for everything."
    )

    # --------------------------------------------------------------------
    # Skeleton once you have the causalbench imports figured out:
    # --------------------------------------------------------------------
    # from causalscbench.data_access.create_dataset import CreateDataset
    # loader = CreateDataset(data_directory=data_directory, dataset_name=dataset_name)
    # expression, interventions, gene_names = loader.load()
    #
    # # Note: CausalBench uses empty string "" for control cells in some
    # # versions and "non-targeting" in others. Normalise to our CONTROL_LABEL.
    # interventions = [iv if iv else CONTROL_LABEL for iv in interventions]
    #
    # dataset = Dataset(
    #     expression=np.asarray(expression, dtype=np.float32),
    #     interventions=list(interventions),
    #     gene_names=list(gene_names),
    # )
    # if subset_genes is not None:
    #     dataset = dataset.subset_genes(subset_genes)
    # if subset_cells is not None and dataset.n_cells > subset_cells:
    #     rng = np.random.default_rng(seed)
    #     # Stratified subsample: keep proportional counts per intervention.
    #     idx = _stratified_subsample(dataset.interventions, subset_cells, rng)
    #     mask = np.zeros(dataset.n_cells, dtype=bool)
    #     mask[idx] = True
    #     dataset = dataset.subset_cells(mask)
    # return dataset
