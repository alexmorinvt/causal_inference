"""Core Dataset abstraction.

The single type everything else consumes. Both synthetic fixtures and
real CausalBench data must produce one of these.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CONTROL_LABEL = "non-targeting"


@dataclass
class Dataset:
    """A single-cell perturbation dataset.

    Attributes
    ----------
    expression
        Array of shape ``(n_cells, n_genes)``. Assumed to be normalised
        (e.g. log1p(CPM/10k)), not raw counts. If you pass raw counts the
        statistical metric will still work but will be dominated by
        depth variation. CausalBench's provided data is already
        normalised in this way.
    interventions
        Length ``n_cells``. Each entry is either ``CONTROL_LABEL``
        (non-targeting guide) or the symbol of the targeted gene.
    gene_names
        Length ``n_genes``. Gene symbols matching the columns of
        ``expression``. Must be unique.
    """

    expression: np.ndarray
    interventions: list[str]
    gene_names: list[str]
    _gene_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.expression.ndim != 2:
            raise ValueError("expression must be 2-D (n_cells, n_genes)")
        n_cells, n_genes = self.expression.shape
        if len(self.interventions) != n_cells:
            raise ValueError(
                f"interventions length {len(self.interventions)} != n_cells {n_cells}"
            )
        if len(self.gene_names) != n_genes:
            raise ValueError(
                f"gene_names length {len(self.gene_names)} != n_genes {n_genes}"
            )
        if len(set(self.gene_names)) != len(self.gene_names):
            raise ValueError("gene_names must be unique")
        self._gene_index = {g: i for i, g in enumerate(self.gene_names)}

    # ------------------------------------------------------------------
    # Shape accessors
    # ------------------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return self.expression.shape[0]

    @property
    def n_genes(self) -> int:
        return self.expression.shape[1]

    # ------------------------------------------------------------------
    # Masks and slices
    # ------------------------------------------------------------------
    def gene_idx(self, gene: str) -> int:
        return self._gene_index[gene]

    def control_mask(self) -> np.ndarray:
        """Boolean mask over cells where no guide was targeted."""
        return np.array(
            [iv == CONTROL_LABEL for iv in self.interventions], dtype=bool
        )

    def intervention_mask(self, gene: str) -> np.ndarray:
        """Boolean mask over cells where ``gene`` was targeted."""
        return np.array([iv == gene for iv in self.interventions], dtype=bool)

    def perturbed_genes(self) -> list[str]:
        """Genes that were targeted in at least one cell and are measured.

        Only these can appear as the *source* of an evaluable predicted
        edge (the statistical metric needs interventional cells for the
        source gene). The biological metric does not need this.
        """
        measured = set(self.gene_names)
        targeted = {iv for iv in self.interventions if iv != CONTROL_LABEL}
        return sorted(targeted & measured)

    # ------------------------------------------------------------------
    # Subsetting
    # ------------------------------------------------------------------
    def subset_genes(self, genes: list[str]) -> "Dataset":
        """Return a new Dataset restricted to ``genes``.

        Genes not present are silently dropped; order is preserved.
        Cells whose intervention target is not in ``genes`` and is not
        the control remain but their intervention column will no longer
        match a measured gene — which is fine for observational use but
        excludes them from the statistical metric as source cells.
        """
        keep = [g for g in genes if g in self._gene_index]
        idxs = np.array([self._gene_index[g] for g in keep])
        return Dataset(
            expression=self.expression[:, idxs].copy(),
            interventions=list(self.interventions),
            gene_names=keep,
        )

    def subset_cells(self, mask: np.ndarray) -> "Dataset":
        mask = np.asarray(mask, dtype=bool)
        return Dataset(
            expression=self.expression[mask].copy(),
            interventions=[iv for iv, m in zip(self.interventions, mask) if m],
            gene_names=list(self.gene_names),
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> str:
        n_ctrl = int(self.control_mask().sum())
        n_perturbed_genes = len(self.perturbed_genes())
        return (
            f"Dataset(n_cells={self.n_cells:,}, n_genes={self.n_genes:,}, "
            f"n_control_cells={n_ctrl:,}, n_perturbed_genes={n_perturbed_genes:,})"
        )
