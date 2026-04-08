from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from typing import List, Tuple

import numpy as np


class MeanDifference(AbstractInferenceModel):

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        expression_matrix: np.ndarray,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        gene_names = np.array(gene_names)
        interventions = list(interventions)

        node_dict = {g: idx for idx, g in enumerate(gene_names)}
        gene_names_set = set(gene_names)

        # Filter cells to only those with a recognized perturbation or non-targeting control
        subset = []
        interventions_ = []
        for idx, iv in enumerate(interventions):
            if iv in gene_names_set or iv == "non-targeting":
                subset.append(idx)
                interventions_.append(iv)
        expression_matrix = expression_matrix[subset, :]

        # Group cell indices by intervention
        gene_to_interventions = dict()
        for i, intervention in enumerate(interventions_):
            gene_to_interventions.setdefault(intervention, []).append(i)

        # Mean expression of each gene across observational cells
        obs_indices = gene_to_interventions.get("non-targeting", [])
        obs_mean = expression_matrix[obs_indices].mean(axis=0)  # [nb_genes]

        # Collect (strength, gene_a, gene_b) for all candidate edges
        scored_edges = []

        for gene_a, pert_indices in gene_to_interventions.items():
            if gene_a == "non-targeting":
                continue
            a_idx = node_dict[gene_a]
            pert_mean = expression_matrix[pert_indices].mean(axis=0)  # [nb_genes]

            # Normalized strength: relative change in B's expression when A is perturbed.
            # obs_mean is used as denominator; genes with zero observational mean are skipped.
            with np.errstate(invalid="ignore", divide="ignore"):
                strength = np.where(
                    obs_mean != 0,
                    (obs_mean - pert_mean),
                    np.nan,
                )

            for b_idx in range(len(gene_names)):
                if b_idx == a_idx:
                    continue
                s = strength[b_idx]
                if not np.isnan(s):
                    scored_edges.append((abs(s), gene_a, gene_names[b_idx]))

        # Return the 1000 edges with the largest absolute normalized strength
        scored_edges.sort(key=lambda x: x[0], reverse=True)
        return [(gene_a, gene_b) for _, gene_a, gene_b in scored_edges[:1000]]
