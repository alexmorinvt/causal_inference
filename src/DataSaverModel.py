import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime

DATASET = "k562"
OUTPUT_DIR = os.path.join("manual_data", "perturbations", DATASET)

class DataSaverModel(AbstractInferenceModel):

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
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        interventions = np.array(list(interventions))

        # Save observational cells
        obs_mask = interventions == "non-targeting"
        obs_df = pd.DataFrame(expression_matrix[obs_mask], columns=gene_names)
        obs_df.to_csv(os.path.join(OUTPUT_DIR, "Observational.csv"), index=False)

        # Save one dataframe per perturbed gene
        perturbed_genes = [
            g for g in np.unique(interventions)
            if g not in ("non-targeting", "excluded")
        ]
        for gene in perturbed_genes:
            pert_mask = interventions == gene
            pert_df = pd.DataFrame(expression_matrix[pert_mask], columns=gene_names)
            pert_df.to_csv(os.path.join(OUTPUT_DIR, f"{gene}_data.csv"), index=False)

        return []
