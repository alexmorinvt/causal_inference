# CausalBench import
from causalscbench.models.abstract_model import AbstractInferenceModel

# 
from typing import List, Tuple
import random

class RandomModel(AbstractInferenceModel):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, expression_matrix, interventions, gene_names, training_regime, seed = 0) -> List[Tuple]:

        random.seed(seed)
        edges = set()
        size = len(gene_names)
        for i in range(size):
            a = gene_names[i]
            for j in range(i + 1, size):
                b = gene_names[j]

                if random.random() < (1. / size):
                    edges.add((a, b))
                    edges.add((b, a))

        return list(edges)
    