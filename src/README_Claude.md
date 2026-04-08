# Note: this file consists entirely of Claude-generated text

Parameters of AbstractInferenceModel.__call__                                                                                                     
                                                                                                                                                    
  expression_matrix: np.ndarray                                                                                                                     

  Shape: [nb_samples, nb_genes] — a 2D float array.

  Each row is one cell (a single sequenced cell from the Perturb-seq experiment). Each column is one gene. The values are log-normalized UMI counts
  — raw UMI counts are first normalized per cell to a common total, then log1p-transformed (sc.pp.log1p). Only genes that were also used as
  perturbation targets (and had ≥100 perturbed cells) are kept as columns, so nb_genes is a few hundred, not the full transcriptome.

  ---
  interventions: List[str]

  Length: nb_samples (one string per row/cell).

  Each string is either:
  - A gene name (e.g. "TP53") — meaning that gene was CRISPRi-knocked down in that cell, or
  - "non-targeting" — meaning that cell is from the control condition (no gene was perturbed; observational data).
  - "excluded" — assigned during preprocessing to cells whose perturbed gene had fewer than 100 cells, meaning those interventions were too sparse
  to be reliable. You can treat these like non-targeting.

  The gene names used here are Ensembl gene IDs (e.g. "ENSG00000141510"), sourced from data_expr_raw.obs["gene_id"] in the preprocessing step — not
  the human-readable HGNC symbols used in gene_names.

  ---
  gene_names: List[str]

  Length: nb_genes — one string per column of expression_matrix.

  These are the column identifiers of the expression matrix, loaded from arr["var_names"] in the .npz file, which comes from the AnnData .var index.
   They are the gene names (HGNC symbols, e.g. "TP53", "MYC") of the genes that were both expressed and used as perturbation targets with sufficient
   coverage. These are the only valid node names for your output edges.

  ---
  training_regime: TrainingRegime

  Type: TrainingRegime enum (causalscbench.models.training_regimes).

  ┌──────────────────────┬────────────────────────────────────────────────────────────────────────┐
  │      Enum value      │                       Meaning for interventions                        │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Observational        │ All entries are "non-targeting" or "excluded" — no intervention signal │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ PartialIntervational │ A random subset of perturbed genes is included alongside controls      │
  ├──────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Interventional       │ All perturbed genes with sufficient cells are included                 │
  └──────────────────────┴────────────────────────────────────────────────────────────────────────┘

  Your model can branch on this to decide whether to use interventions at all.

  ---
  seed: int

  Type: int, default 0. A plain integer passed directly to your RNG (e.g. random.seed(seed) or np.random.seed(seed)) for reproducibility.

  ---
  Return value: List[Tuple[str, str]]

  A list of 2-tuples of gene name strings from gene_names. Each tuple (A, B) represents a directed edge A → B (A causally regulates B) in the
  inferred GRN.

  Key constraints:
  - Both elements of each tuple must be strings present in gene_names
  - (A, B) and (B, A) are distinct edges — the list is treated as a directed graph
  - Duplicates are ignored by the evaluators (they convert the list to a set of pairs internally)
  - There is no required ordering or ranking — all returned edges are treated equally
  - An empty list is valid (predicts no edges)