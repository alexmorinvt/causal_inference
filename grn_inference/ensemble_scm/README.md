# EnsembleSCMFitter

Ensemble of linear cyclic SCMs fit by moment-matching against observed
per-perturbation distributions. The only method here that fits a generative
model — it can recover edges from unperturbed sources by explaining
observational structure, not just intervention shifts.

## Algorithm

1. Initialise N random adjacency matrices `W ∈ R^{G×G}` with small entries.
2. At each of `n_steps`:
   - Sample an arm from `{control} ∪ {perturbed genes}`.
   - Simulate `batch_size` cells per candidate via `x = (I − W)⁻¹ ε` (batched solve).
   - Compute per-candidate `moment_matching_discrepancy(sim, real)` + L1 penalty.
   - `torch.autograd.grad` → manual SGD step on W.
   - **Spectral-radius projection**: rescale any W_k with `ρ(W_k) > spectral_threshold` so `ρ = spectral_threshold`. Without this, W diverges within ~5 steps and the fit is unrecoverable.
3. Aggregate `|W|` across candidates with a generalised mean `(mean_k |W_k|^p)^(1/p)`, zero diagonals, return top-k edges.

## Key hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `spectral_threshold` | 0.80 | Match the synthetic generator's `target_spectral_radius`. Never raise above ~0.9 — the simulator over-amplifies and discrepancy plateaus. |
| `n_steps` | 1000 | Saturated at current threshold; more iterations don't help. |
| `step_size` | 0.01 | Saturated. Smaller steps slow convergence; larger steps destabilise. |
| `l1_lambda` | 1e-4 | Only bites at `spectral_threshold ≤ 0.9`. At 0.80 the sweet spot is 1e-4 to 1e-3. |
| `aggregation_power` | 3.0 | Generalised mean exponent — biased toward max but not pure max. |
| `n_candidates` | 5 | More candidates improve coverage; diminishing returns above ~5. |

## Failure modes

- **Divergence**: without spectral projection, `ρ(W)` crosses 1 by step ~5 and shoots to 10¹³. `scripts/diagnose_divergence.py` shows the trajectory.
- **Discrepancy plateau at 10⁴–10⁵**: `spectral_threshold` too high (typically 0.95). Simulator over-amplifies relative to data. Lower threshold toward 0.8.
- **Dense fitted W**: weak L1 + loose threshold → `|W|` has no natural zeros. Lower threshold (primary fix) or raise L1 modestly.

## Scalability

Runtime scales as O(n_steps × n_candidates × G³) due to the `(I−W)⁻¹` solve.
Practical limits on CPU:
- G ≤ 100: comfortable at n_steps=1000, n_candidates=5
- G ~ 600 (real K562): use n_candidates=3, n_steps=200

## Files

| File | Purpose |
|---|---|
| `simulator.py` | `LinearSCM`, `simulate_control`, `simulate_intervention` |
| `loss.py` | `moment_matching_discrepancy`, `l1_penalty` |
| `fit.py` | `fit_scm_ensemble` — the SGD loop with spectral projection |
| `model.py` | `EnsembleSCMFitter` — Model-protocol entry point |
