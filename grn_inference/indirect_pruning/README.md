# IndirectPruningModel

Shift-based ranker that removes cascade shortcuts before ranking. The
intuition: if `A → B` and `A → C → B` both exist, the observed shift
`S(A, B)` may be driven entirely by the cascade, not a direct edge.
Prune `A → B` if a two-hop alternative explains it nearly as well.

## Algorithm

1. Compute shift matrix `S[i, j] = |mean(B | do(A)) − mean(B | control)|` for all perturbed sources.
2. Keep the top `top_frac` fraction of off-diagonal entries as candidate edges, forming graph H.
3. For each candidate edge `A → B` in H, compute its best two-hop alternative:
   ```
   indirect(A, B) = max over C: (S[A, C] + S[C, B]) / 2
   ```
   where both `A → C` and `C → B` must be in H.
4. Remove `A → B` if `indirect(A, B) / S[A, B] > prune_ratio`.
5. Rank surviving edges by `S[A, B]` and return top-k.

## Key hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `top_frac` | 0.30 | Fraction of pairs kept as candidates. Too low → prune away real edges; too high → cascade shortcuts survive. |
| `prune_ratio` | 0.90 | Remove an edge if the two-hop alternative scores above this fraction of the direct shift. |
| `top_k` | 1000 | Max edges returned. |

## Known limitation

The model only returns edges whose source is a perturbed gene (pruning
uses shift matrix rows). Returns far fewer than `top_k` when most
high-shift edges are pruned and the candidate pool is exhausted.
