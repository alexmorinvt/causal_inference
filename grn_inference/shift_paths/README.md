# ShiftPathsModel

Path-scoring ranker that greedily selects edges by their contribution
as the terminal hop in multi-hop paths to sink genes. Designed to
distinguish direct regulatory edges from cascade shortcuts by weighting
path scores toward the final hop.

## Algorithm

1. Compute shift matrix S; threshold to top `top_frac` fraction → directed graph H.
2. Re-rank nodes by `in_degree − out_degree` (descending). The top node is the most sink-like gene N.
3. For every source G, enumerate simple paths `G → … → N` up to `max_path_length` hops. Score each path with exponentially-decaying weights *largest on the final hop*:
   ```
   score(path of length L) = Σ_{i=0..L-1}  S[v_i, v_{i+1}] / 2^(L-1-i)
   ```
4. Tally scores onto the *final edge* of each best path per source.
5. Greedily select the highest-tally edge, remove it from H, repeat from step 2.

Invalidated cache entries (paths that traversed a removed edge) are evicted lazily.

## Key hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `top_frac` | 0.30 | Candidate graph density. |
| `max_path_length` | 5 | Upper bound on hops considered. |
| `top_k` | 1000 | Max edges returned. |

## Status

Currently excluded from benchmarks — the greedy loop has a correctness
issue under investigation. Do not use until fixed.
