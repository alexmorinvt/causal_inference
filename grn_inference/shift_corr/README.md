# ShiftCorrModel

Mean-shift ranker augmented with within-arm Pearson correlation. The
correlation bonus rewards edges where per-cell residual knockdown of the
source gene tracks proportionally with the response in the target —
a signal that attenuates across cascade hops.

## Algorithm

For each perturbed source `A` and each target `B`:

```
shift[A, B] = |mean(B | do(A)) − mean(B | control)|
corr[A, B]  = Pearson(A, B) within cells from do(A)
score[A, B] = shift[A, B] × (1 + corr_weight × |corr[A, B]|)
```

Rank by score, return top-k.

Setting `corr_weight = 0` reduces exactly to `MeanDifferenceModel`.

## Key hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `corr_weight` | 1.0 | Multiplicative weight on the correlation bonus. |
| `top_k` | 1000 | Max edges returned. |

## Notes

- Only scores edges from perturbed sources (same limitation as Mean Difference).
- The correlation term adds a small but consistent precision boost over Mean Difference on synthetic data at top_k=50–100.
