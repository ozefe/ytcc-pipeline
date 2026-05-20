# formula_batch_size

Formula-recognition batch size. Greedy generation runs the whole batch for the slowest row, so the optimum balances throughput against straggler cost.

- **Knob:** `formula_batch_size`
- **PDF:** `904599.pdf`
- **Runs:** 3 (3 ok, 0 failed)
- **Fastest:** `16` @ 228.74s -- **Slowest:** `4` @ 486.63s (**2.13×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| 4 | ok | 486.63 | 11.43 | 5.43 | 12.59 | -- | 454.16 | -- | 0.07 |
| 8 | ok | 325.30 | 11.50 | 5.25 | 12.74 | -- | 292.77 | -- | 0.07 |
| 16 | ok | 228.74 | 11.80 | 5.43 | 12.87 | -- | 195.63 | -- | 0.07 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| 4 | 8 / 104 / 644 / 2149 | 3080 / 3197 / 3355 / 15229 | 1171 / 4143 / 4145 / 5761 |
| 8 | 16 / 104 / 1597 / 5470 | 3149 / 3340 / 11045 / 15387 | 1221 / 6789 / 6789 / 6801 |
| 16 | 16 / 104 / 1609 / 3316 | 3272 / 3490 / 13858 / 15414 | 1293 / 12039 / 12039 / 12041 |

## Quality

| value | formulas_with_text | formula_latex_chars | formulas_with_image |
|---|---|---|---|
| 4 | 900 | 106311 | 0 |
| 8 | 900 | 106571 | 0 |
| 16 | 900 | 106543 | 0 |

## Notes

- Wall time across 3 successful runs: min=228.74s, median=325.30s, max=486.63s.
