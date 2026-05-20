# formula_bucketed

Sequence-bucketed batching. Groups crops by bbox area so each batch's slowest row doesn't drag the rest through unused decode steps.

- **Knob:** `formula_bucketed`
- **PDF:** `904599.pdf`
- **Runs:** 2 (2 ok, 0 failed)
- **Fastest:** `True` @ 483.31s -- **Slowest:** `False` @ 600.06s (**1.24×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| False | ok | 600.06 | 11.48 | 5.68 | 13.03 | -- | 566.84 | -- | 0.08 |
| True | ok | 483.31 | 11.94 | 5.50 | 12.73 | -- | 450.20 | -- | 0.08 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| False | 8 / 104 / 488 / 10023 | 3622 / 3925 / 4010 / 15743 | 1417 / 4413 / 4413 / 6009 |
| True | 19 / 104 / 781 / 1999 | 3870 / 4106 / 4291 / 16007 | 1489 / 4463 / 4463 / 6081 |

## Quality

| value | formulas_with_text | formula_latex_chars | formulas_with_image |
|---|---|---|---|
| False | 900 | 106865 | 0 |
| True | 900 | 106311 | 0 |
