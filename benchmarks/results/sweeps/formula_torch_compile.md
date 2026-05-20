# formula_torch_compile

torch.compile on the formula model. First call pays ~20-30 s Inductor compilation; subsequent batches run kernel-fused. The wall in this sweep includes the compile cost.

- **Knob:** `formula_torch_compile`
- **PDF:** `904599.pdf`
- **Runs:** 2 (2 ok, 0 failed)
- **Fastest:** `False` @ 482.49s -- **Slowest:** `True` @ 485.64s (**1.01×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| False | ok | 482.49 | 11.78 | 5.32 | 12.78 | -- | 449.63 | -- | 0.07 |
| True | ok | 485.64 | 11.62 | 5.32 | 12.54 | -- | 452.25 | -- | 0.07 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| False | 27 / 104 / 874 / 9284 | 4181 / 4519 / 4664 / 16326 | 1613 / 4609 / 4609 / 6205 |
| True | 15 / 104 / 554 / 6449 | 4352 / 4568 / 4723 / 16525 | 1687 / 4661 / 4661 / 6275 |

## Quality

| value | formulas_with_text | formula_latex_chars | formulas_with_image |
|---|---|---|---|
| False | 900 | 106311 | 0 |
| True | 900 | 106311 | 0 |
