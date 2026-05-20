# formula_enabled

Master toggle for the formula stage. Off: formula crops ship image-only. On: PP-FormulaNet-L recovers LaTeX.

- **Knob:** `formula_enabled`
- **PDF:** `904599.pdf`
- **Runs:** 2 (2 ok, 0 failed)
- **Fastest:** `False` @ 30.08s -- **Slowest:** `True` @ 498.94s (**16.59×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| False | ok | 30.08 | 12.06 | 5.27 | 12.43 | -- | -- | -- | 0.24 |
| True | ok | 498.94 | 11.52 | 5.40 | 12.64 | -- | 466.36 | -- | 0.08 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| False | 8 / 1601 / 1827 / 2039 | 3060 / 11333 / 15195 / 15234 | 2973 / 2977 / 8201 / 8385 |
| True | 16 / 104 / 815 / 2857 | 3014 / 3176 / 3351 / 15207 | 2977 / 5843 / 5843 / 8445 |

## Quality

| value | blocks_formula | formulas_with_text | formula_latex_chars | formulas_with_image |
|---|---|---|---|---|
| False | 900 | 0 | 0 | 900 |
| True | 900 | 900 | 106311 | 0 |
