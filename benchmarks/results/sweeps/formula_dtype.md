# formula_dtype

fp16 vs fp32 inference for PP-FormulaNet-L. fp16 halves VRAM (~1.8 GiB vs ~3.6 GiB) and exploits tensor cores.

- **Knob:** `formula_dtype`
- **PDF:** `904599.pdf`
- **Runs:** 2 (2 ok, 0 failed)
- **Fastest:** `fp16` @ 497.99s -- **Slowest:** `fp32` @ 518.23s (**1.04×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| fp32 | ok | 518.23 | 11.77 | 5.32 | 12.74 | -- | 486.33 | -- | 0.07 |
| fp16 | ok | 497.99 | 11.47 | 5.61 | 12.68 | -- | 465.17 | -- | 0.09 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| fp32 | 4 / 104 / 536 / 2282 | 2778 / 2785 / 2934 / 14909 | 955 / 4933 / 4933 / 6445 |
| fp16 | 8 / 104 / 572 / 8002 | 2780 / 2929 / 3074 / 14970 | 1007 / 4023 / 4023 / 5617 |

## Quality

| value | formulas_with_text | formula_latex_chars | formulas_with_image |
|---|---|---|---|
| fp32 | 900 | 107055 | 0 |
| fp16 | 900 | 106311 | 0 |
