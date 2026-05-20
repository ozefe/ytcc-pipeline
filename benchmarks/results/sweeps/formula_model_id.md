# formula_model_id

PP-FormulaNet-L vs PP-FormulaNet_plus-L. The +L variant has a 2560-token decoder position limit vs L's 1024.

- **Knob:** `formula_model_id`
- **PDF:** `904599.pdf`
- **Runs:** 2 (2 ok, 0 failed)
- **Fastest:** `PaddlePaddle/PP-FormulaNet_plus-L_safetensors` @ 330.66s -- **Slowest:** `PaddlePaddle/PP-FormulaNet-L_safetensors` @ 508.38s (**1.54×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| PaddlePaddle/PP-FormulaNet-L_safetensors | ok | 508.38 | 11.53 | 5.82 | 12.92 | -- | 475.04 | -- | 0.07 |
| PaddlePaddle/PP-FormulaNet_plus-L_safetensors | ok | 330.66 | 11.54 | 5.60 | 12.91 | -- | 297.03 | -- | 0.07 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| PaddlePaddle/PP-FormulaNet-L_safetensors | 15 / 104 / 723 / 10576 | 2402 / 2545 / 2685 / 14548 | 757 / 3751 / 3751 / 6267 |
| PaddlePaddle/PP-FormulaNet_plus-L_safetensors | 16 / 104 / 1600 / 10389 | 2542 / 2723 / 10354 / 14732 | 829 / 3823 / 3823 / 6359 |

## Quality

| value | blocks_formula | formulas_with_text | formula_latex_chars | formulas_with_image |
|---|---|---|---|---|
| PaddlePaddle/PP-FormulaNet-L_safetensors | 900 | 900 | 106311 | 0 |
| PaddlePaddle/PP-FormulaNet_plus-L_safetensors | 900 | 900 | 86510 | 0 |
