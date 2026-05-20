# digital_born_workers

Process-pool size for digital-born block extraction. Each worker opens its own pdf_oxide doc; speed scales until OpenBLAS thread storms cap it on high-core boxes.

- **Knob:** `digital_born_workers`
- **PDF:** `904599.pdf`
- **Runs:** 7 (5 ok, 2 failed)
- **Fastest:** `1` @ 20.99s -- **Slowest:** `2` @ 39.09s (**1.86×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ok | 20.99 | 12.00 | 5.42 | 3.28 | -- | -- | -- | 0.19 |
| 2 | ok | 39.09 | 12.23 | 5.74 | 20.83 | -- | -- | -- | 0.22 |
| 4 | ok | 31.75 | 11.65 | 5.30 | 14.49 | -- | -- | -- | 0.22 |
| 8 | ok | 29.87 | 11.84 | 5.33 | 12.35 | -- | -- | -- | 0.24 |
| 16 | ok | 30.00 | 11.77 | 5.35 | 12.61 | -- | -- | -- | 0.18 |
| 32 | error | 42.19 | 11.86 | 5.58 | -- | -- | -- | -- | -- |
| 64 | error | 21.32 | 11.94 | 5.38 | -- | -- | -- | -- | -- |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| 1 | 58 / 1566 / 1755 / 3706 | 3028 / 8462 / 15189 / 15192 | 2719 / 2719 / 7539 / 8369 |
| 2 | 10 / 257 / 1634 / 7986 | 3051 / 4558 / 15152 / 15255 | 2789 / 2841 / 7381 / 8441 |
| 4 | 31 / 509 / 1666 / 11126 | 3054 / 6064 / 15172 / 15195 | 2841 / 2913 / 7433 / 8493 |
| 8 | 12 / 942 / 1756 / 8612 | 3011 / 9068 / 15179 / 15241 | 2913 / 2945 / 8365 / 8365 |
| 16 | 81 / 1602 / 1795 / 6093 | 3000 / 11148 / 15185 / 15221 | 2945 / 2969 / 8385 / 8385 |
| 32 | 65 / 1678 / 1780 / 3634 | 3000 / 15122 / 26426 / 26664 | 2969 / 2971 / 7445 / 8385 |
| 64 | 36 / 1604 / 1840 / 3867 | 3059 / 10493 / 15185 / 15187 | 2971 / 2971 / 8365 / 8365 |

## Quality

| value | blocks_total | text_chars |
|---|---|---|
| 1 | 1587 | 167805 |
| 2 | 1587 | 167805 |
| 4 | 1587 | 167805 |
| 8 | 1587 | 167805 |
| 16 | 1587 | 167805 |
| 32 |  |  |
| 64 |  |  |

## Failures

- `32` -- **error** -- `BrokenProcessPool('A process in the process pool was terminated abruptly while the future was running or pending.')`
- `64` -- **error** -- `BrokenProcessPool('A process in the process pool was terminated abruptly while the future was running or pending.')`

## Notes

- Wall time across 5 successful runs: min=20.99s, median=30.00s, max=39.09s.
