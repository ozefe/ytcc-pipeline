# page_format

Temp page-render container: PNG (lossless) vs JPEG. Layout downsamples to 800x800 so the JPEG artefacts are erased; the only impact is render wall.

- **Knob:** `page_format`
- **PDF:** `904599.pdf`
- **Runs:** 2 (2 ok, 0 failed)
- **Fastest:** `jpeg` @ 30.00s -- **Slowest:** `png` @ 32.94s (**1.10×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| png | ok | 32.94 | 11.04 | 8.73 | 12.89 | -- | -- | -- | 0.17 |
| jpeg | ok | 30.00 | 11.71 | 5.12 | 12.83 | -- | -- | -- | 0.24 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| png | 58 / 1601 / 1850 / 7928 | 2706 / 10572 / 14820 / 14854 | 2131 / 2203 / 7641 / 7781 |
| jpeg | 36 / 1602 / 1800 / 5268 | 2769 / 11245 / 14882 / 14896 | 2203 / 2255 / 6793 / 7853 |

## Quality

| value | blocks_total | blocks_text | text_chars |
|---|---|---|---|
| png | 1588 | 570 | 167805 |
| jpeg | 1587 | 570 | 167805 |
