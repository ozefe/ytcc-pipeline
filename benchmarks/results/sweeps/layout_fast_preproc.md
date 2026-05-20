# layout_fast_preproc

cv2 preprocessing + producer-thread overlap vs HF's PIL-based AutoImageProcessor. Layout-stage isolated.

- **Knob:** `layout_fast_preproc`
- **PDF:** `904599.pdf`
- **Runs:** 2 (2 ok, 0 failed)
- **Fastest:** `True` @ 30.03s -- **Slowest:** `False` @ 33.53s (**1.12×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| False | ok | 33.53 | 11.99 | 8.55 | 12.68 | -- | -- | -- | 0.24 |
| True | ok | 30.03 | 11.78 | 5.33 | 12.64 | -- | -- | -- | 0.18 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| False | 84 / 1601 / 1780 / 3061 | 2506 / 10579 / 14625 / 14637 | 1545 / 1617 / 7055 / 7195 |
| True | 53 / 1601 / 1797 / 6590 | 2510 / 10728 / 14653 / 14759 | 1617 / 1667 / 6207 / 7267 |

## Quality

| value | blocks_total | blocks_text | blocks_image |
|---|---|---|---|
| False | 1602 | 574 | 37 |
| True | 1587 | 570 | 37 |
