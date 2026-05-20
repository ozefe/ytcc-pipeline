# render_dpi_scanned

Render DPI for scanned PDFs. OCR accuracy is sensitive to source resolution; lower DPI saves render + decode time but loses recall.

- **Knob:** `render_dpi`
- **PDF:** `084016.pdf`
- **Runs:** 5 (5 ok, 0 failed)
- **Fastest:** `100` @ 105.11s -- **Slowest:** `300` @ 119.43s (**1.14×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| 100 | ok | 105.11 | 13.80 | 4.16 | 86.99 | -- | -- | -- | 0.08 |
| 150 | ok | 105.38 | 14.56 | 4.91 | 85.71 | -- | -- | -- | 0.11 |
| 200 | ok | 109.13 | 15.84 | 6.09 | 87.01 | -- | -- | -- | 0.09 |
| 250 | ok | 112.58 | 16.75 | 7.43 | 88.15 | -- | -- | -- | 0.13 |
| 300 | ok | 119.43 | 20.55 | 10.96 | 87.66 | -- | -- | -- | 0.16 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| 100 | 20 / 699 / 1615 / 20943 | 4629 / 25254 / 47296 / 47893 | 2603 / 19433 / 20969 / 20969 |
| 150 | 31 / 706 / 1614 / 1872 | 4734 / 25324 / 47246 / 47946 | 2675 / 17311 / 21023 / 21023 |
| 200 | 43 / 708 / 1612 / 20672 | 4741 / 24964 / 49146 / 50007 | 2729 / 18149 / 23397 / 23397 |
| 250 | 8 / 709 / 1614 / 2620 | 4805 / 25241 / 50287 / 51371 | 2799 / 19205 / 20101 / 20101 |
| 300 | 54 / 714 / 1617 / 2054 | 4863 / 23560 / 51336 / 52668 | 2831 / 15721 / 19881 / 20137 |

## Quality

| value | text_chars | miss | blocks_text |
|---|---|---|---|
| 100 | 140689 | 2 | 821 |
| 150 | 144155 | 2 | 831 |
| 200 | 146074 | 5 | 845 |
| 250 | 145676 | 3 | 855 |
| 300 | 146516 | 4 | 858 |

## Notes

- Wall time across 5 successful runs: min=105.11s, median=109.13s, max=119.43s.
