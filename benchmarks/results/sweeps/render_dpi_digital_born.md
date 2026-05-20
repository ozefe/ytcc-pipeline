# render_dpi_digital_born

Page-render DPI for digital-born PDFs. Layout downsamples to 800x800 internally and pdf_oxide reads the text layer directly, so the quality delta is small but render cost scales with DPI².

- **Knob:** `render_dpi_digital_born`
- **PDF:** `904599.pdf`
- **Runs:** 5 (5 ok, 0 failed)
- **Fastest:** `100` @ 28.94s -- **Slowest:** `300` @ 39.83s (**1.38×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| 100 | ok | 28.94 | 11.14 | 4.64 | 12.81 | -- | -- | -- | 0.25 |
| 150 | ok | 30.29 | 11.81 | 5.37 | 12.85 | -- | -- | -- | 0.17 |
| 200 | ok | 32.90 | 12.88 | 6.61 | 13.08 | -- | -- | -- | 0.23 |
| 250 | ok | 36.28 | 14.08 | 8.46 | 13.45 | -- | -- | -- | 0.19 |
| 300 | ok | 39.83 | 15.64 | 10.83 | 13.07 | -- | -- | -- | 0.19 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| 100 | 41 / 1604 / 1810 / 10896 | 2580 / 10748 / 14662 / 14744 | 1739 / 1811 / 6331 / 7391 |
| 150 | 23 / 1602 / 1817 / 4214 | 2585 / 10904 / 14779 / 14801 | 1811 / 1863 / 6403 / 7463 |
| 200 | 42 / 1599 / 1803 / 5701 | 2645 / 10890 / 14981 / 15061 | 1863 / 1935 / 6455 / 7515 |
| 250 | 12 / 1602 / 1797 / 3223 | 2700 / 10900 / 15303 / 15326 | 1935 / 2009 / 6849 / 7587 |
| 300 | 39 / 1600 / 1779 / 7100 | 2649 / 10800 / 15473 / 15524 | 2009 / 2059 / 7241 / 7657 |

## Quality

| value | blocks_total | text_chars | blocks_text |
|---|---|---|---|
| 100 | 1557 | 167835 | 569 |
| 150 | 1587 | 167805 | 570 |
| 200 | 1598 | 167910 | 568 |
| 250 | 1592 | 168114 | 572 |
| 300 | 1585 | 167824 | 568 |

## Notes

- Wall time across 5 successful runs: min=28.94s, median=32.90s, max=39.83s.
