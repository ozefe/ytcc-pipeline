# layout_batch_size

Layout-model batch size on a digital-born PDF. Tests how many pages PP-DocLayoutV3 processes per GPU forward pass; larger batches mean fewer kernel launches but a higher VRAM peak.

- **Knob:** `layout_batch_size`
- **PDF:** `904599.pdf`
- **Runs:** 6 (6 ok, 0 failed)
- **Fastest:** `8` @ 32.41s -- **Slowest:** `1` @ 35.45s (**1.09×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ok | 35.45 | 11.72 | 10.52 | 12.88 | -- | -- | -- | 0.23 |
| 2 | ok | 33.34 | 11.62 | 8.70 | 12.75 | -- | -- | -- | 0.18 |
| 4 | ok | 33.88 | 11.59 | 9.30 | 12.71 | -- | -- | -- | 0.17 |
| 8 | ok | 32.41 | 11.64 | 7.69 | 12.82 | -- | -- | -- | 0.18 |
| 16 | ok | 33.28 | 11.52 | 9.27 | 12.25 | -- | -- | -- | 0.17 |
| 24 | ok | 34.90 | 12.02 | 7.47 | 15.09 | -- | -- | -- | 0.23 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| 1 | 29 / 1592 / 1833 / 2027 | 1913 / 9726 / 14040 / 14068 | 755 / 827 / 1505 / 1505 |
| 2 | 8 / 1598 / 1822 / 1991 | 1922 / 9980 / 14084 / 14162 | 827 / 901 / 2259 / 2259 |
| 4 | 42 / 1597 / 1767 / 10685 | 2027 / 9953 / 14150 / 14169 | 901 / 953 / 3671 / 3671 |
| 8 | 12 / 1600 / 1871 / 10636 | 2091 / 10167 / 14206 / 14223 | 953 / 1005 / 6443 / 6443 |
| 16 | 28 / 1599 / 1753 / 9227 | 2144 / 10147 / 14293 / 14301 | 1005 / 1077 / 11955 / 11955 |
| 24 | 64 / 1444 / 1788 / 1890 | 2251 / 10383 / 14384 / 14443 | 1077 / 1151 / 14589 / 14769 |

## Quality

| value | blocks_total | text_chars | miss |
|---|---|---|---|
| 1 | 1588 | 167805 | 0 |
| 2 | 1589 | 167805 | 0 |
| 4 | 1588 | 167805 | 0 |
| 8 | 1587 | 167805 | 0 |
| 16 | 1588 | 167805 | 0 |
| 24 | 1589 | 167805 | 0 |

## Notes

- Wall time across 6 successful runs: min=32.41s, median=33.61s, max=35.45s.
