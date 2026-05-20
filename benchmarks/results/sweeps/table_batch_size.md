# table_batch_size

Tables per SLANet+ structure forward pass. Quality is the tables-with-cells count; it should stay constant.

- **Knob:** `table_batch_size`
- **PDF:** `904599.pdf`
- **Runs:** 5 (5 ok, 0 failed)
- **Fastest:** `4` @ 36.70s -- **Slowest:** `1` @ 38.96s (**1.06×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ok | 38.96 | 11.55 | 5.75 | 12.83 | 8.22 | -- | -- | 0.23 |
| 2 | ok | 37.22 | 11.65 | 5.45 | 12.95 | 6.62 | -- | -- | 0.22 |
| 4 | ok | 36.70 | 11.78 | 5.29 | 12.72 | 6.33 | -- | -- | 0.23 |
| 8 | ok | 37.67 | 11.86 | 5.33 | 13.03 | 6.90 | -- | -- | 0.23 |
| 16 | ok | 37.10 | 11.75 | 5.16 | 12.62 | 6.98 | -- | -- | 0.22 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| 1 | 37 / 1593 / 1823 / 10704 | 4860 / 12322 / 16981 / 17007 | 2015 / 2087 / 6605 / 6605 |
| 2 | 53 / 1596 / 1764 / 7174 | 4919 / 12795 / 17034 / 17056 | 2087 / 2139 / 6677 / 6677 |
| 4 | 33 / 1599 / 1779 / 8849 | 4920 / 12863 / 17045 / 17126 | 2139 / 2211 / 6729 / 6729 |
| 8 | 28 / 1597 / 1781 / 2034 | 4921 / 12912 / 17038 / 17140 | 2211 / 2281 / 6801 / 6801 |
| 16 | 12 / 1600 / 1791 / 7667 | 4919 / 12980 / 17034 / 17061 | 2281 / 2335 / 6873 / 6873 |

## Quality

| value | tables_with_cells | tables_image_only | total_cells | cells_with_text |
|---|---|---|---|---|
| 1 | 36 | 0 | 1871 | 1863 |
| 2 | 36 | 0 | 1871 | 1863 |
| 4 | 36 | 0 | 1871 | 1863 |
| 8 | 36 | 0 | 1871 | 1863 |
| 16 | 36 | 0 | 1871 | 1863 |

## Notes

- Wall time across 5 successful runs: min=36.70s, median=37.22s, max=38.96s.
