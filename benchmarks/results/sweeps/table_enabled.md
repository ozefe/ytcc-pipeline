# table_enabled

Master toggle for the table stage. Off: table blocks ship image-only. On: SLANet+ recovers the cell grid + per-cell text.

- **Knob:** `table_enabled`
- **PDF:** `904599.pdf`
- **Runs:** 2 (2 ok, 0 failed)
- **Fastest:** `False` @ 29.94s -- **Slowest:** `True` @ 38.21s (**1.28×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| False | ok | 29.94 | 11.71 | 5.39 | 12.56 | -- | -- | -- | 0.18 |
| True | ok | 38.21 | 11.67 | 5.66 | 12.54 | 7.76 | -- | -- | 0.22 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| False | 8 / 1602 / 1777 / 9635 | 4487 / 12807 / 16659 / 16674 | 1809 / 1881 / 6399 / 6399 |
| True | 55 / 1598 / 1844 / 3103 | 4487 / 12415 / 16616 / 16797 | 1881 / 1933 / 6471 / 6471 |

## Quality

| value | blocks_table | tables_with_cells | tables_image_only | total_cells | cells_with_text |
|---|---|---|---|---|---|
| False | 36 | 0 | 36 | 0 | 0 |
| True | 36 | 36 | 0 | 1871 | 1863 |
