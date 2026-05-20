# references_enabled

Master toggle for reference parsing via an external GROBID server. Off: `Block.reference` stays null. On: parsed citations attached to every reference block GROBID could parse.

- **Knob:** `references_enabled`
- **PDF:** `904599.pdf`
- **Runs:** 2 (2 ok, 0 failed)
- **Fastest:** `False` @ 29.88s -- **Slowest:** `True` @ 61.52s (**2.06×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| False | ok | 29.88 | 11.50 | 5.46 | 12.58 | -- | -- | -- | 0.24 |
| True | ok | 61.52 | 11.49 | 5.30 | 12.55 | -- | -- | 31.85 | 0.22 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| False | 34 / 1604 / 1816 / 2996 | 4948 / 13441 / 17095 / 17245 | 2405 / 2477 / 6997 / 6997 |
| True | 0 / 8 / 1716 / 2666 | 5010 / 5013 / 17070 / 17198 | 2477 / 2529 / 7069 / 7069 |

## Quality

| value | blocks_reference | references_parsed | references_total |
|---|---|---|---|
| False | 44 | 0 | 44 |
| True | 44 | 44 | 44 |
