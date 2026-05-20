# layout_fp16

Run PP-DocLayoutV3 in fp16 vs fp32 on a digital-born PDF. Speed comes from tensor-core throughput; quality is tracked via total detected blocks.

- **Knob:** `layout_fp16`
- **PDF:** `904599.pdf`
- **Runs:** 2 (2 ok, 0 failed)
- **Fastest:** `True` @ 30.23s -- **Slowest:** `False` @ 30.72s (**1.02×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| False | ok | 30.72 | 11.74 | 5.89 | 12.82 | -- | -- | -- | 0.21 |
| True | ok | 30.23 | 11.88 | 5.44 | 12.62 | -- | -- | -- | 0.21 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| False | 40 / 1601 / 1741 / 9440 | 2415 / 10519 / 14521 / 14549 | 1245 / 1379 / 11379 / 12345 |
| True | 15 / 1602 / 1805 / 9528 | 2367 / 10848 / 14552 / 14571 | 1379 / 1453 / 6011 / 6871 |

## Quality

| value | blocks_total | blocks_text | blocks_image |
|---|---|---|---|
| False | 1589 | 570 | 37 |
| True | 1587 | 570 | 37 |
