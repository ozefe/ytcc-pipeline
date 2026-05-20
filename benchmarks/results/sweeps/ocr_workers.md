# ocr_workers

Spawn-pool size for scanned OCR. Each worker owns its own RapidOCR engine (~2 GiB VRAM); too many saturate the device and trigger BFC allocator OOMs.

- **Knob:** `ocr_workers`
- **PDF:** `084016.pdf`
- **Runs:** 3 (3 ok, 0 failed)
- **Fastest:** `8` @ 101.30s -- **Slowest:** `1` @ 409.74s (**4.04×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ok | 409.74 | 18.29 | 9.21 | 382.01 | -- | -- | -- | 0.11 |
| 4 | ok | 148.60 | 18.45 | 11.97 | 117.90 | -- | -- | -- | 0.19 |
| 8 | ok | 101.30 | 19.35 | 9.31 | 72.35 | -- | -- | -- | 0.19 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| 1 | 74 / 121 / 414 / 1877 | 32639 / 32757 / 40390 / 45894 | 773 / 3505 / 3505 / 6283 |
| 4 | 8 / 478 / 1610 / 3934 | 36887 / 55741 / 79593 / 81421 | 845 / 12665 / 12665 / 12665 |
| 8 | 38 / 939 / 1612 / 2406 | 36889 / 56535 / 86207 / 88107 | 917 / 16723 / 24557 / 24557 |

## Quality

| value | text_chars | miss |
|---|---|---|
| 1 | 146516 | 4 |
| 4 | 146516 | 4 |
| 8 | 146516 | 4 |

## Notes

- Wall time across 3 successful runs: min=101.30s, median=148.60s, max=409.74s.
