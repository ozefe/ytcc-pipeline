# ocr_batch_size

RapidOCR recognition batch size (`Rec.batch_size`). Larger batches reduce kernel-launch overhead but trade off latency.

- **Knob:** `ocr_batch_size`
- **PDF:** `084016.pdf`
- **Runs:** 4 (4 ok, 0 failed)
- **Fastest:** `128` @ 118.81s -- **Slowest:** `64` @ 121.28s (**1.02×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| 4 | ok | 119.52 | 20.58 | 11.27 | 87.46 | -- | -- | -- | 0.12 |
| 16 | ok | 119.99 | 20.21 | 8.73 | 90.81 | -- | -- | -- | 0.15 |
| 64 | ok | 121.28 | 20.72 | 9.00 | 91.30 | -- | -- | -- | 0.15 |
| 128 | ok | 118.81 | 20.36 | 9.41 | 88.78 | -- | -- | -- | 0.16 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| 4 | 20 / 714 / 1616 / 2160 | 4866 / 23719 / 51634 / 52670 | 2861 / 17733 / 20165 / 20165 |
| 16 | 44 / 711 / 1614 / 2597 | 4872 / 24427 / 51193 / 52674 | 2861 / 15043 / 20931 / 21187 |
| 64 | 19 / 707 / 1616 / 12740 | 4876 / 24646 / 51879 / 52640 | 2859 / 15937 / 20161 / 20161 |
| 128 | 31 / 714 / 1612 / 4103 | 4877 / 24020 / 51379 / 52820 | 2855 / 16829 / 20042 / 20157 |

## Quality

| value | text_chars | miss |
|---|---|---|
| 4 | 146516 | 4 |
| 16 | 146516 | 4 |
| 64 | 146516 | 4 |
| 128 | 146516 | 4 |

## Notes

- Wall time across 4 successful runs: min=118.81s, median=119.76s, max=121.28s.
