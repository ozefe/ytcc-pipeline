# jpeg_quality

JPEG quality knob (1-100). Affects the temp page renders that the scanned-path OCR reads from disk. Benchmarked on a Turkish scanned PDF because Turkish diacritics (ç, ğ, ı, ş, ü, ö) make OCR character recall sensitive to JPEG compression artefacts; the digital-born path bypasses the rendered image entirely (pdf_oxide reads the embedded text layer) so there's no signal to measure there.

- **Knob:** `jpeg_quality`
- **PDF:** `101123.pdf`
- **Runs:** 7 (7 ok, 0 failed)
- **Fastest:** `40` @ 114.88s -- **Slowest:** `90` @ 116.65s (**1.02×** spread)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| 40 | ok | 114.88 | 17.48 | 9.18 | 88.05 | -- | -- | -- | 0.08 |
| 50 | ok | 116.21 | 17.69 | 8.97 | 89.33 | -- | -- | -- | 0.12 |
| 60 | ok | 115.18 | 18.11 | 8.47 | 88.43 | -- | -- | -- | 0.09 |
| 70 | ok | 114.93 | 17.58 | 8.34 | 88.77 | -- | -- | -- | 0.13 |
| 80 | ok | 115.41 | 18.02 | 8.57 | 88.56 | -- | -- | -- | 0.14 |
| 90 | ok | 116.65 | 17.59 | 8.86 | 89.92 | -- | -- | -- | 0.16 |
| 100 | ok | 115.38 | 17.76 | 8.87 | 88.39 | -- | -- | -- | 0.24 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| 40 | 81 / 703 / 1612 / 3515 | 2004 / 22440 / 46590 / 48195 | 757 / 18707 / 21139 / 21139 |
| 50 | 54 / 702 / 1609 / 2525 | 2033 / 22037 / 46530 / 48133 | 829 / 16859 / 20187 / 20187 |
| 60 | 34 / 701 / 1610 / 1925 | 2034 / 22093 / 46577 / 48293 | 901 / 16271 / 24567 / 24567 |
| 70 | 8 / 701 / 1613 / 1868 | 2036 / 22576 / 46796 / 48260 | 953 / 16835 / 23363 / 23363 |
| 80 | 27 / 705 / 1612 / 5684 | 2061 / 22541 / 46878 / 48420 | 1005 / 18461 / 24477 / 24477 |
| 90 | 23 / 703 / 1612 / 1919 | 2064 / 22071 / 46372 / 48180 | 1095 / 16871 / 21479 / 22503 |
| 100 | 54 / 702 / 1612 / 2350 | 2064 / 22162 / 46536 / 48314 | 1169 / 19225 / 24345 / 24553 |

## Quality

| value | text_chars | miss | blocks_text |
|---|---|---|---|
| 40 | 178612 | 1 | 824 |
| 50 | 178612 | 1 | 824 |
| 60 | 178612 | 1 | 824 |
| 70 | 178612 | 1 | 824 |
| 80 | 178612 | 1 | 824 |
| 90 | 178612 | 1 | 824 |
| 100 | 178612 | 1 | 824 |

## Notes

- Wall time across 7 successful runs: min=114.88s, median=115.38s, max=116.65s.
