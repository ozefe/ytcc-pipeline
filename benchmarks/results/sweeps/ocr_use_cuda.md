# ocr_use_cuda

RapidOCR via ONNXRuntime CUDAExecutionProvider vs CPU. Output is deterministic across providers; only speed and resources change.

- **Knob:** `ocr_use_cuda`
- **PDF:** `084016.pdf`
- **Runs:** 1 (1 ok, 0 failed)

## Speed

| value | status | wall (s) | render (s) | layout (s) | blocks (s) | table (s) | formula (s) | reference (s) | bundle (s) |
|---|---|---|---|---|---|---|---|---|---|
| True | ok | 114.22 | 18.55 | 8.84 | 86.57 | -- | -- | -- | 0.15 |

## Resources (min / median / p95 / max)

| value | CPU% (min/med/p95/max) | RSS MiB (min/med/p95/max) | VRAM MiB (min/med/p95/max) |
|---|---|---|---|
| True | 62 / 713 / 1614 / 1882 | 2042 / 21849 / 48909 / 49867 | 757 / 13907 / 18131 / 18131 |

## Quality

| value | text_chars | miss |
|---|---|---|
| True | 146516 | 4 |
