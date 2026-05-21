# Digital-born vs scanned

The pipeline runs two qualitatively different text-extraction paths. This page describes the auto-detect heuristic, when to override it, and how to deploy as digital-born-only or scanned-only.

## The two paths

| | Digital-born | Scanned |
|---|---|---|
| Text source | `pdf_oxide` text layer | `RapidOCR` over rendered crops |
| Render DPI | 150 (default, via `render_dpi_digital_born`) | 300 (via `render_dpi`) |
| Per-page parallelism | `cfg.digital_born_workers` (default 1) | `cfg.ocr_workers` (default 1) |
| Per-worker cost | one `pdf_oxide.PdfDocument` handle | one RapidOCR engine + CUDA context (~2 GiB) |
| Typical wall on RTX 3090 | ~15s for a 150-page thesis | ~5-10x that |

The two paths share rendering, layout, the formula stage, the table stage, and bundling. They diverge only inside the block stage.

## Auto-detect

When the caller doesn't pass `digital_born=` explicitly, the orchestrator probes the PDF:

```python
from ytcc_pipeline.pdf_io import detect_digital_born

digital_born = detect_digital_born(
    pdf_path,
    sample_pages=cfg.digital_born_sample_pages,    # default 5
    text_ratio=cfg.digital_born_text_ratio,        # default 0.3
    min_text_chars=cfg.digital_born_min_text_chars # default 100
)
```

The heuristic samples up to `sample_pages` indices evenly distributed across the document and counts non-whitespace text-layer characters per page (`pdf_oxide.extract_spans`). The PDF is digital-born if **either** signal fires:

1. **Strict majority** -- at least `text_ratio` of probed pages clear `min_text_chars` non-whitespace characters. Catches standard text-rich academic theses.
2. **Sparse-but-consistent** -- the median probed page has at least `min_text_chars / 2` characters **and** at least one probed page clears the full `min_text_chars` bar. Catches gazette-style PDFs with thin-but-real text layers that the strict signal misses.

The verdict is logged at DEBUG (full character counts) and INFO (the boolean result).

## When to override

Auto-detect works well on academic theses. Override `digital_born=True/False` when:

- You know the document type ahead of time. Saves the 5-page probe (~50 ms).
- The PDF has a text layer but it's garbage (e.g. scanned-then-OCRed with junk text glued on). Pass `digital_born=False` to force OCR.
- The PDF has no text layer but you'd rather get incomplete-but-fast `pdf_oxide` output than wait for OCR. Pass `digital_born=True` -- TEXT/REFERENCE blocks will MISS but layout, image extraction, and formula recognition still work.

```python
process_pdf("paper.pdf", language="en", digital_born=True)   # force digital-born
process_pdf("paper.pdf", language="en", digital_born=False)  # force scanned
process_pdf("paper.pdf", language="en")                      # auto-detect (default)
```

In the FastAPI service the `digital_born` form field maps directly:

```bash
curl -X POST http://localhost:8000/process \
  -F "pdf=@paper.pdf" \
  -F "language=en" \
  -F "digital_born=true"
```

Omit the field for auto-detect.

## Scanned-only deployments

`cfg.scanned_enabled=true` (default) keeps both paths available.

## Digital-born-only deployments (`scanned_enabled=false`)

Setting `cfg.scanned_enabled=false` is a hard guarantee that **no PDF will ever go through OCR** in this process. Concretely:

- Any PDF that resolves to scanned -- whether by explicit `digital_born=False` or by auto-detect -- is rejected **before** any OCR worker spawns.
- `process_pdf` raises `ValueError` with the PDF name in the message.
- The FastAPI service returns HTTP 415 with the same message.
- The ~12 GiB of VRAM that 6 OCR workers would otherwise reserve stays free.
- The layout analyzer never has to be closed and reloaded between requests.

```toml
[pipeline]
scanned_enabled = false
```

Use this when:

- You only ingest digital-born PDFs (e.g. modern arXiv submissions).
- VRAM is tight and you'd rather give the headroom to a bigger formula or table model.
- You want to fail fast on accidental scanned uploads instead of paying the OCR wall.

> [!IMPORTANT]
> The auto-detect path runs **before** the scanned-disabled check in `process_pdf`. The probe is cheap (~50 ms) but you still pay it. Skip it by passing `digital_born=True` explicitly when you know the input is digital-born.

## Why workers are split (`digital_born_workers` vs `ocr_workers`)

The two paths have asymmetric per-worker costs:

- Digital-born workers each open a `pdf_oxide.PdfDocument` -- cheap, ~10 MiB RSS each. 16 workers is fine on a 72-core host.
- Scanned workers each spawn a RapidOCR engine + CUDA context -- ~2 GiB VRAM each. 6 is the max on a 24 GiB 3090 alongside the resident formula recognizer.

Splitting the knobs lets you tune each path independently. Library defaults are conservative (both 1, serial). Production sets `digital_born_workers=16` and `ocr_workers=6`.

> [!NOTE]
> The orchestrator only uses **one** of the two pools per call -- whichever matches `digital_born`. The other knob is ignored.

## Mixed corpora

If your input mix is e.g. 80% digital-born / 20% scanned, the service handles this seamlessly with `scanned_enabled=true`. Per-request behaviour:

1. **Digital-born request**: layout analyzer stays resident, no OCR workers spawn.
2. **Scanned request**: layout analyzer is closed before processing, OCR workers spawn, analyzer reloaded before lock release. Total VRAM peaks at the OCR workers' footprint.

The lock serialisation guarantees only one of these is active at a time.
