"""Per-block-type orchestration -- one module per `BlockType` the pipeline emits.

Each module here owns the "what happens to an X-type block after layout" logic for one
block-type group, dispatching down to the right model wrapper or PDF I/O call:

- `text`: TEXT and REFERENCE routes. Dispatches digital-born to `extract_text_in_bbox`
  (pdf_oxide-backed), scanned to `OcrExtractor` (RapidOCR-backed).
- `image`: IMAGE / FORMULA / TABLE routes' crop-and-save half. Single entry point for
  "save this block's pixels to disk and give me a bundle-relative path." Future
  preprocessing (grayscale, denoise, deskew) lands here.
- `formula`: the FORMULA stage proper. Runs the resident `FormulaRecognizer` across
  every formula crop after the block stage, then splices the recognized LaTeX back into
  the page list and cleans up the on-disk crops.
- `table`: the TABLE stage. Runs the resident `TableEngine` (RapidTable SLANet+) across
  every table crop to recover the cell grid, then extracts per-cell text via pdf_oxide
  (digital-born) or RapidOCR (scanned). Opt-in via `cfg.table_enabled`.
- `reference`: the REFERENCE enrichment stage. Sends every reference-labeled block's
  text to an externally-managed GROBID server in one batched call and attaches the
  parsed `Reference` to each block. Opt-in via `cfg.references_enabled`.

The pipeline's worker layer and serial loop both call into these modules for per-route
work; per-route logic isn't scattered across multiple files.
"""
