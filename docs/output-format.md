# Output format

Every successful `process_pdf` call writes one **tar bundle** containing:

```text
paper.tar
├── document.json              # the schema document
└── images/                    # cropped block images
    ├── 0001-image-{uuid}.png
    ├── 0001-formula-{uuid}.png
    ├── 0014-table-{uuid}.png
    └── ...
```

The tar is **uncompressed** (POSIX format). PNG / JPEG payloads are already compressed; wrapping the archive in gzip would burn CPU for no size gain, and uncompressed streaming lets consumers parse `document.json` before the image bytes arrive.

## Why tar and not zip

Tar is streaming-first: members are written sequentially without seeking back for a central directory, so the bundle can be a pipe, socket, or HTTP response body. `FileResponse` in the FastAPI service ships it directly.

## `document.json` schema

The serialized form mirrors the `Document` / `Page` / `Block` / `Cell` / `Reference` dataclasses. Tuples become arrays, frozen dataclasses become objects, `None` becomes JSON `null`. bbox floats are rounded to two decimals.

### Top-level `Document`

```json
{
  "metadata": {
    "filename": "paper.pdf",
    "sha256": "97fc1c...",
    "byte_size": 4823091,
    "pdf_info": {
      "Title": "...",
      "Author": "...",
      "...": "..."
    }
  },
  "language": "en",
  "digital_born": true,
  "pipeline_version": "0.1.0",
  "pages": [/* Page, ... */]
}
```

- **`metadata.pdf_info`** -- XMP fields read from the PDF, cleaned of UTF-16 BOMs and null bytes. Empty `{}` when the PDF has no XMP block.
- **`language`** -- the ISO 639-1 string the caller passed. Echoed unchanged; only meaningful for scanned PDFs (it picks the OCR model).
- **`digital_born`** -- the resolved verdict (caller override or auto-detect result).
- **`pipeline_version`** -- `ytcc_pipeline.__version__` at the time of the run. Useful when the output format gains fields in a future release.

### `Page`

```json
{
  "page_no": 14,                /* 1-based */
  "width_px": 2483,             /* page size at the effective render DPI */
  "height_px": 3508,
  "blocks": [/* Block, ... */]
}
```

The render DPI varies: digital-born PDFs use `cfg.render_dpi_digital_born` (150 by default), scanned PDFs use `cfg.render_dpi` (300). `width_px`/`height_px` are in that effective DPI, and so are every `Block.bbox` and `Cell.bbox` on the page.

### `Block`

```json
{
  "reading_order": 0,
  "label": "doc_title",
  "type": "text",
  "bbox": [120.5, 88.34, 1880.0, 156.7],
  "confidence": 0.94,
  "text": "Analysis of Strontium-Containing Apatites",
  "image_path": null,
  "miss": false,
  "n_rows": null,
  "n_cols": null,
  "cells": null,
  "reference": null
}
```

| Field | Type | Notes |
|-------|------|-------|
| `reading_order` | int | 0-based position within the page in reading order. |
| `label` | string | The raw PP-DocLayoutV3 label, e.g. `"abstract"`, `"display_formula"`. |
| `type` | string | The `BlockType` enum value: one of `text`, `image`, `reference`, `formula`, `table`. |
| `bbox` | `[x1,y1,x2,y2]` | Pixel coords at the effective render DPI, origin top-left, rounded to two decimals. |
| `confidence` | float | The layout model's detection score. |
| `text` | string \| null | Set for TEXT, REFERENCE, FORMULA (LaTeX); `null` for IMAGE / TABLE / MISS. |
| `image_path` | string \| null | Bundle-relative path (`"images/..."`) for IMAGE, TABLE, FORMULA-on-MISS, TEXT/REFERENCE-on-MISS. `null` otherwise. |
| `miss` | bool | `True` when the primary extraction (text or LaTeX) failed and the block is a fallback. |
| `n_rows`, `n_cols`, `cells` | TABLE-only | `null` on non-TABLE and on TABLE-blocks that fell back to image-only. |
| `reference` | REFERENCE-only | Parsed `Reference` object; `null` when GROBID didn't run or didn't parse anything. |

#### Representation invariants

For every block exactly one of (`text`, `image_path`) is set, except MISS fallbacks which can have both `null` and FORMULA blocks where `image_path` is set on MISS only:

| Block kind | `text` | `image_path` | `miss` |
|------------|--------|--------------|--------|
| TEXT, success | extracted text | `null` | `false` |
| TEXT, MISS | `null` | crop path (if in `bundle_miss_images_for`) or `null` | `true` |
| REFERENCE, success | extracted text | `null` | `false` |
| REFERENCE, MISS | `null` | crop path (if in `bundle_miss_images_for`) or `null` | `true` |
| IMAGE | `null` | crop path | `false` |
| FORMULA, success | LaTeX | `null` (crop deleted after recognition) | `false` |
| FORMULA, MISS | `null` | crop path with `-MISS-` marker | `true` |
| TABLE, structured | `null` | crop path | `false` |
| TABLE, image-only fallback | `null` | crop path | `false` |

> [!NOTE]
> `miss=True` always means "the primary representation is unavailable." For TEXT/REFERENCE/FORMULA this is genuine recognition failure. TABLE blocks never carry `miss=True` -- a structure-recognition failure quietly degrades to image-only with `cells=null` (the crop is still the primary representation, just less structured than ideal).

### `Cell` (TABLE blocks only)

```json
{
  "row_start": 0,
  "row_end": 0,
  "col_start": 1,
  "col_end": 1,
  "bbox": [430.1, 1280.3, 690.8, 1320.0],
  "text": "Strontium"
}
```

- **`row_start`/`row_end`/`col_start`/`col_end`** -- 0-based inclusive indices in the recovered grid. Cells spanning multiple rows or columns have `row_end > row_start` / `col_end > col_start`.
- **`bbox`** -- in the **page's** coord space, not the crop's. Consumers can re-render against the page image directly.
- **`text`** -- per-cell text via `pdf_oxide` (digital-born) or `RapidOCR` (scanned). `null` on extraction failure or genuinely empty cells.

### `Reference` (REFERENCE blocks only, when `references_enabled`)

```json
{
  "title": "PP-DocLayoutV3...",
  "authors": [
    {"name": "J Smith", "surname": "Smith", "forename": "J"},
    {"name": "A Doe", "surname": "Doe", "forename": "A"}
  ],
  "year": "2024",
  "venue": "arXiv preprint",
  "volume": "12",
  "issue": "3",
  "pages": "1-23",
  "publisher": "ACM",
  "doi": "10.1145/...",
  "url": "https://...",
  "pmid": null,
  "arxiv": "2401.12345"
}
```

Every field is optional -- GROBID routinely returns partial parses and the schema preserves that. The original raw string survives on `Block.text` regardless. When GROBID fails to extract any usable fields, `Block.reference` stays `null` and `Block.text` is the sole representation.

## Image filenames

```text
images/{page:04d}-{label}-{uuid}.{ext}            # success
images/{page:04d}-{label}-MISS-{uuid}.{ext}       # MISS fallback
```

- `page` is 1-based, zero-padded to 4 digits (sortable alphabetically).
- `label` is the raw PP-DocLayoutV3 label (`formula`, `table`, `chart`, ...). Grep-friendly.
- `uuid` is a 32-character hex from `uuid.uuid4().hex` (no hyphens).
- `ext` is the configured `cfg.crop_format`: `png` (default) or `jpg`.

The marker is **`-MISS-`**, surrounded by hyphens, so `grep "MISS"` is unambiguous.

## Crop format

`cfg.crop_format` is `"png"` by default. Don't switch to JPEG unless you're OK with quality loss on figures and tables you'll never re-render. JPEG saves are quantised; PNG saves are lossless. This is distinct from `cfg.page_format`, which controls the **temp** page renders consumed by the layout model -- those can safely be JPEG because the layout model downsamples to 800x800 anyway.

## MISS image bundling

`cfg.bundle_miss_images_for` is a `frozenset[BlockType]` -- the block types that get a fallback crop bundled when their primary extraction fails. The default includes every type. Configure narrower sets to save bundle size on text-heavy corpora:

```python
from ytcc_pipeline import PipelineConfig
from ytcc_pipeline.schema import BlockType

cfg = PipelineConfig(bundle_miss_images_for=frozenset({BlockType.FORMULA}))
```

```toml
# config.toml
[pipeline]
bundle_miss_images_for = ["formula"]
```

```bash
# env var (comma-separated, case-insensitive)
YTCC_BUNDLE_MISS_IMAGES_FOR="formula"
```

Set to `[]` / empty string to disable MISS images entirely -- the block's JSON entry still appears with `text=null`, `image_path=null`, `miss=true` so consumers retain reading order and bbox.
