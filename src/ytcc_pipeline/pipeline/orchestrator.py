"""`process_pdf` entry point + the stage-by-stage runner.

This is the only module in the pipeline subpackage that's part of the public API
surface -- `process_pdf` is re-exported at the top of the package.
"""

import logging
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from ytcc_pipeline import __version__
from ytcc_pipeline.bundle import create_bundle
from ytcc_pipeline.config import PipelineConfig
from ytcc_pipeline.models.formula import FormulaRecognizer, make_formula_recognizer
from ytcc_pipeline.models.layout import LayoutAnalyzer, make_analyzer_from_config
from ytcc_pipeline.models.ocr import OcrExtractor, make_ocr_extractor
from ytcc_pipeline.pdf_io.digital_born import detect_digital_born
from ytcc_pipeline.pdf_io.metadata import extract_metadata
from ytcc_pipeline.pdf_io.rendering import render_pages
from ytcc_pipeline.processors.formula import run_formula_stage
from ytcc_pipeline.processors.reference import run_reference_stage
from ytcc_pipeline.processors.table import (
    TableEngine,
    make_table_engine,
    run_table_stage,
)
from ytcc_pipeline.schema import Document

from .blocks import summarize_pages
from .page_processing import process_pages

logger = logging.getLogger(__name__)


def process_pdf(  # noqa: PLR0913  -- public entry point; each kwarg is independently optional and documented in the docstring
    pdf_path: Path | str,
    language: str,
    *,
    digital_born: bool | None = None,
    output_path: Path | str | None = None,
    config: PipelineConfig | None = None,
    analyzer: LayoutAnalyzer | None = None,
    formula_recognizer: FormulaRecognizer | None = None,
    table_engine: TableEngine | None = None,
) -> Path:
    """Run the full pipeline on a single PDF and return the bundle path.

    The pipeline renders pages, runs PP-DocLayoutV3 layout analysis, extracts text
    (pdf_oxide for digital-born PDFs, RapidOCR for scanned), crops image and formula
    blocks, runs PP-FormulaNet-L on the formula crops, optionally recovers cell grids
    for tables (RapidTable SLANet+) and parses bibliographic references via an external
    GROBID server, then packs `document.json` and every saved crop into a single tar
    bundle.

    Args:
        pdf_path: Source PDF file.
        language: ISO 639-1 code (e.g. `"en"`, `"tr"`). Selects the RapidOCR model for
            scanned PDFs; ignored for digital-born.
        digital_born: Override the auto-detected document type. `None` samples the text
            layer and decides automatically.
        output_path: Bundle destination. Defaults to `<pdf-stem>.tar` beside the input.
        config: Pipeline knobs. Defaults to `PipelineConfig()` -- see that module for
            the available performance flags.
        analyzer: Optional pre-loaded layout analyzer to reuse across calls (e.g. the
            FastAPI service holds one across requests). When provided the pipeline skips
            analyzer creation and close -- the caller owns its lifecycle. When `None`
            (library default) a fresh analyzer is created and closed per call.
        formula_recognizer: Optional pre-loaded PP-FormulaNet-L wrapper. Behaves like
            `analyzer` -- injected by the service, owned per call in library mode. When
            `None` AND `config.formula_enabled` is true a fresh recognizer is loaded and
            closed inside this call. When `config.formula_enabled` is false the formula
            stage is a no-op regardless.
        table_engine: Optional pre-loaded RapidTable SLANet+ engine. Same ownership
            semantics as `analyzer` / `formula_recognizer`. When `None` AND
            `config.table_enabled` is true a fresh engine is built and dropped inside
            this call. When `config.table_enabled` is false the table stage is a no-op
            regardless.

    Returns:
        Absolute path to the written tar bundle.

    Raises:
        FileNotFoundError: `pdf_path` does not exist.
        ValueError: `language` is not a supported OCR language, or the PDF resolves to
            scanned while `config.scanned_enabled` is `False`.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.is_file():
        msg = f"PDF not found: {pdf_path}"
        raise FileNotFoundError(msg)

    cfg = config or PipelineConfig()
    output_path = (
        Path(output_path).resolve() if output_path else pdf_path.with_suffix(".tar")
    )

    started = time.perf_counter()
    logger.info(
        "pipeline start: pdf=%s language=%s bytes=%d",
        pdf_path.name,
        language,
        pdf_path.stat().st_size,
    )

    if digital_born is None:
        digital_born = detect_digital_born(
            pdf_path,
            sample_pages=cfg.digital_born_sample_pages,
            text_ratio=cfg.digital_born_text_ratio,
            min_text_chars=cfg.digital_born_min_text_chars,
        )
        logger.info(
            "pipeline detect: pdf=%s digital_born=%s (auto)",
            pdf_path.name,
            digital_born,
        )
    else:
        logger.info(
            "pipeline detect: pdf=%s digital_born=%s (caller)",
            pdf_path.name,
            digital_born,
        )

    if not digital_born and not cfg.scanned_enabled:
        # Reject before any OCR worker spawns and before any temp dir is created.
        # ValueError is the right contract here: the API layer already maps it to HTTP
        # 400 with the message attached, and library callers get a clear failure with
        # the same reason.
        logger.warning(
            "pipeline rejected: pdf=%s reason=scanned_disabled",
            pdf_path.name,
        )
        msg = (
            f"scanned PDFs are disabled in this configuration "
            "(cfg.scanned_enabled=False); "
            f"set scanned_enabled=True to process {pdf_path.name}, "
            "or pass digital_born=True if the PDF is known to have a text layer"
        )
        raise ValueError(msg)

    # Use mkdtemp + manual rmtree rather than TemporaryDirectory: the `retain_temp_dir`
    # debug knob would otherwise have to reach into the TemporaryDirectory's private
    # `_finalizer` to suppress cleanup.
    work_dir = Path(tempfile.mkdtemp(prefix="ytcc_pipeline_"))
    try:
        result = _run_stages(
            pdf_path=pdf_path,
            language=language,
            digital_born=digital_born,
            output_path=output_path,
            cfg=cfg,
            work_dir=work_dir,
            analyzer=analyzer,
            formula_recognizer=formula_recognizer,
            table_engine=table_engine,
        )
    finally:
        if cfg.retain_temp_dir:
            logger.info("retaining temp dir %s for debugging", work_dir)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)

    logger.info(
        "pipeline done: pdf=%s elapsed_s=%.2f bundle=%s bundle_bytes=%d",
        pdf_path.name,
        time.perf_counter() - started,
        result.name,
        result.stat().st_size,
    )
    return result


def _run_stages(  # noqa: PLR0913, PLR0915 -- the stage runner threads the same args through every stage; splitting buys a worse interface
    *,
    pdf_path: Path,
    language: str,
    digital_born: bool,
    output_path: Path,
    cfg: PipelineConfig,
    work_dir: Path,
    analyzer: LayoutAnalyzer | None = None,
    formula_recognizer: FormulaRecognizer | None = None,
    table_engine: TableEngine | None = None,
) -> Path:
    """Run every pipeline stage in sequence against a single PDF.

    Stage order:
        render -> metadata -> layout -> blocks -> table -> formula -> reference
        -> bundle.

    The table, formula, and reference stages are all individually opt-in via
    `cfg.table_enabled` / `cfg.formula_enabled` / `cfg.references_enabled`; disabled
    stages short-circuit to a no-op.

    Ownership semantics for the three injectable resources:

    - `analyzer`: when `None`, a fresh analyzer is built for the call and closed before
      block processing (freeing VRAM for OCR workers on the scanned path). When
      injected, the caller owns the lifecycle and the stage runner leaves it untouched.
    - `formula_recognizer`: same shape -- `None` triggers a per-call load + close when
      `cfg.formula_enabled` is true; an injected instance is reused.
    - `table_engine`: same shape -- `None` triggers a per-call build + drop when
      `cfg.table_enabled` is true; an injected instance is reused.

    The reference stage (GROBID) has no injectable resource: GROBID is an
    externally-managed HTTP service, so a fresh `GrobidClient` is built per call from
    `cfg.grobid_url` / `cfg.grobid_timeout_s`.

    Args:
        pdf_path: Absolute path to the source PDF.
        language: ISO 639-1 code passed to the OCR workers (scanned path only).
        digital_born: Caller-supplied or auto-detected; selects the digital-born or
            scanned block-processing path.
        output_path: Destination path for the tar bundle; overwritten if it exists.
        cfg: Pipeline knobs.
        work_dir: Temp dir for intermediate files (rendered pages, saved crops).
        analyzer: Optional pre-loaded layout analyzer; see ownership note above.
        formula_recognizer: Optional pre-loaded formula recognizer; see ownership note
            above.
        table_engine: Optional pre-loaded table engine; see ownership note above.

    Returns:
        Absolute path to the written tar bundle.
    """
    pages_dir = work_dir / "pages"
    images_dir = work_dir / "images"
    pages_dir.mkdir()
    images_dir.mkdir()

    # Digital-born PDFs can use a lower render DPI without losing content; the text
    # payload comes from pdf_oxide's text layer, not from the rendered pixels. Update
    # `cfg` once so every downstream consumer (rendering, bbox->PDF-point conversion in
    # `extract_text_in_bbox`) sees the effective DPI without an extra parameter on every
    # call.
    if digital_born and cfg.render_dpi_digital_born != cfg.render_dpi:
        cfg = replace(cfg, render_dpi=cfg.render_dpi_digital_born)

    t0 = time.perf_counter()
    page_paths = render_pages(
        pdf_path,
        pages_dir,
        dpi=cfg.render_dpi,
        image_format=cfg.page_format,
        jpeg_quality=cfg.jpeg_quality,
        max_workers=cfg.render_workers,
    )
    logger.info(
        "stage render: pdf=%s pages=%d dpi=%d format=%s elapsed_s=%.2f",
        pdf_path.name,
        len(page_paths),
        cfg.render_dpi,
        cfg.page_format,
        time.perf_counter() - t0,
    )

    # Cheap; run early so I/O failures surface before model loading.
    metadata = extract_metadata(pdf_path)

    # An injected analyzer is the FastAPI service's mode: load once, reuse across many
    # calls. Library mode (analyzer=None) keeps the original behavior -- fresh analyzer
    # per call, closed before block processing so VRAM frees for OCR workers.
    t0 = time.perf_counter()
    owns_analyzer = analyzer is None
    if owns_analyzer:
        analyzer = make_analyzer_from_config(cfg)

    try:
        layout_results = analyzer.analyze(
            page_paths,
            batch_size=cfg.layout_batch_size,
            confidence=cfg.layout_confidence,
        )
    finally:
        if owns_analyzer:
            analyzer.close()

    n_detections = sum(len(detections) for detections in layout_results.values())
    logger.info(
        "stage layout: pdf=%s detections=%d elapsed_s=%.2f analyzer=%s",
        pdf_path.name,
        n_detections,
        time.perf_counter() - t0,
        "owned" if owns_analyzer else "injected",
    )

    t0 = time.perf_counter()
    pages = process_pages(
        page_paths=page_paths,
        layout_results=layout_results,
        digital_born=digital_born,
        images_dir=images_dir,
        cfg=cfg,
        pdf_path=pdf_path,
        language=language,
    )

    summary = summarize_pages(pages)
    logger.info(
        "stage blocks: pdf=%s text=%d image=%d reference=%d formula=%d table=%d "
        "miss=%d elapsed_s=%.2f",
        pdf_path.name,
        summary["text"],
        summary["image"],
        summary["reference"],
        summary["formula"],
        summary["table"],
        summary["miss"],
        time.perf_counter() - t0,
    )
    if summary["miss"]:
        logger.warning(
            "pipeline miss: pdf=%s miss=%d/%d (%.1f%%)",
            pdf_path.name,
            summary["miss"],
            summary["total"],
            100.0 * summary["miss"] / max(1, summary["total"]),
        )

    # Table stage. Owns its engine when the caller didn't inject one. For scanned PDFs
    # we build a single `OcrExtractor` for cell text; digital-born uses pdf_oxide and
    # doesn't need it.
    owns_table = table_engine is None and cfg.table_enabled
    if owns_table:
        table_engine = make_table_engine(
            device=cfg.table_device,
            batch_size=cfg.table_batch_size,
        )

    scanned_ocr: OcrExtractor | None = None
    if cfg.table_enabled and not digital_born:
        try:
            scanned_ocr = make_ocr_extractor(cfg, language)
        except ValueError as exc:
            logger.warning(
                "table scanned OCR skipped: language=%s reason=%s",
                language,
                exc,
            )

    pages = run_table_stage(
        pages,
        pdf_path=pdf_path,
        digital_born=digital_born,
        work_dir=work_dir,
        table_engine=table_engine,
        scanned_ocr=scanned_ocr,
        cfg=cfg,
    )

    owns_formula = formula_recognizer is None and cfg.formula_enabled
    if owns_formula:
        formula_recognizer = make_formula_recognizer(cfg)

    try:
        pages = run_formula_stage(
            pages,
            work_dir=work_dir,
            cfg=cfg,
            formula_recognizer=formula_recognizer,
            pdf_name=pdf_path.name,
        )
    finally:
        if owns_formula and formula_recognizer is not None:
            formula_recognizer.close()

    # Reference stage. External HTTP call to a GROBID server -- gated on
    # `cfg.references_enabled`. No model lifecycle (the server is externally managed);
    # failures fall back to `Block.reference=None` without raising.
    pages = run_reference_stage(pages, pdf_path=pdf_path, cfg=cfg)

    document = Document(
        metadata=metadata,
        language=language,
        digital_born=digital_born,
        pipeline_version=__version__,
        pages=pages,
    )

    t0 = time.perf_counter()
    bundle_path = create_bundle(document, images_dir, output_path)
    logger.info(
        "stage bundle: pdf=%s path=%s bytes=%d elapsed_s=%.2f",
        pdf_path.name,
        bundle_path.name,
        bundle_path.stat().st_size,
        time.perf_counter() - t0,
    )

    return bundle_path
