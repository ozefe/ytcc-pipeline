# Documentation

Reference guide for `ytcc-pipeline` -- a Python library that turns an academic-thesis PDF into a structured JSON document plus a tar bundle of cropped figures, tables, and formulas.

## Topics

- **Quickstart** -- install, first run, library + service modes, recommended config.
- **Architecture** -- stage-by-stage pipeline, module layout, resource lifecycle, the three per-page execution paths.
- **Output format** -- the tar bundle layout, the `document.json` schema, `Block` / `Page` / `Reference` / `Cell` field reference, MISS-image bundling semantics.
- **Configuration** -- `PipelineConfig` knobs, the project TOML, env-var overrides, override precedence.
- **Stages** -- per-stage behaviour, knobs, and skip / no-op semantics for render, metadata, layout, blocks, table, formula, reference, and bundle.
- **Performance** -- recommended config, the auto-DPI lever, fp16, fast_preproc, worker sizing, batch sizes, bucketed formula batching, VRAM budgets, tuning checklist.
- **Digital-born vs scanned** -- the auto-detect heuristic, when to override, scanned-only / digital-only deployments, asymmetric worker costs.
- **API service** -- FastAPI wrapper, lifespan model loading, the `asyncio.Lock` serialisation pattern, `/process` and `/health` contracts.
- **References (GROBID)** -- standing up a GROBID server, `references_enabled`, the parsed `Reference` shape, failure modes.
- **Gotchas** -- common pitfalls, MISS handling, OOM recovery, log conventions, debugging tips.
