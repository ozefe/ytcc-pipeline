# References (GROBID)

The reference stage enriches `REFERENCE` blocks with parsed `Reference` objects via an external GROBID server. The pipeline never spawns the JVM -- you start GROBID separately. This page covers setup and the parsed shape.

## Enable

```toml
[pipeline]
references_enabled = true
grobid_url         = "http://localhost:8070"
grobid_timeout_s   = 60.0
reference_labels   = ["reference", "reference_content"]
```

When disabled (default), `Block.reference` stays `None` on every block and the raw reference string survives on `Block.text`.

## Run GROBID

Docker, one-liner:

```bash
docker run --rm -p 8070:8070 grobid/grobid:0.9.0
```

This uses GROBID's default config (all CRF models preloaded). The pipeline only ever calls `/api/processCitationList` -- everything else is dead weight. If your host can't run Docker or you want to trim startup time + RSS, use the provided helper scripts which generate a citation-only config:

```bash
scripts/grobid_start.sh        # default paths, port 8070
GROBID_PORT=9090 scripts/grobid_start.sh
scripts/grobid_stop.sh
```

The helper sets `modelPreload: false` so unused models (fulltext, header, figure, table, ...) load lazily on first request. Startup drops from ~10s to ~3s and idle RSS shrinks by ~1 GiB.

**Environment overrides for `grobid_start.sh`** (all optional):

| Variable | Default | Notes |
|----------|---------|-------|
| `GROBID_HOME` | `~/grobid` | Source / install directory containing `gradlew` and `grobid-home/`. |
| `JAVA_HOME` | `~/.conda/envs/grobid-env` | OpenJDK 21 install. |
| `GROBID_PORT` | `8070` | Service port. |
| `GROBID_LOG_DIR` | `$GROBID_HOME/logs` | Server log directory. |
| `GROBID_PID_FILE` | `$GROBID_HOME/grobid.pid` | PID file. |
| `GROBID_READY_TIMEOUT` | `90` | Seconds to wait for `/api/isalive`. |

The script is idempotent: starting an already-running instance is a no-op + exit 0.

> [!NOTE]
> `grobid_start.sh` runs the JVM under `setsid` so its PID equals its PGID -- `grobid_stop.sh` signals `-PID` to take down the whole JVM-wrapper-JVM chain at once without needing `pstree`.

## Pipeline behaviour

`run_reference_stage` does one batched POST to `{grobid_url}/api/processCitationList` with every reference block's text. One call per pipeline run regardless of bibliography size. `grobid_timeout_s` is the upper bound on the stage wall.

**Failure modes** are all logged at WARNING and the page list flows through unchanged:

- Server unreachable (DNS, connection refused).
- Server timeout (longer than `grobid_timeout_s`).
- HTTP error status (4xx, 5xx).
- Malformed XML response (`GrobidError` from `parse_response`).

References are an enrichment, not a hard requirement -- the pipeline always completes.

## Lifespan health probe

When the FastAPI service starts with `references_enabled=true`, the lifespan probes `/api/isalive` once and logs the result:

- Reachable -> INFO line, service continues.
- Unreachable -> WARNING line, service still starts. The reference stage will keep logging-and-skipping per request until GROBID comes back.

```python
from ytcc_pipeline.models.grobid import is_grobid_alive
is_grobid_alive("http://localhost:8070", timeout_s=5.0)  # -> True / False
```

`is_grobid_alive` never raises; transport/timeout failures resolve to `False`.

## Parsed `Reference` shape

The output-format guide describes the JSON-level schema. Programmatically:

```python
from ytcc_pipeline.schema import Author, Reference

# All fields optional; GROBID routinely returns partial parses.
Reference(
    title: str | None,
    authors: tuple[Author, ...],
    year: str | None,           # 4-digit, extracted from <imprint><date when=...>
    venue: str | None,
    volume: str | None,
    issue: str | None,
    pages: str | None,          # "from-to" when both present, else single value
    publisher: str | None,
    doi: str | None,
    url: str | None,
    pmid: str | None,
    arxiv: str | None,
)
```

GROBID's `<idno>` types are matched case-insensitively because the project has historically alternated between `"DOI"` / `"doi"` and `"arXiv"` / `"arxiv"` across releases.

## `reference_labels`: which blocks go to GROBID

`PP-DocLayoutV3` emits two bibliography-related labels today: `reference` (single citations) and `reference_content` (multi-citation paragraphs). Both are forwarded by default.

```toml
[pipeline]
reference_labels = ["reference_content"]    # narrow if `reference` blocks are noisy
```

When `reference` blocks come out as multi-reference blobs in your corpus, GROBID returns a degraded single parse on them. Narrow `reference_labels` to skip them; the raw text still survives on `Block.text`.

## When a block has no `reference`

`Block.reference` is `None` in any of these cases (indistinguishable from the schema alone -- check the logs):

- Stage disabled (`references_enabled=false`).
- Block's `label` not in `cfg.reference_labels`.
- Block's `text` was empty after strip.
- GROBID returned the block but `_biblstruct_to_reference` couldn't extract any usable fields (no title, no authors, no year, no identifiers, no venue).
- GROBID raised (transport, timeout, parse) -- the whole stage skipped, every block's `reference` stays `None`.

The raw reference string is always available on `Block.text` regardless.

## Empty `<biblStruct>` handling

If GROBID returns fewer `<biblStruct>` elements than you sent, `parse_response` pads the tail with `None` so the result list lines up with the input. Empty / unparsable elements yield `None` at their original position. This keeps the splice loop in `run_reference_stage` simple -- one-to-one with the input order.
