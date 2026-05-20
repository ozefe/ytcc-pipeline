# Docker images

Pre-built images for three deployment profiles, each available as a slim variant (models fetched on first request) and a baked variant (models pre-downloaded in the image). Published to GitHub Container Registry.

## Image matrix

| Tag | Profile | Models baked | Approx size |
|---|---|---|---|
| `ghcr.io/ozefe/ytcc-pipeline:scanned` | Scanned-optimized | No | ~2 GB |
| `ghcr.io/ozefe/ytcc-pipeline:scanned-baked` | Scanned-optimized | Yes | ~8 GB |
| `ghcr.io/ozefe/ytcc-pipeline:digital-born` | Digital-born-only | No | ~2 GB |
| `ghcr.io/ozefe/ytcc-pipeline:digital-born-baked` | Digital-born-only | Yes | ~8 GB |
| `ghcr.io/ozefe/ytcc-pipeline:digital-born-a100` | A100-tuned | No | ~2 GB |
| `ghcr.io/ozefe/ytcc-pipeline:digital-born-a100-baked` | A100-tuned | Yes | ~8 GB |

Tagged releases (`v0.1.0`, `v0.2.0`, ...) additionally publish the pinned form `<tag>-v<version>` (e.g. `:scanned-baked-v0.1.0`). Pin to a versioned tag in production.

## Choosing a profile

- **`scanned`** -- mixed corpus (digital-born + scanned). Loads RapidOCR engines; runs OCR on the scanned path. Matches `config.scanned.toml`.
- **`digital-born`** -- digital-born only (`scanned_enabled=false`). Rejects scanned PDFs at the API layer with HTTP 415; saves the ~12 GiB VRAM the OCR pool would otherwise need. Matches `config.digital-born.toml`.
- **`digital-born-a100`** -- digital-born only, tuned for A100-80GB. Larger batches (`layout_batch_size=24`, `formula_batch_size=16`), `formula_torch_compile=true`. Matches `config.digital-born-a100.toml`.

## Choosing a variant

- **Baked** -- ~6 GB larger image, but starts serving in seconds and works offline. Pick this for production deployments where instant cold-start matters.
- **Slim** -- ~2 GB image, but the first `/process` call pays a ~30s model download. Pick this for dev / CI / air-gapped setups where you'll bind-mount your own `HF_HOME` cache.

## Quickstart

Each profile ships a compose file that wires up the bundled GROBID sidecar required by the default reference stage:

```bash
docker compose -f docker/compose.scanned.yml up -d
curl -X POST http://localhost:8000/process \
    -F "pdf=@paper.pdf" \
    -F "language=en" \
    -o paper.tar
```

The compose `depends_on` blocks startup until GROBID's `/api/isalive` returns true (~30s on a cold host). The pipeline's `/health` endpoint reports readiness; the compose healthcheck grants up to 120s for the lifespan handler to load every resident model.

## Configuration

Three layers, most-permissive to most-specific:

1. **Bake-in defaults** -- each image ships with the matching `config.toml` already at `/app/config.toml`.
2. **`YTCC_*` env vars** -- override individual fields. Full list in [`docs/configuration.md`](../docs/configuration.md).
3. **Mount your own TOML** -- bind `your.toml` over `/app/config.toml` for full control.

```bash
# Override the formula model via env var
docker run --gpus all -p 8000:8000 \
    -e YTCC_FORMULA_MODEL_ID=PaddlePaddle/PP-FormulaNet_plus-L_safetensors \
    ghcr.io/ozefe/ytcc-pipeline:digital-born-baked

# Mount a custom config
docker run --gpus all -p 8000:8000 \
    -v "$(pwd)/my-config.toml:/app/config.toml:ro" \
    ghcr.io/ozefe/ytcc-pipeline:scanned-baked
```

## Volumes

| Path | Purpose |
|---|---|
| `/app/config.toml` | Mount your own TOML to override the baked one. |
| `/opt/hf_cache` | HuggingFace model cache. Persist with a named volume on slim images to avoid re-downloading models on every container restart. |

The compose files declare a `ytcc-hf-cache` named volume mounted at `/opt/hf_cache` for both reasons.

## GROBID

The reference stage (enabled by default in every shipped config) calls an external GROBID server. The bundled compose files start `grobid/grobid:0.9.0` as a sidecar and point the pipeline at `http://grobid:8070`. To use a pre-existing GROBID instead, drop the `grobid` service from the compose file and set `YTCC_GROBID_URL=http://your-grobid:8070`. To skip reference parsing entirely, set `YTCC_REFERENCES_ENABLED=false` and drop the sidecar.

## Ports

The service binds `0.0.0.0:8000` inside the container; the compose files expose it on the host's `8000`. Change the host side to run multiple profiles side-by-side:

```yaml
services:
  ytcc-pipeline:
    ports:
      - "8001:8000"
```

## Building locally

```bash
# Slim variant for the scanned profile
docker build -f docker/Dockerfile --target runtime \
    --build-arg PROFILE=scanned \
    -t ytcc-pipeline:scanned .

# Baked variant (pre-downloads ~5 GB of models -- needs internet at build time)
docker build -f docker/Dockerfile --target baked \
    --build-arg PROFILE=scanned \
    -t ytcc-pipeline:scanned-baked .
```

Valid `PROFILE` values: `scanned`, `digital-born`, `digital-born-a100`. Each maps to a `config.<profile>.toml` at the repo root.

The build uses BuildKit cache mounts for apt + uv. Repeat builds against the same builder cache complete in seconds for the slim variant; baked rebuilds are dominated by the HF snapshot download (~5 GB once, cached on rebuild via the uv layer cache).

## Troubleshooting

- **`nvidia-container-cli: requirement error`** -- install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host and restart the docker daemon.
- **A100 image OOMs on smaller cards** -- use `digital-born` instead. The A100 profile sets `layout_batch_size=24` and `formula_batch_size=16` which assume ~80 GiB headroom.
- **`/health` reports `model_loaded: false` past the start-period** -- check the container logs; the lifespan handler prints one INFO line per stage of model load. Slim images on a slow connection can exceed 120s.
- **Bundle missing parsed references even though GROBID is up** -- GROBID's citation CRF model loads lazily on the first request, adding ~3-5s to the first `/process` call. Subsequent calls finish in the usual 50-200 ms.
- **Want to run two profiles on the same host** -- edit the host-side port in the second compose file (`"8001:8000"`) and the `container_name` fields, then `docker compose -p ytcc-second -f ... up`.

## Tag convention

Every push to `main` updates the `:<profile>[-baked]` floating tag.
Every git tag `v<X.Y.Z>` additionally publishes `:<profile>[-baked]-v<X.Y.Z>`. Pull requests build the images for verification but never push.
