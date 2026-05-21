# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the three ytcc-pipeline deployment profiles, each
# available as a slim variant (target=runtime) or a pre-baked variant
# (target=baked) that ships the model weights inside the image.
#
# Build:
#   docker build -f docker/Dockerfile --target runtime \
#       --build-arg PROFILE=scanned -t ytcc-pipeline:scanned .
#
# Valid PROFILE values: scanned | digital-born | digital-born-a100.
# Each maps to a config.<profile>.toml at the repo root that is baked in as
# the container's default /app/config.toml.

ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=24.04
ARG PYTHON_VERSION=3.14
ARG PROFILE=scanned


# ---- Base: CUDA + cuDNN runtime + Python 3.14 ------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS base

ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages. `libgl1` + `libglib2.0-0` are pulled in by
# opencv-python-headless's image codec backends; without them every cv2.imread
# call on a JPEG / WebP page render fails inside the container. `curl` is used
# by the HEALTHCHECK and by ad-hoc operator debugging.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        gnupg \
        libgl1 \
        libglib2.0-0 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        "python${PYTHON_VERSION}" \
        "python${PYTHON_VERSION}-venv" \
        "python${PYTHON_VERSION}-dev"

# Stable uid/gid so bind-mounted volumes have predictable ownership across hosts.
# Ubuntu 24.04 base images ship a default `ubuntu` user at UID/GID 1000; remove
# it first so we can claim 1000 for the app user (useradd otherwise exits 4 on
# "UID already in use"). The guarded form keeps the build green on any future
# base-image variant that drops the ubuntu user.
RUN if id -u ubuntu >/dev/null 2>&1; then userdel -r ubuntu; fi \
    && if getent group ubuntu >/dev/null 2>&1; then groupdel ubuntu; fi \
    && groupadd -r -g 1000 app \
    && useradd -r -u 1000 -g app -m -d /home/app -s /bin/bash app


# ---- Builder: resolve + install deps into a self-contained venv ------------
FROM base AS builder

ARG PYTHON_VERSION

WORKDIR /build

# Install metadata first so a pyproject change invalidates the dep cache while
# unrelated source edits don't trigger a full reinstall on subsequent builds.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# `--upgrade-deps` ensures pip + setuptools inside the venv are recent enough
# to resolve the modern wheel set (torch CUDA tags, onnxruntime-gpu) AND to
# understand PEP 735's `--group` flag. pip's wheel cache is mounted from the
# BuildKit cache so repeated builds skip the downloads.
#
# `--group api` (not `.[api]`) because pyproject.toml puts FastAPI / uvicorn /
# python-multipart under `[dependency-groups]` (PEP 735), not under
# `[project.optional-dependencies]`. The optional-extras syntax silently
# installs nothing extra in that case.
RUN --mount=type=cache,target=/root/.cache/pip \
    "python${PYTHON_VERSION}" -m venv --upgrade-deps /opt/venv \
    && /opt/venv/bin/pip install . --group api


# ---- Runtime (slim): copy venv + the chosen profile's config ---------------
FROM base AS runtime

ARG PROFILE

LABEL org.opencontainers.image.title="ytcc-pipeline" \
      org.opencontainers.image.description="PDF processing pipeline for academic theses" \
      org.opencontainers.image.source="https://github.com/ozefe/ytcc-pipeline" \
      org.opencontainers.image.licenses="MIT" \
      io.ytcc.profile="${PROFILE}" \
      io.ytcc.models_baked="false"

ENV PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/opt/hf_cache \
    YTCC_CONFIG=/app/config.toml

COPY --from=builder /opt/venv /opt/venv
COPY "config.${PROFILE}.toml" /app/config.toml

RUN mkdir -p /opt/hf_cache /app/bundles \
    && chown -R app:app /opt/venv /opt/hf_cache /app

WORKDIR /app
USER app

EXPOSE 8000

# Generous start_period because the first /process call on a slim image
# downloads ~5 GiB of model weights. Baked images load from disk in seconds
# and the lifespan handler is ready well within the grace window.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Absolute path -- some container runtimes (notably vast.ai) reset PATH before
# exec, so relying on `/opt/venv/bin` being on PATH for entrypoint resolution
# is fragile. Hardcoding the venv path is the canonical fix.
ENTRYPOINT ["/opt/venv/bin/uvicorn", "ytcc_pipeline.api.app:app", "--host", "0.0.0.0", "--port", "8000"]


# ---- Baked: pre-download every model weight so the image runs offline ------
FROM runtime AS baked

LABEL io.ytcc.models_baked="true"

# Root only for the prebake step; the final layer reverts to the app user.
USER root

COPY docker/prebake_models.py /usr/local/bin/prebake_models.py
RUN HF_HOME=/opt/hf_cache python /usr/local/bin/prebake_models.py \
    && rm /usr/local/bin/prebake_models.py \
    && chown -R app:app /opt/hf_cache /home/app

USER app
