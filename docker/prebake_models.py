"""Pre-download every model weight ytcc-pipeline needs at runtime.

Used by the `baked` Docker build target so the resulting image starts serving requests
without an initial model-download wait. Slim images skip this step and download on first
`/process` call instead.

Idempotent: a second run finds everything cached and exits in milliseconds.
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s prebake :: %(message)s",
)
logger = logging.getLogger("prebake")

# Setdefault, not setenv -- the Dockerfile exports HF_HOME at build time and we respect
# an operator override on local rebuilds.
os.environ.setdefault("HF_HOME", "/opt/hf_cache")
logger.info("HF_HOME=%s", os.environ["HF_HOME"])

# Imports below sit AFTER the env-var setdefault so huggingface_hub + transformers pick
# up the configured cache directory at module load.
from huggingface_hub import snapshot_download  # noqa: E402

# Snapshot every HF repo the pipeline loads at runtime. Switch
# `PP-FormulaNet-L_safetensors` to `PP-FormulaNet_plus-L_safetensors` here if you want
# the plus-L variant baked instead (2560-token decoder vs 1024).
_HF_MODELS: tuple[str, ...] = (
    "PaddlePaddle/PP-DocLayoutV3_safetensors",
    "PaddlePaddle/PP-FormulaNet-L_safetensors",
)


def _prebake_huggingface() -> None:
    """Snapshot every HF model repo into HF_HOME."""
    for repo_id in _HF_MODELS:
        logger.info("downloading %s", repo_id)
        snapshot_download(repo_id=repo_id)


def _prebake_rapidtable() -> None:
    """Trigger SLANet+ ONNX download by constructing a RapidTable on CPU.

    The `rapid_table` package fetches its model into a package-local cache on first
    instantiation. Failure is non-fatal: the baked image is still functional, the table
    model just downloads on the first table-stage call.
    """
    try:
        # rapid_table emits `gpu_id` in the engine config; ONNXRuntime expects
        # `device_id`. The runtime `TableEngine` applies the same fix.
        from rapid_table import ModelType, RapidTable, RapidTableInput  # noqa: PLC0415
        from rapid_table.inference_engine.base import InferSession  # noqa: PLC0415

        ep_cfg = InferSession.engine_cfg["onnxruntime"]["cuda_ep_cfg"]
        if "gpu_id" in ep_cfg:
            ep_cfg["device_id"] = ep_cfg.pop("gpu_id")

        logger.info("instantiating RapidTable on CPU to trigger SLANet+ download")
        RapidTable(
            RapidTableInput(
                model_type=ModelType.SLANETPLUS,
                use_ocr=False,
                engine_cfg={"use_cuda": False},
            ),
        )
    except Exception:
        logger.exception(
            "SLANet+ prebake failed; baked image is still functional but the "
            "table model will be fetched on the first table-stage call",
        )


def main() -> int:
    """Download every weight the pipeline loads at runtime."""
    _prebake_huggingface()
    _prebake_rapidtable()
    logger.info("prebake complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
