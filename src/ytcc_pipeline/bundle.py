"""Build the final tar bundle from a Document and the cropped images."""

import io
import logging
import tarfile
from pathlib import Path

from .schema import Document, to_json

__all__ = ["create_bundle"]

logger = logging.getLogger(__name__)


def create_bundle(
    document: Document,
    images_dir: Path,
    output_path: Path,
) -> Path:
    """Write `document.json` + every file under `images_dir` into a tar.

    The bundle is an uncompressed POSIX tar archive. TAR is streaming-first: members are
    written sequentially without seeking back for a central directory, so the output can
    be a pipe, socket, or HTTP response body. PNG / JPEG crops are already compressed,
    so wrapping them in gzip would burn CPU for no size gain; `document.json` is
    typically small (KB-range) so leaving it uncompressed keeps the bundle
    one-pass-readable without an outer codec.

    Args:
        document: The fully assembled `Document`.
        images_dir: Directory containing all bundled crop files. Files are stored under
            `images/{name}` in the tar in alphabetical order. `document.json` is written
            first so consumers can stream-parse it before buffering crops.
        output_path: Destination path for the tar; overwritten if it exists.

    Returns:
        Path to the written tar.
    """
    output_path = Path(output_path)
    images_dir = Path(images_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir() if p.is_file())
    logger.debug(
        "bundle write: path=%s images=%d",
        output_path.name,
        len(images),
    )

    with tarfile.open(output_path, "w") as tf:
        # document.json first so streaming consumers can parse the index without
        # buffering image bytes.
        json_bytes = to_json(document).encode("utf-8")
        info = tarfile.TarInfo(name="document.json")
        info.size = len(json_bytes)

        tf.addfile(info, io.BytesIO(json_bytes))

        for img_path in images:
            tf.add(img_path, arcname=f"images/{img_path.name}")

    return output_path
