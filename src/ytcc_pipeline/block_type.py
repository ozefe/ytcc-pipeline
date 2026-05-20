"""`BlockType` enum -- kept in its own module to stay zero-dependency.

The schema layer and the config layer both need this enum. Putting it here (rather than
under `schema`) lets `config` import it without pulling in the rest of `schema` and
therefore without pulling in `pdf_io`, which would otherwise create a config <-> pdf_io
import cycle through `pdf_io.rendering`'s use of `config.ImageFormat`.
"""

from enum import StrEnum

__all__ = ["BlockType"]


class BlockType(StrEnum):
    """The five kinds of block in the output document.

    Each block ships exactly one primary representation per the formula / table / text
    contracts -- `text` (LaTeX, plain text) OR `image_path` (saved crop) -- with `miss`
    flagging the rare case where the primary extraction failed.

    - `FORMULA` blocks emit LaTeX in `text` on success (the on-disk crop is deleted and
      `image_path` is cleared). On a MISS the LaTeX is `None` and a `-MISS-` marked
      crop is bundled instead (subject to `cfg.bundle_miss_images_for`). The schema
      permits both fields to be set simultaneously, but the pipeline never produces
      that combination.
    - `TABLE` blocks carry the table's crop in `image_path`, and (when structure
      recognition succeeds) a structured cell grid in `cells` plus `n_rows` / `n_cols`.
      A degenerate structure result leaves `cells=None` and the crop is the sole
      representation -- `miss` stays `False`.
    """

    TEXT = "text"
    IMAGE = "image"
    REFERENCE = "reference"
    FORMULA = "formula"
    TABLE = "table"
