"""Open a pipeline bundle and print a structured summary of its contents.

Reads `document.json` directly out of the tar and walks the page / block tree. Useful as
a starting point for downstream consumers that want to ingest bundles back into Python.

Run from the project root, passing a bundle path:

    python examples/inspect_bundle.py path/to/paper.tar
"""

import json
import sys
import tarfile
from collections import Counter
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python examples/inspect_bundle.py <bundle.tar>")

    bundle_path = Path(sys.argv[1]).resolve()
    if not bundle_path.is_file():
        sys.exit(f"bundle not found: {bundle_path}")

    with tarfile.open(bundle_path) as tf:
        names = tf.getnames()
        fp = tf.extractfile("document.json")
        if fp is None:
            sys.exit(f"bundle missing document.json: {bundle_path}")

        doc = json.loads(fp.read())

    image_names = [n for n in names if n.startswith("images/")]
    miss_count = sum(1 for n in image_names if "-MISS-" in n)

    print(f"Bundle: {bundle_path}")
    print(f"  pipeline_version: {doc['pipeline_version']}")
    print(f"  language:         {doc['language']}")
    print(f"  digital_born:     {doc['digital_born']}")
    print(f"  pages:            {len(doc['pages'])}")
    print(f"  images on disk:   {len(image_names)} ({miss_count} MISS)")

    label_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    total_text_chars = 0
    for page in doc["pages"]:
        for block in page["blocks"]:
            label_counts[block["label"]] += 1
            type_counts[block["type"]] += 1
            if block["text"]:
                total_text_chars += len(block["text"])

    print(f"\nBlock types: {dict(type_counts)}")
    print(f"Total extracted text chars: {total_text_chars}")
    print("\nTop labels:")
    for label, count in label_counts.most_common(10):
        print(f"  {count:>4}  {label}")

    first_page = doc["pages"][0]
    print(
        f"\nFirst page ({first_page['page_no']}) -- {len(first_page['blocks'])} blocks:"
    )
    for block in first_page["blocks"][:5]:
        text = (block["text"] or "")[:60].replace("\n", " ")
        print(
            f"  [{block['reading_order']}] {block['label']:>20} "
            f"{block['type']:>9}  conf={block['confidence']:.2f}  {text!r}",
        )


if __name__ == "__main__":
    main()
