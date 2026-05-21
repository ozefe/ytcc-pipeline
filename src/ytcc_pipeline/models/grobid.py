"""GROBID HTTP client + TEI -> Reference parser.

GROBID is an external service the pipeline talks to over HTTP. Two pieces live in this
module:

- `GrobidClient`: a tiny stdlib wrapper around `POST /api/processCitationList`. One
  batched call per pipeline run; no retries, no connection pooling, no auth.
- `parse_response`: pure TEI -> `Reference` conversion. Handles both shapes GROBID
  returns: the bare `<biblStruct>` from `/api/processCitation` and the namespaced
  `<TEI>` envelope from `/api/processCitationList`.

The module never spawns the JVM, never holds GROBID resources, and never fails the
pipeline. Errors surface as `GrobidError` for the stage to log + skip.
"""

import logging
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from http import HTTPStatus
from typing import TYPE_CHECKING

from ytcc_pipeline.schema import Author, Reference

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["GrobidClient", "GrobidError", "is_grobid_alive", "parse_response"]

logger = logging.getLogger(__name__)

# A four-character all-digit token from TEI `<imprint><date when=...>` is always the
# 4-digit year (per the TEI/W3C date schema).
_YEAR_TOKEN_LEN = 4

# GROBID's TEI envelope uses the standard TEI namespace.
_TEI_NS = "http://www.tei-c.org/ns/1.0"

# `idno[@type=...]` strings GROBID emits today, mapped to `Reference` attribute names.
# The comparison is case-insensitive on our side because GROBID has historically
# alternated between "DOI" / "doi" and "arXiv" / "arxiv" across releases.
_IDNO_TYPE_TO_FIELD: dict[str, str] = {
    "DOI": "doi",
    "URL": "url",
    "PMID": "pmid",
    "ARXIV": "arxiv",
}


class GrobidError(RuntimeError):
    """Raised on GROBID transport, status, timeout, or parse failure.

    The reference stage catches this once, the pipeline always completes; references are
    an enrichment, not a hard requirement.
    """


# --- parsing ----------------------------------------------------------------


def parse_response(
    xml_text: str,
    *,
    expected: int | None = None,
) -> list[Reference | None]:
    """Parse a GROBID citation response into aligned `Reference` records.

    Both response shapes are accepted:

      - `/api/processCitation` returns a bare `<biblStruct>` (one ref).
      - `/api/processCitationList` returns `<TEI>…<listBibl><biblStruct/>...` with one
        `<biblStruct>` per input citation.

    Empty / unparsable `<biblStruct>` elements are yielded as `None` so the result list
    aligns with the input order. If GROBID returned fewer `<biblStruct>` elements than
    `expected`, the tail is padded with `None`.

    Args:
        xml_text: The raw XML body returned by GROBID.
        expected: Pad / truncate the result to this length so it lines up with the input
            citation list. `None` (the default) returns whatever GROBID produced.

    Returns:
        A list of `Reference` or `None` in input order.

    Raises:
        GrobidError: `xml_text` is not valid XML.
    """
    text = xml_text.strip()
    if not text:
        return _pad_or_truncate([], expected)

    try:
        # GROBID is an internal trusted service the operator runs; the response is XML
        # we know the schema of. `defusedxml` would be appropriate for untrusted XML
        # uploads from the public web, not for parsing our own backend.
        root = ET.fromstring(text)  # noqa: S314
    except ET.ParseError as exc:
        msg = f"malformed XML from GROBID: {exc}"
        raise GrobidError(msg) from exc

    if _local_name(root.tag) == "biblStruct":
        bibl_structs: Iterable[ET.Element] = [root]
    else:
        # The TEI envelope nests biblStructs under <listBibl>. Iterating the full tree
        # is cheap and handles both the standard structure and any envelope variations
        # across GROBID versions.
        ns_match = list(root.iter(f"{{{_TEI_NS}}}biblStruct"))
        bibl_structs = ns_match or list(root.iter("biblStruct"))

    parsed = [_biblstruct_to_reference(b) for b in bibl_structs]
    return _pad_or_truncate(parsed, expected)


def _pad_or_truncate(
    values: list[Reference | None],
    expected: int | None,
) -> list[Reference | None]:
    """Pad / truncate `values` to `expected` length (no-op when None)."""
    if expected is None:
        return values

    if len(values) < expected:
        logger.debug(
            "grobid response short: got=%d expected=%d (padding tail with None)",
            len(values),
            expected,
        )
        return values + [None] * (expected - len(values))

    if len(values) > expected:
        logger.debug(
            "grobid response long: got=%d expected=%d (truncating tail)",
            len(values),
            expected,
        )
        return values[:expected]

    return values


def _biblstruct_to_reference(bibl: ET.Element) -> Reference | None:
    """Convert one `<biblStruct>` element to a `Reference`, or `None` if empty."""
    analytic = _child_named(bibl, "analytic")
    monogr = _child_named(bibl, "monogr")

    title = _analytic_title(analytic) or _monogr_title(monogr, levels=("m", "s"))
    venue = _monogr_title(monogr, levels=("j",)) or _monogr_title(
        monogr, levels=("m", "s")
    )
    authors = _authors_from(analytic, monogr)

    fields: dict[str, str | None] = {
        "title": title,
        "venue": venue,
        "year": None,
        "volume": None,
        "issue": None,
        "pages": None,
        "publisher": None,
        "doi": None,
        "url": None,
        "pmid": None,
        "arxiv": None,
    }
    _fill_imprint(monogr, fields)
    _fill_identifiers(bibl, analytic, fields)

    # Empty parse: no title, no authors, no identifiers, no year. Yield None so callers
    # can distinguish "GROBID couldn't extract anything" from "GROBID extracted a
    # partial reference".
    _any_identifier = ("year", "doi", "url", "pmid", "arxiv", "venue")
    if not title and not authors and not any(fields.get(k) for k in _any_identifier):
        return None

    return Reference(
        title=fields["title"],
        authors=authors,
        year=fields["year"],
        venue=fields["venue"],
        volume=fields["volume"],
        issue=fields["issue"],
        pages=fields["pages"],
        publisher=fields["publisher"],
        doi=fields["doi"],
        url=fields["url"],
        pmid=fields["pmid"],
        arxiv=fields["arxiv"],
    )


def _analytic_title(analytic: ET.Element | None) -> str | None:
    """Return the analytic-level article title (`level='a'`) if present."""
    if analytic is None:
        return None

    for title in _children_named(analytic, "title"):
        if title.get("level", "") == "a":
            text = _element_text(title)
            if text:
                return text
    return None


def _monogr_title(monogr: ET.Element | None, *, levels: tuple[str, ...]) -> str | None:
    """Return the first non-empty `monogr/title` with a matching `@level`."""
    if monogr is None:
        return None

    for title in _children_named(monogr, "title"):
        if title.get("level", "") in levels:
            text = _element_text(title)
            if text:
                return text
    return None


def _authors_from(
    analytic: ET.Element | None,
    monogr: ET.Element | None,
) -> tuple[Author, ...]:
    """Return `Author` records from analytic then monogr `<author>` elements."""
    out: list[Author] = []
    for src in (analytic, monogr):
        if src is None:
            continue

        for author_el in _children_named(src, "author"):
            author = _author_from_element(author_el)
            if author is not None:
                out.append(author)

    return tuple(out)


def _author_from_element(author_el: ET.Element) -> Author | None:
    """Extract a single `Author` from an `<author><persName/></author>` tree."""
    pers = _child_named(author_el, "persName")
    if pers is None:
        return None

    surname = _element_text(_child_named(pers, "surname")) or None
    forenames = [_element_text(f) for f in _children_named(pers, "forename")]
    forenames = [f for f in forenames if f]
    if not surname and not forenames:
        return None

    forename = " ".join(forenames) if forenames else None
    name = " ".join(filter(None, forenames + ([surname] if surname else []))).strip()

    return Author(name=name, surname=surname, forename=forename)


def _fill_imprint(monogr: ET.Element | None, fields: dict[str, str | None]) -> None:  # noqa: C901, PLR0912
    """Populate volume / issue / pages / year / publisher from `monogr/imprint`."""
    if monogr is None:
        return

    imprint = _child_named(monogr, "imprint")
    if imprint is None:
        return

    for scope in _children_named(imprint, "biblScope"):
        unit = scope.get("unit", "")
        if unit == "volume":
            fields["volume"] = _element_text(scope) or None
        elif unit == "issue":
            fields["issue"] = _element_text(scope) or None
        elif unit == "page":
            page_from = scope.get("from")
            page_to = scope.get("to")
            if page_from and page_to:
                fields["pages"] = f"{page_from}-{page_to}"
            elif page_from:
                fields["pages"] = page_from
            else:
                fields["pages"] = _element_text(scope) or None

    date = _child_named(imprint, "date")
    if date is not None:
        # `when="2020-05"` and similar: we keep only the 4-digit year because downstream
        # consumers index on year, not full date.
        when = date.get("when") or _element_text(date)
        if when:
            for token in when.split("-"):
                if len(token) == _YEAR_TOKEN_LEN and token.isdigit():
                    fields["year"] = token
                    break
            else:
                fields["year"] = when

    publisher = _child_named(imprint, "publisher")
    if publisher is not None:
        fields["publisher"] = _element_text(publisher) or None


def _fill_identifiers(
    bibl: ET.Element,
    analytic: ET.Element | None,
    fields: dict[str, str | None],
) -> None:
    """Populate DOI / URL / PMID / arXiv from `<idno>` and `<ptr>` elements."""
    seen: set[str] = set()
    sources: list[ET.Element] = [bibl]
    if analytic is not None:
        sources.append(analytic)

    for src in sources:
        for idno in _children_named(src, "idno"):
            target_field = _IDNO_TYPE_TO_FIELD.get((idno.get("type") or "").upper())
            if target_field is None or target_field in seen:
                continue

            value = _element_text(idno)
            if value:
                fields[target_field] = value
                seen.add(target_field)

        for ptr in _children_named(src, "ptr"):
            target = ptr.get("target")
            if target and "url" not in seen:
                fields["url"] = target
                seen.add("url")


# --- HTTP client -------------------------------------------------------------


class GrobidClient:
    """Thin POST wrapper around `/api/processCitationList`.

    One instance per pipeline call; stateless beyond `url` and `timeout_s`. Safe to
    share across threads: `urllib.request` is thread-safe for one-shot calls, but the
    pipeline never does.
    """

    def __init__(self, *, url: str, timeout_s: float) -> None:
        """Build a client that POSTs to `{url}/api/processCitationList`.

        Args:
            url: Base URL of a running GROBID server (no trailing slash required, no
                path).
            timeout_s: Per-request HTTP timeout. One call per pipeline run, so this caps
                the total wall the reference stage can spend regardless of bibliography
                size.

        """
        self._endpoint = url.rstrip("/") + "/api/processCitationList"
        self._timeout_s = timeout_s

    def process_citation_list(self, citations: list[str]) -> list[Reference | None]:
        """Parse a list of raw reference strings in one batched call.

        Args:
            citations: Raw reference strings, one per input block.

        Returns:
            A list aligned to `citations`: each element is a `Reference` or `None` (no
            usable fields extracted). An empty input list short-circuits without HTTP.

        Raises:
            GrobidError: server unreachable, HTTP error, timeout, or malformed XML
                response.
        """
        if not citations:
            return []

        body = urllib.parse.urlencode(
            [("citations", c) for c in citations],
        ).encode("utf-8")

        logger.debug(
            "grobid POST: endpoint=%s citations=%d body_bytes=%d timeout_s=%.1f",
            self._endpoint,
            len(citations),
            len(body),
            self._timeout_s,
        )

        # `self._endpoint` is built from operator-controlled `grobid_url` plus a
        # constant path; the URL scheme is always HTTP/HTTPS.
        #
        # S310 warns about untrusted URL schemes (file:, ftp:), which can't appear here.
        req = urllib.request.Request(  # noqa: S310
            self._endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/xml",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:  # noqa: S310
                payload = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            msg = f"GROBID HTTP {exc.code} from {self._endpoint}: {exc.reason}"
            raise GrobidError(msg) from exc
        except urllib.error.URLError as exc:
            msg = f"GROBID unreachable at {self._endpoint}: {exc.reason}"
            raise GrobidError(msg) from exc
        except TimeoutError as exc:
            msg = (
                f"GROBID timeout after {self._timeout_s:.1f}s "
                f"({len(citations)} citations to {self._endpoint})"
            )
            raise GrobidError(msg) from exc

        logger.debug(
            "grobid response: endpoint=%s payload_bytes=%d",
            self._endpoint,
            len(payload),
        )

        return parse_response(payload, expected=len(citations))


# --- health probe -----------------------------------------------------------


def is_grobid_alive(url: str, *, timeout_s: float = 5.0) -> bool:
    """Probe `{url}/api/isalive` and return `True` if the server is alive.

    Designed for one-shot startup sanity checks (the FastAPI lifespan calls this when
    `references_enabled`). Never raises: any transport / timeout / decoding failure
    resolves to `False` so the caller can log and move on.

    Args:
        url: Base URL of the GROBID server.
        timeout_s: Probe timeout in seconds; defaults to a short value because the call
            is on the startup critical path.

    Returns:
        `True` if the server responded `HTTP 200` to `GET /api/isalive`, `False`
        otherwise.
    """
    endpoint = url.rstrip("/") + "/api/isalive"
    try:
        # Same rationale as the POST: operator-controlled URL with a constant path; the
        # scheme is always HTTP/HTTPS.
        req = urllib.request.Request(endpoint, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return resp.status == HTTPStatus.OK
    except urllib.error.URLError, TimeoutError, OSError:
        # Probe is never-raising by design (callers route on the bool). Keep the
        # underlying transport error visible so operators investigating an unreachable
        # GROBID server can see whether DNS, refusal, or timeout was the cause.
        logger.debug("grobid probe failed: endpoint=%s", endpoint, exc_info=True)
        return False


# --- helpers ----------------------------------------------------------------
#
# `_child_named` / `_children_named` deliberately do NOT call `ET.Element.find` /
# `findall`: the stdlib methods are recursive and XPath-aware (`.//foo`, `foo[@bar]`),
# whereas everything in this module wants exactly-one-level child matching with
# namespace-stripping. Keeping our own helpers makes that distinction obvious at every
# call site.


def _local_name(tag: str) -> str:
    """Strip the namespace prefix from a `lxml`-style `{ns}tag` string."""
    return tag.split("}", 1)[1] if "}" in tag else tag


def _child_named(root: ET.Element, name: str) -> ET.Element | None:
    """Return the first direct child whose local-name equals `name`."""
    for child in root:
        if _local_name(child.tag) == name:
            return child

    return None


def _children_named(root: ET.Element, name: str) -> list[ET.Element]:
    """Return every direct child whose local-name equals `name`."""
    return [c for c in root if _local_name(c.tag) == name]


def _element_text(el: ET.Element | None) -> str:
    """Return whitespace-stripped concatenated text of `el` (`""` for None)."""
    if el is None:
        return ""

    return "".join(el.itertext()).strip()
