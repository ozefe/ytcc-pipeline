"""Tests for `ytcc_pipeline.models.grobid`.

Two surfaces under test:

  - `parse_response`: pure TEI -> `Reference` conversion. Verified with hand-crafted XML
    strings covering both response shapes that GROBID returns (bare `<biblStruct>` vs
    full `<TEI>` document).
  - `GrobidClient`: `urllib`-based POST wrapper. Verified with `unittest.mock.patch`
    against `urllib.request.urlopen`; no real network traffic.

Integration against a live server lives alongside the reference-stage tests under the
`integration` marker.
"""

import email.message
import io
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Self, cast
from unittest.mock import patch

import pytest

from ytcc_pipeline.models.grobid import (
    GrobidClient,
    GrobidError,
    is_grobid_alive,
    parse_response,
)
from ytcc_pipeline.schema import Author, Reference

if TYPE_CHECKING:
    from types import TracebackType


def _require_one(refs: list[Reference | None]) -> Reference:
    """Unwrap a `parse_response` result expected to carry one non-None `Reference`.

    Keeps the per-test assertions narrow and lets pyright narrow the type through the
    helper, instead of repeating `assert ref is not None` in every test body.
    """
    assert len(refs) == 1, f"expected 1 reference, got {len(refs)}"

    ref = refs[0]
    assert ref is not None, "expected a parsed Reference, got None"
    return ref


# --- parse_response ----------------------------------------------------------


def test_parse_response_bare_biblstruct_emits_one_reference() -> None:
    """`/api/processCitation` returns a single bare `<biblStruct>`."""
    xml = """<biblStruct status="extracted">
        <analytic>
            <title level="a" type="main">Title here</title>
            <author>
                <persName>
                    <forename type="first">J</forename>
                    <surname>Smith</surname>
                </persName>
            </author>
            <idno type="DOI">10.1234/abc</idno>
            <ptr target="https://doi.org/10.1234/abc"/>
        </analytic>
        <monogr>
            <title level="j">Journal of Things</title>
            <imprint>
                <biblScope unit="volume">5</biblScope>
                <biblScope unit="issue">2</biblScope>
                <biblScope unit="page" from="12" to="30"/>
                <date type="published" when="2020">2020</date>
            </imprint>
        </monogr>
    </biblStruct>"""

    out = parse_response(xml)

    assert len(out) == 1

    ref = out[0]
    assert isinstance(ref, Reference)
    assert ref.title == "Title here"
    assert ref.authors == (Author(name="J Smith", surname="Smith", forename="J"),)
    assert ref.year == "2020"
    assert ref.venue == "Journal of Things"
    assert ref.volume == "5"
    assert ref.issue == "2"
    assert ref.pages == "12-30"
    assert ref.doi == "10.1234/abc"
    assert ref.url == "https://doi.org/10.1234/abc"


def test_parse_response_namespaced_tei_emits_one_per_biblstruct() -> None:
    """`/api/processCitationList` returns a full `<TEI>` document."""
    xml = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
        <teiHeader/>
        <text><back><div><listBibl>
            <biblStruct xml:id="b0">
                <analytic><title level="a">A</title>
                    <author>
                        <persName>
                            <forename type="first">X</forename>
                            <surname>One</surname>
                        </persName>
                    </author>
                </analytic>
                <monogr><title level="j">J1</title>
                    <imprint><date type="published" when="2020">2020</date></imprint>
                </monogr>
            </biblStruct>
            <biblStruct xml:id="b1">
                <analytic><title level="a">B</title>
                    <author><persName><surname>Two</surname></persName></author>
                </analytic>
                <monogr><title level="j">J2</title>
                    <imprint><date type="published" when="2021">2021</date></imprint>
                </monogr>
            </biblStruct>
        </listBibl></div></back></text>
    </TEI>"""

    out = parse_response(xml)

    assert len(out) == 2

    first, second = out
    assert first is not None
    assert second is not None
    assert first.title == "A"
    assert first.authors == (Author(name="X One", surname="One", forename="X"),)
    assert second.title == "B"
    assert second.authors == (Author(name="Two", surname="Two"),)
    assert second.year == "2021"


def test_parse_response_minimal_biblstruct_keeps_other_fields_none() -> None:
    """Title-only payload still parses; missing fields stay `None` / empty."""
    xml = """
    <biblStruct>
        <analytic>
            <title level="a">Only a title</title>
        </analytic>
    </biblStruct>
    """

    ref = _require_one(parse_response(xml))

    assert ref.title == "Only a title"
    assert ref.authors == ()
    assert ref.year is None
    assert ref.venue is None
    assert ref.doi is None
    assert ref.pages is None


def test_parse_response_empty_string_returns_empty_list() -> None:
    assert parse_response("") == []
    assert parse_response("   ") == []


def test_parse_response_listbibl_with_zero_biblstructs_returns_empty_list() -> None:
    """GROBID's TEI envelope but no parsed citations."""
    xml = """<TEI xmlns="http://www.tei-c.org/ns/1.0">
        <text><back><div><listBibl/></div></back></text>
    </TEI>"""
    assert parse_response(xml) == []


def test_parse_response_pads_to_expected_length_with_none() -> None:
    """`expected=N` aligns the result to input order for batched calls."""
    xml = """
    <biblStruct>
        <analytic>
            <title level="a">Only one</title>
        </analytic>
    </biblStruct>
    """

    out = parse_response(xml, expected=3)

    assert len(out) == 3
    assert out[0] is not None
    assert out[0].title == "Only one"
    assert out[1] is None
    assert out[2] is None


def test_parse_response_treats_empty_biblstruct_as_none() -> None:
    """A `<biblStruct/>` with no extractable fields is yielded as `None`."""
    xml = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
        <text>
            <back>
                <div>
                    <listBibl>
                        <biblStruct xml:id="b0"/>
                        <biblStruct xml:id="b1">
                            <analytic>
                                <title level="a">Real</title>
                            </analytic>
                        </biblStruct>
                    </listBibl>
                </div>
            </back>
        </text>
    </TEI>
    """

    out = parse_response(xml)

    assert len(out) == 2
    assert out[0] is None
    assert out[1] is not None
    assert out[1].title == "Real"


def test_parse_response_extracts_idno_variants() -> None:
    """DOI / PMID / arXiv identifiers come through under typed fields."""
    xml = """<biblStruct>
        <analytic>
            <title level="a">Paper</title>
            <idno type="DOI">10.1/abc</idno>
            <idno type="arXiv">2301.00001</idno>
            <idno type="PMID">12345678</idno>
        </analytic>
    </biblStruct>"""

    ref = _require_one(parse_response(xml))

    assert ref.doi == "10.1/abc"
    assert ref.arxiv == "2301.00001"
    assert ref.pmid == "12345678"


def test_parse_response_prefers_journal_title_for_venue() -> None:
    """When both monogr/title[@level='m'] and journal title exist, journal wins."""
    xml = """<biblStruct>
        <analytic><title level="a">Article</title></analytic>
        <monogr>
            <title level="j">Some Journal</title>
            <title level="m">Some Proceedings</title>
        </monogr>
    </biblStruct>"""

    ref = _require_one(parse_response(xml))

    assert ref.venue == "Some Journal"


def test_parse_response_uses_book_title_when_no_analytic_title() -> None:
    """Book-style entries use monogr title as the reference title."""
    xml = """<biblStruct>
        <monogr>
            <title level="m">A Book</title>
            <author><persName><surname>Author</surname></persName></author>
            <imprint><date type="published" when="2018">2018</date></imprint>
        </monogr>
    </biblStruct>"""

    ref = _require_one(parse_response(xml))

    assert ref.title == "A Book"
    assert ref.year == "2018"


def test_parse_response_raises_grobid_error_on_malformed_xml() -> None:
    with pytest.raises(GrobidError, match="malformed"):
        parse_response("<not-xml")


# --- GrobidClient ------------------------------------------------------------


class _FakeResponse:
    """Stand-in for `urlopen()`'s context-manager return value."""

    def __init__(self, body: str, *, status: int = 200) -> None:
        self.status = status
        self._buf = io.BytesIO(body.encode("utf-8"))

    def read(self) -> bytes:
        return self._buf.read()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        return None


def _fake_response(body: str, status: int = 200) -> _FakeResponse:
    return _FakeResponse(body, status=status)


def test_client_short_circuits_on_empty_input() -> None:
    """An empty input list never touches the network."""
    client = GrobidClient(url="http://nowhere.invalid", timeout_s=1.0)
    with patch("urllib.request.urlopen") as mocked:
        out = client.process_citation_list([])
    assert out == []
    mocked.assert_not_called()


def test_client_posts_form_encoded_citations() -> None:
    """The wire format is `application/x-www-form-urlencoded` with repeated keys."""
    xml = '<biblStruct><analytic><title level="a">T</title></analytic></biblStruct>'
    client = GrobidClient(url="http://localhost:8070", timeout_s=10.0)

    with patch("urllib.request.urlopen", return_value=_fake_response(xml)) as mocked:
        client.process_citation_list(["one", "two"])

    args = mocked.call_args.args
    kwargs = mocked.call_args.kwargs

    req = cast("urllib.request.Request", args[0])
    assert req.full_url == "http://localhost:8070/api/processCitationList"
    assert req.get_method() == "POST"
    assert req.headers["Content-type"] == "application/x-www-form-urlencoded"
    assert isinstance(req.data, bytes)

    body = req.data.decode("utf-8")
    assert "citations=one" in body
    assert "citations=two" in body
    assert kwargs["timeout"] == 10.0


def test_client_returns_aligned_results_with_expected_padding() -> None:
    """Server returning fewer parses than inputs leaves trailing `None`s."""
    xml = """
    <biblStruct>
        <analytic>
            <title level="a">Only one</title>
        </analytic>
    </biblStruct>
    """
    client = GrobidClient(url="http://localhost:8070", timeout_s=10.0)

    with patch("urllib.request.urlopen", return_value=_fake_response(xml)):
        out = client.process_citation_list(["a", "b", "c"])

    assert len(out) == 3
    assert out[0] is not None
    assert out[0].title == "Only one"
    assert out[1] is None
    assert out[2] is None


def test_client_raises_grobid_error_on_unreachable_server() -> None:
    client = GrobidClient(url="http://localhost:9999", timeout_s=1.0)
    boom = urllib.error.URLError("Connection refused")

    with (
        patch("urllib.request.urlopen", side_effect=boom),
        pytest.raises(GrobidError, match="unreachable"),
    ):
        client.process_citation_list(["x"])


def test_client_raises_grobid_error_on_http_error() -> None:
    """HTTPError is a URLError subclass -- same code path, message mentions status."""
    err = urllib.error.HTTPError(
        url="http://localhost:8070/api/processCitationList",
        code=500,
        msg="Internal Server Error",
        hdrs=email.message.Message(),
        fp=None,
    )
    client = GrobidClient(url="http://localhost:8070", timeout_s=10.0)

    with (
        patch("urllib.request.urlopen", side_effect=err),
        pytest.raises(GrobidError, match="500"),
    ):
        client.process_citation_list(["x"])


def test_client_raises_grobid_error_on_timeout() -> None:
    client = GrobidClient(url="http://localhost:8070", timeout_s=0.1)
    with (
        patch("urllib.request.urlopen", side_effect=TimeoutError("read timed out")),
        pytest.raises(GrobidError, match="timeout"),
    ):
        client.process_citation_list(["x"])


def test_client_raises_grobid_error_on_malformed_payload() -> None:
    client = GrobidClient(url="http://localhost:8070", timeout_s=10.0)
    with (
        patch("urllib.request.urlopen", return_value=_fake_response("<not-xml")),
        pytest.raises(GrobidError, match="malformed"),
    ):
        client.process_citation_list(["x"])


# --- is_grobid_alive ---------------------------------------------------------


def test_is_grobid_alive_true_on_200() -> None:
    with patch(
        "urllib.request.urlopen", return_value=_fake_response("true", status=200)
    ):
        assert is_grobid_alive("http://localhost:8070") is True


def test_is_grobid_alive_false_on_non_200() -> None:
    with patch(
        "urllib.request.urlopen", return_value=_fake_response("false", status=503)
    ):
        assert is_grobid_alive("http://localhost:8070") is False


def test_is_grobid_alive_false_on_url_error() -> None:
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        assert is_grobid_alive("http://localhost:9999") is False


def test_is_grobid_alive_false_on_timeout() -> None:
    with patch("urllib.request.urlopen", side_effect=TimeoutError("slow")):
        assert is_grobid_alive("http://localhost:8070", timeout_s=0.01) is False


def test_is_grobid_alive_targets_isalive_endpoint() -> None:
    """The probe hits `/api/isalive` regardless of trailing slashes on the URL."""
    captured: dict[str, str] = {}

    def _capture(
        req: urllib.request.Request, timeout: float | None = None
    ) -> _FakeResponse:
        del timeout  # unused; recorded to keep the signature compatible
        captured["url"] = req.full_url
        return _fake_response("true", status=200)

    with patch("urllib.request.urlopen", side_effect=_capture):
        is_grobid_alive("http://localhost:8070/")

    assert captured["url"] == "http://localhost:8070/api/isalive"
