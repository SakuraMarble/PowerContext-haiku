# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ruff: noqa: TRY003

"""Whole-document PDF extraction and exact page citations through scoped Memory."""

from __future__ import annotations

import hashlib
import io
import json
import re
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel
from pypdf import PageObject, PdfReader
from pypdf._cmap import get_encoding
from pypdf.generic import (
    ByteStringObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    StreamObject,
    TextStringObject,
)

from powercontext._lexical import rank_texts, terms
from powercontext.builtin.artifacts.memory.models import MemoryCitation, MemoryHit
from powercontext.builtin.runtime.models import GetMemoryEntryRequest, SearchMemoryRequest
from powercontext.sources.definitions import SourceProjection
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import ScopedMemoryApplication


class PDFDocumentInput(BaseModel):
    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    page: PDFPageInput


_LIGATURES = str.maketrans(dict(zip("ﬀﬁﬂﬃﬄﬅﬆ", ("ff", "fi", "fl", "ffi", "ffl", "st", "st"), strict=True)))


def _page_text(page: PageObject) -> str:
    text = _text_with_actual_glyphs(page).translate(_LIGATURES)
    if re.search(r"\b\d{1,3}\s+\d{1,3}\b", text) is None:
        return text
    # Plain extraction can split tightly positioned numeric runs (201 5).
    # Accept layout's whitespace only when every non-whitespace character stays
    # in the same order, numeric fragmentation falls, and words do not split.
    try:
        layout = page.extract_text(extraction_mode="layout", layout_mode_strip_rotated=False).translate(_LIGATURES)
    except NotImplementedError:
        return text
    if (
        re.sub(r"\s+", "", layout) == re.sub(r"\s+", "", text)
        and len(re.findall(r"\d+", layout)) < len(re.findall(r"\d+", text))
        and len(re.findall(r"[a-z]+", layout.lower())) <= len(re.findall(r"[a-z]+", text.lower()))
    ):
        return layout
    return text


def _text_with_actual_glyphs(page: PageObject) -> str:  # noqa: C901 - PDF operator state machine
    """Honor single-glyph ActualText spans without changing glyph positioning.

    Some PDF generators map ligatures such as 'ti' to NUL in ToUnicode and
    provide their spelling only in /ActualText (PDF 1.7, section 14.9.4).
    Repair unambiguous missing mappings in a page-local font dictionary.
    Original glyph bytes, positioning and valid Unicode mappings stay intact;
    conflicting contextual replacements cannot overwrite one another.
    """
    content = page.get_contents()
    if content is None or b"/ActualText" not in content.get_data():
        return page.extract_text() or ""
    resources = cast("DictionaryObject", page["/Resources"].get_object())
    fonts = cast("DictionaryObject", resources.get("/Font", DictionaryObject()).get_object())
    font_name, font_size = NameObject(""), 0.0
    graphics = []
    marked = []
    replacements = {}
    for index, (operands, operator) in enumerate(content.operations):
        if operator == b"q":
            graphics.append((font_name, font_size))
        elif operator == b"Q" and graphics:
            font_name, font_size = graphics.pop()
        elif operator == b"Tf":
            font_name, font_size = operands[0], float(operands[1])
        elif operator in {b"BDC", b"BMC"}:
            properties = operands[1] if operator == b"BDC" else DictionaryObject()
            if isinstance(properties, NameObject):
                named = cast("DictionaryObject", resources.get("/Properties", DictionaryObject()).get_object())
                properties = named.get(properties)
            actual = properties.get("/ActualText") if isinstance(properties, DictionaryObject) else None
            marked.append((str(actual) if isinstance(actual, TextStringObject) else None, []))
        elif operator == b"EMC" and marked:
            actual, shows = marked.pop()
            if actual is not None and len(shows) == 1:
                show_index, name, size, raw = shows[0]
                font = fonts.get(name)
                if font is not None:
                    font = cast("DictionaryObject", font.get_object())
                    _, cmap = get_encoding(font)
                    width = cmap.get(-1, 2 if font.get("/Subtype") == "/Type0" else 1)
                    if len(raw) == width:
                        replacements[show_index] = (name, size, raw, actual)
        elif marked and operator in {b"Tj", b"TJ", b"'", b'"'}:
            values = operands[0] if operator == b"TJ" else [operands[-1]]
            for value in values:
                if isinstance(value, TextStringObject):
                    raw = value.get_original_bytes()
                elif isinstance(value, ByteStringObject):
                    raw = bytes(value)
                else:
                    continue
                if raw:
                    for actual, shows in marked:
                        if actual is not None:
                            shows.append((index, font_name, font_size, raw))
    if not replacements:
        return page.extract_text() or ""
    local_fonts = DictionaryObject(fonts)
    grouped = {}
    for name, _, raw, actual in replacements.values():
        grouped.setdefault(name, {}).setdefault(raw, set()).add(actual)
    for name, mappings in grouped.items():
        font = cast("DictionaryObject", fonts[name].get_object())
        _, cmap = get_encoding(font)
        entries = []
        for raw, meanings in mappings.items():
            code = raw.decode("charmap" if len(raw) == 1 else "utf-16-be", errors="surrogatepass")
            old = cmap.get(code, "")
            if len(meanings) != 1 or (old and "\x00" not in old and old != "\ufffd"):
                continue
            actual = next(iter(meanings))
            entries.append(b"<" + raw.hex().encode() + b"> <" + actual.encode("utf-16-be").hex().encode() + b">\n")
        if not entries:
            continue
        mapping = b"\n" + str(len(entries)).encode() + b" beginbfchar\n" + b"".join(entries) + b"endbfchar\n"
        replacement_font = DictionaryObject(font)
        if "/ToUnicode" in font:
            before, after = cast("StreamObject", font["/ToUnicode"]).get_data().rsplit(b"endcmap", 1)
            data = before + mapping + b"endcmap" + after
        else:
            width = len(next(iter(mappings)))
            data = b"begincmap\n1 begincodespacerange\n<" + b"00" * width
            data += b"> <" + b"ff" * width + b">\nendcodespacerange\n" + mapping + b"endcmap\n"
        stream = DecodedStreamObject()
        stream.set_data(data)
        replacement_font[NameObject("/ToUnicode")] = stream
        local_fonts[name] = replacement_font
    local_resources = DictionaryObject(resources)
    local_resources[NameObject("/Font")] = local_fonts
    page[NameObject("/Resources")] = local_resources
    return page.extract_text() or ""


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    if not value.document_id.strip() or not value.content:
        raise ValueError("PDF input requires a document ID and complete bytes")
    digest = hashlib.sha256(value.content).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(value.content), strict=True)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF requires a nonempty password")  # noqa: TRY301 - normalize parser and permission errors
        pages = tuple(
            PDFPageInput(
                document_id=value.document_id, document_sha256=digest, page_number=index, text=_page_text(page)
            )
            for index, page in enumerate(reader.pages, start=1)
        )
    except Exception as error:
        raise ValueError("PDF cannot be parsed completely") from error
    if not pages:
        raise ValueError("PDF contains no physical pages")
    return pages


class _PDFTextProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class: type[PDFPageSource] = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        page = source.page
        return TextEvidence(
            source_type="pdf-page",
            source_id=source.name,
            content=f"{page.document_id}\n{page.text}",
            metadata={
                "document_id": page.document_id,
                "document_sha256": page.document_sha256,
                "page_number": page.page_number,
            },
        )


class _PDFPageDefinition:
    name = "pdf-page"
    version = "1"
    input_class: type[PDFPageInput] = PDFPageInput
    source_class: type[PDFPageSource] = PDFPageSource
    projections: tuple[SourceProjection[PDFPageSource], ...] = (_PDFTextProjection(),)

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        if (
            not value.document_id.strip()
            or value.page_number < 1
            or re.fullmatch(r"[0-9a-f]{64}", value.document_sha256) is None
        ):
            raise ValueError("PDF page identity is invalid")
        parts = json.dumps(
            [value.document_id, value.document_sha256, value.page_number], ensure_ascii=False, separators=(",", ":")
        ).encode()
        identity = "pdf:" + hashlib.sha256(parts).hexdigest()
        return PDFPageSource(name=identity, materialization=SourceMaterialization.CAPTURED, page=value)

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return source.page


PDF_PAGE_SOURCE_DEFINITION = _PDFPageDefinition()


async def search_pdf_pages(
    memory: ScopedMemoryApplication, question: str, limit: int = 10
) -> tuple[MemoryCitation, ...]:
    """Retrieve real Memory hits, expand financial terms, and rerank without a model."""
    if type(limit) is not int or limit < 1:
        raise ValueError("page search limit must be a positive integer")
    expansions = []
    lower = question.lower()
    for trigger, expansion in (
        (r"legal|litigat|lawsuit|battles", "legal proceedings litigation contingencies lawsuits"),
        (r"geograph|regions?|countries|operates", "geographic segments countries international operations"),
        (r"capex|capital expenditure", "capital spending payments property plant equipment cash flows"),
        (r"ebitda|depreciation|amortization|d&a", "operating income depreciation amortization consolidated cash flows"),
        (r"income statement|p&l", "consolidated statements income earnings revenue"),
    ):
        if re.search(trigger, lower):
            expansions.append(expansion)
    normalized_question = re.sub(r"\bfy\s*(\d{4})\b", r"\1", question, flags=re.IGNORECASE)
    # The Memory analyzer preserves identifiers, and its admission gate asks
    # long queries to share several words. Finance questions often describe
    # multiple statements, so retrieve with short keyword views, then rank the
    # union against the complete question. Every candidate remains a real hit.
    queries = []
    for view in (normalized_question, *expansions):
        words = tuple(dict.fromkeys(terms(view, stem=False, preserve_alphanumeric=True)))
        queries.extend(" ".join(words[start : start + 2]) for start in range(0, len(words), 2))
    queries = list(dict.fromkeys(queries)) or [question]
    hits: dict[tuple[str, str, str], MemoryHit] = {}
    for query_text in queries:
        result = await memory.search(SearchMemoryRequest(query=query_text, limit=max(128, limit * 16), mode="fts"))
        for hit in result.hits:
            hits.setdefault((hit.memory_ref.artifact_id, hit.entry_id, hit.entry_version_id), hit)
    candidates = list(hits.values())
    scores = rank_texts(
        normalized_question + " " + " ".join(expansions), [hit.text for hit in candidates], preserve_alphanumeric=True
    )
    order = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
    selected, pages = [], set()
    for index in order:
        hit = candidates[index]
        citation = MemoryCitation(
            memory_ref=hit.memory_ref, entry_id=hit.entry_id, entry_version_id=hit.entry_version_id
        )
        entry = (await memory.get(GetMemoryEntryRequest(citation=citation))).entry
        identities = tuple(
            (source.source_type, source.source_id) for source in entry.sources if source.source_type == "pdf-page"
        )
        if not identities or any(identity in pages for identity in identities):
            continue
        selected.append(citation)
        pages.update(identities)
        if len(selected) == limit:
            break
    return tuple(selected)
