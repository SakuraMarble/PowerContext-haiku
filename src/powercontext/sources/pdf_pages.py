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

from powercontext._lexical import rank_texts
from powercontext.builtin.artifacts.memory.models import MemoryCitation
from powercontext.builtin.runtime.models import SearchMemoryRequest
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


def _financial_views(question: str) -> tuple[str, ...]:
    """Separate accounting concepts so a multi-statement question keeps each view."""
    views = [re.sub(r"\bfy\s*(\d{4})\b", r"\1", question, flags=re.IGNORECASE)]
    for trigger, view in (
        (r"legal|litigat|lawsuit|battles", "legal proceedings litigation contingencies lawsuits"),
        (
            r"geograph|regions?|countries|operates",
            "business divisions reportable segments geographic markets operations",
        ),
        (r"capex|capital expenditure", "capital expenditures spending purchases additions property plant equipment"),
        (
            r"income statement|p\s*&\s*l|ebitda|operating income|profit margin",
            "consolidated statements income earnings operations net sales revenue operating profit",
        ),
        (
            r"cash.?flows?|ebitda|depreciation|amortization|d&a",
            "consolidated statements cash flows operating activities depreciation amortization",
        ),
        (
            r"balance sheet|financial position|receivabl|payabl|working capital",
            "consolidated balance sheets financial position current assets liabilities receivables payables",
        ),
        (r"restructur", "restructuring charges costs plan expenses"),
        (r"dividend", "dividends declared paid cash distributions shareholders"),
        (r"repurchas|buyback", "share stock repurchases treasury shares cash payments"),
    ):
        if re.search(trigger, question, re.IGNORECASE):
            views.append(view)
    return tuple(dict.fromkeys(views))


def _mentions_document_company(question: str, document_id: str) -> bool:
    """Recognize an explicitly named issuer from the stored document identifier.

    Do not route on a fiscal year: a filing can contain earlier-year evidence.
    Unknown identifiers impose no company restriction.
    """
    prefix = re.split(r"(?:19|20)\d{2}", document_id, maxsplit=1)[0]
    issuer = "".join(re.findall(r"[a-z0-9]+", prefix.lower()))
    if issuer in {"", "report", "annualreport", "filing", "document", "control"}:
        return False
    if len(issuer) < 4:
        return re.search(r"\b" + re.escape(issuer) + r"\b", question.lower()) is not None
    return issuer in "".join(re.findall(r"[a-z0-9]+", question.lower()))


async def search_pdf_pages(
    memory: ScopedMemoryApplication, question: str, limit: int = 10
) -> tuple[MemoryCitation, ...]:
    """Fuse public Memory search with scoped, active-page lexical retrieval.

    One current-head snapshot provides corpus statistics and exact citations.
    This avoids index admission/truncation dropping a requested statement. No
    source file, database internals, external model, or precomputed corpus is
    read; both indexed hits and snapshot bodies come from public Memory APIs.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("page search limit must be a positive integer")
    views = _financial_views(question)
    indexed = await memory.search(SearchMemoryRequest(query=views[0], limit=max(128, limit * 8), mode="fts"))
    snapshot = await memory.list()
    records = [
        record
        for record in snapshot.entries
        if any(source.source_type == "pdf-page" for source in record.entry.sources)
    ]
    named = [record for record in records if _mentions_document_company(question, record.entry.text.partition("\n")[0])]
    if named:
        records = named
    bodies = [record.entry.text.partition("\n")[2] for record in records]
    orders = []
    fused = [0.0] * len(records)
    for view_number, view in enumerate(views):
        scores = rank_texts(view, bodies, preserve_alphanumeric=True)
        order = sorted((i for i, score in enumerate(scores) if score > 0), key=lambda i: (-scores[i], i))
        orders.append(order)
        for rank, index in enumerate(order, start=1):
            fused[index] += (2.0 if view_number == 0 else 1.0) / (60 + rank)
    positions = {(record.entry.entry_id, record.entry.entry_version_id): i for i, record in enumerate(records)}
    for rank, hit in enumerate(indexed.hits, start=1):
        index = positions.get((hit.entry_id, hit.entry_version_id))
        if index is not None:
            fused[index] += 0.5 / (60 + rank)
    # Reserve two leading candidates from each requested concept before filling
    # by reciprocal rank. This keeps an income statement beside cash-flow
    # evidence even when the latter repeats many more words from the question.
    reserved = list(
        dict.fromkeys(index for depth in range(2) for order in orders for index in order[depth : depth + 1])
    )
    overall = sorted((i for i, score in enumerate(fused) if score > 0), key=lambda i: (-fused[i], i))
    ordered = list(dict.fromkeys([*reserved, *overall]))
    selected, pages = [], set()
    for index in ordered:
        record = records[index]
        identities = tuple(
            (source.source_type, source.source_id)
            for source in record.entry.sources
            if source.source_type == "pdf-page"
        )
        if any(identity in pages for identity in identities):
            continue
        selected.append(record.citation)
        pages.update(identities)
        if len(selected) == limit:
            break
    return tuple(selected)
