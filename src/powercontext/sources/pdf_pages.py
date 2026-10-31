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
from typing import TYPE_CHECKING

from pydantic import BaseModel
from pypdf import PdfReader

from powercontext._lexical import rank_texts, terms
from powercontext.builtin.artifacts.memory.models import MemoryCitation, MemoryHit
from powercontext.builtin.runtime.models import GetMemoryEntryRequest, SearchMemoryRequest
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
                document_id=value.document_id, document_sha256=digest, page_number=index, text=page.extract_text() or ""
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
    source_class = PDFPageSource
    output_class = TextEvidence

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
    input_class = PDFPageInput
    source_class = PDFPageSource
    projections = (_PDFTextProjection(),)

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
    query = " ".join(terms(question))
    queries = [query or question, *expansions]
    hits: dict[tuple[str, str, str], MemoryHit] = {}
    for query_text in queries:
        result = await memory.search(SearchMemoryRequest(query=query_text, limit=max(100, limit * 10), mode="fts"))
        for hit in result.hits:
            hits.setdefault((hit.memory_ref.artifact_id, hit.entry_id, hit.entry_version_id), hit)
    candidates = list(hits.values())
    scores = rank_texts(question + " " + " ".join(expansions), [hit.text for hit in candidates])
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
