# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exact financial words remain searchable after persisted PDF ingestion."""

import asyncio
import io

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from powercontext.builtin.artifacts.memory import MemoryEntryInput
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import (
    BuiltinConfig,
    GetMemoryEntryRequest,
    SubmitSourceObservation,
    open_builtin_runtime,
)
from powercontext.builtin.scope import ScopeDraft
from powercontext.sources import SourceDefinitionRegistry, manifest_for_definition, project_source_for_transport
from powercontext.sources.pdf_pages import (
    PDF_PAGE_SOURCE_DEFINITION,
    PDFDocumentInput,
    PDFPageInput,
    parse_pdf_document,
    search_pdf_pages,
)
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY


class _PagePipeline:
    async def extract(self, request, /):
        return tuple(
            MemoryEntryInput(
                kind="source-evidence",
                text=source.projection(TEXT_EVIDENCE_PROJECTION_KEY)["content"],
                sources=(source,),
            )
            for source in request.sources
        )


@pytest.mark.parametrize("word", ["Receivables", "Obligations", "3M"])
@pytest.mark.parametrize("verbose", [False, True])
def test_exact_financial_word_survives_pdf_search_after_restart(tmp_path, word, verbose):
    async def run():
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'pdf.db'}"))
        registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
        source = await registry.resolve(
            PDFPageInput(document_id="REPORT", document_sha256="a" * 64, page_number=1, text=f"{word} were disclosed.")
        )
        async with open_builtin_runtime(config, candidate_pipeline=_PagePipeline()) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(ScopeDraft(title="Report", summary="PDF", idempotency_key="pdf"))
            await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
            await runtime.ingestion.submit(
                SubmitSourceObservation(
                    scope_id=scope.scope_id, observation=project_source_for_transport(registry, source)
                )
            )
            assert (await runtime.memory.for_scope(scope.scope_id).flush()).processed
        async with open_builtin_runtime(config) as runtime:
            memory = runtime.memory.for_scope(scope.scope_id)
            question = f"What quantity of {word} was reported in the financial filings?" if verbose else f"{word}?"
            citations = await search_pdf_pages(memory, question, limit=20)
            assert len(citations) == 1
            entry = (await memory.get(GetMemoryEntryRequest(citation=citations[0]))).entry
            assert word in entry.text
            assert any(reference.source_id == source.name for reference in entry.sources)

    asyncio.run(run())


def test_actual_text_recovers_missing_ligatures_without_crossing_page_fonts():
    writer = PdfWriter()
    mapping = DecodedStreamObject()
    mapping.set_data(
        b"begincmap\n1 begincodespacerange\n<00> <ff>\nendcodespacerange\n"
        b"1 beginbfrange\n<20> <7e> <0020>\nendbfrange\n"
        b"1 beginbfchar\n<01> <0000>\nendbfchar\nendcmap\n"
    )
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
        NameObject("/ToUnicode"): mapping,
    })
    font_reference = writer._add_object(font)
    for prefix, actual, suffix in (("Administra", "ti", "ve"), ("adju", "st", "ment"), ("o", "ﬃ", "ce")):
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})
        })
        stream = DecodedStreamObject()
        stream.set_data(
            b"BT /F1 12 Tf 50 700 Td ("
            + prefix.encode()
            + b") Tj /Span << /ActualText <FEFF"
            + actual.encode("utf-16-be").hex().encode()
            + b"> >> BDC <01> Tj EMC ("
            + suffix.encode()
            + b") Tj ET"
        )
        page[NameObject("/Contents")] = stream
    output = io.BytesIO()
    writer.write(output)
    raw = output.getvalue()
    assert "\x00" in PdfReader(io.BytesIO(raw)).pages[0].extract_text()
    pages = parse_pdf_document(PDFDocumentInput(document_id="ligatures", content=raw))
    assert [page.text.strip() for page in pages] == ["Administrative", "adjustment", "office"]


def test_multi_statement_question_keeps_income_evidence_among_cash_flow_pages(tmp_path):
    async def run():
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'statements.db'}"))
        registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
        observations = []
        for page in range(1, 27):
            text = (
                f"Continuation {page}. Consolidated Statement of Cash Flows. "
                "Operating income depreciation amortization capital spending payments property plant equipment cash flows."
                if page <= 25
                else "Consolidated Statements of Income. Net revenue earnings sales cost of goods sold net income operating profit."
            )
            source = await registry.resolve(
                PDFPageInput(document_id="REPORT", document_sha256="b" * 64, page_number=page, text=text)
            )
            observations.append(project_source_for_transport(registry, source))
        async with open_builtin_runtime(config, candidate_pipeline=_PagePipeline()) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(title="Statements", summary="PDF", idempotency_key="statements")
            )
            await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
            for observation in observations:
                await runtime.ingestion.submit(
                    SubmitSourceObservation(scope_id=scope.scope_id, observation=observation)
                )
            memory = runtime.memory.for_scope(scope.scope_id)
            for _ in range(27):
                if not (await memory.flush()).processed:
                    break
            citations = await search_pdf_pages(
                memory,
                "Compute EBITDA less capital expenditures using operating income, depreciation and amortization "
                "from the cash flow statement and income statement.",
                limit=20,
            )
            references = set()
            for citation in citations:
                references.update(
                    ref.source_id for ref in (await memory.get(GetMemoryEntryRequest(citation=citation))).entry.sources
                )
            assert len(citations) <= 20
            assert observations[-1].name in references
            assert references.intersection(observation.name for observation in observations[:-1])

    asyncio.run(run())


def test_named_company_question_does_not_return_another_company_statements(tmp_path):
    async def run():
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'companies.db'}"))
        registry = SourceDefinitionRegistry((PDF_PAGE_SOURCE_DEFINITION,))
        observations, evidence = [], set()
        for number in range(1, 36):
            company = "ALPHA" if number <= 33 else "BETA"
            if number == 1:
                text = "Consolidated Statements of Income. Net sales 100. Operating profit 20."
            elif number == 2:
                text = "Consolidated Statements of Cash Flows. Depreciation 3. Amortization 2."
            elif company == "ALPHA":
                text = f"Product information section {number}. Products, customers and distribution."
            else:
                text = (
                    "EBITDA based on operating income plus depreciation and amortization from the "
                    "income statement and cash flow statement. Unadjusted earnings and capital expenditures."
                )
            source = await registry.resolve(
                PDFPageInput(document_id=f"{company}_2023_10K", document_sha256="c" * 64, page_number=number, text=text)
            )
            observations.append(project_source_for_transport(registry, source))
            if number in {1, 2}:
                evidence.add(source.name)
        async with open_builtin_runtime(config, candidate_pipeline=_PagePipeline()) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(
                ScopeDraft(title="Companies", summary="PDF", idempotency_key="companies")
            )
            await runtime.ingestion.register(manifest_for_definition(PDF_PAGE_SOURCE_DEFINITION))
            for observation in observations:
                await runtime.ingestion.submit(
                    SubmitSourceObservation(scope_id=scope.scope_id, observation=observation)
                )
            memory = runtime.memory.for_scope(scope.scope_id)
            for _ in range(len(observations) + 1):
                if not (await memory.flush()).processed:
                    break
            citations = await search_pdf_pages(
                memory,
                "What is Alpha's FY2023 EBITDA based on operating income plus depreciation and amortization "
                "from its income statement and cash flow statement?",
                limit=2,
            )
            actual = {
                ref.source_id
                for citation in citations
                for ref in (await memory.get(GetMemoryEntryRequest(citation=citation))).entry.sources
            }
            assert actual == evidence

    asyncio.run(run())
