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

"""Select literal query-related intervals while charging the complete wire envelope."""

from __future__ import annotations

import json
import re
from dataclasses import replace

from powercontext._lexical import rank_texts, terms
from powercontext.artifacts import ArtifactAddress
from powercontext.builtin.runtime.models import PreparedContext
from powercontext.builtin.runtime.prepared_context import (
    MemoryEntryAddress,
    PreparedContextBuild,
    PreparedContextFragment,
)
from powercontext.builtin.runtime.prepared_text import TRUST_POLICY


def _identity(address):
    reference = address.memory.artifact
    return (
        address.memory.scope_id,
        reference.family,
        reference.artifact_id,
        reference.revision,
        address.entry_id,
        address.entry_version_id,
    )


def _render(fragments) -> str:
    items = [
        {
            "citation": {
                "scope_id": f.address.memory.scope_id,
                "memory_ref": f.address.memory.artifact.model_dump(mode="json"),
                "entry_id": f.address.entry_id,
                "entry_version_id": f.address.entry_version_id,
            },
            "content": f.text,
            "start": f.start,
            "end": f.end,
        }
        for f in fragments
    ]
    payload = json.dumps({"trust": "untrusted_history", "items": items}, ensure_ascii=False, separators=(",", ":"))
    return (
        TRUST_POLICY
        + "\n\nBEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1\n"
        + payload
        + "\nEND_POWERCONTEXT_PREPARED_CONTEXT_V1"
    )


def _sentences(text: str):
    start = 0
    for boundary in re.finditer(r"(?<=[.!?])\s+|\n+|(?<=[。！？])", text):  # noqa: RUF001 - Chinese sentence punctuation
        end = boundary.start()
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if right > left:
            yield left, right
        start = boundary.end()
    end = len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end > start:
        yield start, end


def select_evidence(request, groups, current_scope_id) -> PreparedContextBuild:  # noqa: C901 - one bounded selection pass
    candidates = []
    full_text = {}
    identities = set()
    for group in groups:
        for hit in group.hits:
            address = MemoryEntryAddress(
                ArtifactAddress(scope_id=group.scope_id or current_scope_id, artifact=hit.memory_ref),
                hit.entry_id,
                hit.entry_version_id,
            )
            key = _identity(address)
            if key in identities:
                continue
            identities.add(key)
            full_text[key] = hit.text
            for start, end in _sentences(hit.text):
                candidates.append(PreparedContextFragment(address, start, end, hit.text[start:end]))
    scores = rank_texts(request.query, [candidate.text for candidate in candidates])
    order = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))
    selected = []

    def merged(fragment):
        options = [[*selected, fragment]]
        for index, old in enumerate(selected):
            if old.address == fragment.address:
                start, end = min(old.start, fragment.start), max(old.end, fragment.end)
                union = replace(old, start=start, end=end, text=full_text[_identity(old.address)][start:end])
                options.append([*selected[:index], union, *selected[index + 1 :]])
        return min(options, key=lambda option: len(_render(option).encode("utf-8")))

    for index in order:
        fragment = candidates[index]
        if any(
            old.address == fragment.address and old.start <= fragment.start and old.end >= fragment.end
            for old in selected
        ):
            continue
        proposal = merged(fragment)
        if len(_render(proposal).encode("utf-8")) <= request.max_bytes:
            selected = proposal

    # If the best sentence itself is too long, preserve the most relevant literal
    # window that fits, rather than falling back to the document's first bytes.
    if not selected and order:
        fragment = candidates[order[0]]
        query_terms = set(terms(request.query))
        matches = [m for m in re.finditer(r"\w+", fragment.text) if set(terms(m.group())) & query_terms]
        center = matches[0].start() if matches else 0
        start = fragment.start + max(0, center - 80)
        lower, upper = 1, fragment.end - start
        while lower <= upper:
            length = (lower + upper) // 2
            candidate = replace(
                fragment,
                start=start,
                end=start + length,
                text=full_text[_identity(fragment.address)][start : start + length],
            )
            if len(_render([candidate]).encode("utf-8")) <= request.max_bytes:
                selected = [candidate]
                lower = length + 1
            else:
                upper = length - 1
    if not selected:
        return PreparedContextBuild(context=PreparedContext(status="empty", content=None, content_bytes=0), origins=())
    content = _render(selected)
    return PreparedContextBuild(
        context=PreparedContext(status="ready", content=content, content_bytes=len(content.encode("utf-8"))),
        origins=tuple(fragment.address for fragment in selected),
        fragments=tuple(selected),
    )
