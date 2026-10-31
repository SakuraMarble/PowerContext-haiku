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

# ruff: noqa: TRY003, TRY004

"""Offline Pi v3 branch evidence and durable, immutable incremental acquisition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, JsonValue

from powercontext.sources.connectors import ConnectorRunCompletion, ConnectorRunSession, ConnectorRunStatus
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence


class PiSessionInput(BaseModel):
    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    session_id: str
    event_id: str
    parent_id: str | None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    event: PiEventInput


class _PiCheckpoint(BaseModel):
    session_id: str
    events: dict[str, str]


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _event_identity(event: PiEventInput) -> str:
    return "pi:" + hashlib.sha256(_canonical([event.session_id, event.event_id]).encode()).hexdigest()


def _invalid_constant(value: str):
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _blocks(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError("Pi message content must be text or a block array")
    parts = []
    for block in value:
        if not isinstance(block, dict):
            raise ValueError("Pi content blocks must be objects")
        if block.get("type") == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError("Pi text blocks must contain a string")
            parts.append(text)
        elif block.get("type") == "toolCall":
            name, call_id = block.get("name"), block.get("id")
            if not isinstance(name, str) or not isinstance(call_id, str):
                raise ValueError("Pi tool calls must identify their name and call ID")
            parts.append(f"[tool_call {name} {call_id}] {_canonical(block.get('arguments', {}))}")
    return "\n".join(parts)


def _visible_event(session_id: str, record: dict[str, Any]) -> PiEventInput | None:
    kind = record["type"]
    if kind == "message":
        message = record.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError("Pi message must include an object with a role")
        role = message["role"]
        if role == "bashExecution":
            command, output = message.get("command", ""), message.get("output", "")
            if not isinstance(command, str) or not isinstance(output, str):
                raise ValueError("Pi shell evidence must be strings")
            body = f"$ {command}\n{output}" if command or output else ""
        else:
            body = _blocks(message.get("content", []))
            if role == "toolResult" and body:
                body = f"[tool_result {message.get('toolName', '')} {message.get('toolCallId', '')}] {body}"
    elif kind in ("compaction", "branch_summary"):
        role, body = kind, record.get("summary", "")
        if not isinstance(body, str):
            raise ValueError("Pi summaries must be text")
    elif kind == "custom_message":
        role, body = "custom", _blocks(record.get("content", []))
    else:
        return None
    if not body:
        return None
    return PiEventInput(
        session_id=session_id,
        event_id=record["id"],
        parent_id=record["parentId"],
        event_type=kind,
        role=role,
        content=f"{role}: {body}",
        record=record,
    )


def _parse(value: PiSessionInput) -> tuple[str, dict[str, dict[str, Any]], tuple[PiEventInput, ...]]:
    records = [
        json.loads(line, parse_constant=_invalid_constant)
        for line in value.content.decode("utf-8-sig").splitlines()
        if line.strip()
    ]
    if not records or not isinstance(records[0], dict):
        raise ValueError("Pi archive requires a session header")
    header = records[0]
    if (
        header.get("type") != "session"
        or type(header.get("version")) is not int
        or header["version"] != 3
        or not isinstance(header.get("id"), str)
        or not header["id"].strip()
    ):
        raise ValueError("Pi archive requires a version 3 session ID")
    nodes: dict[str, dict[str, Any]] = {}
    for record in records[1:]:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"].strip():
            raise ValueError("Pi nodes require unique nonempty IDs")
        if record["id"] in nodes or "parentId" not in record or not isinstance(record.get("type"), str):
            raise ValueError("Pi node ID, type or parent is invalid")
        parent = record["parentId"]
        if parent not in (None, "") and (not isinstance(parent, str) or parent not in nodes):
            raise ValueError("Pi parents must reference an earlier node")
        nodes[record["id"]] = record
    leaf = value.leaf_id if value.leaf_id is not None else next(reversed(nodes), None)
    if leaf is not None and leaf not in nodes:
        raise ValueError("Pi selected leaf does not exist")
    chain = []
    while leaf not in (None, ""):
        record = nodes[leaf]
        chain.append(record)
        leaf = record["parentId"]
    events = tuple(event for row in reversed(chain) if (event := _visible_event(header["id"], row)) is not None)
    return header["id"], nodes, events


def parse_pi_session(value: PiSessionInput) -> tuple[PiEventInput, ...]:
    """Validate the complete archive, then retain a selected branch's visible history."""
    return _parse(value)[2]


class _PiTextProjection:
    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        event = source.event
        return TextEvidence(
            source_type="pi-event",
            source_id=source.name,
            content=event.content,
            metadata={
                key: getattr(event, key) for key in ("session_id", "event_id", "parent_id", "event_type", "role")
            },
        )


class _PiEventDefinition:
    name = "pi-event"
    version = "1"
    input_class = PiEventInput
    source_class = PiEventSource
    projections = (_PiTextProjection(),)

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        return PiEventSource(name=_event_identity(value), materialization=SourceMaterialization.CAPTURED, event=value)

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return source.event


PI_EVENT_SOURCE_DEFINITION = _PiEventDefinition()


class PiSessionConnector:
    name = "pi-session-history"
    version = "1"
    source_definitions = frozenset({"pi-event"})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self.path, self.leaf_id = Path(path), leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        session_id, nodes, events = _parse(PiSessionInput(content=self.path.read_bytes(), leaf_id=self.leaf_id))
        previous = None if session.checkpoint is None else _PiCheckpoint.model_validate(session.checkpoint)
        imported = {} if previous is None else dict(previous.events)
        if previous is not None and previous.session_id != session_id:
            raise ValueError("Pi checkpoint belongs to a different session")
        fingerprints = {key: hashlib.sha256(_canonical(row).encode()).hexdigest() for key, row in nodes.items()}
        for key, digest in imported.items():
            if fingerprints.get(key) != digest:
                raise ValueError("previously imported Pi event was changed or removed")
        for event in events:
            if event.event_id not in imported:
                await session.submit(_event_identity(event), "pi-event", event)
                imported[event.event_id] = fingerprints[event.event_id]
        return ConnectorRunCompletion(
            status=ConnectorRunStatus.COMPLETE, checkpoint={"session_id": session_id, "events": imported}
        )
