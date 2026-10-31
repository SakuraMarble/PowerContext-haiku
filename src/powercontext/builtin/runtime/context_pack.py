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

"""Canonical, independently verifiable full and prefix-delta context packets."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

PACK_SCHEMA = "powercontext.context-pack.v1"
DELTA_SCHEMA = "powercontext.context-pack-delta.v1"


class ContextPackError(ValueError):
    """Invalid UTF-8 chunks, wire values, checksums or delta references."""


def _encode(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _items_wire(items: tuple[bytes, ...]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _validate_items(items: tuple[bytes, ...]) -> None:
    if not isinstance(items, tuple) or any(not isinstance(item, bytes) for item in items):
        raise ContextPackError("chunks must be a tuple of bytes")
    try:
        for item in items:
            item.decode("utf-8")
    except UnicodeError as error:
        raise ContextPackError("each chunk must contain valid UTF-8") from error


def _read_items(value: object) -> tuple[bytes, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ContextPackError("wire chunks must be Base64 string arrays")
    try:
        items = tuple(base64.b64decode(item, validate=True) for item in value)
    except (ValueError, binascii.Error) as error:
        raise ContextPackError("invalid Base64 chunk") from error
    if _items_wire(items) != value:
        raise ContextPackError("noncanonical Base64 chunk")
    _validate_items(items)
    return items


def _validate_digest(value: object) -> None:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ContextPackError("invalid SHA-256 digest")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContextPackError("duplicate JSON field")
        result[key] = value
    return result


def _read_payload(value: bytes, fields: set[str], schema: str) -> dict[str, Any]:
    try:
        payload = json.loads(value.decode("utf-8"), object_pairs_hook=_unique_object)
    except (ValueError, TypeError, UnicodeError) as error:
        raise ContextPackError("invalid context packet JSON") from error
    if not isinstance(payload, dict) or set(payload) != fields or payload.get("schema") != schema:
        raise ContextPackError("invalid context packet fields or schema")
    return payload


@dataclass(frozen=True, slots=True)
class ContextPack:
    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    @classmethod
    def build(cls, items: tuple[bytes, ...]) -> ContextPack:
        _validate_items(items)
        payload = {"schema": PACK_SCHEMA, "items": _items_wire(items)}
        digest = "sha256:" + hashlib.sha256(_encode(payload)).hexdigest()
        payload.update(digest=digest, byte_count=0)
        while (size := len(_encode(payload))) != payload["byte_count"]:
            payload["byte_count"] = size
        return cls(items, digest, size)

    @classmethod
    def from_content(cls, content: str | bytes | None, *, chunk_bytes: int = 512) -> ContextPack:
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ContextPackError("chunk_bytes must be a positive integer")
        try:
            data = content.encode("utf-8") if isinstance(content, str) else content
            if data is None:
                data = b""
            if not isinstance(data, bytes):
                raise ContextPackError("content must be UTF-8 text or bytes")
            data.decode("utf-8")
        except UnicodeError as error:
            raise ContextPackError("content must contain valid UTF-8") from error
        chunks = []
        start = 0
        while start < len(data):
            end = min(start + chunk_bytes, len(data))
            # A continuation byte cannot begin the following independent chunk.
            while end < len(data) and data[end] & 0xC0 == 0x80:
                end -= 1
            if end == start:
                raise ContextPackError("chunk limit cannot contain one UTF-8 character")
            chunks.append(data[start:end])
            start = end
        return cls.build(tuple(chunks))

    def to_bytes(self) -> bytes:
        return _encode({
            "schema": PACK_SCHEMA,
            "items": _items_wire(self.items),
            "digest": self.digest,
            "byte_count": self.byte_count,
        })

    @classmethod
    def from_bytes(cls, value: bytes) -> ContextPack:
        payload = _read_payload(value, {"schema", "items", "digest", "byte_count"}, PACK_SCHEMA)
        _validate_digest(payload["digest"])
        result = cls.build(_read_items(payload["items"]))
        if type(payload["byte_count"]) is not int or payload["byte_count"] != result.byte_count:
            raise ContextPackError("incorrect context packet byte count")
        if payload["digest"] != result.digest:
            raise ContextPackError("incorrect context packet digest")
        return result


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack) -> ContextPackDelta:
        base = ContextPack.from_bytes(base.to_bytes())
        target = ContextPack.from_bytes(target.to_bytes())
        prefix = 0
        for old, new in zip(base.items, target.items, strict=False):
            if old != new:
                break
            prefix += 1
        return cls(base.digest, target.digest, prefix, target.items[prefix:])

    @property
    def byte_count(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        return _encode({
            "schema": DELTA_SCHEMA,
            "base_digest": self.base_digest,
            "target_digest": self.target_digest,
            "prefix_items": self.prefix_items,
            "appended": _items_wire(self.appended),
        })

    @classmethod
    def from_bytes(cls, value: bytes) -> ContextPackDelta:
        payload = _read_payload(
            value, {"schema", "base_digest", "target_digest", "prefix_items", "appended"}, DELTA_SCHEMA
        )
        _validate_digest(payload["base_digest"])
        _validate_digest(payload["target_digest"])
        if type(payload["prefix_items"]) is not int or payload["prefix_items"] < 0:
            raise ContextPackError("prefix_items must be a nonnegative integer")
        return cls(
            payload["base_digest"], payload["target_digest"], payload["prefix_items"], _read_items(payload["appended"])
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta) -> ContextPack:
    base = ContextPack.from_bytes(base.to_bytes())
    delta = ContextPackDelta.from_bytes(delta.to_bytes())
    if delta.base_digest != base.digest:
        raise ContextPackError("delta does not reference the supplied base")
    if delta.prefix_items > len(base.items):
        raise ContextPackError("delta prefix exceeds the base packet")
    target = ContextPack.build(base.items[: delta.prefix_items] + delta.appended)
    if target.digest != delta.target_digest:
        raise ContextPackError("reconstructed target digest does not match")
    return target
