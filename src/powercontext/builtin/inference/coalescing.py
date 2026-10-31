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

"""Share in-flight document embeddings and retain successful vectors in an LRU."""

from __future__ import annotations

import asyncio
import math
from collections import OrderedDict
from dataclasses import dataclass

from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel


@dataclass(frozen=True, slots=True)
class _EmbeddedText:
    vector: EmbeddingVector
    batch_id: int
    usage: InferenceUsage


class CoalescingEmbeddingModel:
    """Coordinate one event loop's callers without coupling their cancellation."""

    def __init__(
        self, model: EmbeddingModel, *, max_batch_size: int = 64, coalesce_ms: float = 5.0, cache_size: int = 1024
    ) -> None:
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        if type(coalesce_ms) not in (int, float) or not math.isfinite(coalesce_ms) or coalesce_ms < 0:
            raise ValueError("coalesce_ms must be a finite nonnegative number")
        self.profile = model.profile
        self._model = model
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[_EmbeddedText]] = {}
        self._pending: dict[str, asyncio.Future[_EmbeddedText]] = {}
        self._flush_task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._next_batch_id = 0

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=())
        values: dict[str, EmbeddingVector] = {}
        awaited: dict[str, asyncio.Future[_EmbeddedText]] = {}
        loop = asyncio.get_running_loop()
        for text in texts:
            if text in values or text in awaited:
                continue
            if text in self._cache:
                values[text] = self._cache[text]
                self._cache.move_to_end(text)
                continue
            future = self._inflight.get(text)
            if future is None:
                future = loop.create_future()
                # A caller may cancel its wait. Consume orphaned failures without
                # cancelling the work another caller can still join.
                future.add_done_callback(lambda f: None if f.cancelled() else f.exception())
                self._inflight[text] = future
                self._pending[text] = future
            awaited[text] = future
        if self._pending and self._flush_task is None:
            self._flush_task = asyncio.create_task(self._flush())
        results = await asyncio.gather(*(asyncio.shield(future) for future in awaited.values()))
        batches: dict[int, InferenceUsage] = {}
        for text, result in zip(awaited, results, strict=True):
            values[text] = result.vector
            batches[result.batch_id] = result.usage
        usage = InferenceUsage(
            requests=sum(item.requests for item in batches.values()),
            input_tokens=_sum_optional(item.input_tokens for item in batches.values()),
            output_tokens=_sum_optional(item.output_tokens for item in batches.values()),
        )
        return EmbeddingResult(vectors=tuple(values[text] for text in texts), usage=usage)

    async def _flush(self) -> None:
        await asyncio.sleep(self._coalesce_seconds)
        pending = tuple(self._pending.items())
        self._pending.clear()
        self._flush_task = None
        for start in range(0, len(pending), self._max_batch_size):
            batch = pending[start : start + self._max_batch_size]
            self._next_batch_id += 1
            task = asyncio.create_task(self._execute(batch, self._next_batch_id))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _execute(self, batch: tuple[tuple[str, asyncio.Future[_EmbeddedText]], ...], batch_id: int) -> None:
        try:
            result = await self._model.embed(tuple(text for text, _ in batch))
            if len(result.vectors) != len(batch):
                raise ValueError("embedding provider returned the wrong number of vectors")  # noqa: TRY301 - fan failure out to the batch's waiters
        except Exception as error:
            for text, future in batch:
                self._inflight.pop(text)
                future.set_exception(error)
            return
        for (text, future), vector in zip(batch, result.vectors, strict=True):
            if self._cache_size:
                self._cache[text] = vector
                self._cache.move_to_end(text)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            self._inflight.pop(text)
            future.set_result(_EmbeddedText(vector, batch_id, result.usage))


def _sum_optional(values) -> int | None:
    available = tuple(value for value in values if value is not None)
    return sum(available) if available else None
