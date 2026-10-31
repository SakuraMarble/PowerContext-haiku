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

"""A persisted propagation intent survives a partial native write failure."""

import asyncio

import pytest

from powercontext.builtin.artifacts.memory import MemoryEntryInput, MemoryService
from powercontext.builtin.persistence.sqlite import SQLiteConfig
from powercontext.builtin.runtime import BuiltinConfig, CaptureSource, RememberMemoryRequest, open_builtin_runtime
from powercontext.builtin.runtime.source_lifecycle import SourceLifecycleEvent
from powercontext.builtin.scope import ScopeDraft


def test_interrupted_native_propagation_resumes_after_restart_without_duplicate_revisions(tmp_path, monkeypatch):
    async def run():
        config = BuiltinConfig(database=SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'replay.db'}"))
        async with open_builtin_runtime(config) as runtime:
            assert runtime.scopes is not None
            scope = await runtime.scopes.create(ScopeDraft(title="Evidence", summary="Replay", idempotency_key="one"))
            lifecycle = runtime.lifecycle.for_scope(scope.scope_id)
            source = await runtime.sources.for_scope(scope.scope_id).capture(
                CaptureSource(source_id="origin", content="Original evidence", metadata={})
            )
            await lifecycle.register_source("origin", source.source_ref, revision=3)
            memory = runtime.memory.for_scope(scope.scope_id)
            citations = []
            for number in range(2):
                result = await memory.remember(
                    RememberMemoryRequest(entries=(MemoryEntryInput(kind="fact", text=f"Independent fact {number}"),))
                )
                assert result.entry is not None
                citations.append(result.entry.citation)
                await lifecycle.register_memory(str(number), result.entry.citation, parents=("origin",), policy="all")

            native_forget = MemoryService.forget
            calls = 0

            async def interrupt_second_write(self, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("interrupted database write")  # noqa: TRY003
                return await native_forget(self, *args, **kwargs)

            monkeypatch.setattr(MemoryService, "forget", interrupt_second_write)
            event = SourceLifecycleEvent(node_id="origin", revision=3, sequence=1, available=False, reason="withdrawn")
            with pytest.raises(RuntimeError, match="interrupted database write"):
                await lifecycle.apply(event)
            partial = await memory.list(include_inactive=True)
            assert {item.state for item in partial.entries} == {"active", "inactive"}

        async with open_builtin_runtime(config) as runtime:
            lifecycle = runtime.lifecycle.for_scope(scope.scope_id)
            memory = runtime.memory.for_scope(scope.scope_id)
            assert await lifecycle.apply(event) == []
            completed = await memory.list(include_inactive=True)
            assert all(item.state == "inactive" for item in completed.entries)
            assert completed.memory_ref is not None and partial.memory_ref is not None
            assert completed.memory_ref.revision == partial.memory_ref.revision + 1
            await lifecycle.apply(event)
            assert (await memory.list(include_inactive=True)).memory_ref == completed.memory_ref
            assert calls == 3
            for number, citation in enumerate(citations):
                assert await lifecycle.resolve(str(number), purpose="audit") == citation
            await lifecycle.apply(event.model_copy(update={"sequence": 2, "available": True, "reason": "restored"}))
            assert {item.entry.entry_version_id for item in (await memory.list()).entries} == {
                citation.entry_version_id for citation in citations
            }

    asyncio.run(run())
