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

"""Durable, scope-local evidence graphs with resumable native state changes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import insert, select, update

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.artifact_governance import ArtifactGovernanceRepository, ArtifactLifecycleState
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_TABLE
from powercontext.builtin.sources import validate_scope_id
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime


class SourceLifecycleError(ValueError):
    """An invalid graph identity, event, reference, or generation request."""


class SourceLifecycleEvent(BaseModel):
    """One strictly typed, monotonically ordered Source availability event."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    node_id: str
    revision: int = Field(gt=0)
    sequence: int = Field(gt=0)
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("blank lifecycle field")  # noqa: TRY003
        return value


class _Node(BaseModel):
    kind: Literal["source", "memory", "topic"]
    reference: dict[str, Any]
    parents: tuple[str, ...] = ()
    policy: Literal["all", "any"] = "all"
    revision: int = 1
    last_event: SourceLifecycleEvent | None = None
    available: bool = True
    applied: bool | None = None

    def ref(self) -> SourceRef | MemoryCitation | ArtifactRef:
        model = {"source": SourceRef, "memory": MemoryCitation, "topic": ArtifactRef}[self.kind]
        return model.model_validate(self.reference)


class _Graph(BaseModel):
    # The graph is deliberately bounded by its scope; no corpus or body data is copied here.
    nodes: dict[str, _Node] = Field(default_factory=dict)

    def propagate(self) -> None:
        # Parents must be registered first. JSON preserves this topological insertion order.
        for node in self.nodes.values():
            if node.kind != "source":
                support = [self.nodes[parent].available for parent in node.parents]
                node.available = all(support) if node.policy == "all" else any(support)


class ScopedSourceLifecycle:
    """One single-writer dependency graph, backed by the Runtime database."""

    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase | None, scope_id: str) -> None:
        self._runtime = runtime
        self._database = database
        self._scope = scope_id

    @property
    def database(self) -> AsyncDatabase:
        if self._database is None:
            raise SourceLifecycleError("lifecycle database unavailable")  # noqa: TRY003
        return self._database

    async def _load(self) -> _Graph:
        async with self.database.transaction() as connection:
            payload = await connection.scalar(
                select(SOURCE_LIFECYCLE_TABLE.c.payload).where(SOURCE_LIFECYCLE_TABLE.c.scope_id == self._scope)
            )
        return _Graph() if payload is None else _Graph.model_validate_json(payload)

    async def _save(self, graph: _Graph) -> None:
        table = SOURCE_LIFECYCLE_TABLE
        payload = graph.model_dump_json().encode()
        async with self.database.transaction() as connection:
            exists = await connection.scalar(select(table.c.scope_id).where(table.c.scope_id == self._scope))
            if exists is None:
                await connection.execute(insert(table).values(scope_id=self._scope, payload=payload))
            else:
                await connection.execute(update(table).where(table.c.scope_id == self._scope).values(payload=payload))

    async def register_source(self, node_id: str, source_ref: SourceRef, *, revision: int = 1) -> None:
        if type(revision) is not int or revision < 1 or not isinstance(source_ref, SourceRef):
            raise SourceLifecycleError("invalid Source registration")  # noqa: TRY003
        await self._register(node_id, _Node(kind="source", reference=source_ref.model_dump(), revision=revision))

    async def register_memory(
        self, node_id: str, citation: MemoryCitation, *, parents: tuple[str, ...], policy: str
    ) -> None:
        if not isinstance(citation, MemoryCitation):
            raise SourceLifecycleError("invalid Memory reference")  # noqa: TRY003
        await self._derivative(node_id, "memory", citation, parents, policy)

    async def register_topic(
        self, node_id: str, artifact_ref: ArtifactRef, *, parents: tuple[str, ...], policy: str
    ) -> None:
        if not isinstance(artifact_ref, ArtifactRef) or artifact_ref.family != "topic-memory":
            raise SourceLifecycleError("invalid Topic Memory reference")  # noqa: TRY003
        await self._derivative(node_id, "topic", artifact_ref, parents, policy)

    async def _derivative(
        self,
        node_id: str,
        kind: Literal["memory", "topic"],
        reference: MemoryCitation | ArtifactRef,
        parents: tuple[str, ...],
        policy: str,
    ) -> None:
        if (
            policy not in {"all", "any"}
            or not isinstance(parents, (tuple, list))
            or not parents
            or any(not isinstance(parent, str) or not parent.strip() for parent in parents)
            or len(set(parents)) != len(parents)
        ):
            raise SourceLifecycleError("invalid dependency policy or parents")  # noqa: TRY003
        await self._register(
            node_id, _Node(kind=kind, reference=reference.model_dump(), parents=tuple(parents), policy=policy)
        )

    async def _register(self, node_id: str, node: _Node) -> None:
        if not isinstance(node_id, str) or not node_id.strip():
            raise SourceLifecycleError("invalid node identity")  # noqa: TRY003
        async with self._runtime._context(self._scope) as context, self._runtime._locked(self._scope):
            graph = await self._load()
            previous = graph.nodes.get(node_id)
            if previous is not None:
                fields = ("kind", "reference", "parents", "policy", "revision")
                if any(getattr(previous, name) != getattr(node, name) for name in fields):
                    raise SourceLifecycleError("node binding conflicts with registration")  # noqa: TRY003
            else:
                if node_id in node.parents or any(parent not in graph.nodes for parent in node.parents):
                    raise SourceLifecycleError("parent must already exist; cycles are forbidden")  # noqa: TRY003
                try:
                    ref = node.ref()
                    if isinstance(ref, SourceRef):
                        await self._runtime.records.for_scope(self._scope).get_source(ref.source_type, ref.source_id)
                    elif isinstance(ref, MemoryCitation):
                        if ref.memory_ref.artifact_id != context.artifacts.memory_artifact_id:
                            raise SourceLifecycleError("Memory belongs to a different identity")  # noqa: TRY003, TRY301
                        await context.artifacts.memory.validate_citation(ref)
                    else:
                        await self._runtime.records.for_scope(self._scope).get_artifact_revision(
                            ref.family, ref.artifact_id, ref.revision
                        )
                except SourceLifecycleError:
                    raise
                except Exception as error:
                    raise SourceLifecycleError("registration reference is not readable") from error  # noqa: TRY003
                graph.nodes[node_id] = node
                graph.propagate()
                await self._save(graph)
            await self._synchronize(graph)

    async def apply(self, event: SourceLifecycleEvent) -> list[str]:
        async with self._runtime._context(self._scope), self._runtime._locked(self._scope):
            graph = await self._load()
            node = graph.nodes.get(event.node_id)
            if node is None or node.kind != "source" or node.revision != event.revision:
                raise SourceLifecycleError("unknown Source or lifecycle revision")  # noqa: TRY003
            if node.last_event is not None and (
                event.sequence < node.last_event.sequence
                or (event.sequence == node.last_event.sequence and event != node.last_event)
            ):
                raise SourceLifecycleError("stale or conflicting event sequence")  # noqa: TRY003
            if event != node.last_event:
                node.last_event = event
                node.available = event.available
                graph.propagate()
                # Commit desired states before native writes. A failed write leaves replayable work.
                await self._save(graph)
            await self._synchronize(graph)
            return sorted(node_id for node_id, item in graph.nodes.items() if item.available)

    async def _synchronize(self, graph: _Graph) -> None:
        for node in graph.nodes.values():
            if node.applied == node.available:
                continue
            reason = "source lifecycle dependency propagation"
            if node.kind == "memory":
                citation = MemoryCitation.model_validate(node.reference)
                async with self._runtime._context(self._scope) as context:
                    service = context.artifacts.memory
                    current = await service.head(citation.memory_ref.artifact_id)
                    entry = await service.validate_citation(citation)
                    operation = service.reactivate if node.available else service.forget
                    # Native state changes are themselves idempotent, including a crash before _save.
                    await operation(current, entries=(entry,), reason=reason)
            elif node.kind == "topic":
                reference = ArtifactRef.model_validate(node.reference)
                governance = ArtifactGovernanceRepository()
                desired = ArtifactLifecycleState.ACTIVE if node.available else ArtifactLifecycleState.DEPRECATED
                async with self.database.transaction() as connection:
                    current = await governance.get(connection, self._scope, reference.family, reference.artifact_id)
                    if current.lifecycle_state != desired:
                        await governance.transition(
                            connection,
                            self._scope,
                            reference.family,
                            reference.artifact_id,
                            current.governance_generation,
                            desired,
                            None,
                        )
            node.applied = node.available
            await self._save(graph)

    async def resolve(self, node_id: str, *, purpose: str) -> SourceRef | MemoryCitation | ArtifactRef:
        if purpose not in {"generation", "audit"}:
            raise SourceLifecycleError("unknown reference purpose")  # noqa: TRY003
        async with self._runtime._context(self._scope):
            graph = await self._load()
            node = graph.nodes.get(node_id)
            if node is None or (purpose == "generation" and not node.available):
                raise SourceLifecycleError("node unavailable for requested purpose")  # noqa: TRY003
            return node.ref()


class SourceLifecycleApplication:
    """Bind lifecycle operations to the Runtime's normal scope and database."""

    def __init__(self, runtime: BuiltinRuntime, database: AsyncDatabase | None) -> None:
        self._runtime = runtime
        self._database = database

    def for_scope(self, scope_id: str) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, self._database, validate_scope_id(scope_id))


__all__ = ["SourceLifecycleApplication", "SourceLifecycleError", "SourceLifecycleEvent"]
