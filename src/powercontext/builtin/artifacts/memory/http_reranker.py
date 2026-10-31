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

"""Identity-preserving reranking against a bounded HTTP scoring service."""

from __future__ import annotations

import math

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import MemoryRerankDecision
from powercontext.builtin.inference import InferenceUsage


class HTTPMemoryReranker:
    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = "http-cross-encoder-v1",
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.endpoint = endpoint
        self.client = client
        self.batch_size = batch_size
        self.policy_id = policy_id

    async def rerank(self, query: str, candidates: tuple[MemoryHit, ...], limit: int, /) -> MemoryRerankDecision:  # noqa: C901 - validate one scoring protocol
        if not candidates or type(limit) is not int or not 1 <= limit <= len(candidates):
            raise ValueError("reranking requires candidates and a valid result limit")
        scores: dict[int, float] = {}
        requests = 0
        for offset in range(0, len(candidates), self.batch_size):
            batch = candidates[offset : offset + self.batch_size]
            expected = set(range(offset + 1, offset + len(batch) + 1))
            response = await self.client.post(
                self.endpoint,
                json={
                    "query": query,
                    "candidates": [
                        {"rank": offset + index, "text": candidate.text}
                        for index, candidate in enumerate(batch, start=1)
                    ],
                },
            )
            requests += 1
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or set(payload) != {"scores"} or not isinstance(payload["scores"], list):
                raise ValueError("reranker response must contain only a scores array")
            observed: set[int] = set()
            for item in payload["scores"]:
                if not isinstance(item, dict) or set(item) != {"rank", "score"}:
                    raise ValueError("reranker scores must contain exactly rank and score")
                rank, score = item["rank"], item["score"]
                if type(rank) is not int or rank not in expected or rank in observed:
                    raise ValueError("reranker scores must cover each batch rank exactly once")
                if type(score) not in (int, float):
                    raise ValueError("reranker score must be a finite number")
                try:
                    finite = math.isfinite(score)
                except OverflowError as error:
                    raise ValueError("reranker score must be a finite number") from error
                if not finite:
                    raise ValueError("reranker score must be a finite number")
                observed.add(rank)
                scores[rank] = score
            if observed != expected:
                raise ValueError("reranker scores do not cover the complete batch")
        selected = tuple(sorted(scores, key=lambda rank: (-scores[rank], rank))[:limit])
        return MemoryRerankDecision(selected_ranks=selected, usage=InferenceUsage(requests=requests))
