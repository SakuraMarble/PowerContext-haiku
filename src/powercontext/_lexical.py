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

"""Small offline lexical ranking helpers; never rewrite the underlying evidence."""

import math
import re
from collections import Counter

_STOP = frozenset([
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "from",
    "by",
    "with",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "has",
    "have",
    "had",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "their",
    "there",
    "what",
    "which",
    "who",
    "when",
    "where",
    "how",
    "can",
    "could",
    "would",
    "should",
    "will",
    "may",
    "must",
    "any",
    "all",
    "not",
    "only",
    "than",
    "then",
    "into",
    "about",
    "between",
    "using",
    "use",
    "used",
    "based",
    "according",
    "give",
    "answer",
    "question",
    "calculate",
    "define",
    "respond",
    "perspective",
    "assuming",
    "shown",
    "details",
    "within",
    "primarily",
])


def _stem(token: str) -> str:
    if len(token) > 5:
        if token.endswith("ies"):
            token = token[:-3] + "y"
        elif token.endswith("ing"):
            token = token[:-3]
        elif token.endswith("ed"):
            token = token[:-2]
        elif token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if token.endswith("ion"):
            token = token[:-3]
        if token.endswith("e"):
            token = token[:-1]
    return token


def terms(text: str, *, stem: bool = True, preserve_alphanumeric: bool = False) -> tuple[str, ...]:
    pattern = r"[a-z0-9]+|[\u4e00-\u9fff]" if preserve_alphanumeric else r"[a-z]+|\d+|[\u4e00-\u9fff]"
    result = [_stem(token) if stem else token for token in re.findall(pattern, text.lower()) if token not in _STOP]
    return tuple(result)


def rank_texts(query: str, documents: list[str], *, preserve_alphanumeric: bool = False) -> list[float]:
    """BM25 scores with deterministic, local corpus statistics and light stemming."""
    if not documents:
        return []
    query_terms = Counter(terms(query, preserve_alphanumeric=preserve_alphanumeric))
    counts = [Counter(terms(text, preserve_alphanumeric=preserve_alphanumeric)) for text in documents]
    document_frequency = Counter(term for count in counts for term in count)
    average_length = sum(sum(count.values()) for count in counts) / len(counts) or 1
    scores = []
    for count in counts:
        normalizer = 1.2 * (0.25 + 0.75 * sum(count.values()) / average_length)
        score = 0.0
        for term, weight in query_terms.items():
            frequency = count[term]
            if frequency:
                inverse = math.log1p((len(counts) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                score += min(weight, 2) * inverse * frequency * 2.2 / (frequency + normalizer)
        scores.append(score)
    return scores
