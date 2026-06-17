# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Common types for semantic KV cache donor discovery.

These types intentionally separate semantic donor discovery from KV
materialization.  A provider may find a semantically related donor, but the KV
cache connector may only report matched tokens to TensorRT-LLM when it has an
engine-valid load plan for those tokens.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


class SemanticKvMaterializationKind(str, Enum):
    """How a semantic lookup result may be materialized by a connector."""

    DISCOVERY_ONLY = "discovery_only"
    EXACT_PREFIX = "exact_prefix"


@dataclass(frozen=True)
class SemanticKvLookupRequest:
    """Semantic lookup input owned by the connector scheduler."""

    request_id: str
    token_ids: Sequence[int]
    prompt_text: str | None
    model_id: str
    namespace: str
    already_computed_tokens: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticKvLookupResult:
    """Provider result for a semantic donor lookup.

    ``DISCOVERY_ONLY`` results are useful for telemetry and routing, but must
    not be counted as matched tokens.  ``EXACT_PREFIX`` means the provider has
    found a donor whose token prefix is exact-equivalent to the request prefix.
    """

    donor_id: str
    similarity: float
    reusable_token_count: int
    materialization_kind: SemanticKvMaterializationKind = (
        SemanticKvMaterializationKind.DISCOVERY_ONLY)
    donor_token_ids: Sequence[int] | None = None
    quality_signals: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True)
class SemanticKvDonor:
    """Completed request metadata that a provider may index as a donor."""

    donor_id: str
    token_ids: Sequence[int]
    prompt_text: str | None
    model_id: str
    namespace: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class SemanticKvProvider(Protocol):
    """Provider contract for semantic KV donor discovery."""

    def lookup(
            self,
            request: SemanticKvLookupRequest) -> SemanticKvLookupResult | None:
        ...

    def register_donor(self, donor: SemanticKvDonor) -> None:
        ...


class LocalSemanticKvProvider:
    """Deterministic provider used by examples and connector tests.

    The provider performs a simple token-set Jaccard search.  It deliberately
    returns ``DISCOVERY_ONLY`` for non-identical token sequences so examples can
    demonstrate semantic discovery without weakening exact KV cache semantics.
    """

    def __init__(self,
                 min_similarity: float = 0.70,
                 max_donors: int = 10_000) -> None:
        self._min_similarity = min_similarity
        self._max_donors = max_donors
        self._donors: OrderedDict[str, SemanticKvDonor] = OrderedDict()

    def lookup(
            self,
            request: SemanticKvLookupRequest) -> SemanticKvLookupResult | None:
        query_tokens = list(request.token_ids)
        query_set = set(query_tokens)
        if not query_set:
            return None

        best: tuple[float, SemanticKvDonor] | None = None
        for donor in self._donors.values():
            if donor.namespace != request.namespace:
                continue
            if donor.model_id != request.model_id:
                continue
            donor_set = set(donor.token_ids)
            if not donor_set:
                continue
            similarity = len(query_set & donor_set) / len(query_set | donor_set)
            if similarity >= self._min_similarity and (
                    best is None or similarity > best[0]):
                best = (similarity, donor)

        if best is None:
            return None

        similarity, donor = best
        donor_tokens = list(donor.token_ids)
        exact = donor_tokens == query_tokens
        return SemanticKvLookupResult(
            donor_id=donor.donor_id,
            similarity=similarity,
            reusable_token_count=len(query_tokens) if exact else 0,
            materialization_kind=(
                SemanticKvMaterializationKind.EXACT_PREFIX
                if exact else SemanticKvMaterializationKind.DISCOVERY_ONLY),
            donor_token_ids=donor_tokens,
            quality_signals={
                "token_jaccard": similarity,
                "exact_tokens": exact,
            },
            reason="local_exact" if exact else "local_semantic_discovery",
        )

    def register_donor(self, donor: SemanticKvDonor) -> None:
        if donor.donor_id in self._donors:
            self._donors.move_to_end(donor.donor_id)
            self._donors[donor.donor_id] = donor
            return

        while len(self._donors) >= self._max_donors:
            self._donors.popitem(last=False)
        self._donors[donor.donor_id] = donor

    def clear_donors(self) -> None:
        self._donors.clear()
