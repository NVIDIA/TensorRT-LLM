# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Map public manager staging views to the shared backend contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import Part, RegionView

from ..base.shared import CacheExtent, Unit

SHARED_CONTRACT_REVISION = "147ed68276e7fb89d5de60002d4e793a78707a8c+cancel-request-v1"
STAGING_EXTENT_NAMESPACE = b"trtllm:shared-kv:staging:1"


@dataclass(frozen=True)
class SharedRuntimeProfile:
    """Explicit assembly facts for the first shared runtime integration profile.

    Args:
        contract_revision: Exact shared contract revision used by the adapter.
        backend_revision: Concrete backend package/build revision for diagnostics.
        backend_kind: Backend implementation family.
        manager_kind: Manager implementation family.
        cache_dtype: Element type of the KV cache.
        attention_backend: Backend that actually writes the cache.
        attention_kind: Attention cache kind.
        layout: Physical layout actually written by the attention backend.
        parallelism: Tensor, data, pipeline and context parallel sizes, in order.
        staging: Owner and location of transfer staging memory.
        committed_whole_blocks: Whether content consists of committed whole blocks.
        extra_features: Features beyond the supported profile, including remapping,
            offload, compression, recurrent state, extra buffer roles or retries.

    These facts are supplied by runtime assembly, not inferred from a layout
    digest. Validation restricts compatibility; it does not certify a backend or
    enable a scheduler path. The backend revision is independent of the contract
    revision and must identify the actual implementation under qualification.
    """

    contract_revision: str
    backend_revision: str
    backend_kind: str
    manager_kind: str
    cache_dtype: str
    attention_backend: str
    attention_kind: str
    layout: str
    parallelism: tuple[int, int, int, int]
    staging: str
    committed_whole_blocks: bool
    extra_features: frozenset[str]

    def validate(self) -> None:
        """Reject unsupported assembly facts before acquiring or exposing resources.

        Raises:
            ValueError: The revision is missing or the profile is unsupported.
        """
        expected = {
            "contract_revision": (self.contract_revision, SHARED_CONTRACT_REVISION),
            "backend_kind": (self.backend_kind, "native_nixl"),
            "manager_kind": (self.manager_kind, "KVCacheManagerV2"),
            "cache_dtype": (self.cache_dtype, "bfloat16"),
            "attention_backend": (self.attention_backend, "TRTLLM"),
            "attention_kind": (self.attention_kind, "mha"),
            "layout": (self.layout, "HND"),
            "staging": (self.staging, "manager_host"),
        }
        for field, (actual, supported) in expected.items():
            if actual != supported:
                raise ValueError(f"unsupported shared runtime {field}: {actual!r}")
        if not isinstance(self.backend_revision, str) or not self.backend_revision.strip():
            raise ValueError("backend_revision must identify a concrete backend build")
        if (
            not isinstance(self.parallelism, tuple)
            or len(self.parallelism) != 4
            or any(type(size) is not int or size != 1 for size in self.parallelism)
        ):
            raise ValueError("shared runtime requires TP=DP=PP=CP=1")
        if self.committed_whole_blocks is not True:
            raise ValueError("shared runtime requires committed whole blocks")
        if not isinstance(self.extra_features, frozenset) or self.extra_features:
            raise ValueError(f"unsupported shared runtime extra_features: {self.extra_features!r}")


def _check_parts(parts: Sequence[Part]) -> None:
    """Validate host-region geometry before deriving slot coordinates.

    Args:
        parts: Public staging allocation descriptions.

    Raises:
        ValueError: A region is malformed or overlaps another region.
    """
    spans = []
    for part in parts:
        for value in (part.address, part.nbytes, part.slot_bytes, part.slots):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError("staging part geometry must contain positive integers")
        if part.slots * part.slot_bytes > part.nbytes:
            raise ValueError("staging part does not cover its slots")
        spans.append((int(part.address), int(part.address) + int(part.nbytes)))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("staging parts overlap")


def build_extent(
    view: RegionView, parts: Sequence[Part], *, name: bytes, is_last: bool
) -> CacheExtent:
    """Translate a ready staging lease to whole opaque content units.

    Args:
        view: Ready view returned by a public staging lease.
        parts: The same lender's fixed staging regions.
        name: Stable versioned namespace, normally ``STAGING_EXTENT_NAMESPACE``;
            never a request ID or physical allocation identity.
        is_last: Whether this extent ends the logical operation.

    Returns:
        An immutable extent preserving every lender row name byte for byte.
        Unit coordinates identify local staging slots, not token block ordinals.

    Raises:
        ValueError: Metadata is absent, out of bounds, misaligned or duplicated.
        TypeError: The extent namespace or final-extent flag is invalid.
    """
    if not isinstance(name, bytes):
        raise TypeError("extent namespace must be bytes")
    if not isinstance(is_last, bool):
        raise TypeError("is_last must be bool")
    _check_parts(parts)
    units = []
    occupied = set()
    for run in view.runs:
        if run.names is None or run.addresses is None or run.part is None:
            raise ValueError("shared extents require staging names, addresses and part indices")
        if run.part >= len(parts):
            raise ValueError("staging run refers to an unknown part")
        if np.any(run.ordinals < 0):
            raise ValueError("staging block ordinals must be nonnegative")
        part = parts[run.part]
        for index, address in enumerate(run.addresses):
            offset = int(address) - int(part.address)
            if offset < 0 or offset % part.slot_bytes:
                raise ValueError("staging row is outside or misaligned with its part")
            slot = offset // int(part.slot_bytes)
            if slot >= part.slots or offset + part.slot_bytes > part.nbytes:
                raise ValueError("staging row exceeds its part capacity")
            coordinate = (run.part, slot)
            if coordinate in occupied:
                raise ValueError("staging rows contain a duplicate physical coordinate")
            occupied.add(coordinate)
            units.append(
                Unit(name=run.names[index].tobytes(), local_group=run.layer_group, local=slot)
            )
    return CacheExtent(name=name, units=tuple(units), is_last=is_last)


def served_masks(view: RegionView, served: frozenset[bytes]) -> tuple[np.ndarray, ...]:
    """Convert delivered whole-unit names to the lender's arrival masks.

    Args:
        view: The ready staging view used to construct the submitted extent.
        served: Names reported by ``Delivered.served``; an empty set is a miss.

    Returns:
        One boolean row mask per run, suitable for ``Lease.mark_arrived``.
        These masks describe delivery only; readiness still comes from the lender
        after local copies and contiguous-prefix checks.

    Raises:
        TypeError: ``served`` is not a frozen set of byte names.
        ValueError: A view lacks names, duplicates names, or a served name was
            absent from the submitted view.
    """
    if not isinstance(served, frozenset) or any(not isinstance(name, bytes) for name in served):
        raise TypeError("served must be a frozenset of opaque byte names")
    masks = view.row_masks()
    known = set()
    for run, mask in zip(view.runs, masks):
        if run.names is None:
            raise ValueError("arrival masks require staging row names")
        for index, row in enumerate(run.names):
            name = row.tobytes()
            if name in known:
                raise ValueError("staging view contains duplicate names")
            known.add(name)
            mask[index] = name in served
    if not served <= known:
        raise ValueError("served names must be a subset of the submitted extent")
    return masks
