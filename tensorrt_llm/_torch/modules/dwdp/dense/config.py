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
"""Settings of dense DWDP.

``DwdpConfig.enable_dense`` turns dense DWDP on for a launch; nothing else of it is configurable.
:class:`DenseConfig` holds what setup and the runtime read: :meth:`DenseConfig.from_dwdp_config`
builds it for a launch, with every other field at the value every deployment uses, and tests
construct it directly to exercise other values.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .types import DEFAULT_MIN_BYTES, SEGMENT_ORDER, DigestSwitches, SegmentKind

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import DwdpConfig


@dataclass(frozen=True)
class DenseConfig:
    """The dense DWDP settings of one rank.

    Attributes:
        enabled: Whether this rank asks for dense DWDP (``DwdpConfig.enable_dense``).
        segments: The segments of each covered layer whose dense weights are sharded, in
            canonical order; always starts with ``ATTN``, whose read window drives the per-layer
            prefetch.  Strings are accepted and converted.  Launches shard both segments.
        min_bytes: Weight storages smaller than this many bytes stay replicated.
        store_chunk_mb: 0: the row store is one memory handle; > 0: whole layers per handle of
            about this many MiB.
        strict: Raise instead of warning where dense DWDP works around a problem: weights setup
            cannot shard cleanly, an original weight storage still alive after the commit (every
            rank raises together), and at run time a demand fetch or a fence fallback.  Off for
            launches.
    """

    enabled: bool = False
    segments: tuple[SegmentKind, ...] = SEGMENT_ORDER
    min_bytes: int = DEFAULT_MIN_BYTES
    store_chunk_mb: int = 0
    strict: bool = False

    def __post_init__(self) -> None:
        segments = _canonical_segments(self.segments)
        object.__setattr__(self, "segments", segments)
        for name in ("min_bytes", "store_chunk_mb"):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"DenseConfig.{name} must be non-negative, got {getattr(self, name)}"
                )

    @classmethod
    def from_dwdp_config(cls, config: DwdpConfig | None) -> DenseConfig:
        """The settings of a launch: ``enabled`` from ``config.enable_dense``, the rest defaults.

        Args:
            config: The launch's ``DwdpConfig``; None turns dense DWDP off.
        """
        return cls(enabled=config is not None and config.enable_dense)

    def digest_switches(self, num_ranks: int) -> DigestSwitches:
        """The settings the layout depends on, for a group of ``num_ranks`` ranks."""
        return DigestSwitches(
            segments=self.segments,
            min_bytes=self.min_bytes,
            store_chunk_mb=self.store_chunk_mb,
            num_ranks=num_ranks,
        )


def _canonical_segments(segments: Iterable[SegmentKind | str]) -> tuple[SegmentKind, ...]:
    """``segments`` as segment kinds in canonical order.

    Raises:
        ValueError: An unknown segment, or no attention segment.
    """
    try:
        kinds = {SegmentKind(kind) for kind in segments}
    except ValueError as exc:
        raise ValueError(f"DenseConfig.segments has an unknown segment: {exc}") from None
    if SegmentKind.ATTN not in kinds:
        raise ValueError("DenseConfig.segments must include the attention segment")
    return tuple(kind for kind in SEGMENT_ORDER if kind in kinds)
