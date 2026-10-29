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
"""Slot checksum watchdog of dense DWDP: ``TRTLLM_DWDP_DENSE_DEBUG=2``.

At H1 of every covered layer, once compute waits for the layer's copies, ``DenseDWDP.wait_layer``
has the watchdog check each slot the layer reads: on the compute stream, the checksum of the slot's
blob is compared with the P1 content checksum of the layer's segment, and a mismatch
adds one to that ``(layer, segment)``'s counter on the device.  Nothing reaches the host there.
``DenseDWDP.layer_done`` reads the counters once per forward, at the last covered layer, and
``DenseDWDP.release`` once more; each read synchronizes the host with the current stream.  Every
check reads its whole blob once more, so the watchdog is a tool for validation runs.

A correct gather leaves exactly the zero-padded blob in a slot: the row store is zero outside the
anchors, and Phase V proved that the gathered anchors equal the originals the P1 checksums were
computed from.  A mismatch therefore means the slot did not hold its layer's bytes when compute
reached the layer: a missing or misbound stream wait, a copy that wrote other bytes, or a row
store that changed after setup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .layout import checksum_bits, device_checksum
from .types import (
    Checksum,
    DenseLayout,
    DenseProtocolError,
    SegmentKey,
    SegmentKind,
    SlotBuffer,
    align_up,
)

#: Bytes per checksum word: a blob is checked up to its end rounded up to a word,
#: which the zero padding of its last row covers.
_WORD_BYTES = 4

#: Most ``(layer, segment)`` pairs one report names.
_MAX_LISTED = 8

#: ``(layer, segment, mismatches)``: a pair whose counter grew since the previous read.
Finding = tuple[int, SegmentKind, int]


class SlotWatchdog:
    """Device-side checks of the slots at H1 against the layers' P1 content checksums.

    Args:
        layers: Covered layers, ascending.
        segments: Enabled segment kinds in canonical order.
        expected: ``(layer, segment) -> (blob_bytes, checksum)`` for every covered layer and
            enabled segment: the blob size ``B`` and the P1 content checksum of the blob, whose
            word origin is the blob start.
        device: Device of the slots.

    Raises:
        ValueError: ``expected`` does not hold exactly the covered ``(layer, segment)`` pairs, or
            a blob size is negative.
    """

    def __init__(
        self,
        layers: Sequence[int],
        segments: Sequence[SegmentKind],
        expected: Mapping[SegmentKey, tuple[int, Checksum]],
        device: torch.device | str,
    ) -> None:
        self._layers = tuple(layers)
        self._segments = tuple(SegmentKind(kind) for kind in segments)
        keys = tuple((layer, kind) for layer in self._layers for kind in self._segments)
        given = {(layer, SegmentKind(kind)): value for (layer, kind), value in expected.items()}
        if set(given) != set(keys):
            missing = sorted(f"{layer}/{kind.value}" for layer, kind in set(keys) - set(given))
            extra = sorted(f"{layer}/{kind.value}" for layer, kind in set(given) - set(keys))
            raise ValueError(
                f"the slot watchdog needs a checksum for every covered (layer, segment): missing "
                f"{missing[:_MAX_LISTED]}, not covered {extra[:_MAX_LISTED]}"
            )
        check_bytes = []
        bits = []
        for layer, kind in keys:
            blob_bytes, checksum = given[(layer, kind)]
            if blob_bytes < 0:
                raise ValueError(f"layer {layer} {kind.value}: negative blob size {blob_bytes}")
            check_bytes.append(align_up(blob_bytes, _WORD_BYTES))
            bits.append(checksum_bits(checksum))
        self._keys = keys
        self._indices = {key: index for index, key in enumerate(keys)}
        self._check_bytes = tuple(check_bytes)
        self._expected: torch.Tensor | None = torch.tensor(bits, dtype=torch.int64, device=device)
        self._counts: torch.Tensor | None = torch.zeros(len(keys), dtype=torch.int64, device=device)
        self._seen = [0] * len(keys)
        self._checks = 0
        self._mismatches = 0

    @classmethod
    def from_layout(cls, layout: DenseLayout, device: torch.device | str) -> SlotWatchdog:
        """The watchdog of a collected layout whose P1 checksums are computed.

        Raises:
            DenseLayoutError: The segment checksums were not computed.
        """
        checksums = layout.checksums()
        expected = {
            key: (segment.blob_bytes, checksum)
            for (key, segment), checksum in zip(layout.segments.items(), checksums)
        }
        return cls(layout.layers, layout.segment_kinds, expected, device)

    @property
    def layers(self) -> tuple[int, ...]:
        """Covered layers, ascending."""
        return self._layers

    @property
    def segments(self) -> tuple[SegmentKind, ...]:
        """Enabled segment kinds in canonical order."""
        return self._segments

    @property
    def checks(self) -> int:
        """Slot checks enqueued so far."""
        return self._checks

    @property
    def mismatches(self) -> int:
        """Checks that found other bytes than their layer's, as of the latest :meth:`poll`."""
        return self._mismatches

    def check(self, layer: int, segment: SegmentKind, slot: SlotBuffer) -> None:
        """Enqueue the check of ``slot``, which must hold ``layer``'s ``segment`` blob.

        Runs on the current stream without synchronizing the host; a mismatch counts on the
        device until the next :meth:`poll`.

        Raises:
            DenseProtocolError: ``(layer, segment)`` is not covered, or called after
                :meth:`release`.
        """
        index = self._indices.get((layer, SegmentKind(segment)))
        if index is None:
            raise DenseProtocolError(
                f"the slot watchdog has no checksum for layer {layer} {SegmentKind(segment).value}"
            )
        counts, expected = self._counts, self._expected
        if counts is None or expected is None:
            raise DenseProtocolError("slot watchdog check after release")
        self._checks += 1
        nbytes = self._check_bytes[index]
        if nbytes:
            sums = device_checksum(slot.blob_view(nbytes))
            counts[index].add_(sums.ne(expected[index]).any())

    def poll(self) -> list[Finding]:
        """Read the counters; synchronizes the host with the current stream.

        Returns:
            ``(layer, segment, mismatches)`` for every pair whose counter grew since the
            previous read, in layer order; empty after :meth:`release`.
        """
        counts = self._counts
        if counts is None:
            return []
        findings = []
        for index, count in enumerate(counts.tolist()):
            new = count - self._seen[index]
            if new:
                self._seen[index] = count
                self._mismatches += new
                layer, kind = self._keys[index]
                findings.append((layer, kind, new))
        return findings

    def release(self) -> None:
        """Drop the device tensors; :attr:`checks` and :attr:`mismatches` keep their values."""
        self._counts = self._expected = None

    @staticmethod
    def describe(findings: Sequence[Finding]) -> str:
        """One line naming the ``(layer, segment)`` pairs of ``findings`` and their counts."""
        listed = ", ".join(
            f"layers.{layer} {kind.value} x{count}" for layer, kind, count in findings[:_MAX_LISTED]
        )
        more = len(findings) - _MAX_LISTED
        return listed + (f" and {more} more" if more > 0 else "")
