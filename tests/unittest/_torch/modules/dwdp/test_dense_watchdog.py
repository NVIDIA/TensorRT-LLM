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
"""CPU tests of the slot checksum watchdog of ``TRTLLM_DWDP_DENSE_DEBUG=2``.

* ``device_checksum`` gives the P1 checksum (``tensor_checksum``) bit for bit, on the CPU too.
* ``SlotWatchdog`` counts the checks that found other bytes per ``(layer, segment)``, reports only
  what grew since its previous read, checks a blob up to its end rounded up to a word, and keeps
  its totals after ``release``; built from a collected layout it accepts exactly the slots a
  correct gather leaves.

The hooks that drive the watchdog are tested in ``test_dense_protocol.py`` (CPU) and
``test_dense_e2e.py`` (GPU); the setup that builds it in ``test_dense_setup.py``.
"""

import pytest
import torch
from test_dense_layout import ToyModel, collect

from tensorrt_llm._torch.modules.dwdp.dense.layout import (
    checksum_bits,
    compute_checksums,
    device_checksum,
    tensor_checksum,
    write_blob,
)
from tensorrt_llm._torch.modules.dwdp.dense.types import (
    RING_DEPTH,
    ROW_ALIGN,
    DenseLayoutError,
    DenseProtocolError,
    SegmentKind,
    SlotSpec,
    allocate_slot_ring,
)
from tensorrt_llm._torch.modules.dwdp.dense.watchdog import SlotWatchdog

pytestmark = pytest.mark.cpu_only

ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE
LAYERS = (1, 2, 3)
KINDS = (ATTN, MOE)
#: Blob bytes per ``(layer, segment)``: whole words, partial words and an empty blob.
BLOB_BYTES = {
    (1, ATTN): 5000,
    (1, MOE): 0,
    (2, ATTN): 4096,
    (2, MOE): 3,
    (3, ATTN): 2 * ROW_ALIGN - 1,
    (3, MOE): 1,
}


def blob(layer: int, kind: SegmentKind) -> torch.Tensor:
    """The bytes a correct gather of ``(layer, kind)`` leaves in its slot."""
    first = 5 * layer + (kind is MOE) + 1
    return (torch.arange(BLOB_BYTES[(layer, kind)]) * 3 + first).remainder(253).to(torch.uint8)


def slot_ring(kind: SegmentKind):
    return allocate_slot_ring(SlotSpec(kind, RING_DEPTH[kind], ROW_ALIGN, 2), "cpu")


def fill(slot, layer: int, kind: SegmentKind) -> None:
    """Leave exactly what a gather of ``(layer, kind)`` leaves: the blob, zeros after it."""
    slot.storage.zero_()
    data = blob(layer, kind)
    slot.blob_view(data.numel()).copy_(data)


def make_watchdog() -> SlotWatchdog:
    expected = {key: (nbytes, tensor_checksum(blob(*key))) for key, nbytes in BLOB_BYTES.items()}
    return SlotWatchdog(LAYERS, KINDS, expected, "cpu")


def test_device_checksum_matches_the_host_checksum_on_cpu():
    """Empty and partial words, misaligned offsets, two chunks, far word origins, wrapping sums."""
    generator = torch.Generator().manual_seed(1234)
    chunk_bytes = (1 << 22) * 4
    data = torch.randint(
        0, 256, (chunk_bytes + ROW_ALIGN + 7,), dtype=torch.uint8, generator=generator
    )
    floats = torch.randn(333, 7, generator=generator)
    # Every word is -1: both unsigned sums are at least 2**63, so their int64 bits are negative.
    ones = torch.full((ROW_ALIGN,), 0xFF, dtype=torch.uint8)
    assert max(checksum_bits(tensor_checksum(ones))) < 0
    cases = [
        (ones, 9),
        (data[:0], 0),
        (data[:1], 0),
        (data[:3], 5),
        (data[:4], 0),
        (data[: ROW_ALIGN + 1], 1023),
        (data[1 : ROW_ALIGN + 2], 0),
        (data, 0),
        (data[3 : chunk_bytes + 3], (1 << 30) + 3),
        (floats, 0),
        (floats.to(torch.bfloat16), 17),
        (floats.to(torch.int32), 0),
    ]
    for tensor, word_offset in cases:
        expected = checksum_bits(tensor_checksum(tensor, word_offset=word_offset))
        actual = device_checksum(tensor, word_offset=word_offset)
        assert actual.device.type == "cpu" and actual.dtype == torch.int64
        assert tuple(actual.tolist()) == expected, f"{tuple(tensor.shape)} at word {word_offset}"
    flipped = data[: ROW_ALIGN + 1].clone()
    flipped[ROW_ALIGN // 2].bitwise_not_()
    assert not torch.equal(device_checksum(flipped), device_checksum(data[: ROW_ALIGN + 1]))
    with pytest.raises(ValueError, match="contiguous"):
        device_checksum(floats.t())
    with pytest.raises(ValueError, match="non-negative"):
        device_checksum(data, word_offset=-1)


def test_checks_count_mismatches_per_layer_and_segment():
    watchdog = make_watchdog()
    assert watchdog.layers == LAYERS and watchdog.segments == KINDS
    rings = {kind: slot_ring(kind) for kind in KINDS}
    for layer in LAYERS:
        for kind in KINDS:
            slot = rings[kind][0]
            fill(slot, layer, kind)
            watchdog.check(layer, kind, slot)
    assert (watchdog.checks, watchdog.poll(), watchdog.mismatches) == (6, [], 0)

    attn, moe = rings[ATTN][1], rings[MOE][1]
    fill(attn, 2, ATTN)  # another layer's blob
    watchdog.check(3, ATTN, attn)
    watchdog.check(3, ATTN, attn)
    fill(moe, 2, MOE)
    moe.blob_view(4)[3] = 1  # past the 3-byte blob, inside its last word: the padding is checked
    watchdog.check(2, MOE, moe)
    moe.blob_view(5)[4] = 1  # past the last word: not checked
    moe.blob_view(4)[3] = 0
    watchdog.check(2, MOE, moe)
    fill(moe, 3, MOE)
    watchdog.check(1, MOE, moe)  # an empty blob checks nothing
    assert watchdog.checks == 11
    assert watchdog.mismatches == 0, "the counters stay on the device until read"
    findings = watchdog.poll()
    assert findings == [(2, MOE, 1), (3, ATTN, 2)]
    assert watchdog.mismatches == 3
    assert SlotWatchdog.describe(findings) == "layers.2 moe x1, layers.3 attn x2"
    assert watchdog.poll() == [], "a read reports only what grew since the previous one"

    fill(attn, 1, ATTN)
    watchdog.check(3, ATTN, attn)
    assert watchdog.poll() == [(3, ATTN, 1)] and watchdog.mismatches == 4

    with pytest.raises(DenseProtocolError, match="no checksum for layer 4 attn"):
        watchdog.check(4, ATTN, attn)
    watchdog.release()
    assert watchdog.poll() == []
    assert (watchdog.checks, watchdog.mismatches) == (12, 4)
    with pytest.raises(DenseProtocolError, match="after release"):
        watchdog.check(1, ATTN, attn)
    watchdog.release()


def test_describe_lists_a_few_findings():
    findings = [(layer, ATTN, 1) for layer in range(1, 12)]
    text = SlotWatchdog.describe(findings)
    assert text.startswith("layers.1 attn x1, layers.2 attn x1")
    assert "layers.8 attn x1 and 3 more" in text and "layers.9 " not in text


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda expected: expected.pop((2, MOE)), r"missing \['2/moe'\]"),
        (lambda expected: expected.update({(4, ATTN): (8, None)}), r"not covered \['4/attn'\]"),
        (lambda expected: expected.update({(1, ATTN): (-1, None)}), "negative blob size"),
    ],
    ids=["missing", "extra", "negative"],
)
def test_the_constructor_needs_one_checksum_per_covered_pair(change, match):
    expected = {key: (nbytes, tensor_checksum(blob(*key))) for key, nbytes in BLOB_BYTES.items()}
    change(expected)
    with pytest.raises(ValueError, match=match):
        SlotWatchdog(LAYERS, KINDS, expected, "cpu")


def test_built_from_a_layout_it_accepts_exactly_the_gathered_blobs():
    """The P1 checksums are what a gather leaves in a slot: the blob, zero-padded."""
    layout = collect(ToyModel())
    with pytest.raises(DenseLayoutError, match="not computed"):
        SlotWatchdog.from_layout(layout, "cpu")
    compute_checksums(layout)
    watchdog = SlotWatchdog.from_layout(layout, "cpu")
    assert watchdog.layers == layout.layers and watchdog.segments == layout.segment_kinds
    rings = {
        kind: allocate_slot_ring(layout.slot_specs[kind], "cpu") for kind in layout.segment_kinds
    }
    for layer in layout.layers:
        for segment in layout.layer_segments(layer):
            slot = rings[segment.segment][layout.slot_index(segment.segment, layer)]
            write_blob(
                segment,
                slot.blob_view(layout.num_ranks * segment.row_bytes),
                num_ranks=layout.num_ranks,
            )
            watchdog.check(layer, segment.segment, slot)
            if segment.blob_bytes:
                slot.blob_view(segment.blob_bytes)[segment.blob_bytes - 1] ^= 0x80
                watchdog.check(layer, segment.segment, slot)
    flipped = [key for key, segment in layout.segments.items() if segment.blob_bytes]
    findings = watchdog.poll()
    assert [(layer, kind) for layer, kind, _ in findings] == flipped
    assert all(count == 1 for _, _, count in findings)
    assert watchdog.checks == len(layout.segments) + len(flipped)
