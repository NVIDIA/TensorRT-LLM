# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The key encoding, the driver boundary's value types, and the in-memory driver's trailer."""

from __future__ import annotations

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.memory import MemoryKind
from tensorrt_llm._torch.disaggregation.blob.keys import FORMAT_VERSION, encode_key, key_stem
from tensorrt_llm._torch.disaggregation.blob.store import (
    DriverCapabilities,
    DriverRejected,
    RegionLayout,
    RegionSpec,
    RowBatch,
    RowIO,
    RowResult,
    RowStatus,
    Span,
    StoreError,
    faulted,
)
from tensorrt_llm._torch.disaggregation.testing.driver_conformance import NAMES, SCOPE
from tensorrt_llm._torch.disaggregation.testing.object_trailer import (
    TRAILER_BYTES,
    TRAILER_VERSION,
    Checksum,
    check_trailer,
    encode_trailer,
)

pytestmark = pytest.mark.cpu_only


# Keys.


def test_keys_use_the_whole_name_and_round_trip_through_hex() -> None:
    keys = {label: encode_key("prod", name) for label, name in NAMES.items()}
    assert len(set(keys.values())) == len(NAMES)
    for label, key in keys.items():
        prefix, version, hexed = key.split("/")
        assert (prefix, version) == ("prod", f"v{FORMAT_VERSION}")
        assert bytes.fromhex(hexed) == NAMES[label]
        assert NAMES[label].startswith(SCOPE) and len(NAMES[label]) == 54


def test_key_prefix_is_a_store_partition_with_a_narrow_charset() -> None:
    for prefix in ("prod", "a_b-1", "A" * 64):
        assert encode_key(prefix, b"\x01").startswith(prefix + "/")
    for prefix in ("", "a/b", "a b", "a.b", "A" * 65, "中"):
        with pytest.raises(ValueError):
            encode_key(prefix, b"\x01")
    with pytest.raises(ValueError):
        encode_key("prod", b"")


# Value types.


def test_a_rejection_is_not_a_store_error() -> None:
    """Code written for store trouble must never swallow a rejection."""
    assert not issubclass(DriverRejected, StoreError)
    assert not issubclass(StoreError, DriverRejected)


def test_row_result_pairs_fault_text_with_fault() -> None:
    assert faulted("timeout").fault == "timeout"
    assert RowResult(RowStatus.READY).concluded and not RowResult(RowStatus.PENDING).concluded
    with pytest.raises(ValueError):
        RowResult(RowStatus.FAULT)
    with pytest.raises(ValueError):
        RowResult(RowStatus.READY, "text")


def test_span_and_layout_validate_their_numbers() -> None:
    with pytest.raises(ValueError):
        Span(-1, 8)
    with pytest.raises(ValueError):
        Span(0, 0)
    assert Span(0x1000, 64).overlaps(Span(0x1020, 64))
    assert not Span(0x1040, 8).overlaps(Span(0x1000, 64))  # adjacent, so back to back
    assert not Span(0x1000, 64).overlaps(Span(0x1040, 8))
    assert Span(0x1010, 16).within(Span(0x1000, 64))
    for slot_bytes, slot_count in ((0, 1), (8, 0), (-8, 1)):
        with pytest.raises(ValueError):
            RegionLayout(slot_bytes, slot_count)
    assert RegionLayout(8, 4).nbytes == 32
    assert RegionLayout(8, 4).memory_kind is MemoryKind.HOST
    assert RegionLayout(8, 4, "kv", MemoryKind.DEVICE).memory_kind is MemoryKind.DEVICE


def test_a_region_spec_is_a_valid_span_whose_layout_fits() -> None:
    spec = RegionSpec(0x1000, 32, RegionLayout(8, 4))
    assert spec.span == Span(0x1000, 32)
    with pytest.raises(ValueError):
        RegionSpec(0x1000, 0, RegionLayout(8, 4))
    with pytest.raises(ValueError):
        RegionSpec(0x1000, 24, RegionLayout(8, 4))  # the layout needs 32 bytes


def test_a_row_is_a_key_and_disjoint_spans_in_the_order_given() -> None:
    spans = [Span(0x2000, 8), Span(0x1000, 8)]
    row = RowIO("k", spans)  # type: ignore[arg-type]
    assert row.spans == (Span(0x2000, 8), Span(0x1000, 8))  # kept in order, not sorted
    assert row.nbytes == 16
    with pytest.raises(ValueError):
        RowIO("", (Span(0x1000, 8),))
    with pytest.raises(ValueError):
        RowIO("k", ())
    with pytest.raises(ValueError):
        RowIO("k", (Span(0x1000, 64), Span(0x1020, 64)))


def test_a_batch_of_slots_builds_nothing_until_read() -> None:
    names = np.array([[0xAB, 0x01], [0xCD, 0x02], [0xEF, 0x03]], dtype=np.uint8)
    slots = np.array([4096, 8192, 12288], dtype=np.int64)
    region = Span(4096, 3 * 4096)
    batch = RowBatch.of_slots(key_stem("p"), names, slots, 4096, region)
    assert len(batch) == 3 and batch.spans_per_row == 1 and batch.row_bytes == 4096
    # Both columns are the caller's arrays: a change to them shows, as nothing was copied or built.
    names[0] = [0x12, 0x34]
    assert np.shares_memory(batch.addresses, slots)
    assert not batch.addresses.flags.writeable  # a driver cannot write through it
    assert batch.addresses[:, 0].tolist() == [4096, 8192, 12288]
    assert batch.span_bytes == (4096,)
    assert batch.keys() == ["p/v1/1234", "p/v1/cd02", "p/v1/ef03"]
    assert batch.extents == (region,)
    assert batch[1] == RowIO("p/v1/cd02", (Span(8192, 4096),))
    assert list(batch) == [batch[0], batch[1], batch[2]]  # read row by row, as a Sequence[RowIO]


def test_a_batch_of_spans_keeps_the_listed_payload_order_without_copying() -> None:
    keys = np.array([[1], [2]], dtype=np.uint8)
    addresses = np.array([[80, 8], [96, 12]], dtype=np.int64)
    extents = (Span(80, 24), Span(8, 8))
    batch = RowBatch.of_spans("s/", keys, addresses, (8, 4), extents)
    assert np.shares_memory(batch.addresses, addresses)
    assert addresses.flags.writeable and not batch.addresses.flags.writeable
    assert batch.spans_per_row == 2 and batch.row_bytes == 12
    assert batch.extents == extents
    assert batch[0] == RowIO("s/01", (Span(80, 8), Span(8, 4)))
    assert list(batch[1:]) == [RowIO("s/02", (Span(96, 8), Span(12, 4)))]


def test_a_slice_of_a_batch_is_the_batch_of_those_rows() -> None:
    names = np.arange(8, dtype=np.uint8).reshape(4, 2)
    batch = RowBatch.of_slots("s/", names, np.arange(4, dtype=np.int64) * 8, 8, Span(0, 32))
    part = batch[1:3]
    assert isinstance(part, RowBatch) and len(part) == 2
    assert part.keys() == batch.keys()[1:3]
    assert part.addresses[:, 0].tolist() == [8, 16] and part.row_bytes == 8
    with pytest.raises(ValueError, match="contiguous"):
        batch[::2]


def test_a_batch_of_rows_is_checked() -> None:
    a = RowIO("a", (Span(0, 4),))
    b = RowIO("b", (Span(10, 4),))
    batch = RowBatch.of([a, b])
    assert batch.keys() == ["a", "b"] and list(batch) == [a, b]
    assert not batch.addresses.flags.writeable  # a driver cannot write through it
    assert batch.span_bytes == (4,) and batch.row_bytes == 4
    assert batch.extents == (Span(0, 14),)  # the smallest span holding every row
    split = RowIO("m", (Span(8, 2), Span(0, 3)))
    other = RowIO("n", (Span(20, 2), Span(4, 3)))
    pair = RowBatch.of([split, other])
    assert pair.spans_per_row == 2 and pair.span_bytes == (2, 3) and pair.row_bytes == 5
    assert pair.extents == (Span(8, 14), Span(0, 7))  # one per span, in span order
    assert pair[1] == other
    with pytest.raises(ValueError, match="at least one row"):
        RowBatch.of([])
    with pytest.raises(ValueError, match="appears twice"):
        RowBatch.of([a, RowIO("a", (Span(8, 4),))])
    with pytest.raises(ValueError, match="same span sizes"):
        RowBatch.of([a, RowIO("c", (Span(8, 6),))])  # rows of different sizes
    with pytest.raises(ValueError, match="same span sizes"):
        RowBatch.of([a, split])  # different numbers of spans


def test_capabilities_that_grant_are_declared_and_later_ones_default_off() -> None:
    with pytest.raises(TypeError):
        DriverCapabilities(get=True, put=True, cancel=False, probe=False)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        DriverCapabilities(get=False, put=False, cancel=False, probe=False, max_rows_per_op=None)
    with pytest.raises(ValueError):
        DriverCapabilities(get=True, put=True, cancel=False, probe=False, max_rows_per_op=0)
    caps = DriverCapabilities(get=True, put=False, cancel=False, probe=False, max_rows_per_op=4)
    assert (caps.hints, caps.absent_is_miss, caps.device_memory, caps.multi_span) == (
        False,
        False,
        False,
        False,
    )
    with pytest.raises(AttributeError):
        caps.put = True  # type: ignore[misc]


# The in-memory driver's trailer.


def test_trailer_round_trip_and_mismatches() -> None:
    trailer = encode_trailer(b"name", 64)
    assert len(trailer) == TRAILER_BYTES
    assert check_trailer(trailer, b"name", 64) is None
    assert check_trailer(trailer, b"other", 64) == "object name mismatch"
    assert check_trailer(trailer, b"name", 65) == "payload length 64, expected 65"
    assert check_trailer(b"\0" * TRAILER_BYTES, b"name", 64) == "bad magic"
    assert check_trailer(trailer[:-1], b"name", 64) == (
        f"trailer is {TRAILER_BYTES - 1} bytes, expected {TRAILER_BYTES}"
    )
    with pytest.raises(ValueError):
        encode_trailer(b"name", 0)


def test_trailer_rejects_another_version() -> None:
    trailer = bytearray(encode_trailer(b"name", 64))
    trailer[4:6] = (TRAILER_VERSION + 1).to_bytes(2, "little")
    assert check_trailer(bytes(trailer), b"name", 64) == (
        f"trailer version {TRAILER_VERSION + 1}, expected {TRAILER_VERSION}"
    )


@pytest.mark.parametrize("checksum", list(Checksum))
def test_every_trailer_byte_is_checked(checksum: Checksum) -> None:
    payload = bytes(range(64))
    trailer = encode_trailer(b"name", 64, checksum=checksum, payload=payload)
    assert check_trailer(trailer, b"name", 64, payload) is None
    for index in range(TRAILER_BYTES):
        flipped = bytearray(trailer)
        flipped[index] ^= 0x01
        assert check_trailer(bytes(flipped), b"name", 64, payload) is not None, index


def test_checksum_covers_the_payload() -> None:
    payload = bytes(range(64))
    trailer = encode_trailer(b"name", 64, checksum=Checksum.CRC32, payload=payload)
    assert check_trailer(trailer, b"name", 64, payload) is None
    corrupted = payload[:-1] + b"\xff"
    assert check_trailer(trailer, b"name", 64, corrupted) == "payload checksum mismatch"
    assert check_trailer(trailer, b"name", 64) == "checksum needs the payload"
    with pytest.raises(ValueError):
        encode_trailer(b"name", 64, checksum=Checksum.CRC32)
    with pytest.raises(ValueError):
        encode_trailer(b"name", 64, checksum=Checksum.CRC32, payload=payload[:-1])


def test_no_checksum_means_a_zero_checksum_field() -> None:
    plain = encode_trailer(b"name", 64)
    assert check_trailer(plain, b"name", 64, bytes(64)) is None
    tampered = plain[:-8] + (1).to_bytes(8, "little")
    assert check_trailer(tampered, b"name", 64) == "checksum set without an algorithm"
