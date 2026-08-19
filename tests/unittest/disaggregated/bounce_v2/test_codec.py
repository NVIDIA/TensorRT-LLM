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
"""Wire codec tests: port of bounceMessageCodecTest.cpp, adapted to v3.

Protocol v3 carries batched ACK entries. Handshake tests are not ported — the
capability handshake stays on the C++ side (serialize_utils-based, not a
control message).
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
from conftest import load_bounce_v2

_b = load_bounce_v2()

# Fixed header field offsets (little-endian "<IHHQQIIIII", 44 bytes):
_VERSION_OFFSET = 4  # u16
_PAYLOAD_BYTES_OFFSET = 36  # u32
_HEADER_SIZE = 44


def decode_ok(blob: bytes, expect_type: "_b.BounceMsgType") -> "_b.BounceMsgHeader":
    assert _b.has_bounce_magic(blob)
    header = _b.decode_header(blob)
    assert header is not None
    assert header.msg_type == int(expect_type)
    return header


def make_runs(entries: list[tuple[int, int, int, int, int, int]]) -> np.ndarray:
    """(bounce_offset, dst_addr, dst_stride, bounce_stride, piece_size, count)."""
    runs = np.zeros(len(entries), dtype=_b.SCATTER_RUN_DTYPE)
    for i, (bo, da, ds, bs, ps, c) in enumerate(entries):
        runs[i] = (bo, da, ds, bs, ps, c)
    return runs


# C++: BounceMessageCodec.WantCarriesChunkSizesAndEndpoint
def test_want_carries_chunk_sizes_and_endpoint() -> None:
    sizes = [4096, 8192, 1024]
    ep = "tcp://10.0.0.3:5170"
    blob = _b.encode_want(42, sizes, ep)
    h = decode_ok(blob, _b.BounceMsgType.WANT)
    assert h.request_id == 42
    assert h.count == 3
    assert h.aux == 3  # num chunks
    decoded = _b.decode_want(blob, h)
    assert decoded is not None
    out, out_ep = decoded
    assert out == sizes
    assert out_ep == ep


# C++: BounceMessageCodec.CancelIsEmptyWantThatStillCarriesEndpoint
def test_cancel_is_empty_want_that_still_carries_endpoint() -> None:
    ep = "tcp://127.0.0.1:9999"
    blob = _b.encode_cancel(42, ep)  # same wire form as an empty-chunk WANT
    h = decode_ok(blob, _b.BounceMsgType.WANT)
    assert h.count == 0  # no chunks -> cancel
    assert h.aux == 0
    decoded = _b.decode_want(blob, h)
    assert decoded is not None
    out, out_ep = decoded
    assert out == []
    assert _b.is_cancel_want(out)  # decoded WANT is recognized as a cancel
    assert out_ep == ep  # endpoint travels so the receiver can bootstrap

    # A real (non-empty) WANT is NOT a cancel.
    wblob = _b.encode_want(43, [4096], ep)
    wh = decode_ok(wblob, _b.BounceMsgType.WANT)
    wdecoded = _b.decode_want(wblob, wh)
    assert wdecoded is not None
    real, _ep2 = wdecoded
    assert not _b.is_cancel_want(real)


# C++: BounceMessageCodec.WantEmptyEndpointRoundTrips
def test_want_empty_endpoint_round_trips() -> None:
    blob = _b.encode_want(7, [16], "")
    h = decode_ok(blob, _b.BounceMsgType.WANT)
    decoded = _b.decode_want(blob, h)
    assert decoded is not None
    out, out_ep = decoded
    assert out == [16]
    assert out_ep == ""


# C++: BounceMessageCodec.GrantRoundTrip
def test_grant_round_trip() -> None:
    credits = [
        _b.CreditEntry(addr=0x3000, length=256, dev_id=1, region_handle=3),
        _b.CreditEntry(addr=0x7000, length=512, dev_id=1, region_handle=7),
    ]
    blob = _b.encode_grant(99, credits)
    h = decode_ok(blob, _b.BounceMsgType.GRANT)
    assert h.request_id == 99
    out = _b.decode_credits(blob, h)
    assert out is not None
    assert len(out) == 2
    assert out[0].region_handle == 3
    assert out[0].dev_id == 1
    assert out[1].addr == 0x7000
    assert out[1].length == 512


# C++: BounceMessageCodec.DataRoundTrip
def test_data_round_trip() -> None:
    # One plain extent + one 3-piece strided run.
    runs = make_runs([(0, 0xD000, 0, 0, 128, 1), (128, 0xE000, 4096, 128, 64, 3)])
    blob = _b.encode_data(7, chunk_idx=2, num_chunks=5, region_handle=4, scatter_runs=runs)
    h = decode_ok(blob, _b.BounceMsgType.DATA)
    assert h.request_id == 7
    assert h.chunk_idx == 2
    assert h.num_chunks == 5
    assert h.region_handle == 4
    out = _b.decode_scatter(blob, h)
    assert out is not None
    assert len(out) == 2
    assert int(out[0]["dst_addr"]) == 0xD000
    assert int(out[0]["count"]) == 1
    assert int(out[1]["bounce_offset"]) == 128
    assert int(out[1]["dst_stride"]) == 4096
    assert int(out[1]["bounce_stride"]) == 128
    assert int(out[1]["piece_size"]) == 64
    assert int(out[1]["count"]) == 3


# C++: BounceMessageCodec.AckRoundTrip — adapted to v3: ACK is a BATCH of
# (chunk_idx, region_handle) entries; header per-chunk fields stay 0.
def test_ack_round_trip_single_entry() -> None:
    blob = _b.encode_ack(7, [(3, 9)])
    h = decode_ok(blob, _b.BounceMsgType.ACK)
    assert h.request_id == 7
    assert h.count == 1
    assert h.chunk_idx == 0  # per-chunk header fields unused in v3
    assert h.region_handle == 0
    out = _b.decode_ack(blob, h)
    assert out == [_b.AckEntry(chunk_idx=3, region_handle=9)]


# v3-specific: batched ACK with >1 entries round-trips in order.
def test_ack_round_trip_batched_entries() -> None:
    entries = [(0, 0x1000), (3, 0x9000), (7, 0x2000), (2, 0xFFFF_FFFF_FFFF)]
    blob = _b.encode_ack(11, entries)
    h = decode_ok(blob, _b.BounceMsgType.ACK)
    assert h.request_id == 11
    assert h.count == len(entries)
    out = _b.decode_ack(blob, h)
    assert out is not None
    assert [(e.chunk_idx, e.region_handle) for e in out] == entries


# C++: BounceMessageCodec.EmptyEntriesSerialize (+ empty ACK batch for v3)
def test_empty_entries_serialize() -> None:
    blob = _b.encode_grant(1, [])
    h = decode_ok(blob, _b.BounceMsgType.GRANT)
    assert h.count == 0
    assert h.payload_bytes == 0
    out = _b.decode_credits(blob, h)
    assert out == []

    ack_blob = _b.encode_ack(1, [])
    ah = decode_ok(ack_blob, _b.BounceMsgType.ACK)
    assert ah.count == 0
    assert _b.decode_ack(ack_blob, ah) == []


# C++: BounceMessageCodec.LargeScatterCountRoundTrips
def test_large_scatter_count_round_trips() -> None:
    # count/payload_bytes are 32-bit: a count well past 65535 must round-trip
    # intact — guards against accidental 16-bit truncation.
    n = 100_000  # 100k * 36B = 3.6 MB payload
    runs = np.zeros(n, dtype=_b.SCATTER_RUN_DTYPE)
    idx = np.arange(n, dtype=np.uint64)
    runs["bounce_offset"] = idx * 64
    runs["dst_addr"] = 0x1000_0000 + idx
    runs["piece_size"] = 64
    runs["count"] = 1
    blob = _b.encode_data(1, chunk_idx=0, num_chunks=1, region_handle=0, scatter_runs=runs)
    h = decode_ok(blob, _b.BounceMsgType.DATA)
    assert h.count == n
    assert h.payload_bytes == n * _b.SCATTER_RUN_DTYPE.itemsize
    out = _b.decode_scatter(blob, h)
    assert out is not None
    assert out.shape[0] == n
    # Spot-check the boundaries and the >16-bit index region.
    assert int(out[0]["dst_addr"]) == 0x1000_0000
    assert int(out[65535]["bounce_offset"]) == 65535 * 64
    assert int(out[n - 1]["dst_addr"]) == 0x1000_0000 + (n - 1)
    assert int(out[n - 1]["count"]) == 1


# C++: BounceMessageCodec.LargeCreditCountRoundTrips
def test_large_credit_count_round_trips() -> None:
    n = 70_000  # > 65535 credits in one GRANT
    credits = [
        _b.CreditEntry(addr=0x2000 + i, length=4096, dev_id=0, region_handle=i) for i in range(n)
    ]
    blob = _b.encode_grant(5, credits)
    h = decode_ok(blob, _b.BounceMsgType.GRANT)
    assert h.count == n
    assert h.payload_bytes == n * 24
    out = _b.decode_credits(blob, h)
    assert out is not None
    assert len(out) == n
    assert out[n - 1].region_handle == n - 1
    assert out[n - 1].addr == 0x2000 + (n - 1)


# v3-specific: a batched ACK past 65535 entries round-trips intact.
def test_large_ack_count_round_trips() -> None:
    n = 70_000
    entries = [(i, i * 64) for i in range(n)]
    blob = _b.encode_ack(5, entries)
    h = decode_ok(blob, _b.BounceMsgType.ACK)
    assert h.count == n
    assert h.payload_bytes == n * 12
    out = _b.decode_ack(blob, h)
    assert out is not None
    assert len(out) == n
    assert out[65536] == _b.AckEntry(chunk_idx=65536, region_handle=65536 * 64)
    assert out[n - 1] == _b.AckEntry(chunk_idx=n - 1, region_handle=(n - 1) * 64)


# C++: BounceMessageCodec.ShortBlobRejected
def test_short_blob_rejected() -> None:
    assert _b.decode_header(b"\x00" * 8) is None
    assert not _b.has_bounce_magic(b"\x00" * 2)


# C++: BounceMessageCodec.BadMagicRejected
def test_bad_magic_rejected() -> None:
    blob = bytearray(_b.encode_ack(1, [(0, 0)]))
    blob[0] = ord("X")  # corrupt magic
    assert _b.decode_header(bytes(blob)) is None
    assert not _b.has_bounce_magic(bytes(blob))


# Version check: a v2 (or any non-v3) blob must be refused — mixed-version
# clusters fall back to the standard path instead of mis-decoding.
def test_version_mismatch_rejected() -> None:
    blob = bytearray(_b.encode_ack(1, [(0, 0)]))
    struct.pack_into("<H", blob, _VERSION_OFFSET, _b.BOUNCE_VERSION - 1)
    assert _b.decode_header(bytes(blob)) is None
    struct.pack_into("<H", blob, _VERSION_OFFSET, _b.BOUNCE_VERSION + 1)
    assert _b.decode_header(bytes(blob)) is None


# C++: BounceMessageCodec.InflatedPayloadBytesRejected
def test_inflated_payload_bytes_rejected() -> None:
    blob = bytearray(_b.encode_ack(1, [(0, 0)]))
    # Corrupt payload_bytes to a value the blob can't satisfy.
    struct.pack_into("<I", blob, _PAYLOAD_BYTES_OFFSET, 4096)
    assert _b.decode_header(bytes(blob)) is None


# C++: BounceMessageCodec.CrossTypeDecodeMismatchRejected
def test_cross_type_decode_mismatch_rejected() -> None:
    # A WANT blob (u32 chunk sizes) decoded as credits (24B) -> size mismatch.
    blob = _b.encode_want(1, [4096, 8192], "tcp://x:1")
    h = _b.decode_header(blob)
    assert h is not None
    assert _b.decode_credits(blob, h) is None


# Truncated / garbage payloads must return None from every typed decoder.
def test_truncated_payload_rejected_by_typed_decoders() -> None:
    for blob, decoder in (
        (_b.encode_grant(1, [_b.CreditEntry(0x3000, 256, 0, 1)]), _b.decode_credits),
        (_b.encode_data(1, 0, 1, 0, make_runs([(0, 0xD000, 0, 0, 128, 1)])), _b.decode_scatter),
        (_b.encode_ack(1, [(0, 1)]), _b.decode_ack),
        (_b.encode_want(1, [4096], "tcp://x:1"), _b.decode_want),
    ):
        h = _b.decode_header(blob)
        assert h is not None
        # Cut into the payload: the header itself no longer validates, and a
        # typed decode against the ORIGINAL header must also refuse.
        truncated = blob[: _HEADER_SIZE + 2]
        assert _b.decode_header(truncated) is None
        assert decoder(truncated, h) is None


def test_garbage_blobs_rejected() -> None:
    rng = np.random.default_rng(1234)
    for size in (0, 1, 3, 4, 43, 44, 100):
        junk = rng.integers(0, 256, size=size, dtype=np.uint8).tobytes()
        assert _b.decode_header(junk) is None
    # A legacy raw endpoint string has no magic.
    assert _b.decode_header(b"tcp://10.0.0.1:5555") is None
    assert not _b.has_bounce_magic(b"tcp://10.0.0.1:5555")


def test_want_truncated_endpoint_rejected() -> None:
    blob = _b.encode_want(1, [4096], "tcp://10.0.0.1:5555")
    h = _b.decode_header(blob)
    assert h is not None
    # Endpoint cut short: header no longer validates; decode_want on the
    # shortened blob with the original header must refuse too.
    short = blob[:-4]
    assert _b.decode_header(short) is None
    assert _b.decode_want(short, h) is None


def test_non_utf8_endpoint_rejected() -> None:
    blob = bytearray(_b.encode_want(1, [], b"\xff\xfe\xfd"))
    h = _b.decode_header(bytes(blob))
    assert h is not None
    assert _b.decode_want(bytes(blob), h) is None  # junk endpoint -> refused


# ---- golden wire bytes: pin the v3 wire format itself ----
#
# The expected hex below is HAND-DERIVED from the documented layout (44-byte
# little-endian header '<IHHQQIIIII' = magic, version, msg_type, request_id,
# region_handle, chunk_idx, num_chunks, count, payload_bytes, aux; 24-byte
# credit '<QIIQ'; 36-byte scatter run; 12-byte ACK entry '<IQ'), NOT from the
# encoder — a symmetric encode/decode bug cannot pass this test.

_GOLDEN_HEADER = "32434e42" + "0300"  # magic 'BNC2' (0x424E4332 LE), version 3

_GOLDEN_MESSAGES: dict[str, str] = {
    # WANT rid=1, sizes=[4096], endpoint "ab":
    # payload = u32 4096 | u32 ep_len 2 | "ab" (10 bytes).
    "want": _GOLDEN_HEADER
    + "0100"  # msg_type WANT=1
    + "0100000000000000"  # request_id 1
    + "0000000000000000"  # region_handle 0
    + "00000000"
    + "00000000"  # chunk_idx, num_chunks
    + "01000000"  # count 1
    + "0a000000"  # payload_bytes 10
    + "01000000"  # aux = num chunks 1
    + "00100000"
    + "02000000"
    + "6162",  # 4096 | len 2 | "ab"
    # CANCEL rid=2, endpoint "x": an empty WANT (count 0, aux 0);
    # payload = u32 ep_len 1 | "x" (5 bytes).
    "cancel": _GOLDEN_HEADER
    + "0100"
    + "0200000000000000"
    + "0000000000000000"
    + "00000000"
    + "00000000"
    + "00000000"  # count 0 -> cancel
    + "05000000"  # payload_bytes 5
    + "00000000"  # aux 0
    + "01000000"
    + "78",  # ep_len 1 | "x"
    # GRANT rid=3, one credit {addr 0x1000, len 256, dev 1, handle 2}.
    "grant": _GOLDEN_HEADER
    + "0200"  # msg_type GRANT=2
    + "0300000000000000"
    + "0000000000000000"
    + "00000000"
    + "00000000"
    + "01000000"  # count 1
    + "18000000"  # payload_bytes 24
    + "00000000"
    + "0010000000000000"  # addr 0x1000
    + "00010000"  # length 256
    + "01000000"  # dev_id 1
    + "0200000000000000",  # region_handle 2
    # DATA rid=4, chunk 1/2, region_handle 0x20, one run
    # {bounce 0, dst 0xD000, dst_stride 0, bounce_stride 0, piece 128, n 1}.
    "data": _GOLDEN_HEADER
    + "0300"  # msg_type DATA=3
    + "0400000000000000"
    + "2000000000000000"  # region_handle 0x20
    + "01000000"  # chunk_idx 1
    + "02000000"  # num_chunks 2
    + "01000000"  # count 1
    + "24000000"  # payload_bytes 36
    + "00000000"
    + "0000000000000000"  # bounce_offset 0
    + "00d0000000000000"  # dst_addr 0xD000
    + "0000000000000000"  # dst_stride 0
    + "00000000"  # bounce_stride 0
    + "80000000"  # piece_size 128
    + "01000000",  # count 1
    # ACK rid=5, batched entries [(1, 0x40), (2, 0x80)].
    "ack": _GOLDEN_HEADER
    + "0400"  # msg_type ACK=4
    + "0500000000000000"
    + "0000000000000000"
    + "00000000"
    + "00000000"
    + "02000000"  # count 2
    + "18000000"  # payload_bytes 24
    + "00000000"
    + "01000000"
    + "4000000000000000"  # entry (1, 0x40)
    + "02000000"
    + "8000000000000000",  # entry (2, 0x80)
}


def _golden_encoders() -> dict[str, bytes]:
    return {
        "want": _b.encode_want(1, [4096], "ab"),
        "cancel": _b.encode_cancel(2, "x"),
        "grant": _b.encode_grant(
            3, [_b.CreditEntry(addr=0x1000, length=256, dev_id=1, region_handle=2)]
        ),
        "data": _b.encode_data(
            4,
            chunk_idx=1,
            num_chunks=2,
            region_handle=0x20,
            scatter_runs=make_runs([(0, 0xD000, 0, 0, 128, 1)]),
        ),
        "ack": _b.encode_ack(5, [(1, 0x40), (2, 0x80)]),
    }


@pytest.mark.parametrize("name", ["want", "cancel", "grant", "data", "ack"])
def test_golden_wire_bytes(name: str) -> None:
    expected_hex = _GOLDEN_MESSAGES[name]
    blob = _golden_encoders()[name]
    assert blob.hex() == expected_hex
    # And the golden bytes decode back (the decoder agrees with the layout).
    assert _b.decode_header(bytes.fromhex(expected_hex)) is not None
