# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded decoding of base64 ``torch.save`` tensor payloads from clients."""

import asyncio
import base64
import io
import pickle
import struct
import threading
import tracemalloc
import zipfile
import zlib
from typing import Callable, Optional

import pytest
import torch

from tensorrt_llm.inputs import utils
from tensorrt_llm.inputs.utils import async_load_base64_tensor, load_base64_tensor

pytestmark = pytest.mark.cpu_only

_LIMIT = 1 << 20


def _encode(value: object, **save_kwargs: object) -> str:
    """Return base64 ``torch.save`` bytes of ``value``."""
    buf = io.BytesIO()
    torch.save(value, buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _rewrite(
    payload: str, replace: Optional[dict] = None, add: tuple = (), **zip_kwargs: object
) -> str:
    """Rewrite a torch.save archive, replacing records by name suffix and adding empty ones.

    A replacement of None drops the record.
    """
    replace = replace or {}
    src = zipfile.ZipFile(io.BytesIO(base64.b64decode(payload)))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", **zip_kwargs) as dst:
        for info in src.infolist():
            suffix = info.filename.split("/", 1)[1]
            data = replace.get(suffix, src.read(info))
            if data is not None:
                dst.writestr(info.filename, data)
        for name in add:
            dst.writestr(name, b"")
    return base64.b64encode(out.getvalue()).decode("ascii")


def _replace_pickle(payload: str, pickled: bytes) -> str:
    """Swap the data.pkl record of a torch.save archive."""
    return _rewrite(payload, {"data.pkl": pickled})


class _BigBytearray:
    """Pickles as ``bytearray(n)``: allowed by weights_only, allocates on load."""

    def __reduce__(self) -> tuple:
        return (bytearray, (1 << 30,))


def _rebuild_pickle(rows: int, stride: tuple, storage_numel: int, wrap: bool) -> bytes:
    """A protocol-2 ``_rebuild_tensor_v2`` pickle of a ``(rows, 2)`` float tensor.

    With ``wrap``, the tensor is passed to ``OrderedDict``, which iterates its rows
    while unpickling; its hooks are then an empty tuple, keeping the opcode counts
    of an ordinary ``torch.save`` pickle.
    """

    def tup(*items: int) -> bytes:
        return b"(" + b"".join(b"J" + i.to_bytes(4, "little") for i in items) + b"t"

    storage = (
        b"((X\x07\x00\x00\x00storagectorch\nFloatStorage\nX\x01\x00\x00\x000"
        b"X\x03\x00\x00\x00cpu" + b"J" + storage_numel.to_bytes(4, "little") + b"tQ"
    )
    hooks = b")" if wrap else b"ccollections\nOrderedDict\n)R"
    rebuild = (
        b"ctorch._utils\n_rebuild_tensor_v2\n"
        + storage
        + b"K\x00"
        + tup(rows, 2)
        + tup(*stride)
        + b"\x89"
        + hooks
        + b"tR"
    )
    if wrap:
        return b"\x80\x02ccollections\nOrderedDict\n" + rebuild + b"\x85R."
    return b"\x80\x02" + rebuild + b"."


def _view_pickle(offset: int, size: tuple, stride: tuple, storage_numel: int) -> bytes:
    """A protocol-2 ``_rebuild_tensor_v2`` pickle of a float view with arbitrary ints."""

    def num(value: int) -> bytes:
        length = (value.bit_length() + 8) // 8
        return b"\x8a" + bytes([length]) + value.to_bytes(length, "little", signed=True)

    return (
        b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n((X\x07\x00\x00\x00storage"
        b"ctorch\nFloatStorage\nX\x01\x00\x00\x000X\x03\x00\x00\x00cpu"
        + num(storage_numel)
        + b"tQ"
        + num(offset)
        + b"("
        + b"".join(num(v) for v in size)
        + b"t("
        + b"".join(num(v) for v in stride)
        + b"t\x89ccollections\nOrderedDict\n)RtR."
    )


def _patch_record_sizes(payload: str, record: str, compress_size: int, file_size: int) -> str:
    """Overwrite one record's declared stored and uncompressed sizes in both its headers."""
    raw = bytearray(base64.b64decode(payload))
    name = f"archive/{record}".encode()
    info = zipfile.ZipFile(io.BytesIO(bytes(raw))).getinfo(name.decode())
    struct.pack_into("<II", raw, info.header_offset + 18, compress_size, file_size)
    entry = raw.find(b"PK\x01\x02")
    while entry != -1:
        name_len = struct.unpack_from("<H", raw, entry + 28)[0]
        if raw[entry + 46 : entry + 46 + name_len] == name:
            struct.pack_into("<II", raw, entry + 20, compress_size, file_size)
        entry = raw.find(b"PK\x01\x02", entry + 4)
    return base64.b64encode(bytes(raw)).decode()


def _spy_reads(monkeypatch) -> list:
    """Record the names of archive records the loader reads."""
    read = zipfile.ZipFile.read
    reads = []

    def spy(self: zipfile.ZipFile, name: str, *args: object) -> bytes:
        reads.append(name)
        return read(self, name, *args)

    monkeypatch.setattr(zipfile.ZipFile, "read", spy)
    return reads


@pytest.mark.parametrize(
    "tensor, save_kwargs",
    [
        (torch.randn(5, 8).to(torch.float16), {}),
        (torch.randn(5, 8).to(torch.bfloat16), {}),
        (torch.randn(5, 8), {}),
        *[(torch.randn(2, 3), {"pickle_protocol": p}) for p in (3, 4, 5)],
        (torch.arange(12.0).reshape(4, 3)[::2], {}),
        (torch.arange(12.0).reshape(4, 3)[1:, 1:], {}),
        (torch.arange(6.0).reshape(3, 2).t(), {}),
        (torch.arange(10.0)[3:7], {}),
        (torch.arange(4.0, requires_grad=True), {}),
        (torch.zeros(0, 4), {}),
    ],
)
def test_saved_tensors_round_trip(monkeypatch, tensor: torch.Tensor, save_kwargs: dict) -> None:
    """Dtypes, pickle protocols, views and empty tensors decode on CPU without torch.load."""

    def no_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("torch.load must not run on client bytes")

    monkeypatch.setattr(torch, "load", no_load)
    payload = _encode(tensor, **save_kwargs)
    # The default device does not leak into the result, even for empty storage.
    with torch.device("meta"):
        decoded = load_base64_tensor(payload, max_bytes=_LIMIT)
    assert decoded.device.type == "cpu" and decoded.dtype == tensor.dtype
    assert torch.equal(decoded, tensor.detach())


def test_async_round_trip(monkeypatch) -> None:
    """The async wrapper decodes in a worker thread, off the event loop."""
    threads = []
    load = utils.load_base64_tensor

    def spy(data: str, *, max_bytes: int) -> torch.Tensor:
        threads.append(threading.get_ident())
        return load(data, max_bytes=max_bytes)

    async def run() -> torch.Tensor:
        threads.append(threading.get_ident())
        return await async_load_base64_tensor(_encode(tensor), max_bytes=_LIMIT)

    monkeypatch.setattr(utils, "load_base64_tensor", spy)
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert torch.equal(asyncio.run(run()), tensor)
    assert len(threads) == 2 and threads[0] != threads[1]


def test_encoded_length_checked_before_decode(monkeypatch) -> None:
    """An over-long string is rejected without decoding it."""

    def no_decode(*args, **kwargs) -> bytes:
        raise AssertionError("an over-long payload was decoded")

    monkeypatch.setattr(utils, "_b64decode_strict", no_decode)
    with pytest.raises(ValueError, match="base64 characters"):
        load_base64_tensor("A" * 5000, max_bytes=1024)


def test_decoded_size_over_limit() -> None:
    """Decoded bytes over the limit are rejected even when the text length passes."""
    with pytest.raises(ValueError, match="is 3 bytes, over the 2-byte limit"):
        load_base64_tensor("AAAA", max_bytes=2)


def test_line_wrapped_base64_near_limit_accepted() -> None:
    """MIME line breaks decode like the unwrapped text and do not count toward the limit."""
    tensor = torch.randn(1000)
    payload = _encode(tensor)
    wrapped = "\r\n".join(payload[i : i + 76] for i in range(0, len(payload), 76))
    limit = len(base64.b64decode(payload))
    assert len(wrapped) > 4 * ((limit + 2) // 3)
    assert torch.equal(load_base64_tensor(wrapped, max_bytes=limit), tensor)


def test_whitespace_retry_releases_the_failed_attempt() -> None:
    """A trailing newline costs a second decode, but not a third copy of the payload."""
    raw = bytes(range(256)) * (1 << 14)
    text = base64.b64encode(raw).decode() + "\n"
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="zip"):
            load_base64_tensor(text, max_bytes=len(raw))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # The decoded slices plus their join: two copies, not three.
    assert peak < 2.5 * len(raw)


@pytest.mark.parametrize("payload", ["not base64!", "QUJD\nRE!G", "QUJDé"])
def test_invalid_base64(payload: str) -> None:
    """Non-alphabet characters are rejected, with or without line breaks."""
    with pytest.raises(ValueError, match="not valid base64"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


def test_non_string_payload() -> None:
    """Bytes are not accepted in place of base64 text."""
    with pytest.raises(TypeError, match="base64 string"):
        load_base64_tensor(b"QUJD", max_bytes=_LIMIT)


def test_base64_decodes_without_strict_mode(monkeypatch) -> None:
    """Python 3.10 has no a2b_base64 strict_mode; validated b64decode is used instead."""
    calls = []
    b64decode = base64.b64decode

    def spy(text: str, validate: bool = False) -> bytes:
        """Record the call and decode as ``base64.b64decode`` does."""
        calls.append(validate)
        return b64decode(text, validate=validate)

    monkeypatch.setattr(utils.sys, "version_info", (3, 10, 0))
    monkeypatch.setattr(utils.base64, "b64decode", spy)
    tensor = torch.randn(4, 4)
    assert torch.equal(load_base64_tensor(_encode(tensor), max_bytes=_LIMIT), tensor)
    with pytest.raises(ValueError, match="not valid base64"):
        load_base64_tensor("not base64!", max_bytes=_LIMIT)
    assert calls and all(calls)


def test_base64_decoded_in_slices(monkeypatch) -> None:
    """Each base64 call sees a bounded slice, so a worker thread yields the GIL between them."""
    lengths = []
    a2b_base64 = utils.binascii.a2b_base64

    def spy(text: str, **kwargs: bool) -> bytes:
        """Record the slice length and decode as ``binascii.a2b_base64`` does."""
        lengths.append(len(text))
        return a2b_base64(text, **kwargs)

    monkeypatch.setattr(utils, "_BASE64_SLICE_CHARS", 64)
    monkeypatch.setattr(utils.binascii, "a2b_base64", spy)
    tensor = torch.randn(100)
    payload = _encode(tensor)
    wrapped = "\n".join(payload[i : i + 76] for i in range(0, len(payload), 76))
    for text in (payload, wrapped):
        assert torch.equal(load_base64_tensor(text, max_bytes=_LIMIT), tensor)
    # A slice may carry up to three characters of an incomplete quartet.
    assert len(lengths) > 2 * len(payload) // 64 and max(lengths) <= 64 + 3


@pytest.mark.parametrize("tail", ["", "\n", "\r\n" * 40])
def test_sliced_decoding_matches_whole_text(monkeypatch, tail: str) -> None:
    """Sliced decoding agrees with whole-text decoding, padding at a slice end included.

    Sizes 40..51 put the padding of a 64-character text exactly at the slice end,
    so data after it must still be refused.
    """
    monkeypatch.setattr(utils, "_BASE64_SLICE_CHARS", 64)
    for size in range(40, 52):
        raw = bytes(range(size))
        text = base64.b64encode(raw).decode()
        wrapped = "\n".join(text[i : i + 16] for i in range(0, len(text), 16))
        assert utils._b64decode_strict(text + tail, strip_whitespace=True) == raw
        assert utils._b64decode_strict(wrapped + tail, strip_whitespace=True) == raw
        if not tail:
            assert utils._b64decode_strict(text) == raw
        if text.endswith("="):
            with pytest.raises(ValueError):
                utils._b64decode_strict(text + tail + "QQ==", strip_whitespace=True)
            if not tail:
                with pytest.raises(ValueError):
                    utils._b64decode_strict(text + "QQ==")


@pytest.mark.parametrize(
    "build, match",
    [
        pytest.param(
            lambda: _encode({"cls": pickle.Pickler}),
            "unsupported pickle opcode|references",
            id="arbitrary-global",
        ),
        pytest.param(
            lambda: _replace_pickle(
                _encode(torch.zeros(2)), pickle.dumps(_BigBytearray(), protocol=2)
            ),
            "bytearray",
            id="allocating-reducer",
        ),
        pytest.param(lambda: _encode([torch.zeros(2)]), "unsupported pickle opcode", id="list"),
        pytest.param(
            lambda: _replace_pickle(_encode(torch.zeros(2)), b"\x80\x02ctorch._utils\n"),
            "malformed",
            id="truncated",
        ),
        pytest.param(
            lambda: _encode(torch.tensor([1 + 2j]).conj().imag), "", id="negative-bit-view"
        ),
        pytest.param(
            # A plain tuple in place of the rebuild GLOBAL is not a callable.
            lambda: _replace_pickle(
                _encode(torch.zeros(8)),
                _rebuild_pickle(4, (2, 1), 8, wrap=False).replace(
                    b"ctorch._utils\n_rebuild_tensor_v2\n",
                    b"X\x1f\x00\x00\x00torch._utils _rebuild_tensor_v2\x85",
                ),
            ),
            "exactly one tensor",
            id="tuple-as-global",
        ),
        pytest.param(
            lambda: _replace_pickle(
                _encode(torch.zeros(8)),
                b"\x80\xff" + _rebuild_pickle(4, (2, 1), 8, wrap=False)[2:],
            ),
            "malformed",
            id="protocol-255",
        ),
        pytest.param(
            lambda: _replace_pickle(
                _encode(torch.zeros(8)),
                b"\x80\x04\x95"
                + (10_000).to_bytes(8, "little")
                + _rebuild_pickle(4, (2, 1), 8, wrap=False)[2:],
            ),
            "malformed",
            id="frame-past-end",
        ),
        pytest.param(
            lambda: _encode(torch.zeros(2, 2, dtype=torch.int32)), "unsupported dtype", id="int32"
        ),
        pytest.param(lambda: _encode(torch.zeros(4, 4).to_sparse()), "", id="sparse"),
        pytest.param(
            lambda: _replace_pickle(
                _encode(torch.zeros(1)), _rebuild_pickle(100_000_000, (0, 0), 1, wrap=True)
            ),
            "rebuild exactly one tensor",
            id="reducer-wrapping-tensor",
        ),
        pytest.param(
            lambda: _encode(torch.zeros(1, 8).expand(2, 8)), "repeats storage", id="expanded-view"
        ),
        pytest.param(
            lambda: _encode(torch.zeros(1, 1024).expand(4096, 1024)),
            "describes",
            id="view-over-limit",
        ),
        pytest.param(
            lambda: _replace_pickle(_encode(torch.zeros(4)), _view_pickle(0, (4,), (1,), _LIMIT)),
            "describes",
            id="storage-over-limit",
        ),
        pytest.param(
            # Four rows of two at stride 3 cover 8 elements but end at element 10.
            lambda: _replace_pickle(
                _encode(torch.zeros(8)), _rebuild_pickle(4, (3, 1), 8, wrap=False)
            ),
            "past its storage",
            id="view-past-storage",
        ),
        *[
            pytest.param(
                lambda o=o, s=s, t=t: _replace_pickle(
                    _encode(torch.zeros(4)), _view_pickle(o, s, t, 4)
                ),
                "size, stride or offset",
                id=f"out-of-range-{name}",
            )
            for name, o, s, t in [
                ("size", 0, (0, 2**64), (1, 1)),
                ("stride", 0, (1, 4), (2**64, 1)),
                ("offset", 2**64, (0,), (1,)),
            ]
        ],
        pytest.param(
            lambda: _replace_pickle(
                _encode(torch.zeros(4)), _view_pickle(0, (1_000_000,) * 4 + (0,), (1,) * 5, 4)
            ),
            "multiply",
            id="zero-dim-huge-product",
        ),
    ],
)
def test_pickle_rejected(build: Callable[[], str], match: str) -> None:
    """Pickles that are not one bounded dense float tensor are refused before allocating."""
    payload = build()
    with pytest.raises(ValueError, match=match):
        load_base64_tensor(payload, max_bytes=_LIMIT)


def test_hand_built_pickle_accepted() -> None:
    """The hand-built rebuild pickle the rejection cases mutate loads when well formed."""
    payload = _replace_pickle(_encode(torch.zeros(8)), _rebuild_pickle(4, (2, 1), 8, wrap=False))
    assert load_base64_tensor(payload, max_bytes=_LIMIT).shape == (4, 2)


def _storage_crc_mismatch() -> str:
    """Overwrite the storage bytes in place, keeping the record's declared CRC."""
    raw = bytearray(base64.b64decode(_encode(torch.zeros(4))))
    info = zipfile.ZipFile(io.BytesIO(bytes(raw))).getinfo("archive/data/0")
    name_len, extra_len = struct.unpack_from("<HH", raw, info.header_offset + 26)
    start = info.header_offset + 30 + name_len + extra_len
    raw[start : start + 16] = struct.pack("<4f", 1, 2, 3, 4)
    return base64.b64encode(bytes(raw)).decode()


def _encrypted_flag() -> str:
    """Set the encryption flag on every record, in both of its headers."""
    raw = bytearray(base64.b64decode(_encode(torch.zeros(4))))
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        header = raw.find(signature)
        while header != -1:
            raw[header + flag_offset] |= 0x1
            header = raw.find(signature, header + 4)
    return base64.b64encode(bytes(raw)).decode()


def _zip64_locator_moved() -> str:
    """Point the zip64 locator 8 bytes before its directory record."""
    raw = bytearray(base64.b64decode(_encode(torch.zeros(2))))
    locator = len(raw) - 22 - 20
    assert raw[locator : locator + 4] == b"PK\x06\x07"
    (reloff,) = struct.unpack_from("<Q", raw, locator + 8)
    struct.pack_into("<Q", raw, locator + 8, reloff - 8)
    return base64.b64encode(bytes(raw)).decode()


@pytest.mark.parametrize(
    "build, max_bytes, match",
    [
        pytest.param(
            lambda: _encode(torch.zeros(2, 2), _use_new_zipfile_serialization=False),
            _LIMIT,
            "zip",
            id="legacy-format",
        ),
        pytest.param(
            lambda: _rewrite(_encode(torch.zeros(16, 16)), compression=zipfile.ZIP_DEFLATED),
            _LIMIT,
            "uncompressed",
            id="deflated",
        ),
        pytest.param(_storage_crc_mismatch, _LIMIT, "CRC", id="crc-mismatch"),
        pytest.param(_encrypted_flag, _LIMIT, "unencrypted", id="encrypted"),
        pytest.param(
            lambda: _rewrite(_encode(torch.zeros(2)), {"data.pkl": None}),
            _LIMIT,
            "only the records",
            id="missing-pickle",
        ),
        pytest.param(_zip64_locator_moved, _LIMIT, "directly precede its locator", id="zip64"),
        pytest.param(
            lambda: base64.b64encode(
                bytes(8) + struct.pack("<4sHHHHIIH", b"PK\x05\x06", 0, 0, 0, 0x4B50, 0x0706, 0, 0)
            ).decode(),
            _LIMIT,
            "",
            id="shorter-than-zip64-locator",
        ),
        pytest.param(
            # A small archive declaring a 4 MiB storage is refused before allocating it.
            lambda: _patch_record_sizes(
                _replace_pickle(_encode(torch.zeros(4)), _view_pickle(0, (4,), (1,), 1 << 20)),
                "data/0",
                4 << 20,
                4 << 20,
            ),
            8 << 20,
            "does not match its tensor",
            id="storage-past-archive",
        ),
        pytest.param(
            # Declared record sizes summing past the limit are refused before reading.
            lambda: _patch_record_sizes(_encode(torch.zeros(4)), "data/0", 2 * _LIMIT, 2 * _LIMIT),
            _LIMIT,
            "expands to",
            id="records-over-limit",
        ),
        pytest.param(
            lambda: _rewrite(_encode(torch.zeros(2)), add=[f"junk/{i}" for i in range(20_000)]),
            8 << 20,
            "central directory is over",
            id="directory-over-limit",
        ),
        pytest.param(
            lambda: _rewrite(_encode(torch.zeros(2)), add=["archive/notes"]),
            _LIMIT,
            "only the records",
            id="unknown-record",
        ),
        pytest.param(
            lambda: _rewrite(_encode(torch.zeros(2)), add=["archive/data/1"]),
            _LIMIT,
            "only the records",
            id="second-storage",
        ),
    ],
)
def test_archive_rejected(build: Callable[[], str], max_bytes: int, match: str) -> None:
    """Archives other than the stored records torch.save writes for one tensor are refused."""
    payload = build()
    with pytest.raises(ValueError, match=match):
        load_base64_tensor(payload, max_bytes=max_bytes)


@pytest.mark.parametrize("record", ["data.pkl", "byteorder", "data/0"])
def test_record_sizes_must_agree(monkeypatch, record: str) -> None:
    """A stored record whose two declared sizes differ is refused before any record is read."""
    payload = _encode(torch.zeros(4096))
    info = zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))).getinfo(f"archive/{record}")
    # zipfile would read compress_size bytes, past the declared file_size.
    forged = _patch_record_sizes(payload, record, info.file_size + 4096, info.file_size)
    reads = _spy_reads(monkeypatch)
    with pytest.raises(ValueError, match="uncompressed"):
        load_base64_tensor(forged, max_bytes=_LIMIT)
    assert reads == []


def test_oversized_pickle_record_rejected_unread(monkeypatch) -> None:
    """A data.pkl record over the pickle limit is refused from its size, before it is read."""
    pickled = pickle.dumps(torch.zeros(1).tolist(), protocol=2)[:-1] + b"N" * 8192 + b"."
    payload = _replace_pickle(_encode(torch.zeros(2)), pickled)
    reads = _spy_reads(monkeypatch)
    with pytest.raises(ValueError, match="pickle is"):
        load_base64_tensor(payload, max_bytes=_LIMIT)
    assert not any(name.endswith("data.pkl") for name in reads)


@pytest.mark.parametrize("value", [b"little\n", b"bigend"])
def test_byteorder_record_must_be_little(monkeypatch, value: bytes) -> None:
    """A longer byteorder record is refused unread; a six-byte one must say little."""
    payload = _rewrite(_encode(torch.zeros(4)), {"byteorder": value})
    reads = _spy_reads(monkeypatch)
    with pytest.raises(ValueError, match="little-endian"):
        load_base64_tensor(payload, max_bytes=_LIMIT)
    assert any(name.endswith("byteorder") for name in reads) == (len(value) == 6)


def _zip_member(name: str, data: bytes, offset: int, method: int = 0, size: int = 0) -> tuple:
    """Local and central directory headers of one record (``size`` for deflated data)."""
    raw = name.encode()
    crc = zlib.crc32(data if not method else bytes(size)) & 0xFFFFFFFF
    usize = size or len(data)
    local = struct.pack(
        "<IHHHHHIIIHH", 0x04034B50, 20, 0, method, 0, 0, crc, len(data), usize, len(raw), 0
    )
    central = struct.pack(
        "<IHHHHHHIIIHHHHHII",
        0x02014B50,
        20,
        20,
        0,
        method,
        0,
        0,
        crc,
        len(data),
        usize,
        len(raw),
        0,
        0,
        0,
        0,
        0,
        offset,
    )
    return local + raw + data, central + raw


def _benign_members() -> list:
    """The records torch.save writes for ``zeros(2, 2)``."""
    src = zipfile.ZipFile(io.BytesIO(base64.b64decode(_encode(torch.zeros(2, 2)))))
    return [(info.filename, src.read(info)) for info in src.infolist()]


def test_legacy_stream_before_zip_ignored() -> None:
    """A legacy torch.save stream at byte 0 ahead of a valid zip is never read."""
    legacy = io.BytesIO()
    torch.save(torch.arange(4), legacy, _use_new_zipfile_serialization=False)
    body, central = bytearray(legacy.getvalue()), bytearray()
    for name, data in _benign_members():
        # The version record claims offset 0, where torch.load would see the legacy magic.
        local, entry = _zip_member(name, data, 0 if name.endswith("/version") else len(body))
        body += local
        central += entry
    count = len(_benign_members())
    eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, count, count, len(central), len(body), 0)
    payload = base64.b64encode(bytes(body + central + eocd)).decode()
    decoded = load_base64_tensor(payload, max_bytes=_LIMIT)
    assert decoded.dtype == torch.float32 and torch.equal(decoded, torch.zeros(2, 2))


def test_hidden_central_directory_rejected() -> None:
    """A second central directory, reachable only through stored offsets, is refused.

    The EOCD overstates the central directory size and pads with a comment, so
    zipfile reads the benign directory shifted back while miniz would read one
    hidden in the last entry's comment, pointing at a deflated 16 MiB storage.
    """
    hidden_numel = 4 << 20
    pickled = _rebuild_pickle(hidden_numel // 2, (2, 1), hidden_numel, wrap=False)
    deflate = zlib.compressobj(9, zlib.DEFLATED, -15)
    compressed = deflate.compress(bytes(4 * hidden_numel)) + deflate.flush()
    body, visible, hidden = bytearray(), [], b""
    for name, data in _benign_members():
        visible.append((name, data, len(body)))
        body += _zip_member(name, data, 0)[0]
    for name, data, method, size in [
        ("archive/data.pkl", pickled, 0, 0),
        ("archive/version", b"3\n", 0, 0),
        ("archive/data/0", compressed, 8, 4 * hidden_numel),
    ]:
        local, entry = _zip_member(name, data, len(body), method, size)
        hidden += entry
        body += local
    entries = [_zip_member(name, data, 0)[1] for name, data, _ in visible]
    shift = sum(len(e) for e in entries)
    central = b""
    for i, (name, data, offset) in enumerate(visible):
        entry = bytearray(_zip_member(name, data, offset + shift)[1])
        if i == len(visible) - 1:
            entry[32:34] = struct.pack("<H", len(hidden))
            entry += hidden
        central += bytes(entry)
    eocd = struct.pack(
        "<IHHHHIIH", 0x06054B50, 0, 0, 3, 3, len(central), len(body) + shift, shift
    ) + bytes(shift)
    payload = base64.b64encode(bytes(body) + central + eocd).decode()
    with pytest.raises(ValueError, match="no comment"):
        load_base64_tensor(payload, max_bytes=_LIMIT)
