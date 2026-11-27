# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded decoding of base64 ``torch.save`` tensor payloads from clients."""

import asyncio
import base64
import binascii
import io
import pickle
import re
import struct
import threading
import tracemalloc
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import pytest
import torch

from tensorrt_llm.inputs import utils
from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.inputs.utils import async_load_base64_tensor, load_base64_tensor

pytestmark = pytest.mark.cpu_only

_LIMIT = 1 << 20
_REBUILD = b"ctorch._utils\n_rebuild_tensor_v2\n"


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
    src = zipfile.ZipFile(io.BytesIO(base64.b64decode(payload)))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", **zip_kwargs) as dst:
        for info in src.infolist():
            data = (replace or {}).get(info.filename.split("/", 1)[1], src.read(info))
            if data is not None:
                dst.writestr(info.filename, data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Duplicate names are added on purpose.
            for name in add:
                dst.writestr(name, b"")
    return base64.b64encode(out.getvalue()).decode("ascii")


def _pickle(
    size: tuple,
    stride: tuple,
    numel: int,
    *,
    wrap: bool = False,
    head: bytes = b"\x80\x02",
    rebuild: bytes = _REBUILD,
) -> bytes:
    """A ``_rebuild_tensor_v2`` pickle of a float view, built opcode by opcode.

    With ``wrap``, the tensor is passed to ``OrderedDict``, which iterates its rows
    while unpickling; its hooks are then an empty tuple, keeping the opcode counts
    of an ordinary ``torch.save`` pickle.
    """

    def num(value: int) -> bytes:
        length = (value.bit_length() + 8) // 8
        return b"\x8a" + bytes([length]) + value.to_bytes(length, "little", signed=True)

    storage = (
        b"((X\x07\x00\x00\x00storagectorch\nFloatStorage\nX\x01\x00\x00\x000X\x03\x00\x00\x00cpu"
    )
    hooks = b")" if wrap else b"ccollections\nOrderedDict\n)R"
    body = rebuild + storage + num(numel) + b"tQ" + num(0)
    body += b"(" + b"".join(map(num, size)) + b"t(" + b"".join(map(num, stride)) + b"t"
    body += b"\x89" + hooks + b"tR"
    if wrap:
        body = b"ccollections\nOrderedDict\n" + body + b"\x85R"
    return head + body + b"."


def _alias(payload: str, record: str) -> str:
    """Store one record under another name, with a Unicode path field naming it back."""
    src = zipfile.ZipFile(io.BytesIO(base64.b64decode(payload)))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            name = info.filename
            copy = zipfile.ZipInfo(name + "~" if name.endswith(record) else name)
            if copy.filename != name:
                stored = binascii.crc32(copy.filename.encode())
                copy.extra = struct.pack("<HHBI", 0x7075, 5 + len(name), 1, stored) + name.encode()
            dst.writestr(copy, src.read(info))
    return base64.b64encode(out.getvalue()).decode("ascii")


def _with_pickle(pickled: bytes) -> str:
    """A torch.save archive of ``zeros(8)`` with its data.pkl record swapped."""
    return _rewrite(_encode(torch.zeros(8)), {"data.pkl": pickled})


def _add_comment(raw: bytearray, _: zipfile.ZipFile) -> None:
    """Give the end-of-central-directory record a one-byte comment."""
    raw[-2:] = struct.pack("<H", 1)
    raw += b"x"


def _patch(payload: str, edit: Callable[[bytearray, zipfile.ZipFile], None]) -> str:
    """Edit the decoded archive bytes in place and re-encode them."""
    raw = bytearray(base64.b64decode(payload))
    edit(raw, zipfile.ZipFile(io.BytesIO(bytes(raw))))
    return base64.b64encode(bytes(raw)).decode()


def _headers(raw: bytearray, archive: zipfile.ZipFile, name: str) -> tuple[int, int]:
    """Offsets of one record's local and central directory headers."""
    central = raw.find(b"PK\x01\x02")
    while raw[central + 46 : central + 46 + len(name)] != name.encode():
        if central < 0:
            raise KeyError(name)
        central = raw.find(b"PK\x01\x02", central + 4)
    return archive.getinfo(name).header_offset, central


def _set_sizes(payload: str, record: str, compress_size: int, file_size: int) -> str:
    """Overwrite one record's declared stored and uncompressed sizes in both its headers."""

    def edit(raw: bytearray, archive: zipfile.ZipFile) -> None:
        local, central = _headers(raw, archive, f"archive/{record}")
        struct.pack_into("<II", raw, local + 18, compress_size, file_size)
        struct.pack_into("<II", raw, central + 20, compress_size, file_size)

    return _patch(payload, edit)


def _encrypt(raw: bytearray, archive: zipfile.ZipFile) -> None:
    """Set the encryption flag of every record."""
    for name in archive.namelist():
        local, central = _headers(raw, archive, name)
        raw[local + 6] |= 1
        raw[central + 8] |= 1


def _deflate_method(raw: bytearray, archive: zipfile.ZipFile) -> None:
    """Mark the storage record deflated in both headers, keeping its equal sizes."""
    local, central = _headers(raw, archive, "archive/data/0")
    struct.pack_into("<H", raw, local + 8, zipfile.ZIP_DEFLATED)
    struct.pack_into("<H", raw, central + 10, zipfile.ZIP_DEFLATED)


def _corrupt(record: str) -> Callable[[bytearray, zipfile.ZipFile], None]:
    """An edit flipping a bit of one record's first byte, keeping its declared CRC."""

    def edit(raw: bytearray, archive: zipfile.ZipFile) -> None:
        local, _ = _headers(raw, archive, f"archive/{record}")
        raw[local + 30 + sum(struct.unpack_from("<HH", raw, local + 26))] ^= 1

    return edit


def _zip64_edit(
    field: int, value: Callable[[int], int], in_record: bool
) -> Callable[[bytearray, zipfile.ZipFile], None]:
    """An edit of one 8-byte field of the zip64 locator, or of the record it points to."""

    def edit(raw: bytearray, archive: zipfile.ZipFile) -> None:
        at = len(raw) - 22 - 20
        if in_record:
            at = struct.unpack_from("<Q", raw, at + 8)[0]
        struct.pack_into("<Q", raw, at + field, value(struct.unpack_from("<Q", raw, at + field)[0]))

    return edit


def _offset_past_archive(raw: bytearray, archive: zipfile.ZipFile) -> None:
    """Point data.pkl's local header at 2**64 - 1 through a zip64 extra field."""
    _, central = _headers(raw, archive, "archive/data.pkl")
    end = central + 46 + len("archive/data.pkl")
    raw[end:end] = struct.pack("<HHQ", 1, 8, 2**64 - 1)
    struct.pack_into("<H", raw, central + 30, 12)
    struct.pack_into("<I", raw, central + 42, 0xFFFFFFFF)
    # The central directory grew by 12 bytes, moving the zip64 record after it.
    _zip64_edit(8, lambda v: v + 12, False)(raw, archive)
    _zip64_edit(40, lambda v: v + 12, True)(raw, archive)
    struct.pack_into("<I", raw, len(raw) - 10, struct.unpack_from("<I", raw, len(raw) - 10)[0] + 12)


def _spy_reads(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record the names of archive records the loader opens; ZipFile.read opens them too.

    Install it after building the payload, whose construction reads records.
    """
    open_ = zipfile.ZipFile.open
    reads = []

    def spy(self: zipfile.ZipFile, name: object, *args: object, **kwargs: object) -> object:
        reads.append(getattr(name, "filename", name))
        return open_(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", spy)
    return reads


class _BigBytearray:
    """Pickles as ``bytearray(n)``: allowed by weights_only, allocates on load."""

    def __reduce__(self) -> tuple:
        return (bytearray, (1 << 30,))


_ACCEPTED = [
    (torch.randn(5, 8).to(torch.float16), {}),
    (torch.randn(5, 8).to(torch.bfloat16), {}),
    *[(torch.randn(2, 3), {"pickle_protocol": p}) for p in (3, 4, 5)],
    (torch.arange(12.0).reshape(4, 3)[1:, 1:], {}),
    (torch.arange(6.0).reshape(3, 2).t(), {}),
    (torch.arange(4.0, requires_grad=True), {}),
    (torch.zeros(0, 4), {}),
    # Storage just over one 1 MiB copy chunk, with distinct values across its end.
    (torch.arange(float((1 << 18) + 1)), {}),
]


@pytest.mark.parametrize("tensor, save_kwargs", _ACCEPTED)
def test_saved_tensors_round_trip(
    monkeypatch: pytest.MonkeyPatch, tensor: torch.Tensor, save_kwargs: dict
) -> None:
    """Dtypes, protocols, views and empty tensors decode on CPU, never through torch.load."""
    monkeypatch.setattr(torch, "load", None)
    payload = _encode(tensor, **save_kwargs)
    with torch.device("meta"):
        decoded = load_base64_tensor(payload, max_bytes=2 * _LIMIT)
    assert decoded.device.type == "cpu" and decoded.dtype == tensor.dtype
    assert torch.equal(decoded, tensor.detach())


def test_hand_built_pickle_accepted() -> None:
    """The hand-built pickle the rejection cases mutate loads when well formed."""
    decoded = load_base64_tensor(_with_pickle(_pickle((4, 2), (2, 1), 8)), max_bytes=_LIMIT)
    assert decoded.shape == (4, 2)


def test_async_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """The async wrapper decodes on the media executor, else on the loop's default one."""
    threads = []
    load = utils.load_base64_tensor

    def spy(data: str, *, max_bytes: int) -> torch.Tensor:
        threads.append(threading.current_thread().name)
        return load(data, max_bytes=max_bytes)

    async def run(pool: Optional[ThreadPoolExecutor]) -> torch.Tensor:
        with BaseMediaIO.use_executor(pool):
            return await async_load_base64_tensor(_encode(tensor), max_bytes=_LIMIT)

    monkeypatch.setattr(utils, "load_base64_tensor", spy)
    tensor = torch.arange(12.0).reshape(3, 4)
    with ThreadPoolExecutor(1, thread_name_prefix="media_test") as pool:
        assert torch.equal(asyncio.run(run(pool)), tensor)
    assert torch.equal(asyncio.run(run(None)), tensor)
    main = threading.current_thread().name
    assert threads[0].startswith("media_test") and threads[1] not in (main, threads[0])


def test_line_wrapped_base64_near_limit_accepted() -> None:
    """MIME line breaks decode like the unwrapped text and do not count toward the limit."""
    tensor = torch.randn(1000)
    payload = _encode(tensor)
    wrapped = "\r\n".join(payload[i : i + 76] for i in range(0, len(payload), 76))
    limit = len(base64.b64decode(payload))
    assert len(wrapped) > 4 * ((limit + 2) // 3)
    assert torch.equal(load_base64_tensor(wrapped, max_bytes=limit), tensor)


_BASE64_REJECTED = {
    "bytes": (b"QUJD", _LIMIT, TypeError, "base64 string"),
    "encoded-over-limit": ("A" * 5000, 1024, ValueError, "base64 characters"),
    # Within the line-wrapping allowance, but none of it is whitespace.
    "data-in-wrapping-allowance": ("A" * 1400, 1024, ValueError, "base64 data is over"),
    "decoded-over-limit": ("AAAA", 2, ValueError, "is 3 bytes, over the 2-byte limit"),
    "alphabet": ("not base64!", _LIMIT, ValueError, "not valid base64"),
    # The retry's error is reported, not the strict pass's refusal of line breaks.
    "wrapped-truncated": ("QUJD\nREVG\nRA", _LIMIT, ValueError, "not valid base64: (?!Only)"),
    "non-ascii": ("QUJDé", _LIMIT, ValueError, "not valid base64"),
}


@pytest.mark.parametrize(
    "text, limit, error, match", _BASE64_REJECTED.values(), ids=_BASE64_REJECTED
)
def test_base64_rejected(
    monkeypatch: pytest.MonkeyPatch, text: object, limit: int, error: type, match: str
) -> None:
    """Non-strings, bad characters and sizes over the limit are refused; long text undecoded."""
    monkeypatch.setattr(
        utils, "_b64decode_strict", None if limit == 1024 else utils._b64decode_strict
    )
    with pytest.raises(error, match=match):
        load_base64_tensor(text, max_bytes=limit)


def test_base64_decodes_without_strict_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Python 3.10 has no a2b_base64 strict_mode; validated b64decode is used instead."""
    calls = []

    def spy(text: str, validate: bool = False) -> bytes:
        """Python 3.10's b64decode, which ignores characters after the padding."""
        calls.append(validate)
        if validate and not re.fullmatch("[A-Za-z0-9+/]*={0,2}", text):
            raise binascii.Error("Non-base64 digit found")
        return binascii.a2b_base64(text)

    monkeypatch.setattr(utils.sys, "version_info", (3, 10, 0))
    monkeypatch.setattr(utils.base64, "b64decode", spy)
    tensor = torch.randn(4, 4)
    assert torch.equal(load_base64_tensor(_encode(tensor), max_bytes=_LIMIT), tensor)
    with pytest.raises(ValueError, match="not valid base64"):
        load_base64_tensor("not base64!", max_bytes=_LIMIT)
    assert utils._b64decode_strict("QUI=") == b"AB"
    with pytest.raises(ValueError):
        utils._b64decode_strict("QUI==")
    assert calls and all(calls)


def test_base64_decoded_in_slices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each base64 call sees a bounded slice, so a worker thread yields the GIL between them."""
    lengths = []
    a2b_base64 = utils.binascii.a2b_base64

    def spy(text: str, **kwargs: bool) -> bytes:
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
def test_sliced_decoding_matches_whole_text(monkeypatch: pytest.MonkeyPatch, tail: str) -> None:
    """Sliced decoding agrees with whole-text decoding, padding at a slice end included.

    Sizes 40..51 put the padding of a 64-character text exactly at the slice end,
    so data after it must still be refused.
    """
    monkeypatch.setattr(utils, "_BASE64_SLICE_CHARS", 64)
    for size in range(40, 52):
        raw = bytes(range(size))
        text = base64.b64encode(raw).decode()
        wrapped = "\n".join(text[i : i + 16] for i in range(0, len(text), 16))
        for strip, encoded in [(True, text + tail), (True, wrapped + tail), (not tail, text)]:
            assert utils._b64decode_strict(encoded, strip_whitespace=strip) == raw
            if text.endswith("="):
                with pytest.raises(ValueError):
                    utils._b64decode_strict(encoded + "QQ==", strip_whitespace=strip)


def test_whitespace_retry_releases_the_failed_attempt() -> None:
    """A line break near the end costs a second decode, but not a third copy of the payload."""
    raw = bytes(range(256)) * (1 << 14)
    encoded = base64.b64encode(raw).decode()
    text = encoded[:-4] + "\n" + encoded[-4:]
    if tracemalloc.is_tracing():
        pytest.skip("tracemalloc is already tracing; its peak would include the caller's.")
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="zip"):
            load_base64_tensor(text, max_bytes=len(raw))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # The decoded slices plus their join: two copies, not three.
    assert peak < 2.5 * len(raw)


_FRAMED = b"\x80\x04\x95" + (10_000).to_bytes(8, "little")
_GOOD = _pickle((4, 2), (2, 1), 8)
_BODY = _GOOD[2:]
# An outer frame holding an inner frame that holds the whole body.
_NESTED = b"\x80\x04" + b"".join(
    b"\x95" + n.to_bytes(8, "little") for n in (9 + len(_BODY), len(_BODY))
)
_TUPLE_GLOBAL = b"X\x1f\x00\x00\x00torch._utils _rebuild_tensor_v2\x85"
_PICKLE_REJECTED = {
    "allocating-global": (pickle.dumps(_BigBytearray(), protocol=2), "bytearray"),
    "opcode": (
        pickle.dumps([1.0], protocol=2),
        "unsupported pickle opcode EMPTY_LIST; .* on its own",
    ),
    "truncated": (b"\x80\x02" + _REBUILD, "malformed"),
    "missing-protocol": (_pickle((4, 2), (2, 1), 8, head=b""), "malformed"),
    "memo-on-empty-stack": (_pickle((4, 2), (2, 1), 8, head=b"\x80\x02q\x00"), "malformed"),
    "stack-underflow": (b"\x80\x02R.", "malformed"),
    "non-string-stack-global": (b"\x80\x04K\x01K\x02\x93.", "malformed"),
    "protocol": (_pickle((4, 2), (2, 1), 8, head=b"\x80\xff"), "malformed"),
    "frame-past-end": (_pickle((4, 2), (2, 1), 8, head=_FRAMED), "malformed"),
    "frame-inside-frame": (_NESTED + _BODY, "malformed"),
    # The frame ends two bytes into GLOBAL's "torch._utils" argument.
    "opcode-across-frame": (b"\x80\x04\x95" + struct.pack("<Q", 2) + _BODY, "malformed"),
    "escaped-global": (
        _pickle((4, 2), (2, 1), 8, rebuild=b"ctorch\\x2e_utils\n_rebuild_tensor_v2\n"),
        "malformed",
    ),
    "empty-tuple-hooks": (_GOOD.replace(b"ccollections\nOrderedDict\n)R", b")"), "exactly one"),
    "four-field-storage-id": (_GOOD.replace(b"X\x03\x00\x00\x00cpu", b""), "exactly one"),
    "non-storage-global-in-id": (
        _GOOD.replace(b"ctorch\nFloatStorage\n", b"ccollections\nOrderedDict\n"),
        "exactly one",
    ),
    "tuple-as-global": (_pickle((4, 2), (2, 1), 8, rebuild=_TUPLE_GLOBAL), "exactly one"),
    # OrderedDict(tensor) would unbind 1e8 zero-stride rows while unpickling.
    "reducer-wrapping-tensor": (_pickle((10**8, 2), (0, 0), 1, wrap=True), "exactly one"),
    "negative-stride": (_pickle((4, 2), (-2, 1), 8), "exactly one"),
    "negative-offset": (_GOOD.replace(b"tQ\x8a\x01\x00", b"tQ\x8a\x01\xff"), "exactly one"),
    "size-stride-lengths": (_pickle((4, 2), (2,), 8), "exactly one"),
    "out-of-range-size": (_pickle((0, 2**64), (1, 1), 4), "size, stride or offset"),
    "logical-over-limit": (_pickle((1 << 18, 2), (0, 0), 1), "describes"),
    "storage-over-limit": (_pickle((4,), (1,), _LIMIT), "describes"),
    "zero-dim-huge-product": (_pickle((10**6,) * 4 + (0,), (1,) * 5, 4), "multiply"),
    "repeats-storage": (_pickle((4, 2), (0, 1), 2), "repeats storage"),
    # Both fit in the 8-element storage, but their rows overlap.
    "overlapping-rows": (_pickle((4, 2), (1, 1), 8), "interleaves or repeats"),
    "broadcast-row": (_pickle((2, 2), (0, 1), 8), "interleaves or repeats"),
    # Four rows of two at stride 3 cover 8 elements but end at element 10.
    "past-storage": (_pickle((4, 2), (3, 1), 8), "past its storage"),
}


@pytest.mark.parametrize("pickled, match", _PICKLE_REJECTED.values(), ids=_PICKLE_REJECTED)
def test_pickle_rejected(monkeypatch: pytest.MonkeyPatch, pickled: bytes, match: str) -> None:
    """Pickles that are not one bounded dense float tensor are refused before allocating."""
    payload = _with_pickle(pickled)
    monkeypatch.setattr(torch, "empty", None)
    with pytest.raises(ValueError, match=match):
        load_base64_tensor(payload, max_bytes=_LIMIT)


@pytest.mark.parametrize(
    "tensor, match",
    [
        (torch.zeros(2, dtype=torch.int32), "unsupported dtype"),
        (torch.tensor([1 + 2j]).conj().imag, "unsupported pickle opcode"),
    ],
    ids=["int32", "negative-bit-view"],
)
def test_unsupported_saved_tensor_rejected(tensor: torch.Tensor, match: str) -> None:
    """Integer tensors and unresolved negative-bit views are refused."""
    with pytest.raises(ValueError, match=match):
        load_base64_tensor(_encode(tensor), max_bytes=_LIMIT)


_ZEROS = _encode(torch.zeros(4))
_SHORT = base64.b64encode(
    bytes(8) + struct.pack("<4sHHHHIIH", b"PK\x05\x06", 0, 0, 0, 0x4B50, 0x0706, 0, 0)
).decode()
_ARCHIVE_REJECTED = {
    "legacy-format": (_encode(torch.zeros(4), _use_new_zipfile_serialization=False), "zip"),
    "deflated": (_rewrite(_ZEROS, compression=zipfile.ZIP_DEFLATED), "uncompressed"),
    "deflated-equal-sizes": (_patch(_ZEROS, _deflate_method), "uncompressed"),
    "encrypted": (_patch(_ZEROS, _encrypt), "unencrypted"),
    "crc-mismatch": (_patch(_ZEROS, _corrupt("data/0")), "CRC"),
    "version-crc-mismatch": (_patch(_ZEROS, _corrupt("version")), "CRC"),
    "offset-past-archive": (_patch(_ZEROS, _offset_past_archive), "outside the archive"),
    "missing-pickle": (_rewrite(_ZEROS, {"data.pkl": None}), "only the records"),
    "unknown-record": (_rewrite(_ZEROS, add=["archive/notes"]), "only the records"),
    "duplicate-record": (_rewrite(_ZEROS, add=["archive/version"]), "only the records"),
    # zipfile would read archive/version~ as archive/version; torch.load would not.
    # Python 3.10's zipfile ignores the field, leaving the unknown name.
    "unicode-path-alias": (_alias(_ZEROS, "/version"), "stored as is|only the records"),
    "directory-over-limit": (_rewrite(_ZEROS, add=[f"x/{i}" for i in range(400)]), "is over"),
    "zip64-directory-over-limit": (
        _patch(_ZEROS, _zip64_edit(40, lambda _: 8192, True)),
        "is over",
    ),
    "zip64-record-moved": (_patch(_ZEROS, _zip64_edit(8, lambda v: v - 8, False)), "precede"),
    "zip64-record-unsigned": (_patch(_ZEROS, _zip64_edit(0, lambda v: v ^ 1, True)), "malformed"),
    "shorter-than-zip64-locator": (_SHORT, "zip"),
    # A comment could hide a second central directory that other readers find.
    "eocd-comment": (_patch(_ZEROS, _add_comment), "no comment"),
    # Declared record sizes summing past the limit are refused before reading.
    "records-over-limit": (_set_sizes(_ZEROS, "data/0", 2 * _LIMIT, 2 * _LIMIT), "expands to"),
    # A small archive declaring a 512 KiB storage is refused before allocating it.
    "storage-past-archive": (
        _set_sizes(
            _rewrite(_ZEROS, {"data.pkl": _pickle((4,), (1,), 1 << 17)}), "data/0", *[1 << 19] * 2
        ),
        "does not match its tensor",
    ),
}


@pytest.mark.parametrize("payload, match", _ARCHIVE_REJECTED.values(), ids=_ARCHIVE_REJECTED)
def test_archive_rejected(payload: str, match: str) -> None:
    """Archives other than the stored records torch.save writes for one tensor are refused."""
    with pytest.raises(ValueError, match=match):
        load_base64_tensor(payload, max_bytes=_LIMIT)


def test_crc_less_archive_rejected() -> None:
    """A file saved without CRC-32s is refused with the torch option that writes them."""
    enabled = torch.serialization.get_crc32_options()
    torch.serialization.set_crc32_options(False)
    try:
        payload = _encode(torch.zeros(4))
    finally:
        torch.serialization.set_crc32_options(enabled)
    with pytest.raises(ValueError, match=r"set_crc32_options\(True\)"):
        load_base64_tensor(payload, max_bytes=_LIMIT)


@pytest.mark.parametrize("max_bytes", [0, -1, 2**63, 1.5, True])
def test_invalid_max_bytes_rejected(max_bytes: object) -> None:
    """max_bytes must be a positive int that the size bounds can hold."""
    with pytest.raises(ValueError, match="max_bytes"):
        load_base64_tensor(_ZEROS, max_bytes=max_bytes)


def test_record_sizes_must_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stored record whose two declared sizes differ is refused before any record is read."""
    # zipfile would read compress_size bytes, past the declared file_size.
    size = (
        zipfile.ZipFile(io.BytesIO(base64.b64decode(_ZEROS))).getinfo("archive/data.pkl").file_size
    )
    forged = _set_sizes(_ZEROS, "data.pkl", size + 4096, size)
    reads = _spy_reads(monkeypatch)
    with pytest.raises(ValueError, match="uncompressed"):
        load_base64_tensor(forged, max_bytes=_LIMIT)
    assert reads == []


@pytest.mark.parametrize(
    "record, value, match",
    [
        ("data.pkl", b"N" * 8192, "data.pkl is"),
        ("byteorder", b"little\n", "little-endian"),
        ("byteorder", b"bigend", "little-endian"),
    ],
    ids=["pickle-over-limit", "byteorder-long", "byteorder-big"],
)
def test_metadata_record_rejected(
    monkeypatch: pytest.MonkeyPatch, record: str, value: bytes, match: str
) -> None:
    """Oversized metadata records are refused unread; a six-byte byteorder must say little."""
    payload = _rewrite(_ZEROS, {record: value})
    reads = _spy_reads(monkeypatch)
    with pytest.raises(ValueError, match=match):
        load_base64_tensor(payload, max_bytes=_LIMIT)
    assert any(name.endswith(record) for name in reads) == (value == b"bigend")
