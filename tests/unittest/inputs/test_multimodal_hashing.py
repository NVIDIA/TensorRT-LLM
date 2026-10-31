# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Streamed multimodal content hashing must keep the previous hash identity.

`_REFERENCE_SERIALIZE` below is a frozen copy of the earlier `serialize_item`,
which built every payload with `tobytes()`. Each test checks that the streamed
hash is byte-identical to hashing that reference output.
"""

import hashlib
import struct

import numpy as np
import pytest
import torch
from blake3 import blake3
from PIL import Image

from tensorrt_llm.inputs import multimodal_data as md
from tensorrt_llm.inputs.multimodal import _HASH_SCHEME_TAG, apply_mm_hashes
from tensorrt_llm.inputs.multimodal_data import (
    _VIDEO_HASH_METADATA_FIELDS,
    AudioData,
    VideoData,
    serialize_item,
)

pytestmark = pytest.mark.cpu_only


def _u8(value):
    return value.to_bytes(1, "big", signed=False)


def _u32(value):
    return value.to_bytes(4, "big", signed=False)


def _u64(value):
    return value.to_bytes(8, "big", signed=False)


def _len_prefixed(payload):
    return _u64(len(payload)) + payload


def _reference_serialize(obj):
    """Frozen copy of the previous `tobytes()`-based `serialize_item`."""
    if isinstance(obj, str):
        return _u8(0x01) + _len_prefixed(obj.encode("utf-8"))
    if isinstance(obj, bytes):
        return _u8(0x02) + _len_prefixed(obj)
    if isinstance(obj, bool):
        return _u8(0x05) + _u8(1 if obj else 0)
    if isinstance(obj, int):
        nbytes = (obj.bit_length() + 8) // 8
        return _u8(0x03) + _u8(nbytes) + obj.to_bytes(nbytes, "big", signed=True)
    if isinstance(obj, float):
        return _u8(0x04) + struct.pack(">d", obj)
    if isinstance(obj, Image.Image):
        width, height = obj.size
        payload = np.array(obj.convert("RGBA")).tobytes()
        return (
            _u8(0x10)
            + _len_prefixed(obj.mode.encode("utf-8"))
            + _u32(width)
            + _u32(height)
            + _len_prefixed(payload)
        )
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().contiguous().numpy()
        array = np.ascontiguousarray(obj)
        parts = [
            _u8(0x11),
            _len_prefixed(array.dtype.str.encode("utf-8")),
            _u8(array.ndim),
        ]
        parts.extend(_u64(dim) for dim in array.shape)
        parts.append(_len_prefixed(array.tobytes()))
        return b"".join(parts)
    if isinstance(obj, (tuple, list)):
        parts = [_u8(0x20), _u64(len(obj))]
        parts.extend(_reference_serialize(item) for item in obj)
        return b"".join(parts)
    if isinstance(obj, dict):
        parts = [_u8(0x22), _u64(len(obj))]
        for key in sorted(obj):
            parts.append(_reference_serialize(key))
            parts.append(_reference_serialize(obj[key]))
        return b"".join(parts)
    if isinstance(obj, np.generic):
        return _reference_serialize(obj.item())
    raise ValueError(f"Unsupported object type: {type(obj)}")


def _reference_content(item):
    """Bytes the previous `_update_hash` fed to the hasher for one item."""
    out = [_HASH_SCHEME_TAG]
    if isinstance(item, AudioData):
        out += [b"<audio>", _reference_serialize((item.samples, item.sample_rate))]
    elif isinstance(item, VideoData):
        meta = {k: item.metadata[k] for k in _VIDEO_HASH_METADATA_FIELDS if k in item.metadata}
        out += [b"<video>", _reference_serialize(meta)]
        if item.raw_bytes_hash is not None:
            out += [b"<raw_bytes>", item.raw_bytes_hash.encode("utf-8")]
        else:
            for frame in item.frames:
                out += [b"<frame>", _reference_serialize(frame)]
        if item.audio is not None:
            out += [b"<audio>", _reference_serialize((item.audio.samples, item.audio.sample_rate))]
    else:
        out.append(_reference_serialize(item))
    return b"".join(out)


class _BytesOnlySha256:
    """Custom hasher that only accepts `bytes`, as `ContentHasher` promises."""

    def __init__(self):
        self._hasher = hashlib.sha256()

    def update(self, data):
        if type(data) is not bytes:
            raise TypeError(f"expected bytes, got {type(data).__name__}")
        self._hasher.update(data)

    def hexdigest(self):
        return self._hasher.hexdigest()


class _BytesSubclass(bytes):
    def __bytes__(self):
        return b"overridden conversion"


_HASH_LIBS = [blake3, hashlib.sha256, _BytesOnlySha256, hashlib.blake2b, hashlib.blake2s]
_HASH_IDS = ["blake3", "sha256", "bytes-only", "blake2b", "blake2s"]
_UUIDS = [None, "", "uid-한글-" + "a" * 80]

_SCALAR_CASES = [
    pytest.param(item, id=f"scalar-{index}")
    for index, item in enumerate(
        [
            "",
            "한글",
            b"",
            b"abc",
            False,
            True,
            0,
            -129,
            2**70,
            1.25,
            -0.0,
            np.bool_(False),
            np.int64(-5),
            np.float32(1.25),
            np.bytes_(b"abc"),
            _BytesSubclass(b"original payload"),
        ]
    )
]

_DTYPES = [
    np.bool_,
    np.int8,
    np.uint8,
    np.int16,
    np.int32,
    np.int64,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.dtype(">i4"),
    np.dtype(">f8"),
    np.dtype("U3"),
    np.dtype("S2"),
    np.dtype("datetime64[ns]"),
    np.dtype("timedelta64[s]"),
    np.dtype([("a", "<i2"), ("b", ">f4")]),
    np.dtype(object),
]


def _make_array(dtype, shape):
    size = int(np.prod(shape)) if shape else 1
    raw = np.arange(size) % 7
    dtype = np.dtype(dtype)
    if dtype.kind in "US":
        values = np.array([str(v) * (v % 3 + 1) for v in raw], dtype=dtype)
    elif dtype.kind in "mM":
        values = raw.astype(np.int64).astype(dtype)
    elif dtype.names:
        values = np.zeros(size, dtype=dtype)
        values["a"] = raw
        values["b"] = raw * 0.5
    else:
        values = raw.astype(dtype)
    return values.reshape(shape)


_ARRAY_CASES = []
for _dtype in _DTYPES:
    for _shape in [(), (0,), (0, 3), (3, 0, 2), (2, 3), (4, 6)]:
        _ARRAY_CASES.append(
            pytest.param(_make_array(_dtype, _shape), id=f"{np.dtype(_dtype).str}-{_shape}")
        )
    _base = _make_array(_dtype, (4, 6))
    _ARRAY_CASES.append(pytest.param(_base[:, ::2], id=f"{np.dtype(_dtype).str}-strided"))
    _ARRAY_CASES.append(pytest.param(_base.T, id=f"{np.dtype(_dtype).str}-transposed"))
    _ARRAY_CASES.append(
        pytest.param(np.asfortranarray(_base), id=f"{np.dtype(_dtype).str}-fortran")
    )

_TENSOR_CASES = [
    pytest.param(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4), id="f32"),
    pytest.param(torch.arange(12, dtype=torch.int64).reshape(3, 4).t(), id="i64-transposed"),
    pytest.param(torch.tensor([True, False, True]), id="bool"),
    pytest.param(torch.arange(6, dtype=torch.float16), id="f16"),
    pytest.param(torch.zeros((0, 3), dtype=torch.float32), id="zero-size"),
    pytest.param(torch.tensor(3.5), id="scalar"),
    pytest.param(torch.ones(4, requires_grad=True), id="requires-grad"),
]


def _image(mode, size):
    image = Image.new("RGB", size)
    if size[0] and size[1]:
        pixels = np.arange(size[0] * size[1] * 3, dtype=np.uint8).reshape(size[1], size[0], 3)
        image = Image.fromarray(pixels, "RGB")
    return image.convert(mode)


_IMAGE_CASES = [
    pytest.param(_image(mode, size), id=f"{mode}-{size[0]}x{size[1]}")
    for mode in ["RGB", "RGBA", "L", "LA", "P", "1", "I"]
    for size in [(5, 3), (1, 1), (0, 0), (0, 4), (4, 0)]
]

_NESTED_CASES = [
    pytest.param(
        {
            "frames": [np.arange(6, dtype=np.uint8).reshape(2, 3)],
            "fps": np.float32(2),
            "name": "clip",
            "raw": b"\x00\x01",
            "flags": (True, np.bool_(False), 7, -3, 2**70, np.int64(-5), 1.25),
        },
        id="dict",
    ),
    pytest.param(
        [torch.ones(2, 2), (np.zeros((0, 2)), _image("RGB", (2, 2))), []],
        id="list-of-tuples",
    ),
    pytest.param(
        (np.array(["2026-01-01"], dtype="M8[D]"), np.array([], dtype="m8[ns]")), id="datetime"
    ),
    pytest.param({"empty": {}, "empty_list": [], "image": _image("L", (0, 0))}, id="empty"),
]


def _audio(samples):
    return AudioData(samples=samples, sample_rate=16000)


def _video(frames, raw_bytes_hash=None, audio=None):
    meta = {"frames_indices": [0, 1], "fps": 30.0, "duration": 1.0, "unused": "x"}
    return VideoData(frames=frames, metadata=meta, audio=audio, raw_bytes_hash=raw_bytes_hash)


_MODALITY_CASES = [
    pytest.param("audio", _audio(np.linspace(-1, 1, 33, dtype=np.float32)), id="audio-array"),
    pytest.param("audio", _audio(torch.linspace(-1, 1, 33)), id="audio-tensor"),
    pytest.param("audio", _audio(np.zeros((0,), dtype=np.float32)), id="audio-empty"),
    pytest.param("audio", _audio(np.ones((2, 5), dtype=np.int16)[:, ::2]), id="audio-strided"),
    pytest.param(
        "video", _video([_image("RGB", (4, 3)), _image("RGB", (4, 3))]), id="video-pil-frames"
    ),
    pytest.param(
        "video",
        _video([torch.rand(3, 4, 4), torch.rand(3, 4, 4).transpose(1, 2)]),
        id="video-tensor-frames",
    ),
    pytest.param(
        "video",
        _video(np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)),
        id="video-array-frames",
    ),
    pytest.param(
        "video",
        _video([_image("RGB", (0, 0))], audio=_audio(np.zeros((0,), dtype=np.float32))),
        id="video-empty-frame-with-audio",
    ),
    pytest.param(
        "video",
        _video([_image("RGB", (4, 3))], raw_bytes_hash="ab" * 32, audio=_audio(np.ones(8))),
        id="video-raw-bytes-with-audio",
    ),
]

_ALL_ITEM_CASES = (
    [pytest.param("image", *p.values, id=p.id) for p in _SCALAR_CASES]
    + [pytest.param("image", *p.values, id=f"array-{p.id}") for p in _ARRAY_CASES]
    + [pytest.param("image", *p.values, id=f"tensor-{p.id}") for p in _TENSOR_CASES]
    + [pytest.param("image", *p.values, id=f"image-{p.id}") for p in _IMAGE_CASES]
    + [pytest.param("image", *p.values, id=f"nested-{p.id}") for p in _NESTED_CASES]
    + _MODALITY_CASES
)


@pytest.mark.parametrize(
    "item", [*_SCALAR_CASES, *_ARRAY_CASES, *_TENSOR_CASES, *_IMAGE_CASES, *_NESTED_CASES]
)
def test_serialize_item_matches_reference(item):
    assert serialize_item(item) == _reference_serialize(item)


def _assert_streamed_hash_matches_reference(modality, item, hash_lib, uuid):
    reference = hash_lib()
    if uuid is not None:
        reference.update(b"<uuid>" + uuid.encode("utf-8") + b"</uuid><content>")
    reference.update(_reference_content(item))
    if uuid is not None:
        reference.update(b"</content>")

    uuids = None if uuid is None else {modality: [uuid]}
    hashes, actual_uuids = apply_mm_hashes({modality: [item]}, uuids, hash_lib=hash_lib)

    assert hashes[modality] == [reference.hexdigest()]
    assert actual_uuids == uuids


@pytest.mark.parametrize("hash_lib", _HASH_LIBS, ids=_HASH_IDS)
@pytest.mark.parametrize("uuid", _UUIDS, ids=["no-uuid", "empty-uuid", "unicode-long-uuid"])
@pytest.mark.parametrize("modality,item", _ALL_ITEM_CASES)
def test_streamed_hash_matches_reference(modality, item, hash_lib, uuid):
    _assert_streamed_hash_matches_reference(modality, item, hash_lib, uuid)


@pytest.mark.parametrize("size", [(5, 3), (1, 1), (0, 0), (0, 4), (4, 0)])
@pytest.mark.parametrize("hash_lib", _HASH_LIBS, ids=_HASH_IDS)
@pytest.mark.parametrize("uuid", _UUIDS, ids=["no-uuid", "empty-uuid", "unicode-long-uuid"])
def test_float_image_hash_matches_reference(size, hash_lib, uuid):
    item = _image("F", size)
    try:
        item.convert("RGBA")
    except ValueError:
        with pytest.raises(ValueError):
            _reference_serialize(item)
        with pytest.raises(ValueError):
            serialize_item(item)
        with pytest.raises(ValueError):
            apply_mm_hashes({"image": [item]}, hash_lib=hash_lib)
    else:
        assert serialize_item(item) == _reference_serialize(item)
        _assert_streamed_hash_matches_reference("image", item, hash_lib, uuid)


@pytest.mark.parametrize("hash_lib", _HASH_LIBS, ids=_HASH_IDS)
def test_array_payload_buffer_contract(hash_lib, monkeypatch):
    hasher = hash_lib()
    original_update = md._update_serialized_item
    updates = []

    def record_update(update, item):
        updates.append(update)
        original_update(update, item)

    monkeypatch.setattr(md, "_update_serialized_item", record_update)
    item = np.arange(12, dtype=np.float32)
    md._hash_serialized_item(hasher, item)
    reference = hash_lib()
    reference.update(_reference_serialize(item))
    assert hasher.hexdigest() == reference.hexdigest()
    if hash_lib is not _BytesOnlySha256:
        assert updates[0] == hasher.update


@pytest.mark.parametrize(
    "item,expected",
    [
        (
            torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
            "a58f461450a5e0d040e65f5ca33b3650bedf25115dfcd795b47c15c3e3ef96ec",
        ),
        (
            np.arange(24, dtype=np.int16).reshape(4, 6)[:, ::2],
            "488e181d756628893937686d1747e77723733d9a2a73cdda99b8a8d6c7900b11",
        ),
        (
            Image.new("RGB", (5, 3), (10, 20, 30)),
            "65fa5adbfedf9a284d2a3979d5d2781d42034243ede81b4bd382fa1a17abf2b9",
        ),
        (
            {
                "frames": [np.arange(6, dtype=np.uint8).reshape(2, 3)],
                "fps": np.float32(2),
            },
            "a2dffa10413eb3fef1f2d7499e69cb4df2c017b79d162ad7eae3086df9198cf9",
        ),
    ],
    ids=["tensor", "noncontiguous_array", "pil_image", "nested"],
)
def test_streamed_hash_pinned_digests(item, expected):
    reference = blake3()
    reference.update(_reference_content(item))
    assert reference.hexdigest() == expected

    hashes, _ = apply_mm_hashes({"image": [item]})

    assert hashes["image"] == [expected]
