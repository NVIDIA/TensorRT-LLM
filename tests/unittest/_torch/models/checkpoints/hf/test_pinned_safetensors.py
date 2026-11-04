# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from safetensors import safe_open

from tensorrt_llm._torch.models.checkpoints.hf import pinned_safetensors
from tensorrt_llm._torch.models.checkpoints.hf.pinned_safetensors import PinnedSafetensorsFile
from tensorrt_llm._utils import prefer_pinned

# Every SafeTensors dtype the reader maps that this torch build has.
_DTYPES = {
    name: getattr(torch, attr)
    for name, attr in pinned_safetensors._DTYPE_ATTRS.items()
    if isinstance(getattr(torch, attr, None), torch.dtype)
}


def _random_bits(dtype: torch.dtype, shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Random bytes viewed as ``dtype``; keeps NaN and denormal patterns a value compare would hide."""
    numel = 1
    for dim in shape:
        numel *= dim
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randint(
        0, 2 if dtype is torch.bool else 256, (nbytes,), dtype=torch.uint8, generator=generator
    )
    return raw.view(dtype).reshape(shape)


def _write_shard(path: str, tensors: dict[str, tuple[str, torch.Tensor]]) -> None:
    header, blobs, offset = {}, [], 0
    for key, (name, tensor) in tensors.items():
        blob = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        header[key] = {
            "dtype": name,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", sorted(_DTYPES))
def test_get_tensor_matches_safe_open_bit_for_bit(tmp_path, monkeypatch, name):
    monkeypatch.setattr(pinned_safetensors, "_READ_BYTES", 1000)  # ragged multi-chunk reads
    dtype = _DTYPES[name]
    tensors = {
        "small": (name, _random_bits(dtype, (3, 5), 0)),  # below _PIN_MIN_BYTES: stays pageable
        "scalar": (name, _random_bits(dtype, (), 1)),
        "large": (name, _random_bits(dtype, (130, 1024), 2)),
    }
    path = str(tmp_path / "shard.safetensors")
    _write_shard(path, tensors)
    with PinnedSafetensorsFile(path) as ours, safe_open(path, framework="pt", device="cpu") as ref:
        assert sorted(ours.keys()) == sorted(ref.keys())
        for key in ref.keys():
            try:
                expected = ref.get_tensor(key)
            except Exception as exc:  # the installed safetensors/torch cannot read this dtype
                pytest.skip(f"safe_open cannot read {name}: {exc}")
            got = ours.get_tensor(key)
            assert (got.dtype, got.shape) == (expected.dtype, expected.shape)
            assert torch.equal(
                got.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)
            )
        assert not ours.get_tensor("small").is_pinned()


@pytest.mark.cpu_only
def test_partial_preadv_results_are_resumed(tmp_path, monkeypatch):
    path = str(tmp_path / "shard.safetensors")
    weights = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)  # 64 KiB
    _write_shard(path, {"w": ("F32", weights)})
    real_preadv = os.preadv
    monkeypatch.setattr(
        os,
        "preadv",
        lambda fd, buffers, offset: real_preadv(fd, [memoryview(buffers[0])[:7]], offset),
    )
    with PinnedSafetensorsFile(path) as shard:
        assert torch.equal(shard.get_tensor("w"), weights)


@pytest.mark.cpu_only
def test_missing_bytes_raise_instead_of_returning_a_partial_tensor(tmp_path):
    path = str(tmp_path / "shard.safetensors")
    ints = torch.arange(16, dtype=torch.int32)
    _write_shard(path, {"a": ("I32", ints), "b": ("I32", ints)})
    with open(path, "r+b") as f:
        f.truncate(os.path.getsize(path) - 1)  # last byte of "b" is gone
    with PinnedSafetensorsFile(path) as shard:
        assert torch.equal(shard.get_tensor("a"), ints)
        with pytest.raises(OSError, match="short read"):
            shard.get_tensor("b")


@pytest.mark.cpu_only
def test_unsupported_dtype_fails_only_its_own_tensor(tmp_path):
    path = str(tmp_path / "shard.safetensors")
    ints = torch.arange(4, dtype=torch.int32)
    _write_shard(path, {"ok": ("I32", ints), "odd": ("NOT_A_DTYPE", ints)})
    with PinnedSafetensorsFile(path) as shard:
        assert torch.equal(shard.get_tensor("ok"), ints)
        with pytest.raises(TypeError, match="NOT_A_DTYPE"):
            shard.get_tensor("odd")


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "header",
    [b"\xff" * 8, json.dumps({"t": {"dtype": "U8", "shape": [1]}}).encode()],
    ids=["bad_json", "missing_offsets"],
)
def test_failed_open_does_not_leak_the_file_descriptor(tmp_path, header):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"x")
    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises((KeyError, ValueError, OSError)):
        PinnedSafetensorsFile(str(path)).__enter__()
    assert len(os.listdir("/proc/self/fd")) == before


@pytest.mark.cpu_only
def test_header_length_beyond_the_file_is_rejected(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    with pytest.raises(ValueError, match="exceeds the file"):
        PinnedSafetensorsFile(str(path)).__enter__()


@pytest.mark.skipif(not prefer_pinned(), reason="pinned host memory needs CUDA")
def test_large_tensors_are_pinned_and_exact_under_concurrent_reads(tmp_path):
    path = str(tmp_path / "shard.safetensors")
    tensors = {
        f"w{i}": ("BF16", _random_bits(torch.bfloat16, (256, 1024), i)) for i in range(8)
    }  # 512 KiB each, above _PIN_MIN_BYTES
    _write_shard(path, tensors)
    with PinnedSafetensorsFile(path) as ours, safe_open(path, framework="pt", device="cpu") as ref:
        with ThreadPoolExecutor(4) as pool:
            got = dict(zip(tensors, pool.map(ours.get_tensor, tensors)))
        for key, tensor in got.items():
            assert tensor.is_pinned()
            assert torch.equal(tensor.view(torch.uint8), ref.get_tensor(key).view(torch.uint8))
