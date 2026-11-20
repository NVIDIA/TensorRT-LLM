# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
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
    with PinnedSafetensorsFile(path) as shard:
        with open(path, "r+b") as f:
            f.truncate(os.path.getsize(path) - 1)  # last byte of "b" is gone after open
        assert torch.equal(shard.get_tensor("a"), ints)
        with pytest.raises(OSError, match="short read"):
            shard.get_tensor("b")


@pytest.mark.cpu_only
def test_data_offsets_beyond_the_file_are_rejected_at_open(tmp_path):
    path = str(tmp_path / "shard.safetensors")
    _write_shard(path, {"a": ("I32", torch.arange(16, dtype=torch.int32))})
    with open(path, "r+b") as f:
        f.truncate(os.path.getsize(path) - 1)
    with pytest.raises(ValueError, match="data_offsets outside the file"):
        PinnedSafetensorsFile(path).__enter__()


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
@pytest.mark.parametrize(
    "header_len, file_size", [(64, 10), (1 << 30, 1 << 31)], ids=["beyond_file", "above_cap_sparse"]
)
def test_oversized_header_length_is_rejected(tmp_path, header_len, file_size):
    path = tmp_path / "bad.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_len) + b"{}")
        # Shorter than the header, or sparse: fits in the file but exceeds the cap.
        f.truncate(file_size)
    with pytest.raises(ValueError, match="invalid SafeTensors header length"):
        PinnedSafetensorsFile(str(path)).__enter__()


@pytest.mark.cpu_only
def test_partial_header_reads_are_resumed(tmp_path, monkeypatch):
    path = str(tmp_path / "shard.safetensors")
    ints = torch.arange(16, dtype=torch.int32)
    _write_shard(path, {"a": ("I32", ints)})
    real_pread = os.pread
    monkeypatch.setattr(os, "pread", lambda fd, n, offset: real_pread(fd, min(n, 3), offset))
    with PinnedSafetensorsFile(path) as shard:
        assert torch.equal(shard.get_tensor("a"), ints)


@pytest.mark.cpu_only
def test_eof_during_header_read_raises(tmp_path):
    path = str(tmp_path / "shard.safetensors")
    with open(path, "wb") as f:
        f.write(b"\x00" * 7)
    with pytest.raises(OSError, match="truncated SafeTensors header"):
        PinnedSafetensorsFile(path).__enter__()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "entries, error",
    [
        ({"a": [0, 64], "b": [0, 64]}, "overlapping"),
        ({"a": [0, 64], "b": [32, 96]}, "overlapping"),
        ({"a": [0, 60], "b": [60, 124]}, "does not match"),
    ],
    ids=["duplicate", "overlap", "length_mismatch"],
)
def test_inconsistent_data_offsets_are_rejected_at_open(tmp_path, entries, error):
    path = str(tmp_path / "shard.safetensors")
    header = {
        key: {"dtype": "I32", "shape": [16], "data_offsets": offsets}
        for key, offsets in entries.items()
    }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)) + encoded + bytes(128))
    with pytest.raises(ValueError, match=error):
        PinnedSafetensorsFile(path).__enter__()


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


_I = slice(None)
_SLICE_INDEXES = {
    "int": 5,
    "neg_int": -3,
    "slice": slice(2, 9),
    "neg_slice": slice(-7, -2),
    "none_bounds": slice(None, None),
    "open_stop": slice(4, None),
    "step": slice(1, 20, 3),
    "step_only": slice(None, None, 4),
    "empty": slice(5, 5),
    "empty_reversed": slice(9, 2),
    "oob_slice": slice(-100, 100),
    "ellipsis": Ellipsis,
    "tuple_ints": (3, 2),
    "tuple_neg_ints": (-1, -2, -3),
    "tuple_slices": (slice(1, 6), slice(None, 3)),
    "tuple_mixed": (slice(0, 8, 2), 1, slice(-2, None)),
    "trailing_step": (slice(2, 7), slice(None), slice(None, None, 2)),
    "lead_ellipsis": (Ellipsis, 1),
    "mid_ellipsis": (slice(3, 5), Ellipsis, slice(1, 3)),
    "trail_ellipsis": (-2, Ellipsis),
    "int_then_ellipsis": (4, Ellipsis, 0),
    "empty_trailing": (slice(1, 4), slice(2, 2)),
    "empty_tuple": (),
}


def _slice_shard(tmp_path, name):
    path = str(tmp_path / "shard.safetensors")
    tensors = {
        "t": (name, _random_bits(_DTYPES[name], (23, 4, 6), 7)),
        "pad": (name, _random_bits(_DTYPES[name], (5,), 8)),
    }
    _write_shard(path, tensors)
    return path


def _bits(tensor):
    """Raw bytes of ``tensor``; empty tensors can keep non-unit strides, so they map to no bytes."""
    if tensor.numel() == 0:
        return torch.empty(0, dtype=torch.uint8)
    return tensor.contiguous().reshape(-1).view(torch.uint8)


def _assert_same_as_safe_open(got, expected):
    assert (got.dtype, got.shape) == (expected.dtype, expected.shape)
    assert got.is_contiguous()
    assert got.storage_offset() == 0
    assert got.untyped_storage().nbytes() == got.numel() * got.element_size()
    assert torch.equal(_bits(got), _bits(expected))


def _has_step(index):
    items = index if isinstance(index, tuple) else (index,)
    return any(isinstance(i, slice) and i.step is not None for i in items)


def _check_index(ours, ref, key, index):
    """``ours`` must equal torch indexing of the full tensor, and ``safe_open``'s own slice
    wherever the installed safetensors accepts the index and defines it the same way.

    safetensors ignores slice steps and rejects Ellipsis, empty, out-of-range and zero-row
    indexes, so the full tensor indexed by torch is the reference for those.
    """
    got = ours.get_slice(key)[index]
    _assert_same_as_safe_open(got, ref.get_tensor(key)[index])
    if _has_step(index):
        return
    try:
        theirs = ref.get_slice(key)[index]
    except Exception:  # not supported by the installed safetensors
        return
    _assert_same_as_safe_open(got, theirs)


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", ["BF16", "F32", "I8", "F8_E4M3"])
@pytest.mark.parametrize("index", list(_SLICE_INDEXES.values()), ids=list(_SLICE_INDEXES))
def test_get_slice_matches_safe_open_bit_for_bit(tmp_path, monkeypatch, name, index):
    if name not in _DTYPES:
        pytest.skip(f"this torch has no {name}")
    monkeypatch.setattr(pinned_safetensors, "_READ_BYTES", 100)  # ragged multi-chunk reads
    path = _slice_shard(tmp_path, name)
    with PinnedSafetensorsFile(path) as ours, safe_open(path, framework="pt", device="cpu") as ref:
        ours_slice, ref_slice = ours.get_slice("t"), ref.get_slice("t")
        assert ours_slice.get_shape() == ref_slice.get_shape()
        assert ours_slice.get_dtype() == ref_slice.get_dtype() == name
        _check_index(ours, ref, "t", index)


@pytest.mark.cpu_only
def test_get_slice_accepts_numpy_integers(tmp_path):
    path = _slice_shard(tmp_path, "I8")
    with PinnedSafetensorsFile(path) as ours, safe_open(path, framework="pt", device="cpu") as ref:
        full = ref.get_tensor("t")
        got = ours.get_slice("t")[np.int64(3), np.int32(1) : np.int64(4)]
        assert torch.equal(_bits(got), _bits(full[3, 1:4]))


@pytest.mark.cpu_only
def test_get_slice_zero_dim_and_zero_size_tensors(tmp_path):
    path = str(tmp_path / "shard.safetensors")
    tensors = {
        "scalar": ("F32", _random_bits(torch.float32, (), 0)),
        "no_rows": ("I32", torch.empty((0, 4), dtype=torch.int32)),
        "no_cols": ("I32", torch.empty((3, 0), dtype=torch.int32)),
    }
    _write_shard(path, tensors)
    cases = {
        "scalar": [Ellipsis, ()],
        "no_rows": [slice(None), Ellipsis, (slice(None), 1), slice(0, 3)],
        "no_cols": [slice(None), 1, (slice(1, 3), slice(None)), (Ellipsis, slice(None, None, 2))],
    }
    with PinnedSafetensorsFile(path) as ours, safe_open(path, framework="pt", device="cpu") as ref:
        for key, indexes in cases.items():
            assert ours.get_slice(key).get_shape() == ref.get_slice(key).get_shape()
            for index in indexes:
                _check_index(ours, ref, key, index)
        with pytest.raises(IndexError):
            ours.get_slice("scalar")[0]
        with pytest.raises(IndexError):
            ours.get_slice("no_rows")[0]


@pytest.mark.cpu_only
def test_get_slice_pins_under_the_get_tensor_rule(tmp_path):
    path = str(tmp_path / "shard.safetensors")
    weights = _random_bits(torch.bfloat16, (64, 1024), 3)  # 2 KiB per row
    _write_shard(path, {"w": ("BF16", weights)})
    with PinnedSafetensorsFile(path) as shard:
        big = shard.get_slice("w")[:40]  # 80 KiB: at or above _PIN_MIN_BYTES
        small = shard.get_slice("w")[3:6]  # 6 KiB: stays pageable
        assert big.is_pinned() == prefer_pinned()
        assert not small.is_pinned()
        # A trailing-dim view is copied into a fresh block, pinned under the same rule.
        copied = shard.get_slice("w")[:, :512]  # 64 KiB
        assert copied.is_pinned() == prefer_pinned()
        assert torch.equal(_bits(copied), _bits(weights[:, :512]))


@pytest.mark.cpu_only
def test_get_slice_reads_only_the_needed_byte_range(tmp_path, monkeypatch):
    path = str(tmp_path / "shard.safetensors")
    weights = _random_bits(torch.int32, (50, 8), 4)  # 32 bytes per row
    _write_shard(path, {"pre": ("I32", weights[:7]), "w": ("I32", weights)})
    reads = []
    real_preadv = os.preadv

    def recording_preadv(fd, buffers, offset):
        reads.append((offset, sum(len(b) for b in buffers)))
        return real_preadv(fd, buffers, offset)

    monkeypatch.setattr(os, "preadv", recording_preadv)
    with PinnedSafetensorsFile(path) as shard:
        data_start = shard._entries["w"][2]
        cases = {
            "int": (10, (10 * 32, 32)),
            "neg_int": (-1, (49 * 32, 32)),
            "slice": (slice(4, 9), (4 * 32, 5 * 32)),
            "step": (slice(2, 30, 7), (2 * 32, (3 * 7 + 1) * 32)),
            "trailing_only": ((slice(6, 8), slice(1, 3)), (6 * 32, 2 * 32)),
            "empty": (slice(8, 8), None),
        }
        for index, span in cases.values():
            reads.clear()
            shard.get_slice("w")[index]
            if span is None:
                assert reads == []
            else:
                assert reads == [(data_start + span[0], span[1])]


@pytest.mark.cpu_only
def test_get_slice_survives_short_reads(tmp_path, monkeypatch):
    path = str(tmp_path / "shard.safetensors")
    weights = _random_bits(torch.float32, (40, 9), 5)
    _write_shard(path, {"w": ("F32", weights)})
    real_preadv = os.preadv
    monkeypatch.setattr(
        os,
        "preadv",
        lambda fd, buffers, offset: real_preadv(fd, [memoryview(buffers[0])[:7]], offset),
    )
    with PinnedSafetensorsFile(path) as shard:
        assert torch.equal(_bits(shard.get_slice("w")[5:30:4, 2:]), _bits(weights[5:30:4, 2:]))


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "index",
    [[0, 1], torch.tensor([0, 1]), None, (None, 0), (0, None), True, (False,), 1.5, "a"]
    + [slice(0.5, 2), slice(None, None, True), slice(torch.tensor(1), None)],
    ids=lambda v: f"{type(v).__name__}-{v}"[:40],
)
def test_get_slice_rejects_unsupported_index_types(tmp_path, monkeypatch, index):
    path = str(tmp_path / "shard.safetensors")
    _write_shard(path, {"w": ("I32", torch.arange(24, dtype=torch.int32).reshape(4, 6))})

    def forbidden(*args):
        raise AssertionError("an unsupported index must not read the file")

    with PinnedSafetensorsFile(path) as shard:
        monkeypatch.setattr(os, "preadv", forbidden)
        with pytest.raises(TypeError, match="index|slice"):
            shard.get_slice("w")[index]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "index, error",
    [
        (4, IndexError),
        (-5, IndexError),
        ((0, 0, 0), IndexError),
        ((Ellipsis, Ellipsis), IndexError),
        ((slice(None), 6), IndexError),
        (slice(None, None, 0), ValueError),
        (slice(None, None, -1), ValueError),
    ],
)
def test_get_slice_out_of_range_indexes_raise(tmp_path, index, error):
    path = str(tmp_path / "shard.safetensors")
    _write_shard(path, {"w": ("I32", torch.arange(24, dtype=torch.int32).reshape(4, 6))})
    with PinnedSafetensorsFile(path) as shard:
        with pytest.raises(error):
            shard.get_slice("w")[index]


@pytest.mark.cpu_only
def test_get_slice_unknown_key_and_unsupported_dtype(tmp_path):
    path = str(tmp_path / "shard.safetensors")
    ints = torch.arange(4, dtype=torch.int32)
    _write_shard(path, {"ok": ("I32", ints), "odd": ("NOT_A_DTYPE", ints)})
    with PinnedSafetensorsFile(path) as shard:
        with pytest.raises(KeyError):
            shard.get_slice("missing")
        odd = shard.get_slice("odd")
        assert odd.get_dtype() == "NOT_A_DTYPE"
        assert odd.get_shape() == [4]
        with pytest.raises(TypeError, match="NOT_A_DTYPE"):
            odd[:2]
        assert torch.equal(shard.get_slice("ok")[1:3], ints[1:3])


@pytest.mark.cpu_only
def test_get_slice_after_close_raises(tmp_path):
    path = str(tmp_path / "shard.safetensors")
    _write_shard(path, {"w": ("I32", torch.arange(24, dtype=torch.int32).reshape(4, 6))})
    with PinnedSafetensorsFile(path) as shard:
        view = shard.get_slice("w")
        assert view[1].shape == (6,)
    assert view.get_shape() == [4, 6]
    assert view.get_dtype() == "I32"
    with pytest.raises(ValueError, match="after file close"):
        view[0]
    with pytest.raises(ValueError, match="after file close"):
        view[:]


@pytest.mark.cpu_only
def test_get_slice_results_outlive_the_file(tmp_path):
    path = str(tmp_path / "shard.safetensors")
    weights = _random_bits(torch.bfloat16, (16, 8), 6)
    _write_shard(path, {"w": ("BF16", weights)})
    with PinnedSafetensorsFile(path) as shard:
        got = shard.get_slice("w")[2:10:2, 1:5]
    assert torch.equal(_bits(got), _bits(weights[2:10:2, 1:5]))
