# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TLLM_K3_EXPERT_PREAD's shard reader returns exactly what safetensors' own reader returns."""

import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tensorrt_llm._torch.models.modeling_kimi_linear import read_safetensors_slices


def _buffer(nbytes):
    return torch.empty(nbytes, dtype=torch.uint8)


def _bytes(tensor):
    return tensor.reshape(-1).view(torch.uint8)


def _tensors():
    g = torch.Generator().manual_seed(0)
    return {
        "layers.0.experts.0.w": torch.randint(0, 255, (64, 33), dtype=torch.uint8, generator=g),
        "layers.0.experts.0.s": torch.randint(-128, 127, (7,), dtype=torch.int8, generator=g),
        "layers.0.experts.1.w": torch.randn(3, 5, generator=g).to(torch.bfloat16),
        "layers.0.experts.10.w": torch.randn(17, generator=g).to(torch.float16),
        "layers.0.experts.11.w": torch.randn(2, 2, 2, generator=g),
        "layers.0.experts.12.w": torch.randint(0, 2**30, (9,), dtype=torch.int32, generator=g),
        "layers.0.experts.13.w": torch.randint(0, 2**40, (3,), dtype=torch.int64, generator=g),
        "layers.0.experts.14.w": torch.randn(31, generator=g).to(torch.float8_e4m3fn),
        # Wider than the 1 MiB merge gap, so the requested tensors span several ranges.
        "layers.0.filler": torch.zeros(3 << 20, dtype=torch.uint8),
        "layers.1.experts.0.w": torch.randint(0, 255, (1000,), dtype=torch.uint8, generator=g),
        "layers.1.experts.0.empty": torch.zeros(0, 4, dtype=torch.bfloat16),
    }


def test_reads_match_safe_open(tmp_path):
    tensors = _tensors()
    path = os.path.join(tmp_path, "shard.safetensors")
    save_file(tensors, path)
    wanted = [name for name in tensors if name != "layers.0.filler"]
    got = read_safetensors_slices(path, wanted + ["not.in.this.shard"], _buffer)
    assert got is not None and set(got) == set(wanted)
    with safe_open(path, framework="pt", device="cpu") as fh:
        for name in wanted:
            ref = fh.get_tensor(name)
            assert got[name].dtype == ref.dtype, name
            assert got[name].shape == ref.shape, name
            assert torch.equal(_bytes(got[name]), _bytes(ref)), name


def test_views_share_one_buffer(tmp_path):
    path = os.path.join(tmp_path, "shard.safetensors")
    save_file(_tensors(), path)
    buffers = []

    def buffer(nbytes):
        buffers.append(torch.empty(nbytes + 4096, dtype=torch.uint8))
        return buffers[-1]

    got = read_safetensors_slices(path, ["layers.0.experts.0.w", "layers.1.experts.0.w"], buffer)
    assert len(buffers) == 1
    base = buffers[0].untyped_storage().data_ptr()
    for tensor in got.values():
        assert tensor.untyped_storage().data_ptr() == base


def test_unknown_dtype_and_short_file_return_none(tmp_path):
    path = os.path.join(tmp_path, "shard.safetensors")
    save_file(
        {"a": torch.zeros(4, dtype=torch.float64), "b": torch.ones(1024, dtype=torch.uint8)}, path
    )
    assert read_safetensors_slices(path, ["a"], _buffer) is None
    assert read_safetensors_slices(path, ["b"], _buffer) is not None
    os.truncate(path, os.path.getsize(path) - 16)
    assert read_safetensors_slices(path, ["b"], _buffer) is None
