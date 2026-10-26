# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSPARK_GLUE_CPP: trtllm::minimax_m31_dspark_advance / _indices (kernels/minimaxM31/dsparkGlueSm107.cu) bit for bit
the catalog Triton kernels they replace (catalog/spec_decode/dspark_ctx_advance.py, dspark_forward_indices.py), on
every output, through the catalog wrappers (C++ path vs the forced Triton path); and the kill switch."""

import os
import subprocess
import sys

import pytest
import torch

import tensorrt_llm  # noqa: F401  (loads the trtllm op library)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode import (
    dspark_ctx_advance as AD,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode import (
    dspark_forward_indices as IX,
)

_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or not hasattr(torch.ops.trtllm, "minimax_m31_dspark_advance")
    or not hasattr(torch.ops.trtllm, "minimax_m31_dspark_indices"),
    reason="needs an SM107 GPU and the DSPARK_GLUE_CPP ops",
)


def _ri(g, lo, hi, shape, dtype=torch.int64):
    return torch.randint(lo, hi, shape, generator=g, device="cuda", dtype=dtype)


def _both(fn, monkeypatch, mod):
    """(C++ result, Triton result) of fn() through the catalog wrapper."""
    got = fn()
    monkeypatch.setattr(mod, "_GLUE_CPP_OFF", True)
    ref = fn()
    monkeypatch.setattr(mod, "_GLUE_CPP_OFF", False)
    return got, ref


@_GPU
@pytest.mark.parametrize("width", [2048, 8192, 65536])
@pytest.mark.parametrize("n", [1, 7, 22, 56])
@pytest.mark.parametrize("nacc_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("parallel", [False, True])
def test_advance_bitwise(monkeypatch, width, n, nacc_dtype, parallel):
    g = torch.Generator(device="cuda").manual_seed(width + 31 * n)
    nslots = nrows = 64
    page, block_size = 128, 7
    slots = torch.randperm(nslots - 1, generator=g, device="cuda")[:n].to(torch.int64)
    if n > 2:
        slots[-1] = nslots - 1  # a dummy (padding) request
    rows = torch.randperm(nrows, generator=g, device="cuda")[:n].to(torch.int64)
    nacc = _ri(g, 0, 9, (n,), nacc_dtype)
    max_ctx = width * page - 1000
    ctx0 = _ri(g, 0, max_ctx, (nslots,))
    ctx0[:2] = max_ctx  # at the clamp
    counts = _ri(g, 0, width + 1, (nrows,))
    counts[0] = 0  # nothing allocated
    base = _ri(g, 0, width // 4, (nslots,))
    tables = _ri(g, 0, 2**31 - 1, (nrows, width), torch.int32)

    def run():
        ctx, read = ctx0.clone(), torch.zeros_like(tables)
        args = (
            slots,
            rows,
            nacc,
            ctx,
            max_ctx,
            nslots - 1,
            counts,
            base,
            tables,
            read,
            page,
            block_size,
        )
        out = AD.advance_ctx_and_read_view(*args, parallel=parallel)
        torch.cuda.synchronize()
        return out, ctx, read

    got, ref = _both(run, monkeypatch, AD)
    if not parallel:  # the engagement keys behind the "[M3 dispatch] DSPARK_GLUE_CPP ..." lines
        assert ("minimax_m31_dspark_advance", None) in AD._glue_logged
        assert (
            "minimax_m31_dspark_advance",
            "killed (TRTLLM_M31_DISABLE_DSPARK_GLUE_CPP=1)",
        ) in AD._glue_logged
    for a, b in zip(got, ref):
        assert a.dtype == b.dtype and torch.equal(a, b)


@_GPU
@pytest.mark.parametrize("B", [1, 7, 22, 56])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_indices_bitwise(monkeypatch, B, dtype, strided):
    g = torch.Generator(device="cuda").manual_seed(7 * B + int(strided))
    block, pages_per_slot, page = 7, 8192, 128
    num_ctx = _ri(g, 0, 2**20, (B,), dtype)
    num_ctx[0] = 0  # (after - 1) % page at the wrap
    batch_idx = _ri(g, 0, 64, (B,), dtype)
    q = _ri(g, 0, 2**20, (B, 2 * block), dtype)
    qpos = q[:, ::2] if strided else q[:, :block].contiguous()
    if dtype == torch.int64:
        num_ctx[-1] = 2**33 + 5  # the int64 -> int32 cast keeps the low 32 bits
        qpos[-1, -1] = -(2**34) - 3

    def run():
        out = IX.forward_indices(num_ctx, batch_idx, qpos, block, pages_per_slot, page)
        torch.cuda.synchronize()
        return out

    got, ref = _both(run, monkeypatch, IX)
    assert ("minimax_m31_dspark_indices", None) in IX._glue_logged
    for a, b in zip(got, ref):
        assert a.dtype == b.dtype and torch.equal(a, b)


def test_kill_switch():
    code = (
        "import torch\n"
        "from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode import dspark_ctx_advance as a, "
        "dspark_forward_indices as i\n"
        "print(a._GLUE_CPP_OFF, i._GLUE_CPP_OFF, a._glue_cpp_op('x', torch.device('cuda', 0)) is None)"
    )
    env = {**os.environ, "TRTLLM_M31_DISABLE_DSPARK_GLUE_CPP": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split()[-3:] == ["True", "True", "True"]
