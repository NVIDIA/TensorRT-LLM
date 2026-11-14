# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family ops minimax_m31_dspark_advance / _advance_par / _indices (csrc/m31/dsparkGlueSm107.cu) through the
ops/dspark wrappers. The advance ops (one launch and ADV_PAR) bit for bit a pure-torch reference of the
documented contract on every output (the Triton kernels are deleted); _indices vs the stock torch lines of
dflash_forward (its Triton fallback is deleted too); the kill switch (None: the caller's stock path); no Triton import
in the ported modules."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_ctx_advance as AD,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_forward_indices as IX,
)

_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the DSpark glue ops run on sm_107 (Rubin) only",
)


def _ri(g, lo, hi, shape, dtype=torch.int64):
    return torch.randint(lo, hi, shape, generator=g, device="cuda", dtype=dtype)


def _ref_advance(
    slots, rows, nacc, ctx, max_ctx, dummy, counts, base, tables, read, page, block_size
):
    """The documented contract (dsparkGlueSm107.h) in torch: ctx / read updated in place, returns out."""
    slots, rows = slots.long(), rows.long()
    new = torch.clamp_max(ctx[slots] + nacc.long(), max_ctx)
    keep = slots != dummy
    ctx[slots[keep]] = new[keep]
    if slots.numel():
        ctx[dummy] = 0
    new = torch.where(keep, new, torch.zeros_like(new))
    num = torch.minimum(new, torch.clamp_min(counts[rows] * page - block_size, 0))
    b = base[slots]
    rebase = torch.minimum(b, torch.div(num, page, rounding_mode="floor"))
    width = tables.shape[1]
    if width:
        j = torch.arange(width, device=tables.device)
        src = torch.clamp_max(j[None, :] + rebase[:, None], width - 1)
        read[rows] = torch.gather(tables[rows], 1, src)
    return torch.clamp_min(num - b * page, 0)


_GPU_PAR = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU and the DSPARK_GLUE_CPP ADV_PAR op",
)


@_GPU
@_GPU_PAR
@pytest.mark.parametrize("width", [0, 1, 511, 513, 2048, 8192, 65536])
@pytest.mark.parametrize("n", [1, 7, 22, 56, 129])
@pytest.mark.parametrize("nacc_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("parallel", [False, True])
def test_advance_bitwise(monkeypatch, width, n, nacc_dtype, parallel):
    g = torch.Generator(device="cuda").manual_seed(width + 31 * n)
    nslots = nrows = 160
    page, block_size = 128, 7
    slots = torch.randperm(nslots - 1, generator=g, device="cuda")[:n].to(torch.int64)
    if n > 2:
        slots[-1] = nslots - 1  # a dummy (padding) request
    if n > 3:
        slots[0] = nslots - 1  # request 0 on the dummy slot (it also zeroes ctx_len[dummy])
    rows = torch.randperm(nrows, generator=g, device="cuda")[:n].to(torch.int64)
    nacc = _ri(g, 0, 9, (n,), nacc_dtype)
    max_ctx = max(width, 16) * page - 1000
    ctx0 = _ri(g, 0, max_ctx, (nslots,))
    ctx0[:2] = max_ctx  # at the clamp
    ctx0[nslots - 1] = 12345  # the dummy slot's stale length (zeroed by request 0)
    counts = _ri(g, 0, max(width, 16) + 1, (nrows,))
    counts[0] = 0  # nothing allocated
    base = _ri(g, 0, max(width, 16) // 4, (nslots,))
    base[slots[1 % n]] = 2**40 // page  # a context shorter than its base
    tables = _ri(g, 0, 2**31 - 1, (nrows, width), torch.int32)

    def run(fn, **kw):
        ctx, read = ctx0.clone(), torch.full_like(tables, -7)
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
        out = fn(*args, **kw)
        torch.cuda.synchronize()
        return out, ctx, read

    name = "minimax_m31_dspark_advance_par" if parallel else "minimax_m31_dspark_advance"
    got = run(AD.advance_ctx_and_read_view, parallel=parallel)
    ref = run(_ref_advance)
    assert got[0] is not None and (name, None) in AD._glue_logged
    for a, b in zip(got, ref):
        assert a.dtype == b.dtype and torch.equal(a, b)
    # Killed: nothing computed or written, None (the caller's stock path), the engagement key logged.
    monkeypatch.setattr(AD, "_GLUE_CPP_OFF", True)
    out, ctx, read = run(AD.advance_ctx_and_read_view, parallel=parallel)
    assert out is None and torch.equal(ctx, ctx0) and bool((read == -7).all())
    assert (name, "killed (TRTLLM_M31_DISABLE_DSPARK_GLUE_CPP=1)") in AD._glue_logged


@_GPU
@_GPU_PAR
@pytest.mark.parametrize("parallel", [False, True])
def test_advance_strided_tables_and_pdl_chain(parallel):
    """Row-strided tables / read view (column stride 1), int32 slots / rows (the wrapper widens them), and two
    back-to-back steps on one stream in a CUDA graph (PDL launches chained on their own outputs)."""
    g = torch.Generator(device="cuda").manual_seed(5)
    n, nslots, nrows, width, page, block_size = 8, 12, 10, 700, 64, 3
    slots = torch.randperm(nslots - 1, generator=g, device="cuda")[:n].int()
    slots[-1] = nslots - 1
    rows = torch.randperm(nrows, generator=g, device="cuda")[:n].int()
    nacc = _ri(g, 1, 5, (n,), torch.int32)
    max_ctx = width * page
    ctx0 = _ri(g, 0, max_ctx, (nslots,))
    counts = _ri(g, 0, width + 1, (nrows,))
    base = _ri(g, 0, 6, (nslots,))
    big = _ri(g, 0, 2**31 - 1, (nrows, width + 37), torch.int32)
    tables = big[:, 5 : 5 + width]
    ref_ctx, ref_read = (
        ctx0.clone(),
        torch.full((nrows, width), -3, dtype=torch.int32, device="cuda"),
    )
    outs_ref = [
        _ref_advance(
            slots,
            rows,
            nacc,
            ref_ctx,
            max_ctx,
            nslots - 1,
            counts,
            base,
            tables,
            ref_read,
            page,
            block_size,
        )
        for _ in range(2)
    ]
    ctx = ctx0.clone()
    read_big = torch.full((nrows, width + 9), -3, dtype=torch.int32, device="cuda")
    read = read_big[:, :width]
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
    AD.advance_ctx_and_read_view(*args, parallel=parallel)  # warm-up outside capture
    ctx.copy_(ctx0)
    read.fill_(-3)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s):
        outs = [AD.advance_ctx_and_read_view(*args, parallel=parallel) for _ in range(2)]
    torch.cuda.current_stream().wait_stream(s)
    ctx.copy_(ctx0)
    read.fill_(-3)
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(outs, outs_ref):
        assert torch.equal(a, b)
    assert torch.equal(ctx, ref_ctx) and torch.equal(read, ref_read)


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

    got = run()
    assert ("minimax_m31_dspark_indices", None) in IX._glue_logged
    ref = _ref_indices(num_ctx, batch_idx, qpos, block, pages_per_slot, page)
    for a, b in zip(got, ref, strict=True):
        assert a.dtype == b.dtype and torch.equal(a, b)
    monkeypatch.setattr(IX, "_GLUE_CPP_OFF", True)
    assert run() is None  # killed: the caller's stock lines


def _ref_indices(num_ctx, batch_idx, qpos, block, pages_per_slot, page):
    """dflash_forward's stock paged-prelude lines (torch)."""
    B = qpos.shape[0]
    seqlens = num_ctx[:B].to(torch.int32)
    after = seqlens + block
    indptr = torch.arange(
        0, (B + 1) * pages_per_slot, pages_per_slot, dtype=torch.int32, device="cuda"
    )
    last = ((after - 1) % page) + 1
    append = (seqlens.view(-1, 1) + torch.arange(block, dtype=torch.int32, device="cuda")).reshape(
        -1
    )
    return (
        seqlens,
        batch_idx.to(torch.int32),
        after,
        indptr,
        last,
        append,
        qpos.reshape(-1).to(torch.int32),
    )


def test_kill_switch():
    code = (
        "import torch\n"
        "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import "
        "dspark_ctx_advance as a, dspark_forward_indices as i\n"
        "print(a._GLUE_CPP_OFF, i._GLUE_CPP_OFF, a._glue_cpp_op('x', torch.device('cuda', 0)) is None)"
    )
    env = {**os.environ, "TRTLLM_M31_DISABLE_DSPARK_GLUE_CPP": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split()[-3:] == ["True", "True", "True"]


def test_no_triton_in_the_ported_modules():
    """No Triton in the served DSpark glue paths (the ported modules import no triton, define no kernel)."""
    import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark as pkg

    root = Path(pkg.__file__).parent
    for name in ("dspark_ctx_advance.py", "dspark_qk_norm_rope.py", "dspark_forward_indices.py"):
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] == "triton" for a in node.names), name
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "triton", name
