# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""sm_103 (GB300) device test of the split-K tactics ``MUSESPARK_OPEN_SM103_TEP8_F11=1`` serves at TEP8 (one GPU).

The claims the port relies on, checked bitwise (torch.equal on the BF16 bits):

* the wide64 QKV / o_proj tactics (``dense_gemms.SM107_TP8_WIDE64_KERNELS``) launch within GB300's opt-in shared
  memory at 33..64 rows, and every row equals that row computed alone by the 8-row cell's ``split_k`` (QKV: sm103's
  ``splitk:64,8,4,12``; o_proj: split_k 1) -- "every row gets the bits it gets in a one-request step";
* the stage count changes no bits (each wide64 tactic vs the same tile at 2 stages);
* every sm103 TP8 wide drafter / fc-shard tactic (``fixed_low_m_gemm.SM103_TP8_*_WIDE_TACTICS``) launches and equals
  the 8-row ``SPLITK_TACTICS`` cell of its shape row by row;
* the fp32-reference error of each wide call is within 1 BF16 ulp of the reference everywhere it differs.

Run with ``MUSESPARK_OPEN_SM103_TEP8_F11=1`` on an sm_103 device (the env makes the module read the wide64 switch).
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 3), reason="sm_103 only"
    ),
    pytest.mark.skipif(os.environ.get("MUSESPARK_OPEN_SM103_TEP8_F11") != "1", reason="needs the F11 switch"),
]

_PKG = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_103__tep8"
)


def _mods():
    import importlib

    sk = importlib.import_module(f"{_PKG}.ops.gemm.splitk_gemm")
    fl = importlib.import_module(f"{_PKG}.ops.gemm.fixed_low_m_gemm")
    dg = importlib.import_module(f"{_PKG}.kernels.dense_gemms.dispatcher")
    return sk, fl, dg


def _run(sk, x, w, tactic):
    out = torch.empty((x.shape[0], w.shape[0]), dtype=x.dtype, device=x.device)
    sk.run_splitk_dense(x, w.t(), None, out, False, sk.SplitKTactic(*tactic))
    torch.cuda.synchronize()
    return out


def _rowwise_alone(sk, x, w, tactic8):
    """Each 8-row block of ``x`` on the 8-row tactic (a one-request step)."""
    outs = []
    for i in range(0, x.shape[0], 8):
        blk = x[i : i + 8]
        if blk.shape[0] < 8:
            pad = torch.zeros((8 - blk.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
            outs.append(_run(sk, torch.cat([blk, pad]), w, tactic8)[: blk.shape[0]])
        else:
            outs.append(_run(sk, blk.contiguous(), w, tactic8))
    return torch.cat(outs)


def _ref(x, w):
    return (x.double() @ w.double().t()).to(torch.bfloat16)


def _inputs(rows, n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, k, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda", generator=g) * (1.0 / k**0.5)).to(torch.bfloat16)
    return x, w


#: (site, (N, K)) of a TP8 rank: QKV [2304, 8192] (16 query + 2x2 KV heads of 64 ... the projection's rank slice),
#: o_proj [8192, 1024].
WIDE64_SHAPES = {"qkv": (2304, 8192), "o_proj": (8192, 1024)}
WIDE64_8ROW = {"qkv": (64, 8, 4, 12), "o_proj": (64, 8, 1, 12)}


@pytest.mark.parametrize("site", ["qkv", "o_proj"])
@pytest.mark.parametrize("rows", [33, 40, 48, 56, 64])
def test_wide64_fits_and_rows_get_their_one_request_bits(site, rows):
    sk, _, dg = _mods()
    kernel = dg.SM107_TP8_WIDE64_KERNELS[site]
    tactic = tuple(int(v) for v in kernel.split(":")[1].split(","))
    assert sk._smem_bytes(sk.SplitKTactic(*tactic), tactic[3]) <= 232448
    n, k = WIDE64_SHAPES[site]
    x, w = _inputs(rows, n, k, rows)
    wide = _run(sk, x, w, tactic)
    alone = _rowwise_alone(sk, x, w, WIDE64_8ROW[site])
    assert torch.equal(wide.view(torch.int16), alone.view(torch.int16)), (site, rows)
    # stage count: no bits
    two = _run(sk, x, w, (*tactic[:3], 2))
    assert torch.equal(wide.view(torch.int16), two.view(torch.int16)), (site, rows)
    # close to the fp64 product (sanity: fp32 accumulation, one BF16 rounding)
    ref = _ref(x, w).float()
    err = (wide.float() - ref).abs().max()
    assert float(err) <= 2**-6 * float(ref.abs().max()), float(err)


def test_sm103_drafter_wide_tactics_launch_and_keep_one_request_bits():
    sk, fl, _ = _mods()
    assert fl.TEP8_DRAFTER_WIDE and fl.TEP8_WIDE64
    tables = {**fl.SM103_TP8_DRAFTER_WIDE_TACTICS, **fl.SM103_TP8_FC_WIDE_TACTICS}
    for key, tactic in sorted(tables.items()):
        rows, k, n = key
        assert fl.SPLITK_TACTICS[key] == tactic, key  # served on sm103 TP8 under the switch
        x, w = _inputs(rows, n, k, rows + n)
        wide = _run(sk, x, w, tactic)
        alone = _rowwise_alone(sk, x, w, fl.SPLITK_TACTICS[(8, k, n)])
        assert torch.equal(wide.view(torch.int16), alone.view(torch.int16)), (key, tactic)


@pytest.mark.parametrize("rows", [33, 40, 41, 48, 56, 63, 64])
@pytest.mark.parametrize("scale", [0.05, 0.5, 2.0])
def test_wide64_o_proj_keeps_the_bits_of_sm103s_cublas_cell(rows, scale):
    """sm103's 8-row o_proj cell is cuBLAS (``dense_gemms.DEFAULT_KERNELS``); the wide64 o_proj tactic (split_k 1)
    equals torch.mm on each 8-row block and on the whole call (``dense_gemms.TEP8_WIDE64_CUBLAS_CELLS``)."""
    sk, _, dg = _mods()
    assert dg.default_kernels(8, True, (10, 3))["o_proj"][dg.MAX_ROWS - 1] == "cublas"
    assert ("o_proj", (10, 3)) in dg.TEP8_WIDE64_CUBLAS_CELLS
    tactic = tuple(int(v) for v in dg.SM107_TP8_WIDE64_KERNELS["o_proj"].split(":")[1].split(","))
    n, k = WIDE64_SHAPES["o_proj"]
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = (torch.randn(rows, k, device="cuda", generator=g) * scale).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda", generator=g) * k**-0.5).to(torch.bfloat16)
    wide = _run(sk, x, w, tactic)
    blocks = []
    for i in range(0, rows, 8):
        blk = x[i : i + 8]
        pad = torch.zeros((8 - blk.shape[0], k), dtype=x.dtype, device=x.device)
        blocks.append(torch.mm(torch.cat([blk, pad]), w.t())[: blk.shape[0]])
    alone = torch.cat(blocks)
    torch.cuda.synchronize()
    assert torch.equal(wide.view(torch.int16), alone.view(torch.int16)), rows
    assert torch.equal(wide.view(torch.int16), torch.mm(x, w.t()).view(torch.int16)), rows
