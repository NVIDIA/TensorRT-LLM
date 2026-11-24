# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 index_key_norm entry (the DSA indexer's k LayerNorm, one fixed reduction order).

The raw key is the wk column view of a packed TP16 projection row, as the target passes it. The op is gated
three ways: within one bf16 ulp of the plain torch LayerNorm (its catalog reference); bit for bit against a
Triton copy of the kernel torch.compile builds for the module at the configuration most ranks pick
(XBLOCK=8, num_warps=2; another configuration, XBLOCK=1, is shown to differ, so the gate can see an order
change); and bit for bit across launch shapes and across processes (what two TP ranks compute).
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention.dsa_index_key_norm import (  # noqa: E501
    index_key_norm_into,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_index_key_norm import (  # noqa: E501
    index_key_norm as op,
)

__extra_import_path__ = ["../../../..", "."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _validating import certifying  # noqa: E402,F401

pytestmark = inputs.on_sm103


def _build(spec, seed: int):
    gen = torch.Generator().manual_seed(seed)
    n = spec["tokens"]
    x = torch.randn(n, spec["row_stride"], generator=gen)
    if spec["data"] == "offset":
        x = x + 50 * torch.randn(n, 1, generator=gen)
    elif spec["data"] == "flat":
        x[::2] = 0
        x[1::2] = 1.5
    elif spec["data"] == "wide":
        x = x * torch.pow(10.0, torch.randint(-4, 5, (n, 1), generator=gen).float())
    packed = x.to(torch.bfloat16).cuda()
    col = spec["column"]
    weight = (1 + 0.2 * torch.randn(128, generator=gen)).float().cuda()
    bias = (0.1 * torch.randn(128, generator=gen)).float().cuda()
    return dict(x=packed[:, col : col + 128], weight=weight, bias=bias, eps=spec["eps"])


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """The bf16 spacing at |x| (8 significant bits)."""
    e = torch.floor(torch.log2(x.abs().clamp_min(2.0**-126)))
    return torch.pow(2.0, e - 7)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    got = op(**args)
    want = op.reference(**args)
    torch.cuda.synchronize()
    assert got.shape == want.shape == (cell.spec["tokens"], 128) and got.dtype == torch.bfloat16
    # The two reduce in different orders: the fp32 statistics differ in their last bits, which moves an output by
    # at most one bf16 step, plus an absolute 1e-5 where the output cancels to near zero (x near the mean, or the
    # bias cancelling the scaled value); and only a few outputs in ten thousand move at all.
    g, w = got.float(), want.float()
    excess = (g - w).abs() - _bf16_ulp(w)
    assert float(excess.max()) <= 1e-5, float(excess.max())
    moved = int((got.view(torch.int16) != want.view(torch.int16)).sum())
    assert moved <= max(2, got.numel() // 10000), moved


# ---------------------------------------------------------------------------------------------
# The kernel torch.compile builds for the module's forward, at two of its launch configurations.
# ---------------------------------------------------------------------------------------------

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")
libdevice = pytest.importorskip("torch._inductor.runtime.triton_helpers").libdevice


@triton.jit
def _inductor_layer_norm(
    in_ptr0, in_ptr1, in_ptr2, out_ptr2, ks0, xnumel, EPS: tl.constexpr, XBLOCK: tl.constexpr
):
    # The body Inductor generates for LayerNorm.forward (triton_per_fused__to_copy_native_layer_norm_0),
    # with its eps as a constexpr.
    R0_BLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + ks0 * x0), xmask, eviction_policy="evict_first", other=0.0).to(
        tl.float32
    )
    tmp24 = tl.load(in_ptr1 + (r0_1), None, eviction_policy="evict_last")
    tmp26 = tl.load(in_ptr2 + (r0_1), None, eviction_policy="evict_last")
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp9 = (tl.full([1, 1], 128, tl.int32)).to(tl.float32)
    tmp10 = tmp8 / tmp9
    tmp11 = tmp2 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp17 = tmp1 - tmp10
    tmp18 = tl.full([1, 1], 128.0, tl.float32)
    tmp19 = tmp16 / tmp18
    tmp20 = tl.full([1, 1], EPS, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp27.to(tl.float32)
    tl.store(out_ptr2 + (r0_1 + 128 * x0), tmp28, xmask)


def _inductor(x, weight, bias, eps, xblock):
    n = x.shape[0]
    out = torch.empty(n, 128, dtype=torch.bfloat16, device=x.device)
    _inductor_layer_norm[(triton.cdiv(n, xblock),)](
        x, weight, bias, out, x.stride(0), n, EPS=eps, XBLOCK=xblock, num_warps=2, num_stages=1
    )
    return out


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_bitwise_to_the_compiled_module_kernel(cell):
    args = _build(cell.spec, inputs.seed_of(cell) + 1)
    got = op(**args)
    want = _inductor(args["x"], args["weight"], args["bias"], args["eps"], 8)
    torch.cuda.synchronize()
    assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (
        f"{int((got.view(torch.int16) != want.view(torch.int16)).sum())} elements differ"
    )


def test_another_compiled_configuration_differs():
    """XBLOCK=1 sums each row in another order: on enough rows some outputs differ from XBLOCK=8, so the bitwise
    gate above sees a reduction-order change."""
    diff = 0
    for seed in range(4):
        args = _build(
            dict(tokens=16384, row_stride=3040, column=2112, eps=1e-5, data="randn"), seed
        )
        a = _inductor(args["x"], args["weight"], args["bias"], args["eps"], 8)
        b = _inductor(args["x"], args["weight"], args["bias"], args["eps"], 1)
        diff += int((a.view(torch.int16) != b.view(torch.int16)).sum())
    assert diff > 0


# ---------------------------------------------------------------------------------------------
# Rank consistency: one result whatever the launch shape, the process or the device.
# ---------------------------------------------------------------------------------------------


def test_same_bits_for_every_launch_shape():
    args = _build(dict(tokens=3000, row_stride=3040, column=2112, eps=1e-5, data="randn"), 7)
    outs = []
    for rows_per_cta in (4, 8, 16, 32):
        for use_pdl in (False, True):
            out = torch.empty(3000, 128, dtype=torch.bfloat16, device="cuda")
            index_key_norm_into(
                args["x"],
                args["weight"],
                args["bias"],
                args["eps"],
                out,
                3000,
                use_pdl,
                rows_per_cta,
            )
            outs.append(out)
    torch.cuda.synchronize()
    for out in outs[1:]:
        assert torch.equal(out.view(torch.int16), outs[0].view(torch.int16))


_CHILD = textwrap.dedent(
    """
    import hashlib, sys, torch
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_index_key_norm import (
        index_key_norm,
    )
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(16384, 3040, generator=gen).to(torch.bfloat16).cuda()[:, 2112:2240]
    w = (1 + 0.2 * torch.randn(128, generator=gen)).float().cuda()
    b = (0.1 * torch.randn(128, generator=gen)).float().cuda()
    y = index_key_norm(x, w, b, 1e-5)
    torch.cuda.synchronize()
    print("HASH", hashlib.sha256(y.cpu().view(torch.int16).numpy().tobytes()).hexdigest())
    """
)


def test_same_bits_across_processes_and_devices():
    """Two processes (on two GPUs when there are two), as two TP ranks would: identical bits."""
    devices = ["0", "1"] if torch.cuda.device_count() > 1 else ["0", "0"]
    hashes = []
    for dev in devices:
        # A fresh process, not another rank of a launcher this test may run under (MPI / PMIx / Slurm).
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("OMPI_", "PMIX_", "PMI_", "SLURM_", "MPI_"))
        }
        visible = env.get("CUDA_VISIBLE_DEVICES")
        env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[int(dev)] if visible else dev
        out = subprocess.run(
            [sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, timeout=900
        )
        assert out.returncode == 0, out.stderr[-2000:]
        hashes.append([line for line in out.stdout.splitlines() if line.startswith("HASH")][-1])
    assert hashes[0] == hashes[1], hashes
