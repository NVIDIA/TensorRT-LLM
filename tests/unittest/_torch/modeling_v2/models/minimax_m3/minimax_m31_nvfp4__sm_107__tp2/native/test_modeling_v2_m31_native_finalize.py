# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE_FINALIZE: trtllm::minimax_m31_moe_finalize is bit for bit the Triton kernels/m3_moe_finalize it replaces: the
routed experts' finalize from the trtllm-gen runner's unfinalized layout (fc2 permuted rows, expanded -> permuted
index with -1 = dropped, bf16 or fp32 weights), with and without the shared expert's add, for every work-split table
entry, at decode and prefill row counts, with denormal products and dropped slots."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tensorrt_llm._torch._experimental.modeling_v2.catalog.fakes import m31_fake_ops
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels.m3_moe_finalize import (  # noqa: E501
    m3_moe_finalize,
)

_ops = torch.ops.trtllm
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(_ops, "minimax_m31_moe_finalize"),
    reason="needs a CUDA device and the native finalize op",
)
H = 6144


def _parts(tokens, topk, kind, seed, wdtype):
    g = torch.Generator(device="cuda").manual_seed(seed)
    p = tokens * topk + 128
    fc2 = (torch.randn(p, H, generator=g, device="cuda") * 2).to(torch.bfloat16)
    perm = (
        torch.randperm(p, generator=g, device="cuda")[: tokens * topk]
        .to(torch.int32)
        .view(tokens, topk)
    )
    w = torch.rand(tokens, topk, generator=g, device="cuda") / topk
    if kind == "dropped":
        perm = torch.where(torch.rand(perm.shape, generator=g, device="cuda") < 0.05, -1, perm).to(
            torch.int32
        )
    if kind == "denormal":
        w = w * 2.0**-126
    shared = (torch.randn(tokens, H, generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    return fc2, w.to(wdtype).contiguous(), perm.contiguous(), shared


def _bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


@pytest.mark.parametrize("tokens", [1, 49, 176, 224, 448, 2048, 4096])
@pytest.mark.parametrize("topk", [4, 8])
@pytest.mark.parametrize("kind", ["randn", "dropped", "denormal"])
@pytest.mark.parametrize("wdtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_shared", [False, True])
def test_every_cfg_is_the_triton_finalize(tokens, topk, kind, wdtype, with_shared):
    fc2, w, perm, shared = _parts(tokens, topk, kind, tokens * 31 + topk, wdtype)
    sh = shared if with_shared else None
    ref = m3_moe_finalize(fc2, w, perm, sh)
    for cfg in (-1, 0, 1, 2, 3, 4, 5, 6, 7):
        assert _bits_equal(_ops.minimax_m31_moe_finalize(fc2, w, perm, sh, cfg), ref), f"cfg {cfg}"


def test_graph_replay():
    fc2, w, perm, shared = _parts(2048, 8, "dropped", 5, torch.bfloat16)
    ref = m3_moe_finalize(fc2, w, perm, shared)
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        _ops.minimax_m31_moe_finalize(fc2, w, perm, shared, 4)
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=st):
        out = _ops.minimax_m31_moe_finalize(fc2, w, perm, shared, 4)
    for _ in range(3):
        gr.replay()
    torch.cuda.synchronize()
    assert _bits_equal(out, ref)


@pytest.mark.parametrize("with_shared", [False, True])
def test_fake_impl_matches_the_op(with_shared):
    """The registered fake (catalog m31_fake_ops) gives the op's output shape and dtype."""
    m31_fake_ops.register()
    fc2, w, perm, shared = _parts(176, 8, "randn", 3, torch.bfloat16)
    sh = shared if with_shared else None
    real = _ops.minimax_m31_moe_finalize(fc2, w, perm, sh, -1)
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fake = _ops.minimax_m31_moe_finalize(
            mode.from_tensor(fc2),
            mode.from_tensor(w),
            mode.from_tensor(perm),
            None if sh is None else mode.from_tensor(sh),
            -1,
        )
    assert (tuple(fake.shape), fake.dtype) == (tuple(real.shape), real.dtype)
