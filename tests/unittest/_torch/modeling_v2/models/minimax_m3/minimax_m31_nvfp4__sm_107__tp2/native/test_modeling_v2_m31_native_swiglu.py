# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE_SWIGLU: trtllm::minimax_m31_swiglu_oai_vec is bit for bit the Triton silu_and_mul it replaces (limit 7, alpha
1.702, beta 1) on every bf16 gate pattern x the special up values and the reverse, random cells at the TP2 widths
(shared expert d 1536, dense d 6144) and the pair fallback (d % 8 != 0), for every vector width and with or without
the programmatic launch, eager and replayed from a CUDA graph."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.silu_and_mul
from tensorrt_llm._torch._experimental.modeling_v2.catalog.fakes import m31_fake_ops

_ops = torch.ops.trtllm
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(_ops, "minimax_m31_swiglu_oai_vec"),
    reason="needs a CUDA device and the native SwiGLU op",
)
LIMIT, ALPHA, BETA = 7.0, 1.702, 1.0


def _triton(x):
    return _ops.silu_and_mul(x, swiglu_limit=LIMIT, swiglu_alpha=ALPHA, swiglu_beta=BETA)


def _bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def _specials(n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    vals = [0.0, -0.0, 1.0, -1.0, 6.99, 7.0, 7.01, -7.0, -7.01, 1e-38, -1e-38, 1e-40, 88.0, -88.0, 3e38, -3e38,
            float("inf"), float("-inf"), float("nan")]  # fmt: skip
    rows = [torch.full((n,), v, dtype=torch.bfloat16, device="cuda") for v in vals]
    rows += [
        (s * torch.randn(n, generator=g, device="cuda")).to(torch.bfloat16)
        for s in (0.05, 1.0, 4.0, 30.0)
    ]
    rows.append(
        torch.randint(-(1 << 15), 1 << 15, (n,), generator=g, device="cuda")
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    return rows


@pytest.mark.parametrize("nv", [1, 2, 4])
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("gate_side", [True, False])
def test_every_bf16_pattern(nv, pdl, gate_side):
    allp = (
        torch.arange(1 << 16, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16)
    )
    for j, other in enumerate(_specials(1 << 16, 1)):
        g, u = (allp, other) if gate_side else (other, allp)
        for rows in (1, 256):
            x = torch.cat([g.view(rows, -1), u.view(rows, -1)], 1).contiguous()
            assert _bits_equal(_ops.minimax_m31_swiglu_oai_vec(x, pdl, nv), _triton(x)), (
                f"set {j} rows {rows}"
            )


@pytest.mark.parametrize("d", [1536, 6144, 1532])
@pytest.mark.parametrize("rows", [1, 8, 49, 224, 448, 2048, 4096])
@pytest.mark.parametrize("nv", [1, 2, 4])
@pytest.mark.parametrize("pdl", [False, True])
def test_tp2_shapes(d, rows, nv, pdl):
    g = torch.Generator(device="cuda").manual_seed(d * 7 + rows)
    x = (1.5 * torch.randn(rows, 2 * d, generator=g, device="cuda")).to(torch.bfloat16)
    assert _bits_equal(_ops.minimax_m31_swiglu_oai_vec(x, pdl, nv), _triton(x))


def test_graph_replay():
    x = (1.5 * torch.randn(2048, 2 * 1536, device="cuda")).to(torch.bfloat16)
    ref = _triton(x)
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        _ops.minimax_m31_swiglu_oai_vec(x, True, 2)
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=st):
        out = _ops.minimax_m31_swiglu_oai_vec(x, True, 2)
    for _ in range(3):
        gr.replay()
    torch.cuda.synchronize()
    assert _bits_equal(out, ref)


@pytest.mark.parametrize("rows,d", [(1, 1536), (2048, 6144)])
def test_fake_impl_matches_the_op(rows, d):
    """The registered fake (catalog m31_fake_ops) gives the op's output shape and dtype."""
    m31_fake_ops.register()
    x = torch.randn(rows, 2 * d, device="cuda").to(torch.bfloat16)
    real = _ops.minimax_m31_swiglu_oai_vec(x, True, 1)
    with FakeTensorMode(allow_non_fake_inputs=True) as mode:
        fake = _ops.minimax_m31_swiglu_oai_vec(mode.from_tensor(x), True, 1)
    assert (tuple(fake.shape), fake.dtype) == (tuple(real.shape), real.dtype)
