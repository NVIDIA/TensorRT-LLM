# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE_SWIGLU (family ``minimax_m3/ops/native_swiglu``): the row-grid mode (nv 0) of
``trtllm::minimax_m31_swiglu_oai_vec`` is bit for bit the Triton ``silu_and_mul`` at limit 7 / alpha 1.702 / beta 1
(the DEP8 dense / drafter MLP rows, d 12288, with clamped, signed-zero, subnormal and non-finite values; PDL on and
off; inside a CUDA graph); other widths too (d not a multiple of 8 takes the pair shape); and an installed
GatedMLP's activation is the op's, with the torch SwiGLU-OAI for operands outside the contract."""

import types

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import native_swiglu
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the native SwiGLU-OAI op runs on sm_107 (Rubin) only",
)

D = 12288


def _triton(x):
    return torch.ops.trtllm.silu_and_mul(x, swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_beta=1.0)


def _bits(t):
    return t.contiguous().view(torch.int16)


def _gu(rows, d, gen):
    x = (torch.randn(rows, 2 * d, generator=gen, device="cuda") * 4.0).to(torch.bfloat16)
    x[:, 1] = 9.0  # gate above the limit
    x[:, d + 2] = -11.0  # up below -limit
    x[:, 3] = -0.0
    x[:, d + 4] = 1e-40
    x[:, 5] = float("-inf")
    x[:, d + 6] = float("inf")
    return x


@pytest.mark.parametrize("rows", [1, 2, 8, 84, 96, 112, 128])
@pytest.mark.parametrize("pdl", [True, False])
def test_the_row_grid_is_bitwise_the_triton_kernel(rows, pdl):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    x = _gu(rows, D, gen)
    got = native_swiglu.swiglu_op()(x, pdl, 0)
    want = _triton(x)
    assert torch.equal(torch.isnan(got), torch.isnan(want))
    keep = ~torch.isnan(want)
    assert torch.equal(_bits(got)[keep], _bits(want)[keep])


@pytest.mark.parametrize("d", [12, 1000, 1536])
def test_other_widths_are_bitwise_the_triton_kernel(d):
    gen = torch.Generator(device="cuda").manual_seed(d)
    x = (torch.randn(16, 2 * d, generator=gen, device="cuda") * 4.0).to(torch.bfloat16)
    assert torch.equal(_bits(native_swiglu.swiglu_op()(x, True, 0)), _bits(_triton(x)))


def test_the_row_grid_replays_inside_a_cuda_graph():
    gen = torch.Generator(device="cuda").manual_seed(3)
    x = _gu(96, D, gen)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        native_swiglu.swiglu_oai(x)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = native_swiglu.swiglu_oai(x)
    x.copy_(_gu(96, D, gen))
    graph.replay()
    torch.cuda.synchronize()
    want = _triton(x)
    keep = ~torch.isnan(want)
    assert torch.equal(_bits(out)[keep], _bits(want)[keep])


def _mlp(alpha=1.702, beta=1.0, limit=7.0, fp8=False):
    """A GatedMLP shell carrying only what ``serves`` / ``_apply_activation`` read."""
    mlp = GatedMLP.__new__(GatedMLP)
    torch.nn.Module.__init__(mlp)
    mlp.activation = F.silu
    mlp.split_gate_up = False
    mlp.swiglu_alpha, mlp.swiglu_beta, mlp.swiglu_limit = alpha, beta, limit
    mlp.down_proj = types.SimpleNamespace(has_fp8_qdq=fp8, has_w4a8_nvfp4_fp8=False)
    return mlp


def test_install_takes_the_swiglu_oai_mlps_only():
    mlps = [
        _mlp(),
        _mlp(alpha=None, beta=None),
        _mlp(fp8=True),
        torch.nn.Linear(2, 2),
        None,
        _mlp(),
    ]
    assert native_swiglu.install(mlps) == 2
    assert native_swiglu.install(mlps) == 0  # once per instance
    gen = torch.Generator(device="cuda").manual_seed(5)
    x = _gu(84, D, gen)
    keep = ~torch.isnan(_triton(x))
    assert torch.equal(_bits(mlps[0]._apply_activation(x))[keep], _bits(_triton(x))[keep])
    odd = x.float()  # outside the contract: the torch SwiGLU-OAI, not the Triton kernel
    torch.testing.assert_close(
        mlps[0]._apply_activation(odd),
        native_swiglu.swiglu_oai_torch(odd),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_the_torch_form_is_close_to_the_triton_kernel():
    gen = torch.Generator(device="cuda").manual_seed(7)
    x = (torch.randn(32, 2 * D, generator=gen, device="cuda") * 4.0).to(torch.bfloat16)
    torch.testing.assert_close(
        native_swiglu.swiglu_oai_torch(x).float(), _triton(x).float(), atol=2e-2, rtol=2e-2
    )
