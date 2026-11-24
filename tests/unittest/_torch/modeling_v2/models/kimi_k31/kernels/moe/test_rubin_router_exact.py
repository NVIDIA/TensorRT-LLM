# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kimi K3.1 small-batch MoE front end: the exact router logits.

``rubin_router_exact.k3_router_logits`` vs ``trtllm::dsv3_router_gemm_op``: identical FP32 bits
for ``M <= 16``, eager and in a replayed CUDA graph.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU",
)

_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="needs SM100/SM103/SM107",
)

E, H, LAT = 896, 7168, 3584


def _hidden(m, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, H), device="cuda", generator=gen)
    x[:, ::7] *= 30.0
    x[:, 1::11] = 0.0
    return x.to(torch.bfloat16)


def _gate(seed):
    gen = torch.Generator(device="cuda").manual_seed(1000 + seed)
    return (torch.randn((E, H), device="cuda", generator=gen) * 0.04).to(torch.bfloat16)


def _bits(t):
    return t.contiguous().view(torch.int32)


@_GPU
@pytest.mark.parametrize("m", [1, 2, 7, 8, 9, 14, 16])
@pytest.mark.parametrize("seed", [0, 1])
def test_router_exact_bitwise(m, seed):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
        rubin_router_exact,
    )

    w = _gate(seed)
    x = _hidden(m, seed)
    ref = torch.ops.trtllm.dsv3_router_gemm_op(x, w.t(), bias=None, out_dtype=torch.float32)
    out = rubin_router_exact.k3_router_logits(x, w)
    assert out.shape == (m, E) and out.dtype == torch.float32
    assert torch.equal(_bits(out), _bits(ref))


@_GPU
def test_router_exact_cuda_graph():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
        rubin_router_exact,
    )

    w = _gate(3)
    x = _hidden(14, 3)
    rubin_router_exact.k3_router_logits(x, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = rubin_router_exact.k3_router_logits(x, w)
    for seed in range(3):
        x.copy_(_hidden(14, 10 + seed))
        graph.replay()
        torch.cuda.synchronize()
        ref = torch.ops.trtllm.dsv3_router_gemm_op(x, w.t(), bias=None, out_dtype=torch.float32)
        assert torch.equal(_bits(out), _bits(ref))


@_GPU
def test_router_exact_can_run():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
        rubin_router_exact,
    )

    w = _gate(0)
    assert rubin_router_exact.can_run(_hidden(16, 0), w)
    assert not rubin_router_exact.can_run(_hidden(17, 0), w)
    assert not rubin_router_exact.can_run(_hidden(4, 0).float(), w)
    assert not rubin_router_exact.can_run(_hidden(4, 0), w[:448])
    assert not rubin_router_exact.can_run(_hidden(4, 0), w.float())
