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
"""Kimi-K3 router logits kernel (``kimi_k3_router_logits.py``) vs ``trtllm::dsv3_router_gemm_op``.

The kernel must produce the same FP32 bits as ``dsv3_router_gemm_op`` for every ``M <= 16``,
eager, in a replayed CUDA graph, and launched with PDL right behind a producer of its input.
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE,
    IS_CUTLASS_DSL_RUBIN_AVAILABLE,
)


def _capability():
    return torch.cuda.get_device_capability() if torch.cuda.is_available() else None


pytestmark = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE
    or not (
        _capability() in ((10, 0), (10, 3))
        or (_capability() == (10, 7) and IS_CUTLASS_DSL_RUBIN_AVAILABLE)
    ),
    reason="needs CuTe DSL and an SM100/SM103 GPU, or SM107 with a Rubin-capable CuTe DSL",
)

E, H = 896, 7168


def _kernel_module():
    from tensorrt_llm._torch.moe.kernels.blackwell import kimi_k3_router_logits

    return kimi_k3_router_logits


def _hidden(m, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, H), device="cuda", generator=gen)
    # Large and zero columns stress the accumulation order.
    x[:, ::7] *= 30.0
    x[:, 1::11] = 0.0
    return x.to(torch.bfloat16)


def _gate(seed):
    gen = torch.Generator(device="cuda").manual_seed(1000 + seed)
    return (torch.randn((E, H), device="cuda", generator=gen) * 0.04).to(torch.bfloat16)


def _reference(x, w):
    return torch.ops.trtllm.dsv3_router_gemm_op(x, w.t(), bias=None, out_dtype=torch.float32)


def _logits(x, w):
    out = torch.empty((x.shape[0], E), dtype=torch.float32, device=x.device)
    _kernel_module().run_router_logits(x, w, out)
    return out


def _bits(t):
    return t.contiguous().view(torch.int32)


@pytest.mark.parametrize("m", list(range(1, 17)))
@pytest.mark.parametrize("seed", [0, 1])
def test_bitwise_vs_dsv3_router_gemm(m, seed):
    w = _gate(seed)
    x = _hidden(m, seed)
    out = _logits(x, w)
    assert torch.equal(_bits(out), _bits(_reference(x, w)))


def test_bitwise_adversarial_values():
    """Cancellation-heavy inputs: rows of huge values with alternating signs, denormal-scale
    rows and a weight with mixed magnitudes, where any change of summation order shows."""
    m = 16
    w = _gate(5)
    w[:, ::3] *= 64.0
    w[::5, :] *= 1e-3
    x = _hidden(m, 5).float()
    x[0] = 1e4 * torch.where(torch.arange(H, device="cuda") % 2 == 0, 1.0, -1.0)
    x[1] = 1e-30
    x[2, : H // 2] = 0.0
    x[3] = x[0] * -1.0 + 1.0
    out = _logits(x.to(torch.bfloat16), w)
    assert torch.equal(_bits(out), _bits(_reference(x.to(torch.bfloat16), w)))


def test_cuda_graph_replay():
    w = _gate(3)
    x = _hidden(14, 3)
    out = torch.empty((14, E), dtype=torch.float32, device="cuda")
    _kernel_module().run_router_logits(x, w, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _kernel_module().run_router_logits(x, w, out)
    for seed in range(3):
        x.copy_(_hidden(14, 10 + seed))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(out), _bits(_reference(x, w)))


@pytest.mark.parametrize("m", [1, 9, 16])
def test_pdl_behind_producer(m):
    """Launched with PDL right after the kernel that writes its input (the weight copy runs
    before the wait; the activation is read only after it)."""
    w = _gate(7)
    src = _hidden(m, 7)
    x = torch.empty_like(src)
    out = torch.empty((m, E), dtype=torch.float32, device="cuda")
    for _ in range(3):
        x.zero_()
        # Producer: a short chain whose last kernel writes x.
        torch.mul(src, 1.0, out=x)
        _kernel_module().run_router_logits(x, w, out)
    torch.cuda.synchronize()
    assert torch.equal(_bits(out), _bits(_reference(src, w)))


def test_can_run():
    mod = _kernel_module()
    w = _gate(0)
    assert mod.can_run(_hidden(16, 0), w)
    assert mod.can_run(_hidden(1, 0), w)
    assert not mod.can_run(_hidden(17, 0), w)
    assert not mod.can_run(_hidden(4, 0)[:0], w)
    assert not mod.can_run(_hidden(4, 0).float(), w)
    assert not mod.can_run(_hidden(4, 0), w[:448])
    assert not mod.can_run(_hidden(4, 0), w.float())
    assert not mod.can_run(_hidden(4, 0), w.t().contiguous().t())
    assert not mod.can_run(_hidden(8, 0)[::2], w)
    assert not mod.can_run(_hidden(4, 0).cpu(), w)


@pytest.mark.parametrize("rubin_dsl", [False, True], ids=["stock_dsl", "rubin_dsl"])
def test_sm107_selected_only_with_rubin_cute_dsl(monkeypatch, rubin_dsl):
    """On SM107 the kernel serves only with a Rubin-capable CuTe DSL; otherwise the MoE gate
    keeps ``dsv3_router_gemm_op``."""
    from tensorrt_llm._torch import cute_dsl_utils

    mod = _kernel_module()
    monkeypatch.setattr(mod, "_device_capability", lambda device_index: (10, 7))
    monkeypatch.setattr(cute_dsl_utils, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", rubin_dsl)
    assert mod.can_run(_hidden(4, 0), _gate(0)) == rubin_dsl


@pytest.mark.parametrize("m", [1, 7, 16])
def test_op_bitwise(m):
    import tensorrt_llm._torch.custom_ops  # noqa: F401  (registers the trtllm ops)

    if not hasattr(torch.ops.trtllm, "kimi_k3_router_logits"):
        pytest.skip("trtllm::kimi_k3_router_logits is not registered")
    w = _gate(2)
    x = _hidden(m, 2)
    out = torch.ops.trtllm.kimi_k3_router_logits(x, w)
    assert out.shape == (m, E) and out.dtype == torch.float32
    assert torch.equal(_bits(out), _bits(_reference(x, w)))
