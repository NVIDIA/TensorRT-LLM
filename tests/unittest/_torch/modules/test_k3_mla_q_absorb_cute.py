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
"""Bit-exactness of the tcgen05 MLA FP8 query absorption (``k3_mla_q_absorb``).

Every output byte (absorbed half and q_pe half) must equal the Triton kernel's
(``mla_fp8_q_absorb_nope`` without ``TRTLLM_K3_MLA_FP8_Q_ABSORB_CUTE``), including saturating,
infinite and NaN inputs.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="needs an SM100/SM103/SM107 GPU",
)

H, NOPE, ROPE, LORA = 96, 128, 64, 512


def _cute():
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_mla_q_absorb

    return k3_mla_q_absorb


def _triton(q, w, out):
    from tensorrt_llm._torch.modules import mla_fp8_q_absorb as mod

    use_cute = mod._USE_CUTE
    mod._USE_CUTE = False
    try:
        mod.mla_fp8_q_absorb_nope(q, w, out, NOPE, ROPE)
    finally:
        mod._USE_CUTE = use_cute


def _inputs(num_tokens, seed, special=True, num_heads=H):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(num_tokens, num_heads, NOPE + ROPE, device="cuda", generator=g) * 3).to(
        torch.bfloat16
    )
    w = (torch.randn(num_heads, LORA, NOPE, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
    if special:
        flat = q.view(-1)
        flat[::7919] = 3e4  # saturates the absorbed outputs of its row
        flat[1::104729] = float("inf")
        flat[2::130363] = float("-inf")
        flat[3::150001] = float("nan")
    return q, w


def _outs(num_tokens, num_heads=H):
    shape = (num_tokens, num_heads, LORA + ROPE)
    ref = torch.full(shape, 0x55, dtype=torch.uint8, device="cuda")
    out = torch.full(shape, 0xAA, dtype=torch.uint8, device="cuda")
    return ref, out


@pytest.mark.parametrize("num_tokens", [1, 7, 128, 129, 224, 448, 449, 896, 1344, 1792])
def test_matches_triton(num_tokens):
    cq = _cute()
    q, w = _inputs(num_tokens, num_tokens)
    ref, out = _outs(num_tokens)
    _triton(q, w, ref)
    assert cq.can_run(q, w, out, NOPE, ROPE)
    cq.k3_mla_q_absorb(q, w, out, NOPE, ROPE)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


@pytest.mark.parametrize("tile", [(128, False), (256, False), (128, True), (256, True)])
@pytest.mark.parametrize("use_pdl", [True, False])
def test_tiles(tile, use_pdl):
    cq = _cute()
    num_tokens = 300
    q, w = _inputs(num_tokens, 3)
    ref, out = _outs(num_tokens)
    _triton(q, w, ref)
    cq.k3_mla_q_absorb(q, w, out, NOPE, ROPE, tile=tile, use_pdl=use_pdl)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def test_strided_q_and_fp8_out():
    """q as the head-split view of a wider row; ``out`` as FP8 rather than its uint8 view."""
    cq = _cute()
    num_tokens = 200
    q_full, w = _inputs(num_tokens + 3, 5)
    base = torch.zeros(num_tokens, H * (NOPE + ROPE) + 64, dtype=torch.bfloat16, device="cuda")
    base[:, : H * (NOPE + ROPE)] = q_full[3:].reshape(num_tokens, -1)
    q = base[:, : H * (NOPE + ROPE)].view(num_tokens, H, NOPE + ROPE)
    ref, _ = _outs(num_tokens)
    _triton(q, w, ref)
    out = torch.zeros(num_tokens, H, LORA + ROPE, dtype=torch.float8_e4m3fn, device="cuda")
    assert cq.can_run(q, w, out, NOPE, ROPE)
    cq.k3_mla_q_absorb(q, w, out, NOPE, ROPE)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.uint8), ref)


def test_other_head_count():
    cq = _cute()
    num_tokens, num_heads = 96, 64
    q, w = _inputs(num_tokens, 9, num_heads=num_heads)
    ref, out = _outs(num_tokens, num_heads)
    _triton(q, w, ref)
    cq.k3_mla_q_absorb(q, w, out, NOPE, ROPE)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def test_dispatch_and_cuda_graph(monkeypatch):
    """The env-gated dispatch in ``mla_fp8_q_absorb_nope``, captured and replayed in a graph."""
    from tensorrt_llm._torch.modules import mla_fp8_q_absorb as mod

    monkeypatch.setattr(mod, "_USE_CUTE", True)
    num_tokens = 640
    q, w = _inputs(num_tokens, 11)
    ref, out = _outs(num_tokens)
    mod.mla_fp8_q_absorb_nope(q, w, out, NOPE, ROPE)  # eager: compiles every tile variant
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        mod.mla_fp8_q_absorb_nope(q, w, out, NOPE, ROPE)
    q2, _ = _inputs(num_tokens, 12)
    q.copy_(q2)
    out.fill_(0)
    graph.replay()
    torch.cuda.synchronize()
    _triton(q, w, ref)
    assert torch.equal(out, ref)


def test_can_run_rejects():
    cq = _cute()
    q, w = _inputs(8, 0, special=False)
    _, out = _outs(8)
    assert not cq.can_run(q.float(), w, out, NOPE, ROPE)
    assert not cq.can_run(q, w.transpose(1, 2).contiguous().transpose(1, 2), out, NOPE, ROPE)
    assert not cq.can_run(q[:, :, :], w, out[:4], NOPE, ROPE)
    assert not cq.can_run(q.cpu(), w, out, NOPE, ROPE)
