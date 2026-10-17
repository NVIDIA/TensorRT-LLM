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
"""The hop_routing routed_moe fork against routed_moe on one GPU (real GLM-5.2 NVFP4 weights).

routed_moe computes the routing from the logits; the fork reads the same routing
(the native runner's ids and weights for those logits, packed as the hop_routing hop
writes them). Given that, every output must be routed_moe's bit for bit:

* the routing tables the kernel builds (``DEBUG_ROUTING`` dump: top-k ids
  and weights, slot experts, slot masks, slot count);
* the raw tuple (FC2 rows and expert scales) and the finalized output;
* the routed_moe_into_hop FP32 accumulator within the FP32 reorder bound (routed_moe's own output
  class; the atomic adds are run-to-run nondeterministic), and identical to
  routed_moe's from the same rows;
* padded rows leave the live rows unchanged; CUDA-graph replay with fresh
  routing; host-only declines (wrong routing shape or dtype) before any
  device work.

Inputs: real captured logits and post-norm hidden states of layer 10
(``GLM_MOE_DSA_REAL_MOE_INPUTS_DIR``), plus random activations with the real router GEMM.
"""

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "routed_moe"))
import hop_routing_oracle as orc  # noqa: E402
import routed_moe_fixtures as fx  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)

DEBUG_ROUTING = 1
REORDER_REL = 2.0**-20
LAYER = int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_LAYER", "10"))


@pytest.fixture(scope="module")
def layer():
    torch.manual_seed(0)
    return fx.build_native_layer(
        layer=LAYER, tp_rank=int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_RANK", "0"))
    )


@pytest.fixture(scope="module")
def runners(layer):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.hop_routing.routed_moe import (  # noqa: E501
        HopRoutedMoe,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.routed_moe import (  # noqa: E501
        RoutedMoe,
    )

    return RoutedMoe.from_native(layer.module, layer.bias), HopRoutedMoe(layer.module, layer.bias)


def _pack(logits, bias):
    ids, w = orc.native_routing(logits, bias)
    rows = ids.shape[0]
    routing = torch.empty(2, rows, 8, dtype=torch.int32, device="cuda")
    routing[0].copy_(ids)
    routing[1].copy_(w.float().view(torch.int32))
    return routing


def _real(layer, rows, count, offset=0):
    win = fx.real_windows(fx.load_real_capture(layer.layer), rows, stride=rows * 7)
    out = []
    for x, lg in win[offset : offset + count]:
        out.append((x.cuda().contiguous(), lg.float().cuda().contiguous()))
    return out


def _random(layer, rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, fx.HIDDEN, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    return x, layer.router_logits(x).float().contiguous()


def _cases(layer, rows):
    return _real(layer, rows, 3, offset=rows) + [_random(layer, rows, 40 + rows)]


@pytest.mark.parametrize("rows", [1, 2, 6, 8])
def test_routing_tables_raw_and_finalized_equal_routed_moe(layer, runners, rows):
    routed_runner, fork = runners
    for i, (x, logits) in enumerate(_cases(layer, rows)):
        routing = _pack(logits, layer.bias)
        dbg_a = torch.zeros(2 * 1024 * 1024, dtype=torch.int32, device="cuda")
        dbg_b = torch.zeros_like(dbg_a)
        routed_runner(x, logits, debug=DEBUG_ROUTING, debug_buffer=dbg_a)
        fork(x, routing, debug=DEBUG_ROUTING, debug_buffer=dbg_b)
        raw_a = routed_runner(x, logits, do_finalize=False)
        raw_b = fork(x, routing, do_finalize=False)
        fin_a = routed_runner(x, logits)
        fin_b = fork(x, routing)
        torch.cuda.synchronize()
        assert torch.equal(dbg_a[:2048], dbg_b[:2048]), f"case {i}: routing tables differ"
        for a, b in zip(raw_a, raw_b):
            assert torch.equal(a, b), f"case {i}: raw tuple differs"
        assert torch.equal(fin_a, fin_b), f"case {i}: finalized output differs"


def test_fp32_accumulator_matches_routed_moe_class(layer, runners):
    routed_runner, fork = runners
    rows = 6
    if fork.schedule(rows, False, True) != "xc":
        pytest.skip("FP32 accumulator output needs the expert-cluster schedule")
    for x, logits in _real(layer, rows, 2, offset=3):
        routing = _pack(logits, layer.bias)
        fc2, scale, idx = routed_runner(x, logits, do_finalize=False)
        terms = scale.double()[:, :, None] * fc2[idx.long()].double()
        exact = terms.sum(1)
        bound = REORDER_REL * terms.abs().sum(1)
        for _ in range(4):
            acc = torch.zeros(rows, fx.HIDDEN, dtype=torch.float32, device="cuda")
            fork(x, routing, red_out=acc)
            torch.cuda.synchronize()
            assert bool(((acc.double() - exact).abs() <= bound).all())


@pytest.mark.parametrize("live", [1, 5])
def test_padded_rows_do_not_change_live_rows(layer, runners, live):
    _, fork = runners
    ((x, logits),) = _real(layer, 8, 1, offset=1)
    full = fork(x, _pack(logits, layer.bias), do_finalize=False)
    part = fork(
        x[:live].contiguous(), _pack(logits[:live].contiguous(), layer.bias), do_finalize=False
    )
    torch.cuda.synchronize()
    assert torch.equal(full[0][: live * 8], part[0])
    assert torch.equal(full[1][:live], part[1])


def test_cuda_graph_replay(layer, runners):
    _, fork = runners
    rows = 6
    cases = _real(layer, rows, 4, offset=2)
    x = cases[0][0].clone()
    routing = _pack(cases[0][1], layer.bias)
    fork(x, routing, do_finalize=False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fork(x, routing, do_finalize=False)
    for xn, ln in cases[1:]:
        x.copy_(xn)
        routing.copy_(_pack(ln, layer.bias))
        graph.replay()
        torch.cuda.synchronize()
        got = [t.clone() for t in captured]
        want = fork(x, routing, do_finalize=False)
        torch.cuda.synchronize()
        for a, b in zip(got, want):
            assert torch.equal(a, b)
    del graph


def test_declines_before_device_work(layer):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.hop_routing import (  # noqa: E501
        routed_moe as hop_routed_mod,
    )

    x, logits = _random(layer, 6, 3)
    good = _pack(logits, layer.bias)
    bad = [
        None,
        good[:, :5].contiguous(),
        good.to(torch.int64),
        good.transpose(1, 2).contiguous().transpose(1, 2),
    ]
    torch.cuda.synchronize()
    for routing in bad:
        before = torch.cuda.memory_allocated()
        assert (
            hop_routed_mod.routed_output(layer.module, layer.bias, x, logits, routing, False)
            is None
        )
        assert torch.cuda.memory_allocated() == before
    assert hop_routed_mod.routing_reason(good, x) == ""
