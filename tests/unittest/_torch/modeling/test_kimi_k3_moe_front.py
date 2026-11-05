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
"""Issue order of the Kimi-K3 MoE front: up to ``_ROUTER_FIRST_MAX_TOKENS`` tokens the router GEMM
is issued on the main stream before latent_down on the latent stream, with the latent fork event
recorded before both.

The modules are stand-ins (elementwise ops on the GPU), so the test checks the plumbing: the
issue order and streams, the token-count bound, the result, and CUDA graph capture.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models import modeling_kimi_linear
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiK3MoERuntime
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HIDDEN = 64


class _SharedExperts(nn.Module):
    """Stand-in shared expert: ``2 x``."""

    def __init__(self):
        super().__init__()
        self.down_proj = nn.Identity()

    def forward(self, x, final_all_reduce_params=None):
        return x * 2.0


class _RoutedExperts:
    """Stand-in routed experts (BF16 input, no sorting routing kernel): join
    ``input_ready_event``, return ``3 x + logits``."""

    all_reduce = None
    comm = None
    # The backend surface the eager forward reads for the per-launch SM
    # reserve: this stand-in sizes no persistent grid to shrink.
    backend = SimpleNamespace(reserve_sms_for_concurrent_kernels=lambda num_sms: None)

    def nvfp4_input_scale(self, num_tokens):
        return None

    def routes_with_sort_tables(self, num_tokens):
        return False

    def __call__(self, value, router_logits, all_rank_num_tokens=None, input_ready_event=None):
        if input_ready_event is not None:
            input_ready_event.wait()
        return value * 3.0 + router_logits


def _runtime(log):
    runtime = KimiK3MoERuntime.__new__(KimiK3MoERuntime)
    nn.Module.__init__(runtime)
    main = {}

    def compute_logits(value):
        log.append(("router", torch.cuda.current_stream() == main["stream"]))
        return value + 1.0

    def projection(value, module):
        log.append(
            ("latent_down" if module is runtime.routed_expert_down_proj else "latent_up", None)
        )
        return value * 1.0

    runtime.gate = SimpleNamespace(compute_logits=compute_logits)
    runtime.routed_experts = _RoutedExperts()
    runtime.top_k, runtime.num_experts, runtime.experts_per_rank = 16, 896, 112
    runtime.routed_expert_down_proj = nn.Identity()
    runtime.routed_expert_up_proj = nn.Identity()
    runtime.routed_expert_norm = nn.Identity()
    runtime.shared_experts = _SharedExperts()
    runtime._shared_all_reduce = False
    runtime._reduce_routed_output = False
    runtime._defer_shared_allreduce = False
    runtime._shared_gemms_can_lead = True
    # _shared_sms as __init__ builds it here: the stand-in backend reserves none.
    runtime._shared_sms = 0
    runtime.moe_routed_event = None
    runtime.shared_expert_stream = torch.cuda.Stream()
    runtime.latent_down_stream = torch.cuda.Stream()
    for name in (
        "moe_main_event",
        "moe_shared_event",
        "moe_latent_fork_event",
        "moe_latent_ready_event",
        "moe_shared_gemm_event",
    ):
        setattr(runtime, name, torch.cuda.Event())
    runtime._can_fuse_router_down = lambda value: False
    runtime._routed_projection = projection
    real_record = runtime.moe_latent_fork_event.record

    def record_fork(*args):
        log.append(("fork", None))
        return real_record(*args)

    runtime.moe_latent_fork_event = SimpleNamespace(
        record=record_fork, wait=runtime.moe_latent_fork_event.wait
    )
    return runtime, main


@pytest.mark.parametrize("num_tokens", [1, 64, 65, 256])
def test_router_issued_before_latent_down(num_tokens):
    log = []
    runtime, main = _runtime(log)
    hidden = torch.randn(num_tokens, HIDDEN, device="cuda")
    main["stream"] = torch.cuda.current_stream()
    with torch.inference_mode(), with_multi_stream(True):
        routed, shared = runtime(hidden, return_parts=True)
    torch.cuda.synchronize()

    events = [name for name, _ in log if name != "latent_up"]
    if num_tokens <= modeling_kimi_linear._ROUTER_FIRST_MAX_TOKENS:
        assert events == ["fork", "router", "latent_down"]
    else:
        assert events == ["fork", "latent_down", "router"]
    assert all(on_main for name, on_main in log if name == "router")
    torch.testing.assert_close(routed, hidden * 3.0 + (hidden + 1.0), rtol=0, atol=0)
    torch.testing.assert_close(shared, hidden * 2.0, rtol=0, atol=0)


def test_router_first_graph_capture():
    """The front captured in a CUDA graph follows new inputs on replay."""
    log = []
    runtime, main = _runtime(log)
    static = torch.randn(8, HIDDEN, device="cuda")
    side = torch.cuda.Stream()
    with torch.inference_mode(), with_multi_stream(True):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            main["stream"] = side
            runtime(static, return_parts=True)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            main["stream"] = torch.cuda.current_stream()
            routed, shared = runtime(static, return_parts=True)
        for seed in range(2):
            torch.manual_seed(seed)
            static.copy_(torch.randn_like(static))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(routed, static * 3.0 + (static + 1.0), rtol=0, atol=0)
            torch.testing.assert_close(shared, static * 2.0, rtol=0, atol=0)
