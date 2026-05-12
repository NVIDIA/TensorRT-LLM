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

from types import SimpleNamespace
from unittest.mock import patch

import torch

from tensorrt_llm._torch.modules.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from tensorrt_llm._torch.modules.fused_moe.moe_scheduler import ExternalCommMoEScheduler


class _SeparatedRouting:

    top_k = 1
    experts_per_token = 1
    requires_separated_routing = True

    def __init__(self):
        self.calls = []

    def apply(self, router_logits, input_ids):
        self.calls.append((router_logits, input_ids))
        num_tokens = router_logits.shape[0]
        return (
            torch.zeros((num_tokens, self.top_k), dtype=torch.int64),
            torch.ones((num_tokens, self.top_k), dtype=torch.float32),
        )


class _RecordingBackend:

    def __init__(self):
        self.token_selected_experts = None
        self.token_final_scales = None

    def _supports_load_balancer(self):
        return False

    def quantize_input(self, x, post_quant_comm=False):
        return x, None

    def run_moe(self, *, x, token_selected_experts, token_final_scales, x_sf, **kwargs):
        self.token_selected_experts = token_selected_experts
        self.token_final_scales = token_final_scales
        return x


class _FakeMoe:

    def __init__(self, backend, routing_method):
        self.backend = backend
        self.routing_method = routing_method
        self.apply_router_weight_on_input = False
        self.layer_load_balancer = None
        self.comm = None
        self.num_slots = 4
        self.layer_idx = 0
        self.use_dp = False
        self.enable_dummy_allreduce = False
        self.parallel_size = 1
        self.reduce_results = False

    def _load_balancer_start_wait_gpu_stage(self, is_first_call):
        pass

    def _using_load_balancer(self):
        return False

    def _load_balancer_start_set_cpu_stage(self, is_last_call):
        pass

    def _load_balancer_done_set_cpu_stage(self, is_last_call):
        pass


def test_external_comm_scheduler_honors_routing_requires_separation():
    routing_method = _SeparatedRouting()
    backend = _RecordingBackend()
    scheduler = ExternalCommMoEScheduler(_FakeMoe(backend, routing_method))

    x = torch.randn(2, 8)
    router_logits = torch.randn(2, 4)
    input_ids = torch.tensor([7, 11], dtype=torch.int32)

    output = scheduler._forward_chunk_impl(
        x,
        router_logits,
        output_dtype=torch.bfloat16,
        all_rank_num_tokens=[2],
        use_dp_padding=False,
        is_first_call=True,
        is_last_call=True,
        do_finalize=True,
        input_ids=input_ids,
    )

    assert output is x
    assert routing_method.calls == [(router_logits, input_ids)]
    assert backend.token_selected_experts is not None
    assert backend.token_selected_experts.dtype == torch.int32
    assert backend.token_selected_experts.shape == (2, 1)
    assert backend.token_final_scales is not None


def test_external_comm_trtllm_gen_kwargs_skip_router_logits_when_routing_is_separated():
    backend = object.__new__(TRTLLMGenFusedMoE)
    moe = SimpleNamespace(
        backend=backend,
        routing_method=SimpleNamespace(requires_separated_routing=True),
        comm=None,
    )
    scheduler = ExternalCommMoEScheduler(moe)
    router_logits = torch.randn(2, 4)

    with patch.object(TRTLLMGenFusedMoE, "_supports_load_balancer", return_value=False):
        kwargs = scheduler._get_backend_kwargs(router_logits=router_logits)

    assert kwargs["router_logits"] is None
