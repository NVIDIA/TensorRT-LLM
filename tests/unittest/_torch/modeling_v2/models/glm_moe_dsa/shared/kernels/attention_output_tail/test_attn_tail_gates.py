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
"""attention_output_tail eligibility gates and registered-API state.

* The static capability check declines attention-DP modules and o_proj
  tensor-parallel widths other than 8 explicitly, before any other check or
  GPU work, and the engaged decision logs its per-call domain.
* The registered functional entry point never falls back to module-global
  counters for the non-default fused engine: without a caller state it uses
  zeroed counters private to the call, eagerly and under CUDA-graph replay.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail import (  # noqa: E501
    kernels,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.kernels import (  # noqa: E501
    TailConfig,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.tail import (  # noqa: E501
    ENGAGED_DOMAIN,
    SUPPORTED_TP_SIZE,
    AttnTail,
    _static_reason,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tail_reference  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="attention_output_tail targets SM100/SM103",
)

LOCAL_HEADS, V_DIM, C_DIM, HIDDEN = 8, 256, 512, 6144


def _module(tp_size, attention_dp):
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=8, tp_size=8, rank=0, enable_attention_dp=attention_dp)
    mapping_o = Mapping(world_size=tp_size, tp_size=tp_size, rank=0)
    # Anything past the gates would fail on this stub, so a returned reason
    # proves the gate fired first.
    return SimpleNamespace(mapping=mapping, mapping_o=mapping_o)


def test_attention_dp_declines():
    assert _static_reason(_module(8, True)) == "attention data parallelism"


@pytest.mark.parametrize("tp_size", [1, 2, 4, 16])
def test_other_tp_sizes_decline(tp_size):
    reason = _static_reason(_module(tp_size, False))
    assert reason == (f"o_proj tensor-parallel size {tp_size} is not {SUPPORTED_TP_SIZE}")


def test_declined_layer_takes_native_path_without_side_effects():
    tail = AttnTail("adp")
    meta = SimpleNamespace(num_contexts=0, num_tokens=6)
    assert not tail.arm(_module(8, True), meta)
    assert tail.static_reason == "attention data parallelism"
    assert tail._pending is None and not tail._armed
    assert tail._state is None
    assert tail.engaged_calls == 0 and tail.declined_calls == 1
    # The op-side hook then declines and touches nothing.
    latent = torch.zeros(6, LOCAL_HEADS, C_DIM, device="cuda").bfloat16()
    out = torch.zeros(6, LOCAL_HEADS * V_DIM, device="cuda").bfloat16()
    assert not tail.defer_w_uv(None, meta, latent, out)
    assert tail.project(None, out, None) is None


def test_engaged_domain_text():
    assert "GEN-only" in ENGAGED_DOMAIN and "TP8" in ENGAGED_DOMAIN
    assert "1-8 token rows" in ENGAGED_DOMAIN


def test_registered_fused_entry_has_no_module_state():
    torch.manual_seed(5)
    w_uv = (torch.randn(LOCAL_HEADS, V_DIM, C_DIM, device="cuda") * 0.02).bfloat16()
    w_o = (torch.randn(HIDDEN, LOCAL_HEADS * V_DIM, device="cuda") * 0.02).bfloat16()
    latent = torch.randn(6, LOCAL_HEADS, C_DIM, device="cuda").bfloat16()
    fused = TailConfig(oproj_engine="fused")
    entry = dispatch.get_impl("attention_output_tail", "attn_tail")
    assert entry is kernels.attn_tail
    states_before = dict(kernels._DEFAULT_STATES)
    first = entry(latent, w_uv, w_o, 6, config=fused)
    second = entry(latent, w_uv, w_o, 6, config=fused)
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.cute_fused import (  # noqa: E501
        new_state,
    )

    own = new_state(latent.device)
    explicit = entry(latent, w_uv, w_o, 6, config=fused, state=own)
    torch.cuda.synchronize()
    assert kernels._DEFAULT_STATES == states_before
    assert torch.equal(first, second) and torch.equal(first, explicit)
    native = tail_reference.native_attn_tail(latent, w_uv, w_o, 6)
    torch.testing.assert_close(first, native, rtol=2**-6, atol=2**-6)

    # Graph replay: the per-call counters are re-zeroed by every replay.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = entry(latent, w_uv, w_o, 6, config=fused)
    for step in range(16):
        latent.copy_(torch.randn_like(latent, dtype=torch.float32).bfloat16())
        graph.replay()
        torch.cuda.synchronize()
        expected = entry(latent, w_uv, w_o, 6, config=fused, state=own)
        torch.cuda.synchronize()
        assert torch.equal(captured, expected), step
    assert kernels._DEFAULT_STATES == states_before
    del graph
