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
"""attention_output_tail_prefetch inside the target's ``GlmMLA.forward`` (native DSA attention →
attention_output_tail_prefetch tail).

Runs with the plan that runs only ``attention_output_tail`` and ``attention_output_tail_prefetch``
(opt-in: ``GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_PREFETCH_ADAPTER_PATCH_APPLIED=1``). Builds TP8 rank 0 of
real GLM-5.2 layers 2 (full indexer) and 3 (shared indexer) with the attention_output_tail test
harness (real weights, FP8 KV cache filled by a real prefill, GEN metadata as the decoder runner
prepares it) and runs GEN-only ``GlmMLA.forward`` as a decode step through the target's DSA
attention op:

* eager R = 6 and R = 1: with a successor weight registered (a router-shaped
  tensor, as the layer adoption registers the MoE gate weight), the
  target's ``forward_absorption_generation`` forks the staging exactly once
  per layer call (right before the attention backend call) and ``project``
  joins it; the attention_output_tail_prefetch output is bit-identical to the attention_output_tail
  kernels of ``TAIL_PREFETCH_SCHEDULE`` applied to the latent the attention produced (staging
  changes no result) and within the frozen attention_output_tail budget against the FP64
  oracle;
* CUDA-graph capture of the GEN step with the fork/join inside, replayed
  with new inputs, within budget of a native eager rerun.

Fixture labels: weights real, activations synthetic.
"""

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "attention_output_tail"))
import test_attn_tail_mla_forward as tail_forward  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.attention_output_tail.kernels import (  # noqa: E402, E501
    TAIL_PREFETCH_SCHEDULE,
    attn_tail,
)

pytestmark = [
    *tail_forward.pytestmark[:2],
    pytest.mark.skipif(
        os.environ.get("GLM_MOE_DSA_ATTENTION_OUTPUT_TAIL_PREFETCH_ADAPTER_PATCH_APPLIED") != "1",
        reason="needs attention_output_tail and attention_output_tail_prefetch selected (opt-in)",
    ),
    pytest.mark.threadleak(enabled=False),
]

LAYERS = tail_forward.LAYERS
PREFETCH_PATHS = "attention_output_tail,attention_output_tail_prefetch"


@pytest.mark.parametrize("rows", [6, 1])
def test_eager_forward_forks_joins_and_matches_attn_tail(rows):
    stack = tail_forward._stack(rows, PREFETCH_PATHS)
    router = torch.randn(256, 6144, device="cuda").to(torch.bfloat16)
    for layer in LAYERS:
        assert stack.attn[layer]._tail_prefetch
        assert stack.attn_tail_tails[layer].config == TAIL_PREFETCH_SCHEDULE
        stack.attn_tail_tails[layer].set_successor_weights([router])
    inputs = {layer: stack.gen_inputs(layer, step=0) for layer in LAYERS}
    native, lat_native = tail_forward._run(stack, False, inputs, record=True)
    forks = {layer: stack.attn_tail_tails[layer].prefetch_forks for layer in LAYERS}
    attention_output_tail, lat = tail_forward._run(stack, True, inputs, record=True)
    for layer in LAYERS:
        tail = stack.attn_tail_tails[layer]
        attn = stack.attn[layer]
        assert tail.prefetch_forks == forks[layer] + 1, layer
        assert not tail._prefetch.forked and not tail._armed
        ref = attn_tail(
            lat[layer], attn.v_b_proj, attn.o_proj.weight, rows, config=TAIL_PREFETCH_SCHEDULE
        )
        assert torch.equal(attention_output_tail[layer], ref), layer
        e_tail_prefetch, e_nat = tail_forward._check_budget(
            stack, layer, attention_output_tail[layer], native[layer], lat[layer]
        )
        print(
            f"[attention_output_tail_prefetch-mla-forward] rows={rows} layer={layer} "
            f"err attention_output_tail_prefetch={e_tail_prefetch['rel_rms']:.3e} native={e_nat['rel_rms']:.3e}"
        )


def test_cuda_graph_capture_and_replay():
    stack = tail_forward._stack(6, PREFETCH_PATHS)
    router = torch.randn(256, 6144, device="cuda").to(torch.bfloat16)
    for layer in LAYERS:
        stack.attn_tail_tails[layer].set_successor_weights([router])
    pos = stack.position_ids()
    static_in = {layer: stack.gen_inputs(layer, step=0).clone() for layer in LAYERS}
    stack.set_attn_tail(True)
    with stack.scope():
        for _ in range(2):
            for layer in LAYERS:
                stack.forward(layer, static_in[layer], pos, tail_forward._no_ar())
        torch.cuda.synchronize()
        forks = {layer: stack.attn_tail_tails[layer].prefetch_forks for layer in LAYERS}
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = {
                layer: stack.forward(layer, static_in[layer], pos, tail_forward._no_ar())
                for layer in LAYERS
            }
    for layer in LAYERS:
        assert stack.attn_tail_tails[layer].prefetch_forks == forks[layer] + 1
        assert not stack.attn_tail_tails[layer]._prefetch.forked
    for step in range(1, 5):
        for layer in LAYERS:
            static_in[layer].copy_(stack.gen_inputs(layer, step=step))
        graph.replay()
        torch.cuda.synchronize()
        got = {layer: captured[layer].clone() for layer in LAYERS}
        native, lat = tail_forward._run(
            stack,
            False,
            {layer: static_in[layer].clone() for layer in LAYERS},
            record=True,
            position_ids=pos,
        )
        for layer in LAYERS:
            e_tail_prefetch, e_nat = tail_forward._check_budget(
                stack, layer, got[layer], native[layer], lat[layer]
            )
            print(
                f"[attention_output_tail_prefetch-mla-graph] step={step} layer={layer} "
                f"err attention_output_tail_prefetch={e_tail_prefetch['rel_rms']:.3e} "
                f"native={e_nat['rel_rms']:.3e}"
            )
    del graph
