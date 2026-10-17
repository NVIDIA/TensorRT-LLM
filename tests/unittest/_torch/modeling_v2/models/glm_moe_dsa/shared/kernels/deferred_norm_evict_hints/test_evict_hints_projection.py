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
"""Single-GPU tests of the deferred_norm_evict_hints projection against F's deferred_norm projection (real weights).

For every deferred_norm_evict_hints projection configuration (``CONFIGS``) and every layer
the projection serves (``GLM_MOE_DSA_DEFERRED_NORM_EVICT_HINTS_TEST_LAYERS``: a shared-indexer
target layer; the target's deferred-norm scope is ``shared``, so full-indexer layers keep their
native projection), the fused-A output
is bit-identical to ``DeferredNormProjection``'s on the same deferred packet, for R in
1..8 and residual scales from 1e-3 to 30, in eager mode and in a CUDA graph
with PDL. ``precompile_served`` covers the consumer's launch configuration.

Fixture label: weights real (checkpoint), activations synthetic.
"""

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "absorbed_query"))
sys.path.insert(0, os.path.join(_HERE, "..", "deferred_norm"))

import absorbed_query_fixture  # noqa: E402
from test_deferred_norm_projection import _module, make_packet, make_residual  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not absorbed_query_fixture.checkpoint_available(),
    reason="needs a GPU and the GLM-5.2 checkpoint",
)

LAYERS = [
    int(x)
    for x in os.environ.get("GLM_MOE_DSA_DEFERRED_NORM_EVICT_HINTS_TEST_LAYERS", "7").split(",")
]
CONFIGS = os.environ.get(
    "GLM_MOE_DSA_DEFERRED_NORM_EVICT_HINTS_TEST_PROJ_CFGS",
    "maxrreg=112;maxrreg=0,maxntid=576;maxrreg=112,copy_hint=evict_first;"
    "maxrreg=112,split_chunk=2,split_delay_ns=2500;maxrreg=0;copy_warps=1;prod_warp=1;"
    "prod_warp=1,split_chunk=1,split_delay_ns=3000;"
    "prod_warp=1,split_chunk=2,split_delay_ns=2500,copy_hint=evict_first;"
    "prod_warp=1,defer_tail=8,defer_wait=1;prod_warp=1,split_chunk=1,defer_tail=3,defer_wait=1;"
    "prod_warp=1,defer_tail=5,split_delay_ns=2000;",
).split(";")


def _pair(layer, cfg_text):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormProjection,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.projection import (  # noqa: E501
        projection_eligibility,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm_evict_hints import (  # noqa: E501
        EvictHintsProjConfig,
        EvictHintsProjection,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm_evict_hints.projection import (  # noqa: E501
        parse_cfg,
    )

    attn, gamma = _module(layer)
    ok, reason, kw = projection_eligibility(attn)
    assert ok, reason
    deferred_norm = DeferredNormProjection(attn, kw_weight=kw)
    hop_config = EvictHintsProjection(attn, parse_cfg(cfg_text, EvictHintsProjConfig), kw_weight=kw)
    assert hop_config.supported() is None
    return attn, gamma, deferred_norm, hop_config


def _bits(t):
    return None if t is None else t.view(torch.int16)


def _equal(a, b):
    return all((x is None and y is None) or torch.equal(_bits(x), _bits(y)) for x, y in zip(a, b))


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("layer", LAYERS)
def test_served_configs_cover_the_consumer(layer, cfg):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm_evict_hints.proj_kernel import (  # noqa: E501
        is_compiled,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm_evict_hints.projection import (  # noqa: E501
        EvictHintsProjConfig,
        parse_cfg,
        precompile_served,
        served_kernel_configs,
    )

    config = parse_cfg(cfg, EvictHintsProjConfig)
    device = torch.cuda.current_device()
    assert precompile_served(device, config) is None
    _, _, _, hop_config = _pair(layer, cfg)
    assert hop_config.served_config() in served_kernel_configs(device, config)
    assert is_compiled(**hop_config.served_config())


@pytest.mark.parametrize("scale", [1.0, 1e-3, 30.0])
@pytest.mark.parametrize("rows", [1, 2, 6, 8])
@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("layer", LAYERS)
def test_bit_identical_to_deferred_norm(layer, cfg, rows, scale):
    _, gamma, deferred_norm, hop_config = _pair(layer, cfg)
    with torch.inference_mode():
        x = make_residual(rows, seed=100 + rows + layer, scale=scale)
        packet = make_packet(x, gamma)
        ref = deferred_norm.project(packet)
        got = hop_config.project(packet)
        torch.cuda.synchronize()
        assert _equal(got, ref)


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("layer", LAYERS)
def test_graph_with_pdl_bit_identical(layer, cfg):
    """deferred_norm_evict_hints then deferred_norm projection of 4 packets in one CUDA graph, replayed with
    fresh packets (PDL chain between the projections)."""
    _, gamma, deferred_norm, hop_config = _pair(layer, cfg)
    with torch.inference_mode():
        packets = [make_packet(make_residual(6, seed=500 + i), gamma) for i in range(4)]

        def body():
            return [(hop_config.project(p), deferred_norm.project(p)) for p in packets]

        body()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = body()
        for trial in range(3):
            for i, p in enumerate(packets):
                fresh = make_packet(make_residual(6, seed=600 + 10 * trial + i), gamma)
                p.a_frag.copy_(fresh.a_frag)
                p.sumsq.copy_(fresh.sumsq)
            graph.replay()
            torch.cuda.synchronize()
            for (got, ref), p in zip(outs, packets):
                assert _equal(got, ref)
                assert _equal(got, deferred_norm.project(p))
