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
"""Routing that also writes the backend's ``moe_sort`` tables (``try_route_with_sort``).

``DeepSeekV3MoeRoutingMethod.apply_with_sort`` serves ungrouped fused top-16 routing of decode
batches with ``trtllm::noaux_tc_route_sort``: the ids and weights of ``noaux_tc_op`` plus
this rank's ``moe_sort`` outputs. A ConfigurableMoE whose backend consumes them then runs no
``moe_sort``. Both paths compute every routed row identically; only the row order within an
expert's tile and the order of the BF16 scatter-adds into a token's output row can differ. So
output rows with at most two local experts (``0 + a + b`` rounds the same in either order) must
be bit-identical, the others equal up to the BF16 accumulation order.
"""

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod
from tensorrt_llm._utils import get_sm_version

NUM_EXPERTS = 896
TOP_K = 16
HIDDEN = 512
INTER = 256
SCALE = 2.5
GROUP = 16


def _routing(bias, top_k=TOP_K, n_group=1, topk_group=1, is_fused=True):
    return DeepSeekV3MoeRoutingMethod(
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=SCALE,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=is_fused,
    )


@pytest.mark.parametrize(
    "kwargs",
    [dict(n_group=8, topk_group=4), dict(top_k=8), dict(is_fused=False)],
    ids=["grouped", "top8", "unfused"],
)
def test_apply_with_sort_declines_other_routing(kwargs):
    """Only ungrouped fused top-16 routing has a sorting kernel."""
    bias = torch.zeros(NUM_EXPERTS)
    logits = torch.zeros(4, NUM_EXPERTS)
    assert _routing(bias, **kwargs).apply_with_sort(logits, 0, 112) is None


def test_apply_with_sort_declines_cpu_logits():
    bias = torch.zeros(NUM_EXPERTS)
    assert _routing(bias).apply_with_sort(torch.zeros(4, NUM_EXPERTS), 0, 112) is None


@pytest.mark.parametrize(
    "kwargs",
    [dict(n_group=8, topk_group=4), dict(top_k=8), dict(is_fused=False)],
    ids=["grouped", "top8", "unfused"],
)
def test_serves_with_sort_declines_other_routing(kwargs):
    """The routing configuration decides at construction; no token count is served then."""
    routing = _routing(torch.zeros(NUM_EXPERTS), **kwargs)
    assert not routing.serves_with_sort(8, NUM_EXPERTS, 0, 112, torch.device("cuda"))


def _cute_dsl_available() -> bool:
    if not torch.cuda.is_available() or get_sm_version() not in (100, 103, 107):
        return False
    from tensorrt_llm._torch.cute_dsl_utils import (
        IS_CUTLASS_DSL_AVAILABLE,
        IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    )

    # The routing kernel runs on SM107 only with a Rubin-capable CuTe DSL.
    return IS_CUTLASS_DSL_AVAILABLE and (get_sm_version() != 107 or IS_CUTLASS_DSL_RUBIN_AVAILABLE)


cute_dsl = pytest.mark.skipif(
    not _cute_dsl_available(), reason="needs SM100/SM103/SM107 and CuTe DSL"
)


def _logits(m, seed):
    """Router-like FP32 logits (popular experts, tokens of a request alike) and a bias."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dim = 256
    directions = torch.randn((NUM_EXPERTS, dim), device="cuda", generator=gen)
    directions = directions / directions.norm(dim=1, keepdim=True)
    popularity = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.7
    requests = torch.randn(((m + 7) // 8, dim), device="cuda", generator=gen)
    hidden = requests.repeat_interleave(8, dim=0)[:m] * 0.85
    hidden = hidden + torch.randn((m, dim), device="cuda", generator=gen) * 0.5
    hidden = hidden / hidden.norm(dim=1, keepdim=True)
    logits = (hidden @ directions.t()) * 22.0 + popularity - 1.5
    bias = (torch.rand((NUM_EXPERTS,), device="cuda", generator=gen) - 0.5) * 0.19
    return logits.contiguous(), bias


def _nvfp4_moe(moe_backend, routing):
    """Single-rank NVFP4 + SiTU ConfigurableMoE with random experts."""
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.moe.fused_moe import ConfigurableMoE, SiTuActivation, create_moe
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = NUM_EXPERTS
    pretrained_config.hidden_size = HIDDEN
    pretrained_config.intermediate_size = INTER
    pretrained_config.torch_dtype = torch.bfloat16
    moe = create_moe(
        routing_method=routing,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        dtype=torch.bfloat16,
        reduce_results=True,
        model_config=ModelConfig(
            pretrained_config=pretrained_config, mapping=Mapping(), moe_backend=moe_backend
        ),
        override_quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=GROUP),
        layer_idx=0,
        communication_method=None,
        activation=SiTuActivation(gate_softcap=4.0, linear_softcap=25.0),
        allow_backend_degradation=False,
    ).cuda()
    assert isinstance(moe, ConfigurableMoE)

    gen = torch.Generator().manual_seed(17)
    weights = {}
    for expert_id in range(NUM_EXPERTS):
        for name, rows, cols in (
            ("w1", INTER, HIDDEN),
            ("w3", INTER, HIDDEN),
            ("w2", HIDDEN, INTER),
        ):
            prefix = f"{expert_id}.{name}"
            weights[f"{prefix}.weight"] = torch.randint(
                0, 256, (rows, cols // 2), generator=gen, dtype=torch.uint8
            )
            weights[f"{prefix}.weight_scale"] = (
                torch.randint(120, 132, (rows, cols // GROUP), generator=gen, dtype=torch.int32)
                .to(torch.float32)
                .div(126.0)
                .to(torch.float8_e4m3fn)
            )
            # Gate/up/down outputs of order one, so SiTU and the FP4 requantization of its
            # output do not flush to zero.
            weights[f"{prefix}.weight_scale_2"] = torch.tensor(0.03, dtype=torch.float32)
            weights[f"{prefix}.input_scale"] = torch.tensor(1.0, dtype=torch.float32)
    moe.backend.load_weights([weights])
    moe.post_load_weights()
    return moe


def _local_counts(ids, offset, num_local):
    return ((ids >= offset) & (ids < offset + num_local)).sum(dim=1)


def _assert_same_rows(actual, expected, local_counts):
    """Rows with <= 2 local experts bit-identical, the others up to the BF16 add order."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype == torch.bfloat16
    assert torch.isfinite(expected.float()).all()
    assert expected.abs().amax() > 0, "degenerate test: all-zero MoE output"
    exact = local_counts <= 2
    mismatch = (actual[exact].view(torch.int16) != expected[exact].view(torch.int16)).any(dim=1)
    assert not mismatch.any(), f"{int(mismatch.sum())} of {int(exact.sum())} exact rows differ"
    if (~exact).any():
        a, e = actual[~exact].float(), expected[~exact].float()
        # Each of the up to 16 BF16 adds rounds by at most half an ulp of the running sum.
        tol = (local_counts[~exact].float() * 2.0**-8).unsqueeze(1) * e.abs().amax(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        assert ((a - e).abs() <= tol).all(), f"max diff {(a - e).abs().max().item():.3e}"


@cute_dsl
@pytest.mark.parametrize("m", [1, 8, 64, 200])
def test_cute_dsl_runs_on_sorted_routing(monkeypatch, m):
    """CUTEDSL: the routing kernel's tables replace ``moe_sort``; same output."""
    logits, bias = _logits(m, seed=m)
    moe = _nvfp4_moe("CUTEDSL", _routing(bias))
    x = (torch.randn((m, HIDDEN), device="cuda") * 0.5).to(torch.bfloat16)

    real_moe_sort = torch.ops.trtllm.moe_sort
    sorts = []

    def counting_moe_sort(**kwargs):
        sorts.append(kwargs["tile_tokens_dim"])
        return real_moe_sort(**kwargs)

    monkeypatch.setattr(torch.ops.trtllm, "moe_sort", counting_moe_sort)
    with torch.inference_mode():
        sorted_out = moe(x, logits)
        torch.cuda.synchronize()
        assert sorts == [], f"moe_sort ran although the routing produced tables: {sorts}"
        monkeypatch.setattr(moe.backend, "try_route_with_sort", lambda router_logits: None)
        expected = moe(x, logits)
        torch.cuda.synchronize()
    assert sorts == [128]
    ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, SCALE)[1]
    _assert_same_rows(sorted_out, expected, _local_counts(ids, 0, NUM_EXPERTS))


@cute_dsl
def test_sorted_routing_cuda_graph_replay(monkeypatch):
    """Routing kernel + CUTEDSL captured in one CUDA graph follow new logits on replay."""
    m = 32
    logits, bias = _logits(m, seed=40)
    moe = _nvfp4_moe("CUTEDSL", _routing(bias))
    # The MoE declares which forwards the sorting kernel routes.
    assert moe.sort_routing_applies
    assert moe.routes_with_sort_tables(m) and moe.routes_with_sort_tables(256)
    assert not moe.routes_with_sort_tables(257)
    x = (torch.randn((m, HIDDEN), device="cuda") * 0.5).to(torch.bfloat16)
    static_logits = logits.clone()
    with torch.inference_mode():
        moe(x, static_logits)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = moe(x, static_logits)
        for seed in range(3):
            fresh, _ = _logits(m, seed=50 + seed)
            static_logits.copy_(fresh)
            graph.replay()
            torch.cuda.synchronize()
            ids = torch.ops.trtllm.noaux_tc_op(static_logits, bias, 1, 1, TOP_K, SCALE)[1]
            with monkeypatch.context() as patch:
                patch.setattr(moe.backend, "try_route_with_sort", lambda router_logits: None)
                expected = moe(x, static_logits)
            torch.cuda.synchronize()
            _assert_same_rows(out, expected, _local_counts(ids, 0, NUM_EXPERTS))
