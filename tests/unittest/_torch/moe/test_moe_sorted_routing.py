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


@cute_dsl
@pytest.mark.parametrize(
    "kwargs",
    [dict(n_group=8, topk_group=4), dict(top_k=8), dict(is_fused=False)],
    ids=["grouped", "top8", "unfused"],
)
def test_apply_with_sort_declines_other_routing(kwargs):
    """Only ungrouped fused top-16 routing has a sorting kernel: on a device the kernel
    runs on (CUDA logits), these decline by configuration, not by the CPU fallback."""
    bias = torch.zeros(NUM_EXPERTS, device="cuda")
    logits = torch.zeros(4, NUM_EXPERTS, device="cuda")
    assert _routing(bias, **kwargs).apply_with_sort(logits, 0, 112) is None


def _fc12_available() -> bool:
    if not torch.cuda.is_available() or get_sm_version() != 107:
        return False
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    return IS_CUTLASS_DSL_RUBIN_AVAILABLE


BACKENDS = [
    pytest.param("CUTEDSL", marks=cute_dsl),
    pytest.param(
        "CUTEDSL_FC12",
        marks=pytest.mark.skipif(
            not _fc12_available(), reason="CUTEDSL_FC12 needs SM107 and CuTe DSL internal"
        ),
    ),
]


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


@pytest.mark.parametrize("moe_backend", BACKENDS)
@pytest.mark.parametrize("m", [1, 8, 64, 200])
def test_backend_runs_on_sorted_routing(monkeypatch, moe_backend, m):
    """The routing kernel's tables replace ``moe_sort``; same output."""
    logits, bias = _logits(m, seed=m)
    moe = _nvfp4_moe(moe_backend, _routing(bias))
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


class _SharedExperts(torch.nn.Module):
    """Stand-in shared expert: ``0.5 x`` partial, all-reduce adds one."""

    def __init__(self):
        super().__init__()
        self.down_proj = torch.nn.Module()
        self.down_proj.all_reduce = lambda value: value + 1.0

    def forward(self, value, final_all_reduce_params=None):
        out = value * 0.5
        if final_all_reduce_params is None or final_all_reduce_params.enable_allreduce:
            out = self.down_proj.all_reduce(out)
        return out


def _runtime(moe, logits):
    """``KimiK3MoERuntime`` around ``moe`` with identity latent projections and both streams."""
    from tensorrt_llm._torch.models.modeling_kimi_linear import KimiK3MoERuntime

    runtime = KimiK3MoERuntime.__new__(KimiK3MoERuntime)
    runtime._shared_sms = 0
    runtime.latent_up_rows = None
    runtime._own_latent_all_reduce = None
    runtime._moe_front_ws = None
    runtime._moe_front_plan = None
    runtime._min_latency = False
    runtime._moe_tail_wexp = None
    runtime._shared_frags = None
    runtime._shared_ws = None
    runtime._shared_world = 0
    runtime._column_parallel_tail_supported = False
    runtime._latent_up_mxfp8_supported = False
    torch.nn.Module.__init__(runtime)
    runtime.gate = type("Gate", (), {"compute_logits": staticmethod(lambda value: logits)})()
    runtime.routed_experts = moe
    runtime.top_k, runtime.num_experts, runtime.experts_per_rank = TOP_K, NUM_EXPERTS, NUM_EXPERTS
    runtime.routed_expert_down_proj = torch.nn.Identity()
    runtime.routed_expert_norm = torch.nn.Identity()
    runtime.routed_expert_up_proj = torch.nn.Identity()
    runtime.shared_experts = _SharedExperts()
    runtime._shared_all_reduce = True
    runtime._reduce_routed_output = False
    runtime._defer_shared_allreduce = False
    runtime._shared_gemms_can_lead = True
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
    runtime._routed_projection = lambda value, projection: value * 1.0
    return runtime


@cute_dsl
@pytest.mark.parametrize("m", [8, 64])
def test_shared_gemms_lead_routed_experts(monkeypatch, m):
    """From enough routed rows, the routed experts wait for the shared GEMMs and take the input
    the latent chain quantized; the output equals the concurrent order's."""
    moe_backend = "CUTEDSL"
    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    logits, bias = _logits(m, seed=80 + m)
    moe = _nvfp4_moe(moe_backend, _routing(bias))
    runtime = _runtime(moe, logits)
    # m * 16 routed rows on the single rank: both sizes lead at a threshold of one row and
    # neither at an unreachable one.
    min_rows = {True: 1, False: 1 << 30}
    monkeypatch.setattr(modeling_kimi_linear, "_SHARED_GEMMS_LEAD_MIN_ROUTED_ROWS", min_rows[True])
    plan = runtime._front_plan(m)
    assert plan.shared_first and plan.nvfp4_scale is not None and not plan.latent_down_nvfp4

    quantized_inputs = []
    real_quantize = moe.quantize_nvfp4_input

    def counting_quantize(latent):
        quantized_inputs.append(latent.shape[0])
        return real_quantize(latent)

    monkeypatch.setattr(moe, "quantize_nvfp4_input", counting_quantize)
    ready_events = []
    real_moe_forward = moe.forward

    def recording_forward(*args, **kwargs):
        ready_events.append(kwargs.get("input_ready_event"))
        return real_moe_forward(*args, **kwargs)

    monkeypatch.setattr(moe, "forward", recording_forward)
    hidden = (torch.randn((m, HIDDEN), device="cuda") * 0.5).to(torch.bfloat16)
    outputs = {}
    for lead in (False, False, True):
        monkeypatch.setattr(
            modeling_kimi_linear, "_SHARED_GEMMS_LEAD_MIN_ROUTED_ROWS", min_rows[lead]
        )
        assert runtime._front_plan(m).shared_first == lead
        quantized_inputs.clear()
        ready_events.clear()
        with torch.inference_mode(), with_multi_stream(True):
            routed, shared = runtime(hidden, return_parts=True)
        torch.cuda.synchronize()
        assert quantized_inputs == ([m] if lead else [])
        # Output equality holds in either order; the routed experts waited behind
        # the shared GEMMs' event exactly when they lead.
        assert ready_events == [
            runtime.moe_shared_gemm_event if lead else runtime.moe_latent_ready_event
        ]
        torch.testing.assert_close(shared, hidden * 0.5 + 1.0, rtol=0, atol=0)
        outputs.setdefault(lead, []).append(routed)

    # Same kernels on the same inputs: bit-identical wherever the backend reproduces itself
    # (scatter-adds of tokens with several local experts may land in another order).
    ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, SCALE)[1]
    if torch.equal(outputs[False][0], outputs[False][1]):
        assert torch.equal(outputs[True][0], outputs[False][0])
    else:
        _assert_same_rows(outputs[True][0], outputs[False][0], _local_counts(ids, 0, NUM_EXPERTS))


@pytest.mark.parametrize("blocker", [None, "comm", "eplb", "router_weight_on_input", "chunked"])
def test_nvfp4_input_scale_only_where_the_forward_takes_the_input_as_is(blocker):
    """``ConfigurableMoE.nvfp4_input_scale`` declares an NVFP4 input only where the scheduler
    takes it unchanged: no comm strategy, EPLB or router weight folded into the input, one chunk."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE

    scale = torch.tensor(1.0)
    moe = SimpleNamespace(
        comm=object() if blocker == "comm" else None,
        # The stub lists ``routed_ids_unchanged`` independently so the property
        # under test reads it (a comm strategy, EPLB or the router-weight fold
        # makes the ids reach the backend changed) without needing a full
        # ConfigurableMoE. The "chunked" blocker leaves it True on purpose: the
        # property must decline on the chunk test alone.
        routed_ids_unchanged=blocker not in ("comm", "eplb", "router_weight_on_input"),
        apply_router_weight_on_input=blocker == "router_weight_on_input",
        moe_max_num_tokens=4 if blocker == "chunked" else 64,
        backend=SimpleNamespace(prequantized_nvfp4_input_scale=lambda: scale),
    )
    expected = scale if blocker is None else None
    assert ConfigurableMoE.nvfp4_input_scale(moe, 8) is expected


@pytest.mark.parametrize(
    "num_tokens, comm, max_tokens, expected",
    [
        (8, None, 32, True),
        (32, None, 32, True),
        (33, None, 32, False),
        (0, None, 32, False),
        (8, "comm", 32, False),
        (8, None, 0, False),
    ],
)
def test_pushes_output_where_the_backend_pushes_and_no_comm(num_tokens, comm, max_tokens, expected):
    """``ConfigurableMoE.pushes_output`` takes an output push only at the token counts the backend
    declares (``output_push_max_tokens``, 0 by default) and without a comm strategy."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
    from tensorrt_llm._torch.moe.fused_moe.impl_base import MoEImplBase

    assert MoEImplBase.output_push_max_tokens(object()) == 0
    moe = SimpleNamespace(
        comm=comm, backend=SimpleNamespace(output_push_max_tokens=lambda: max_tokens)
    )
    assert ConfigurableMoE.pushes_output(moe, num_tokens) is expected


@pytest.mark.parametrize("supplied", ["prerouted", "sorted_routing", "output_push"])
@pytest.mark.parametrize("blocker", ["comm", "chunked"])
def test_scheduler_rejects_caller_supplied_args_it_cannot_honour(blocker, supplied):
    """The scheduler raises instead of dropping caller-supplied routing or an output push on a
    forward with more than one chunk or a comm strategy."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import ExternalCommMoEScheduler

    moe = SimpleNamespace(
        comm=object() if blocker == "comm" else None,
        use_dp=False,
        calculate_num_chunks=lambda _: 2 if blocker == "chunked" else 1,
        backend=SimpleNamespace(capabilities=SimpleNamespace(supports_moe_lora=False)),
        determine_communication_method=lambda *_: None,
    )
    scheduler = ExternalCommMoEScheduler(moe)
    with pytest.raises(ValueError, match="one chunk without communication"):
        scheduler.forward(
            torch.zeros(4, 8),
            torch.zeros(4, 16),
            do_finalize=True,
            output_dtype=torch.bfloat16,
            all_rank_num_tokens=[4],
            use_dp_padding=False,
            **{supplied: object()},
        )


@cute_dsl
@pytest.mark.parametrize("m", [1, 8, 16])
def test_latent_down_nvfp4_output_feeds_routed_experts(monkeypatch, m):
    """latent_down's NVFP4 epilogue output (routed only on SM107, forced here on any device the
    low-M GEMM runs on) is the input the routed experts' own quantization makes of latent_down's
    BF16 output at the same GEMM tactic: the same E2M1 codes and linear scale factors, and the
    same routed output."""
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm
    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
    from tensorrt_llm._torch.utils import Fp4QuantizedTensor

    linear_cls = modeling_kimi_linear._Fp8BlockScaleWeightReadLinear
    in_features = 1024
    logits, bias = _logits(m, seed=90 + m)
    moe = _nvfp4_moe("CUTEDSL", _routing(bias))
    runtime = _runtime(moe, logits)
    monkeypatch.setattr(modeling_kimi_linear, "_SHARED_GEMMS_LEAD_MIN_ROUTED_ROWS", 1 << 30)
    # The SM107 route's range; without a route the NVFP4 output runs the BF16 GEMM's tactic.
    monkeypatch.setattr(modeling_kimi_linear, "_low_m_gemm_nvfp4_max_m", lambda n, k: 256)
    # SM107 selects the low-M GEMM only for the measured K3 weight shapes; serve this one too.
    monkeypatch.setattr(fp8_block_scale_low_m_gemm, "max_m_for", lambda n, k: 256)
    gen = torch.Generator(device="cuda").manual_seed(m)
    weight = torch.randn((HIDDEN, in_features), device="cuda", generator=gen) / in_features**0.5
    down = linear_cls(*linear_cls.quantize_weight(weight.to(torch.bfloat16)), HIDDEN)
    down.post_load_weights()
    runtime.routed_expert_down_proj = down
    runtime._routed_projection = lambda value, projection: projection(value)
    hidden = (torch.randn((m, in_features), device="cuda", generator=gen) * 0.5).to(torch.bfloat16)

    routed_inputs = []
    real_forward = moe.forward

    def recording_forward(x, *args, **kwargs):
        routed_inputs.append(x)
        return real_forward(x, *args, **kwargs)

    monkeypatch.setattr(moe, "forward", recording_forward)
    outputs = []
    for epilogue in (True, False):
        if not epilogue:
            monkeypatch.setattr(down, "_nvfp4_out_max_m", 0)
        plan = runtime._front_plan(m)
        assert plan.latent_down_nvfp4 == epilogue and not plan.shared_first
        with torch.inference_mode(), with_multi_stream(True):
            routed, _ = runtime(hidden, return_parts=True)
        torch.cuda.synchronize()
        outputs.append(routed)

    handed, latent = routed_inputs
    assert isinstance(handed, Fp4QuantizedTensor) and not handed.is_sf_swizzled
    assert latent.dtype == torch.bfloat16
    expected = moe.quantize_nvfp4_input(latent)
    assert handed.fp4_tensor.shape == expected.fp4_tensor.shape == (m, HIDDEN // 2)
    assert torch.equal(handed.fp4_tensor, expected.fp4_tensor)
    assert torch.equal(handed.scaling_factor.view(-1), expected.scaling_factor.view(-1))
    ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, SCALE)[1]
    _assert_same_rows(outputs[0], outputs[1], _local_counts(ids, 0, NUM_EXPERTS))


@cute_dsl
def test_grouped_gemm_backend_keeps_every_sm():
    """The CuteDSL grouped GEMM does not shrink its grid for concurrent kernels, so the shared
    expert keeps every SM and may lead the routed experts there."""
    logits, bias = _logits(8, seed=95)
    moe = _nvfp4_moe("CUTEDSL", _routing(bias))
    assert moe.backend.reserve_sms_for_concurrent_kernels(16) == 0


@pytest.mark.skipif(not _fc12_available(), reason="CUTEDSL_FC12 needs SM107 and CuTe DSL internal")
@pytest.mark.parametrize("m", [8, 64])
def test_shared_expert_runs_beside_fc12_on_reserved_sms(monkeypatch, m):
    """Beside the routed grid the shared expert never leads: its GEMMs keep to the SMs the routed
    kernel leaves free (torch's SM carveout, set only while it is issued on the side stream) and
    the routed kernel (fused FC12; the small-batch kernel up to its token limit) launches with
    that SM reserve. The routed output equals the unreserved kernel's."""
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops
    from tensorrt_llm._torch.models import modeling_kimi_linear
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_small_batch_moe_custom_op as small_batch

    shared_sms = modeling_kimi_linear._SHARED_EXPERT_SMS_BESIDE_ROUTED_GRID
    logits, bias = _logits(m, seed=97 + m)
    moe = _nvfp4_moe("CUTEDSL_FC12", _routing(bias))
    runtime = _runtime(moe, logits)
    hidden = (torch.randn((m, HIDDEN), device="cuda") * 0.5).to(torch.bfloat16)
    with torch.inference_mode(), with_multi_stream(True):
        unreserved, _ = runtime(hidden, return_parts=True)
    torch.cuda.synchronize()

    assert moe.backend.reserve_sms_for_concurrent_kernels(shared_sms) == shared_sms
    runtime._shared_sms = shared_sms
    monkeypatch.setattr(modeling_kimi_linear, "_SHARED_GEMMS_LEAD_MIN_ROUTED_ROWS", 1)
    assert not runtime._front_plan(m).shared_first

    reserves = []
    real_reserved = cute_dsl_custom_ops.reserved_max_active_clusters

    def recording_reserved(full, cluster_size, reserve):
        reserves.append(reserve)
        return real_reserved(full, cluster_size, reserve)

    monkeypatch.setattr(cute_dsl_custom_ops, "reserved_max_active_clusters", recording_reserved)
    real_small_batch_ctas = small_batch._persistent_ctas

    def recording_small_batch_ctas(device_index, reserve):
        reserves.append(reserve)
        return real_small_batch_ctas(device_index, reserve)

    monkeypatch.setattr(small_batch, "_persistent_ctas", recording_small_batch_ctas)
    carveouts = []
    real_shared = runtime.shared_experts.forward

    def recording_shared(value, final_all_reduce_params=None):
        carveouts.append(torch._C._get_sm_carveout_experimental())
        return real_shared(value, final_all_reduce_params)

    monkeypatch.setattr(runtime.shared_experts, "forward", recording_shared)
    before = torch._C._get_sm_carveout_experimental()
    with torch.inference_mode(), with_multi_stream(True):
        routed, shared = runtime(hidden, return_parts=True)
    torch.cuda.synchronize()
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    assert carveouts == [total - shared_sms]
    assert torch._C._get_sm_carveout_experimental() == before
    assert reserves and set(reserves) == {shared_sms}
    torch.testing.assert_close(shared, hidden * 0.5 + 1.0, rtol=0, atol=0)
    ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, SCALE)[1]
    _assert_same_rows(routed, unreserved, _local_counts(ids, 0, NUM_EXPERTS))

    # In order on one stream (no CUDA graph) there is nothing to share SMs with: the
    # carveout goes away and FC12 keeps every SM.
    carveouts.clear()
    reserves.clear()
    with torch.inference_mode(), with_multi_stream(False):
        runtime(hidden, return_parts=True)
    assert carveouts == [before]
    assert reserves == [0]
