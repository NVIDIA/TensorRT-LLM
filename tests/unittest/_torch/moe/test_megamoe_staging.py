# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests of the MegaMoE CuteDSL backend's input staging."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl as mm

pytestmark = pytest.mark.cpu_only

IMPL = mm.TrtllmCutedslMegaMoeNvfp4Impl


@pytest.fixture
def preparation_backend(monkeypatch):
    """The backend on CPU tensors, with only quantization and launch endpoints replaced."""

    class Quantize:
        def __call__(self, x, *args, **kwargs):
            return torch.zeros(len(x), 32, dtype=torch.uint8), torch.zeros(
                len(x), 4, dtype=torch.uint8
            )

        def into(self, x, scale, activation, sf, *args):
            activation.zero_()
            sf.zero_()

    def stage(x, scale, activation, sf, ids, weights, out_ids, out_weights, *args):
        activation.zero_()
        sf.zero_()
        out_ids[: len(ids)].copy_(ids)
        out_weights[: len(ids)].copy_(weights)

    monkeypatch.setattr(mm, "_MEGAMOE_PRIME_LADDER", False)
    monkeypatch.setattr(torch.ops.trtllm, "fp4_quantize", Quantize())
    monkeypatch.setattr(torch.ops.trtllm, "megamoe_stage_inputs", stage)
    backend = IMPL.__new__(IMPL)
    torch.nn.Module.__init__(backend)
    for name, value in dict(
        _active_launch_max_tokens=8,
        _active_launch_output_dtype=torch.bfloat16,
        _quantized_into=None,
        _prestaged=None,
        _handed_out_route_staging=None,
        _shared_expert=None,
        _slot_of_expert=None,
        _replicated=False,
        _padding_drop_buckets=set(),
        ep_size=2,
        ep_rank=0,
        tactic_autotune=False,
        _genphase_direct_input_buckets=set(),
        _maxt_buckets=(8, 16),
        max_num_tokens=16,
        _topk_idx_dtype=torch.int64,
        _weights_created=True,
        _symm_providers={16: object()},
        _local_staging_cache={},
        _launch_buffers={},
        dtype=torch.bfloat16,
        hidden_size=64,
        routing_method=SimpleNamespace(experts_per_token=2),
        fc31_input_scale=torch.ones(1),
        fc31_act_scale=None,
        quant_method=object(),
        layer_load_balancer=None,
    ).items():
        setattr(backend, name, value)

    def build(**kwargs):
        bucket, dtype = kwargs["launch_max_T"], kwargs["output_dtype"]
        buffers = mm._MegaMoeBuffers(
            activation=torch.zeros(bucket, 32, dtype=torch.uint8),
            activation_sf=torch.zeros(bucket, 4, dtype=torch.uint8),
            topk_idx_local=torch.zeros(bucket, 2, dtype=torch.int64),
            topk_weights=torch.zeros(bucket, 2),
            combine_output=torch.zeros(bucket, 64, dtype=dtype),
            shared_workspace=torch.zeros(1),
            peer_offsets=(0, 0),
        )
        backend._local_staging_cache[
            mm._StagingKey(bucket, 2, 64, str(kwargs["device"]), dtype)
        ] = {}
        return buffers

    backend._build_buffers = build
    backend._build_weight_view = lambda: None
    backend._stage_inputs = lambda **kwargs: None
    backend._launch_megamoe_kernel = lambda **kwargs: kwargs["combine_output"]
    backend._acquire_buffers(
        top_k=2, hidden=64, device=torch.device("cpu"), output_dtype=torch.bfloat16, launch_max_T=8
    )
    return backend


@pytest.mark.parametrize("drops", [False, True])
@pytest.mark.parametrize("static_slots", [False, True])
def test_routes_replaced_after_the_fused_route_quant_are_staged_without_the_activation(
    preparation_backend, monkeypatch, static_slots, drops
):
    """Regression: routes replaced after the DSV4 fused route+quant (perfect router, calibrator
    replay) made run_moe raise; it now keeps the staged activation and stages only the routes."""
    from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoERunContext

    rows, top_k, experts = 3, 2, 4
    backend = preparation_backend
    backend._fuses_route_quant = True
    backend.routing_method = SimpleNamespace(
        experts_per_token=top_k,
        top_k=top_k,
        e_score_correction_bias=torch.zeros(experts),
        routed_scaling_factor=1.0,
    )
    if static_slots:
        backend._slot_of_expert = torch.tensor([3, 2, 1, 0], dtype=torch.int32)
    # A bucket whose kernel drops masked routes: the fused launch stages it under the word.
    backend._padding_drop_buckets = {8} if drops else set()
    word = torch.tensor([1], dtype=torch.int32)
    monkeypatch.setattr(mm.live_rows, "current_word", lambda device: word)
    words = []

    def route_quant(logits, bias, x, scale, k, scaling, dtype, *, staged, slot_of_expert, **kw):
        words.append(kw.get("live_row_word"))
        ids_out, weights_out, x_out, sf_out = staged
        ids = torch.zeros(rows, k, dtype=torch.int32)
        ids_out[:rows] = ids if slot_of_expert is None else slot_of_expert[ids.long()]
        ids_out[rows:] = -1
        weights_out[:rows] = 1.0
        weights_out[rows:] = 0.0
        x_out[:rows] = 7
        sf_out[:rows] = 5
        return ids, weights_out[:rows], x_out[:rows], sf_out[:rows], None

    staged_routes, stage_activation = [], []
    monkeypatch.setattr(mm, "dsv4_route_nvfp4_quant", route_quant, raising=False)
    monkeypatch.setattr(
        torch.ops.trtllm,
        "megamoe_stage_routes",
        lambda ids, *args: staged_routes.append(ids.clone()),
        raising=False,
    )
    del backend._stage_inputs  # the real staging, down to the route op
    stage = backend._stage_inputs
    backend._stage_inputs = lambda **kw: stage_activation.append(kw["stage_activation"]) or stage(
        **kw
    )
    x = torch.zeros(rows, 64, dtype=torch.bfloat16)

    def forward(routes=None):
        backend.set_adaptive_launch_tokens(8, output_dtype=torch.bfloat16)
        fused = backend.try_fused_route_quant(x, torch.zeros(rows, experts))
        context = MoERunContext(
            x=fused.x,
            x_sf=fused.x_sf,
            token_selected_experts=fused.token_selected_experts if routes is None else routes,
            token_final_scales=fused.token_final_scales,
            output_dtype=torch.bfloat16,
        )
        backend.run_moe(context)

    forward()
    assert not staged_routes and not stage_activation  # The fused launch staged the bucket.
    assert len(words) == 1 and words[0] is (word if drops else None)
    replaced = torch.tensor([[1, 2], [3, 0], [2, 1]], dtype=torch.int32)
    forward(replaced)
    assert stage_activation == [False]
    expected = backend.slots_of_experts(replaced) if static_slots else replaced
    assert len(staged_routes) == 1 and torch.equal(staged_routes[0], expected)
    assert backend._quantized_into is None and backend._prestaged is None


_GLM_SM107 = (
    "tensorrt_llm._torch.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__high_throughput"
)


def _glm_sm107_moe(*names):
    """The GLM-5.2 SM107 MoE modules, or a skip where the op or the target is absent."""
    from tensorrt_llm._torch.modeling_v2._native import NoTarget
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as shared

    if not shared.IS_MEGAMOE_OP_AVAILABLE:
        pytest.skip("the MegaMoE CuteDSL op is not available")
    try:
        return [importlib.import_module(f"{_GLM_SM107}.ops.moe.{name}") for name in names]
    except NoTarget as error:
        pytest.skip(f"this installation does not carry the GLM-5.2 SM107 target: {error}")


def test_rubin_direct_launch_spends_the_quantize_input_preparation(preparation_backend):
    """Regression: a GLM-5.2 SM107 direct launch kept the output dtype and the staged-rows record
    of quantize_input, so a later launch into a bucket without a staging raised at its check."""
    from tensorrt_llm._torch.moe.fused_moe.impl_contract import MoERunContext

    (megamoe,) = _glm_sm107_moe("megamoe")
    backend = preparation_backend
    backend.__class__ = megamoe.RubinMegaMoE
    backend._pending_direct_stage = backend._deferred_topk_tokens = None
    backend._launch_defer, backend._maxt_buckets = False, (4, 8, 16)
    backend.set_adaptive_launch_tokens(None)  # a forward that set neither the hint nor a dtype
    cpu = torch.device("cpu")
    backend._acquire_buffers(
        top_k=2, hidden=64, device=cpu, output_dtype=torch.bfloat16, launch_max_T=16
    )
    x, ids, weights = torch.zeros(3, 64).bfloat16(), torch.zeros(3, 2).int(), torch.ones(3, 2)

    def launch(quantized, sf):
        context = MoERunContext(ids, weights, quantized, sf, output_dtype=torch.bfloat16)
        backend.run_moe(context)

    for _ in range(2):  # direct decodes, in RubinFusedCommScheduler's order
        rows = torch.zeros(8, 32, dtype=torch.uint8), torch.zeros(8, 4, dtype=torch.uint8)
        stage = megamoe.DirectStage(8, 3, ids, weights, *rows)
        quantized = backend.quantize_input_into(x, stage)
        backend.set_adaptive_launch_tokens(8)
        launch(*quantized)
    backend.set_adaptive_launch_tokens(4, output_dtype=torch.bfloat16)  # a bucket never launched
    launch(*backend.quantize_input(x))


def test_rubin_glm_buckets_read_the_kernel_properties_by_name(monkeypatch):
    """Regression: GLM-5.2 SM107 multi-rank init failed unpacking the kernel-properties record."""
    from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as shared
    from tensorrt_llm._torch.utils import ActivationType

    runner, megamoe = _glm_sm107_moe("megamoe_runner", "megamoe")

    class GenerationPhaseRunner:
        def __init__(self, local_rank, output_dtype, **kwargs):
            self.max_tokens_per_rank = kwargs["max_tokens_per_rank"]

        def unique_id(self):
            return "generation-phase"

        def _default_tactic(self, num_tokens):
            return None

        def _resolve_kernel_descriptor(self, tactic, allow_fallback):
            return SimpleNamespace(get_workspace_sizes=lambda: (0, 2048)), None

    generic = shared.MegaMoeKernelProperties(4096, False, True)
    monkeypatch.setattr(
        shared, "query_megamoe_kernel_properties", lambda **kw: generic, raising=False
    )
    monkeypatch.setattr(runner, "RubinMegaMoERunner", GenerationPhaseRunner)
    providers = []
    monkeypatch.setattr(
        shared,
        "get_megamoe_symm_provider",
        lambda **kw: providers.append(kw["shared_workspace_bytes"]) or object(),
        raising=False,
    )
    ladder = (32, 64, 128, 192, 256, 1024, 4096)
    backend = SimpleNamespace(
        _ep_pg=object(),
        routing_method=SimpleNamespace(experts_per_token=8),
        _maxt_buckets=ladder,
        _profiling_scratch_provider_ref=None,
        _launch_buffers={},
        ep_size=16,
        ep_rank=0,
        compute_slot_count=16,
        hidden_size=6144,
        intermediate_size_per_partition=2048,
        expand_intermediate_size_per_partition=4096,
        apply_topk_in_fc1=True,
        act_clamp=None,
        activation_type=ActivationType.Swiglu,
        act_alpha=None,
        act_beta=None,
        _use_in_kernel_fc2_reduce=lambda max_T: False,
        combine_format="bf16",
        _genphase_direct_input_buckets=set(),
        _padding_drop_buckets=set(),
        dtype=torch.bfloat16,
        _symm_providers={},
        tactic_autotune=False,
    )
    megamoe.RubinMegaMoE._alloc_symm_provider(backend)
    generation_phase = set(runner.GEN_TOKEN_BUCKETS) & set(ladder)
    assert backend._genphase_direct_input_buckets == generation_phase
    assert backend._padding_drop_buckets == set(ladder) - generation_phase
    assert providers == [2048 if bucket in generation_phase else 4096 for bucket in ladder]
    for bucket in (64, 256):
        properties = runner.query_rubin_kernel_properties(max_tokens_per_rank=bucket)
        assert isinstance(properties, shared.MegaMoeKernelProperties)
