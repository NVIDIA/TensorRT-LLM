# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-path plumbing tests for routed-expert MoE LoRA.

The op-level tests exercise the fused_moe op and _extract_moe_lora_tensors in
isolation, so they cannot catch a regression where lora_params never reaches
the routed-expert call in the real model.forward to self.experts to
CutlassFusedMoE.run_moe path.

These CPU-only tests (no GPU or built C++ op required) assert, at each hop,
that a non-empty lora_params is forwarded:

  1. MixtralMoE.forward to the routed self.experts call.
  2. ConfigurableMoE.forward_impl to scheduler.forward.
  3. ExternalCommMoEScheduler._build_run_context to the CutlassFusedMoE
     run_moe context, and not to backends that cannot carry LoRA.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import tensorrt_llm._torch.models.modeling_glm as glm_module
import tensorrt_llm._torch.models.modeling_minimaxm3 as minimaxm3_module
import tensorrt_llm._torch.models.modeling_step3p7 as step3p7_module
from tensorrt_llm._torch.models.modeling_afmoe import AfmoeDecoderLayer, AfmoeMoE
from tensorrt_llm._torch.models.modeling_glm import Glm4DecoderLayer, Glm4MoE
from tensorrt_llm._torch.models.modeling_minimaxm2 import MiniMaxM2DecoderLayer, MiniMaxM2MoE
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3DecoderLayer, MiniMaxM3MoE
from tensorrt_llm._torch.models.modeling_mixtral import MixtralMoE
from tensorrt_llm._torch.models.modeling_step3p7 import (
    ClampedGatedMLP,
    Step3p7Attention,
    Step3p7DecoderLayer,
    Step3p7MoE,
)
from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.moe.fused_moe.fused_moe_deepgemm import DeepGemmFusedMoE
from tensorrt_llm._torch.moe.fused_moe.moe_scheduler import ExternalCommMoEScheduler
from tensorrt_llm._torch.peft.lora.layer import LoraModuleType

pytestmark = pytest.mark.cpu_only


# A unique sentinel so the assertions can verify object identity rather than
# mere truthiness; any drop/replace along the chain fails the identity check.
_LORA_PARAMS_SENTINEL = {"num_seqs": 1, "_marker": object()}


def test_mixtral_moe_forward_passes_lora_params_to_routed_experts():
    """MixtralMoE.forward must forward lora_params to routed experts."""
    num_tokens, hidden_dim = 4, 8

    hidden_states = torch.randn(num_tokens, hidden_dim)
    router_logits = torch.randn(num_tokens, 2)
    expert_out = torch.randn(num_tokens, hidden_dim)

    experts = MagicMock(return_value=expert_out)

    # Call the unbound method against a lightweight stand-in so we do not need
    # to construct real weights or CUDA streams.
    fake_self = SimpleNamespace(
        hidden_dim=hidden_dim,
        gate=MagicMock(return_value=router_logits),
        experts=experts,
    )
    attn_metadata = SimpleNamespace(all_rank_num_tokens=[num_tokens])

    MixtralMoE.forward(
        fake_self,
        hidden_states,
        attn_metadata,
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    experts.assert_called_once()
    assert experts.call_args.kwargs.get("lora_params") is _LORA_PARAMS_SENTINEL, (
        "MixtralMoE.forward dropped lora_params on the routed-expert call; "
        "routed-expert MoE LoRA would be silently disabled."
    )


@pytest.mark.parametrize(
    "moe_cls",
    [AfmoeMoE, MiniMaxM2MoE, MiniMaxM3MoE],
)
def test_model_moe_forward_passes_lora_params_to_routed_experts(
    moe_cls: type[torch.nn.Module],
) -> None:
    """MoE model wrappers must forward lora_params to routed experts."""
    num_tokens, hidden_dim = 4, 8
    hidden_states = torch.randn(num_tokens, hidden_dim)
    router_logits = torch.randn(num_tokens, 2)
    experts = MagicMock(return_value=torch.randn_like(hidden_states))
    fake_self = SimpleNamespace(
        gate=MagicMock(return_value=router_logits),
        experts=experts,
        shared_experts=None,
        allreduce=None,
    )
    if moe_cls is MiniMaxM2MoE:
        fake_self.hidden_dim = hidden_dim

    moe_cls.forward(
        fake_self,
        hidden_states,
        SimpleNamespace(all_rank_num_tokens=[num_tokens]),
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    experts.assert_called_once()
    assert experts.call_args.kwargs.get("lora_params") is _LORA_PARAMS_SENTINEL


def test_glm_moe_forward_passes_lora_params_to_routed_experts() -> None:
    """Glm4MoE.compute_routed_output must forward lora_params to experts."""
    num_tokens, hidden_dim = 4, 8
    hidden_states = torch.randn(num_tokens, hidden_dim)
    experts = MagicMock(return_value=torch.randn_like(hidden_states))
    fake_self = SimpleNamespace(
        use_dp=False,
        gate=MagicMock(return_value=torch.randn(num_tokens, 2)),
        experts=experts,
    )

    Glm4MoE.compute_routed_output(
        fake_self,
        hidden_states,
        hidden_states_fp4=None,
        all_rank_num_tokens=[num_tokens],
        do_finalize=True,
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    experts.assert_called_once()
    assert experts.call_args.kwargs.get("lora_params") is _LORA_PARAMS_SENTINEL


def test_step3p7_moe_forward_passes_lora_params_to_routed_experts() -> None:
    """Step3p7MoE.forward must forward lora_params to routed experts."""
    num_tokens, hidden_dim = 4, 8
    hidden_states = torch.randn(num_tokens, hidden_dim)
    experts = MagicMock(return_value=torch.randn_like(hidden_states))
    fake_self = SimpleNamespace(
        hidden_size=hidden_dim,
        need_fp32_gate=False,
        gate=MagicMock(return_value=torch.randn(num_tokens, 2)),
        experts=experts,
        _use_python_clamp=False,
        _moe_lora_enabled=True,
        routed_scaling_factor=1.0,
    )

    Step3p7MoE.forward(
        fake_self,
        hidden_states,
        SimpleNamespace(all_rank_num_tokens=[num_tokens]),
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    experts.assert_called_once()
    assert experts.call_args.kwargs.get("lora_params") is _LORA_PARAMS_SENTINEL


def _make_decoder_layer(decoder_cls: type[torch.nn.Module]) -> tuple[torch.nn.Module, MagicMock]:
    """Build a lightweight decoder whose real forward reaches a mocked MoE."""
    decoder = decoder_cls.__new__(decoder_cls)
    torch.nn.Module.__init__(decoder)

    hidden_states = torch.randn(4, 8)
    residual = torch.randn_like(hidden_states)
    decoder.input_layernorm = MagicMock(return_value=(hidden_states, residual))
    decoder.self_attn = MagicMock(return_value=hidden_states)
    decoder.post_attention_layernorm = MagicMock(return_value=(hidden_states, residual))

    moe = MagicMock(return_value=hidden_states)
    if decoder_cls is AfmoeDecoderLayer:
        decoder.pre_mlp_layernorm = MagicMock(return_value=(hidden_states, residual))
        decoder.post_mlp_layernorm = MagicMock(return_value=hidden_states)
        decoder.moe_enabled = True
        decoder.mlp = moe
    elif decoder_cls is MiniMaxM2DecoderLayer:
        decoder.block_sparse_moe = moe
    elif decoder_cls is MiniMaxM3DecoderLayer:
        decoder.pre_feed_forward_fusion = False
        decoder.post_feed_forward_fusion = False
        decoder.next_layer_layernorm = None
        decoder.block_sparse_moe = moe
    elif decoder_cls is Glm4DecoderLayer:
        glm_moe = Glm4MoE.__new__(Glm4MoE)
        torch.nn.Module.__init__(glm_moe)
        glm_moe.forward = moe
        decoder.mlp = glm_moe
        decoder.disable_attn_allreduce = False
        decoder.layer_idx = 0
        decoder.fusion_config = SimpleNamespace(PRE_MOE_FUSION=False, POST_MOE_FUSION=False)
        decoder.mapping = SimpleNamespace(tp_size=1, is_multi_node=lambda: True)
        decoder.next_layer_layernorm = None
    elif decoder_cls is Step3p7DecoderLayer:
        decoder.moe = moe
        decoder.share_expert = MagicMock(return_value=torch.zeros_like(hidden_states))
        decoder.allreduce = None
    else:
        raise AssertionError(f"Unsupported decoder class: {decoder_cls.__name__}")

    return decoder, moe


@pytest.mark.parametrize(
    "decoder_cls",
    [
        AfmoeDecoderLayer,
        MiniMaxM2DecoderLayer,
        MiniMaxM3DecoderLayer,
        Glm4DecoderLayer,
        Step3p7DecoderLayer,
    ],
)
def test_decoder_layer_passes_lora_params_to_moe(
    decoder_cls: type[torch.nn.Module],
) -> None:
    """Decoder layers must preserve request LoRA state at the MoE boundary."""
    decoder, moe = _make_decoder_layer(decoder_cls)
    hidden_states = torch.randn(4, 8)

    decoder.forward(
        position_ids=torch.arange(4),
        hidden_states=hidden_states,
        attn_metadata=SimpleNamespace(all_rank_num_tokens=[4]),
        residual=torch.randn_like(hidden_states),
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    moe.assert_called_once()
    assert moe.call_args.kwargs.get("lora_params") is _LORA_PARAMS_SENTINEL
    if decoder_cls is Step3p7DecoderLayer:
        assert decoder.share_expert.call_args.kwargs.get("lora_params") is _LORA_PARAMS_SENTINEL


def test_afmoe_dense_mlp_receives_lora_params() -> None:
    """AFMoE dense layers must forward request LoRA state to GatedMLP."""
    hidden_states = torch.randn(2, 4)
    residual = torch.randn_like(hidden_states)
    mlp = MagicMock(return_value=hidden_states)
    fake_self = SimpleNamespace(
        input_layernorm=MagicMock(return_value=(hidden_states, residual)),
        self_attn=MagicMock(return_value=hidden_states),
        post_attention_layernorm=MagicMock(return_value=hidden_states),
        pre_mlp_layernorm=MagicMock(return_value=(hidden_states, residual)),
        moe_enabled=False,
        mlp=mlp,
        post_mlp_layernorm=MagicMock(return_value=hidden_states),
    )

    AfmoeDecoderLayer.forward(
        fake_self,
        position_ids=torch.arange(2),
        hidden_states=hidden_states,
        attn_metadata=SimpleNamespace(),
        residual=residual,
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    assert mlp.call_args.kwargs["lora_params"] is _LORA_PARAMS_SENTINEL


def test_glm_dense_mlp_receives_lora_params() -> None:
    """GLM dense layers must forward request LoRA state to GatedMLP."""
    hidden_states = torch.randn(2, 4)
    residual = torch.randn_like(hidden_states)
    mlp = MagicMock(return_value=hidden_states)
    fake_self = SimpleNamespace(
        fusion_config=SimpleNamespace(PRE_MLP_FUSION=False, POST_MLP_FUSION=False),
        post_attention_layernorm=MagicMock(return_value=(hidden_states, residual)),
        mlp=mlp,
        mlp_tp_size=1,
        next_layer_layernorm=None,
    )

    Glm4DecoderLayer.forward_mlp(
        fake_self,
        hidden_states,
        residual,
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    assert mlp.call_args.kwargs["lora_params"] is _LORA_PARAMS_SENTINEL


def test_minimax_m3_dense_mlp_receives_lora_params() -> None:
    """MiniMax-M3 dense layers must forward request LoRA state to GatedMLP."""
    hidden_states = torch.randn(2, 4)
    residual = torch.randn_like(hidden_states)
    mlp = MagicMock(return_value=hidden_states)
    fake_self = SimpleNamespace(
        _apply_pre_feed_forward_norm=MagicMock(return_value=(hidden_states, residual)),
        mlp=mlp,
        _feed_forward_all_reduce_params=MagicMock(return_value=None),
        _apply_next_layer_layernorm=MagicMock(return_value=(hidden_states, residual)),
    )

    MiniMaxM3DecoderLayer.forward_mlp(
        fake_self,
        hidden_states,
        residual,
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    assert mlp.call_args.kwargs["lora_params"] is _LORA_PARAMS_SENTINEL


def test_step3p7_dense_mlp_receives_lora_params() -> None:
    """Step3p7 dense layers must forward request LoRA state to GatedMLP."""
    hidden_states = torch.randn(2, 4)
    residual = torch.randn_like(hidden_states)
    mlp = MagicMock(return_value=hidden_states)
    fake_self = SimpleNamespace(
        input_layernorm=MagicMock(return_value=(hidden_states, residual)),
        self_attn=MagicMock(return_value=hidden_states),
        post_attention_layernorm=MagicMock(return_value=(hidden_states, residual)),
        moe=None,
        mlp=mlp,
    )

    Step3p7DecoderLayer.forward(
        fake_self,
        position_ids=torch.arange(2),
        hidden_states=hidden_states,
        attn_metadata=SimpleNamespace(),
        residual=residual,
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    assert mlp.call_args.kwargs["lora_params"] is _LORA_PARAMS_SENTINEL


def test_feed_forward_moe_wrappers_combine_routed_and_shared_lora() -> None:
    """AFMoE, GLM, and MiniMax-M3 must retain both adapted MoE branches."""
    hidden_states = torch.zeros(2, 4)
    routed = torch.ones_like(hidden_states)
    shared = torch.full_like(hidden_states, 2.0)

    af_shared = MagicMock(return_value=shared.clone())
    af_self = SimpleNamespace(
        gate=MagicMock(return_value=torch.zeros(2, 2)),
        experts=MagicMock(return_value=routed.clone()),
        shared_experts=af_shared,
        allreduce=None,
    )
    af_output = AfmoeMoE.forward(
        af_self,
        hidden_states,
        SimpleNamespace(all_rank_num_tokens=[2]),
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    def run_both(routed_fn, shared_fn, *_args, **_kwargs):
        return routed_fn(), shared_fn()

    glm_shared = MagicMock(return_value=shared.clone())
    glm_self = SimpleNamespace(
        use_dp=True,
        shared_experts=glm_shared,
        shared_output_scale=None,
        compute_routed_output=MagicMock(return_value=routed.clone()),
        event_dict=MagicMock(),
        aux_stream=object(),
        mapping=SimpleNamespace(tp_size=1),
        top_k=1,
    )
    with patch.object(glm_module, "maybe_execute_in_parallel", side_effect=run_both):
        glm_output = Glm4MoE.forward(
            glm_self,
            hidden_states,
            all_rank_num_tokens=[2],
            lora_params=_LORA_PARAMS_SENTINEL,
        )

    m3_shared = MagicMock(return_value=shared.clone())
    m3_self = SimpleNamespace(
        gate=MagicMock(return_value=torch.zeros(2, 2)),
        experts=MagicMock(return_value=routed.clone()),
        shared_experts=m3_shared,
        event_dict=MagicMock(),
        aux_stream=object(),
        allreduce=None,
    )
    with patch.object(minimaxm3_module, "maybe_execute_in_parallel", side_effect=run_both):
        m3_output = MiniMaxM3MoE.forward(
            m3_self,
            hidden_states,
            SimpleNamespace(all_rank_num_tokens=[2]),
            lora_params=_LORA_PARAMS_SENTINEL,
        )

    for output in (af_output, glm_output, m3_output):
        torch.testing.assert_close(output, torch.full_like(hidden_states, 3.0))
    for shared_expert in (af_shared, glm_shared, m3_shared):
        assert shared_expert.call_args.kwargs["lora_params"] is _LORA_PARAMS_SENTINEL


def test_step3p7_clamped_mlp_applies_gate_up_and_down_lora() -> None:
    """The clamped path must preserve both projection adapter contributions."""
    hidden_states = torch.zeros(1, 2)
    gate_up_lora = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
    down_lora = torch.full_like(hidden_states, 3.0)

    def forward_with_base(base_forward, lora_layers, x, params, layer_idx):
        assert params is _LORA_PARAMS_SENTINEL
        assert layer_idx == 4
        assert lora_layers == ("split_gate_up", "fused_gate_up")
        return base_forward() + gate_up_lora

    def down_projection(x, *, all_reduce_params, lora_params, layer_idx):
        assert all_reduce_params == "reduce"
        assert lora_params is _LORA_PARAMS_SENTINEL
        assert layer_idx == 4
        return x + down_lora

    fake_self = SimpleNamespace(
        swiglu_limit=5.0,
        layer_idx=4,
        _uneven_tp_blocks_lora=False,
        gate_up_proj=MagicMock(return_value=torch.zeros_like(gate_up_lora)),
        splitted_gate_up_lora="split_gate_up",
        fused_gate_up_lora="fused_gate_up",
        down_proj=MagicMock(side_effect=down_projection),
    )

    with patch.object(step3p7_module.LoraLayer, "forward_with_base", side_effect=forward_with_base):
        output = ClampedGatedMLP.forward(
            fake_self,
            hidden_states,
            final_all_reduce_params="reduce",
            lora_params=_LORA_PARAMS_SENTINEL,
        )

    expected = torch.full_like(hidden_states, torch.nn.functional.silu(torch.tensor(1.0)) * 2 + 3)
    torch.testing.assert_close(output, expected)


def test_step3p7_moe_lora_uses_clamp_capable_expert_path() -> None:
    """Step3p7 must not bypass routed-expert LoRA through its Python path."""
    hidden_states = torch.randn(4, 8)
    experts = MagicMock(return_value=torch.randn_like(hidden_states))
    python_clamped_forward = MagicMock(return_value=torch.randn_like(hidden_states))
    fake_self = SimpleNamespace(
        hidden_size=hidden_states.shape[-1],
        need_fp32_gate=False,
        gate=MagicMock(return_value=torch.randn(4, 2)),
        experts=experts,
        _use_python_clamp=True,
        _clamp_weights_loaded=True,
        _moe_lora_enabled=True,
        _python_clamped_moe_forward=python_clamped_forward,
        routed_scaling_factor=1.0,
    )

    Step3p7MoE.forward(
        fake_self,
        hidden_states,
        SimpleNamespace(all_rank_num_tokens=[4]),
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    python_clamped_forward.assert_not_called()
    assert experts.call_args.kwargs.get("lora_params") is _LORA_PARAMS_SENTINEL


def test_step3p7_cutlass_moe_lora_applies_routed_scale_once() -> None:
    """Separated routing and the common output path must not both scale MoE output."""
    routed_scaling_factor = 3.0
    text_config = SimpleNamespace(
        hidden_size=4,
        moe_num_experts=2,
        moe_top_k=2,
        moe_intermediate_size=8,
        moe_router_scaling_factor=routed_scaling_factor,
        need_fp32_gate=False,
        torch_dtype=torch.float32,
    )
    model_config = SimpleNamespace(
        pretrained_config=text_config,
        mapping=SimpleNamespace(enable_attention_dp=True, tp_size=1),
        lora_config=object(),
    )
    experts = MagicMock()

    with (
        patch.object(step3p7_module, "Linear", return_value=MagicMock()),
        patch.object(step3p7_module, "_select_python_expert_path", return_value=(1.0, False, "")),
        patch.object(step3p7_module, "has_moe_lora_targets", return_value=True),
        patch.object(step3p7_module, "create_moe", return_value=experts) as create_moe,
    ):
        moe = Step3p7MoE(model_config, layer_idx=0, aux_stream_dict={})

    routing_method = create_moe.call_args.kwargs["routing_method"]
    moe.router_bias.router_bias.data.zero_()
    router_logits = torch.zeros(1, text_config.moe_num_experts)
    moe.gate.return_value = router_logits
    moe._use_python_clamp = True
    moe._clamp_weights_loaded = True

    def separated_experts(
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        _, routing_weights = routing_method.apply(logits)
        return routing_weights.sum(dim=-1, keepdim=True).expand_as(hidden_states)

    experts.side_effect = separated_experts
    hidden_states = torch.ones(1, text_config.hidden_size)

    routed = moe(
        hidden_states,
        SimpleNamespace(all_rank_num_tokens=[1]),
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    torch.testing.assert_close(
        routed,
        torch.full_like(hidden_states, routed_scaling_factor),
    )


def test_step3p7_attention_applies_lora_to_qkv_and_output() -> None:
    """Step3p7 attention must retain non-zero QKV and output LoRA contributions."""
    hidden_states = torch.zeros(1, 4)
    lora_params = {"enabled_modules": {"attn_qkv", "attn_dense"}}
    qkv_lora_contribution = torch.ones_like(hidden_states)
    output_lora_contribution = torch.full_like(hidden_states, 2.0)

    def forward_with_base(base_forward, lora_layers, x, params, layer_idx):
        assert params is lora_params
        assert layer_idx == 3
        assert lora_layers == ("split_qkv_lora", "fused_qkv_lora")
        return base_forward() + qkv_lora_contribution

    def output_projection(x, *, lora_params, layer_idx):
        assert lora_params is not None
        assert lora_params["enabled_modules"] == {"attn_qkv", "attn_dense"}
        assert layer_idx == 3
        return x + output_lora_contribution

    fake_self = SimpleNamespace(
        rope_fusion=True,
        text_config=SimpleNamespace(num_hidden_layers=8),
        layer_idx=3,
        sliding_window=None,
        qkv_proj=MagicMock(return_value=torch.zeros_like(hidden_states)),
        splitted_qkv_lora="split_qkv_lora",
        fused_qkv_lora="fused_qkv_lora",
        apply_rope=MagicMock(side_effect=lambda qkv, _k, _v, _positions: (qkv, qkv, qkv)),
        convert_qkv=MagicMock(side_effect=lambda q, k, v: (q, k, v)),
        forward_impl=MagicMock(side_effect=lambda q, _k, _v, *_args, **_kwargs: q),
        use_head_wise_gate=False,
        o_proj=MagicMock(side_effect=output_projection),
    )

    with patch.object(step3p7_module.LoraLayer, "forward_with_base", side_effect=forward_with_base):
        output = Step3p7Attention.forward(
            fake_self,
            position_ids=torch.arange(1),
            hidden_states=hidden_states,
            attn_metadata=SimpleNamespace(),
            lora_params=lora_params,
        )

    torch.testing.assert_close(output, torch.full_like(hidden_states, 3.0))
    assert fake_self.forward_impl.call_args.kwargs["has_lora"] is True
    assert fake_self.o_proj.call_args.kwargs["lora_params"] is lora_params


def test_configurable_moe_forward_impl_forwards_lora_params_to_scheduler():
    """ConfigurableMoE.forward_impl must forward lora_params to the scheduler
    so routed-expert MoE LoRA is not dropped."""
    x = torch.randn(4, 8)
    router_logits = torch.randn(4, 2)

    scheduler = MagicMock()
    scheduler.forward = MagicMock(return_value=torch.zeros_like(x))

    fake_self = SimpleNamespace(
        scheduler=scheduler,
        enable_dwdp=False,
        repeat_idx=0,
        repeat_count=1,
    )

    ConfigurableMoE.forward_impl(
        fake_self,
        x,
        router_logits,
        lora_params=_LORA_PARAMS_SENTINEL,
    )

    scheduler.forward.assert_called_once()
    assert scheduler.forward.call_args.kwargs.get("lora_params") is _LORA_PARAMS_SENTINEL, (
        "ConfigurableMoE.forward_impl dropped lora_params before the scheduler; "
        "routed-expert MoE LoRA would be silently disabled on the default path."
    )


def _make_external_comm_scheduler(backend_cls):
    """Build an ExternalCommMoEScheduler whose moe.backend is an uninitialized
    instance of backend_cls, sufficient for building a run context without
    constructing weights."""
    backend = backend_cls.__new__(backend_cls)
    moe = SimpleNamespace(
        backend=backend,
        comm=None,
        enable_alltoall=False,
        mapping=SimpleNamespace(tp_size=1),
        routing_method=SimpleNamespace(top_k=2),
    )
    scheduler = ExternalCommMoEScheduler.__new__(ExternalCommMoEScheduler)
    scheduler.moe = moe
    return scheduler


def _build_run_context(scheduler):
    return scheduler._build_run_context(
        x=torch.randn(4, 8),
        x_sf=None,
        token_selected_slots=torch.zeros(4, 2, dtype=torch.int32),
        token_final_scales=torch.ones(4, 2),
        router_logits=None,
        do_finalize=True,
        output_dtype=torch.bfloat16,
        all_rank_num_tokens=None,
        lora_params=_LORA_PARAMS_SENTINEL,
    )


def test_scheduler_threads_lora_params_to_cutlass_run_context():
    """The run context handed to CutlassFusedMoE.run_moe must carry
    lora_params."""
    ctx = _build_run_context(_make_external_comm_scheduler(CutlassFusedMoE))

    assert ctx.lora_params is _LORA_PARAMS_SENTINEL, (
        "Scheduler dropped lora_params before CutlassFusedMoE.run_moe; "
        "routed-expert MoE LoRA would be silently disabled."
    )


def test_scheduler_does_not_thread_lora_params_to_non_lora_backend():
    """Backends that do not declare supports_moe_lora must get lora_params
    cleared, so an adapter never silently no-ops inside their kernel."""
    ctx = _build_run_context(_make_external_comm_scheduler(DeepGemmFusedMoE))

    assert ctx.lora_params is None, (
        "lora_params must only reach backends declaring supports_moe_lora; "
        f"DeepGemmFusedMoE does not fuse it. Got: {ctx.lora_params}"
    )


def _moe_lora_params_for_layer(layer_idx):
    """A minimal lora_params carrying a routed-expert MoE module (moe_h_to_4h)
    for layer_idx, enough for _moe_lora_active."""
    module_id = int(LoraModuleType.from_string("moe_h_to_4h"))
    return {
        "num_seqs": 1,
        layer_idx: {module_id: {"adapter_size": None, "weight_pointers": None}},
    }


def test_cutlass_moe_lora_active_detects_layer_modules():
    """_moe_lora_active is the predicate the multi-chunk guard relies on: True
    only when this layer has a routed-expert MoE LoRA module."""
    backend = CutlassFusedMoE.__new__(CutlassFusedMoE)
    backend.layer_idx = 3

    assert backend._moe_lora_active(_moe_lora_params_for_layer(3)) is True
    # No MoE modules for this layer, or empty params, means inactive.
    assert backend._moe_lora_active(_moe_lora_params_for_layer(5)) is False
    assert backend._moe_lora_active(None) is False
    assert backend._moe_lora_active({"num_seqs": 1}) is False


def test_scheduler_rejects_multichunk_with_moe_lora():
    """The ConfigurableMoE scheduler must reject multi-chunk execution when
    routed-expert MoE LoRA is active, with an actionable message rather than a
    deep C++ kernel failure."""
    backend = CutlassFusedMoE.__new__(CutlassFusedMoE)
    backend.layer_idx = 0

    moe = SimpleNamespace(
        backend=backend,
        calculate_num_chunks=lambda *_args, **_kw: 2,
    )
    scheduler = ExternalCommMoEScheduler.__new__(ExternalCommMoEScheduler)
    scheduler.moe = moe

    x = torch.randn(8, 4)
    router_logits = torch.randn(8, 2)

    with pytest.raises(NotImplementedError, match="multi-chunk"):
        scheduler.forward(
            x,
            router_logits,
            do_finalize=True,
            output_dtype=torch.bfloat16,
            all_rank_num_tokens=None,
            use_dp_padding=False,
            lora_params=_moe_lora_params_for_layer(0),
        )
