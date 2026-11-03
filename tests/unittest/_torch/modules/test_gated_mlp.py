# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Callable
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.locality_domain.policy import LocalityDomainPolicy
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules import gated_mlp as gated_mlp_module
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import Linear as _RealLinear
from tensorrt_llm._torch.visual_gen import config as visual_gen_config


def _make_gate_up_proj(
    projected: torch.Tensor,
    *,
    partitioned: bool,
) -> nn.Module:
    gate_up_proj = nn.Module()
    gate_up_proj.has_nvfp4 = True
    gate_up_proj.has_bias = False
    gate_up_proj.partition_plan = SimpleNamespace(enabled=partitioned)
    gate_up_proj.can_use_cute_dsl_nvfp4_swiglu_blackwell = Mock(return_value=not partitioned)
    gate_up_proj.forward = Mock(return_value=projected)
    return gate_up_proj


def _make_down_proj() -> nn.Module:
    down_proj = nn.Module()
    down_proj.has_fp8_qdq = False
    down_proj.has_w4a8_nvfp4_fp8 = False
    # The unfused branch calls ``_can_fuse_swiglu_fp8_quant``, whose first
    # check reads this attribute, so a stand-in without it raises
    # AttributeError before the assertions below are reached.
    down_proj.has_fp8_block_scales = False
    down_proj.forward = Mock(side_effect=lambda value, **kwargs: value + 1)
    return down_proj


@pytest.mark.cpu_only
def test_diffusion_config_without_locality_domain_policy() -> None:
    config = visual_gen_config.DiffusionModelConfig()
    mlp = GatedMLP(
        hidden_size=8,
        intermediate_size=16,
        bias=False,
        dtype=torch.bfloat16,
        config=config,
    )

    assert mlp.gate_up_proj.weight.shape == (32, 8)
    assert mlp.down_proj.weight.shape == (8, 16)
    for projection in (mlp.gate_up_proj, mlp.down_proj):
        assert not projection._locality_domain_policy.enabled
        assert not projection.partition_plan.enabled
        assert projection._locality_domain_weight_shards is None


def test_gate_up_partition_falls_back_to_swiglu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mlp = GatedMLP(hidden_size=2, intermediate_size=2, bias=False)
    mlp.use_cute_dsl_blockscaling_mm = True
    projected = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mlp.gate_up_proj = _make_gate_up_proj(projected, partitioned=True)
    mlp.down_proj = _make_down_proj()

    gate, up = projected.chunk(2, dim=-1)
    activated = F.silu(gate) * up
    swiglu = Mock(side_effect=lambda value, **kwargs: F.silu(value[..., :2]) * value[..., 2:])
    monkeypatch.setattr(gated_mlp_module, "swiglu", swiglu)
    fused_gate_up_swiglu = Mock(side_effect=AssertionError("Blackwell fused op must not run"))
    monkeypatch.setattr(mlp, "_fused_gate_up_swiglu", fused_gate_up_swiglu)

    inputs = torch.tensor([[5.0, 6.0]])
    output = mlp(inputs)

    assert not mlp._can_fuse_gate_up_swiglu()
    mlp.gate_up_proj.forward.assert_called_once_with(inputs)
    # ``GatedMLP`` forwards all three SwiGLU shape parameters on every call;
    # they are None here because this layer is plain SwiGLU.
    swiglu.assert_called_once_with(
        projected, swiglu_limit=None, swiglu_alpha=None, swiglu_beta=None
    )
    mlp.down_proj.forward.assert_called_once()
    down_args, down_kwargs = mlp.down_proj.forward.call_args
    torch.testing.assert_close(down_args[0], activated)
    assert down_kwargs == {"all_reduce_params": None, "layer_idx": None}
    fused_gate_up_swiglu.assert_not_called()
    torch.testing.assert_close(output, activated + 1)


@pytest.mark.parametrize(
    "swiglu_limit, expected",
    [(None, True), (float("inf"), True), (7.0, False)],
)
def test_swiglu_limit_controls_gate_up_fusion_capability(
    swiglu_limit: float | None,
    expected: bool,
) -> None:
    mlp = GatedMLP(
        hidden_size=2,
        intermediate_size=2,
        bias=False,
        swiglu_limit=swiglu_limit,
        use_cute_dsl_blockscaling_mm=True,
    )

    assert mlp.gate_up_proj.use_cute_dsl_nvfp4_swiglu_blackwell is expected


@pytest.mark.parametrize(
    ("activation", "expected"),
    [
        pytest.param(F.silu, True, id="plain-swiglu"),
        pytest.param(lambda value: value, False, id="custom-swiglu-oai"),
    ],
)
def test_activation_controls_fp8_quant_fusion_capability(
    monkeypatch: pytest.MonkeyPatch,
    activation: Callable[[torch.Tensor], torch.Tensor],
    expected: bool,
) -> None:
    mlp = GatedMLP(
        hidden_size=2,
        intermediate_size=2,
        bias=False,
        activation=activation,
    )
    down_proj = nn.Module()
    down_proj.has_fp8_block_scales = True
    down_proj.use_cute_dsl_blockscaling_mm = True
    down_proj.disable_deep_gemm = False
    mlp.down_proj = down_proj
    monkeypatch.setattr(gated_mlp_module, "get_sm_version", lambda: 107)
    monkeypatch.setattr(gated_mlp_module, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", True)

    assert mlp._can_fuse_swiglu_fp8_quant() is expected


@pytest.mark.parametrize(
    ("mlp_kwargs", "expected_enabled"),
    [
        pytest.param({}, False, id="default"),
        pytest.param(
            {"enable_locality_domain_bf16_linear": True},
            True,
            id="explicit-opt-in",
        ),
    ],
)
def test_locality_domain_bf16_linear_opt_in_propagates_to_both_projections(
    monkeypatch: pytest.MonkeyPatch,
    mlp_kwargs: dict[str, bool],
    expected_enabled: bool,
) -> None:
    policy = LocalityDomainPolicy(enabled=True)
    config = ModelConfig(
        use_cute_dsl_bf16_gemm=True,
        locality_domain_policy=policy,
    )
    linear_constructor = Mock(side_effect=[nn.Module(), nn.Module()])
    # GatedMLP.__init__ calls Linear._calc_shard for uneven-TP sharding;
    # the Mock must delegate to the real staticmethod or the arithmetic
    # downstream operates on Mock objects.
    linear_constructor._calc_shard = _RealLinear._calc_shard
    monkeypatch.setattr(gated_mlp_module, "Linear", linear_constructor)

    GatedMLP(
        hidden_size=8,
        intermediate_size=16,
        bias=False,
        dtype=torch.bfloat16,
        config=config,
        **mlp_kwargs,
    )

    assert linear_constructor.call_count == 2
    gate_up_call, down_call = linear_constructor.call_args_list
    assert gate_up_call.args == (8, 32)
    assert down_call.args == (16, 8)
    for projection_call in (gate_up_call, down_call):
        assert projection_call.kwargs["use_cute_dsl_bf16_gemm"] is True
        assert projection_call.kwargs["enable_locality_domain_bf16_linear"] is expected_enabled
        assert projection_call.kwargs["locality_domain_policy"] is policy


def test_deepseek_dense_layer_enables_locality_domain_bf16_mlp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm._torch.models import modeling_deepseekv3
    from tensorrt_llm._torch.utils import AuxStreamType
    from tensorrt_llm.models.modeling_utils import QuantConfig

    policy = LocalityDomainPolicy(enabled=True)
    pretrained_config = SimpleNamespace(
        hidden_size=128,
        moe_intermediate_size=64,
        n_routed_experts=None,
        n_shared_experts=0,
        num_experts_per_tok=1,
        model_type="deepseek_v3",
        first_k_dense_replace=3,
        moe_layer_freq=1,
        intermediate_size=256,
        torch_dtype=torch.bfloat16,
        rms_norm_eps=1e-6,
    )
    config = ModelConfig(
        pretrained_config=pretrained_config,
        quant_config=QuantConfig(),
        skip_create_weights_in_init=True,
        use_cute_dsl_bf16_gemm=True,
        locality_domain_policy=policy,
    )
    monkeypatch.setattr(
        modeling_deepseekv3,
        "DeepseekV3Attention",
        Mock(return_value=nn.Identity()),
    )
    monkeypatch.setattr(
        modeling_deepseekv3,
        "can_access_peer",
        Mock(return_value=False),
    )
    gated_mlp_constructor = Mock(return_value=nn.Identity())
    monkeypatch.setattr(modeling_deepseekv3, "GatedMLP", gated_mlp_constructor)

    modeling_deepseekv3.DeepseekV3DecoderLayer(
        config,
        layer_idx=0,
        aux_stream_dict={AuxStreamType.Attention: None},
    )

    assert gated_mlp_constructor.call_count == 1
    call = gated_mlp_constructor.call_args
    assert call.kwargs["config"] is config
    assert call.kwargs["enable_locality_domain_bf16_linear"] is True


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="needs SM100/SM103/SM107",
)
@pytest.mark.parametrize("rows", [6, 12])
@pytest.mark.parametrize("backend", ["deep_gemm", "cute_dsl", "disable_deep_gemm"])
@torch.inference_mode()
def test_fp8_block_scale_mlp_small_m_matches_deep_gemm(rows, backend):
    """A tuned shared-expert-sized FP8 MLP is at least as close to the exact product as its
    untuned forward."""
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    if backend != "deep_gemm" and (
        torch.cuda.get_device_capability() != (10, 7) or not IS_CUTLASS_DSL_RUBIN_AVAILABLE
    ):
        pytest.skip("The CuTe DSL candidate requires Rubin helpers on SM107")
    torch.manual_seed(0)
    hidden, intermediate = 5120, 1152
    config = ModelConfig(
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128)
    )
    mlp = GatedMLP(
        hidden_size=hidden,
        intermediate_size=intermediate,
        bias=False,
        dtype=torch.bfloat16,
        config=config,
        use_cute_dsl_blockscaling_mm=backend == "cute_dsl",
    ).cuda()
    if backend == "disable_deep_gemm":
        for linear in (mlp.gate_up_proj, mlp.down_proj):
            linear.disable_deep_gemm = True

    def block_scaled(out_features, in_features):
        weight = torch.randn(out_features, in_features) * in_features**-0.5
        blocks = weight.view(out_features // 128, 128, in_features // 128, 128)
        scale = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax((1, 3)) / 448)))
        quantized = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn)
        dequantized = (quantized.double() * scale[:, None, :, None]).view(out_features, in_features)
        return {
            "weight": quantized.view(out_features, in_features),
            "weight_scale_inv": scale,
        }, dequantized.cuda()

    gate, gate_ref = block_scaled(intermediate, hidden)
    up, up_ref = block_scaled(intermediate, hidden)
    down, down_ref = block_scaled(hidden, intermediate)
    mlp.gate_up_proj.load_weights([gate, up])
    mlp.down_proj.load_weights([down])
    for linear in (mlp.gate_up_proj, mlp.down_proj):
        linear.post_load_weights()
    x = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    reference = (F.silu(x.double() @ gate_ref.t()) * (x.double() @ up_ref.t())) @ down_ref.t()

    def rms_error(output):
        return (output.double() - reference).square().mean().sqrt().item()

    AutoTuner.get().clear_cache()
    plain = mlp(x)
    with autotune():
        mlp(x)
    tuned = mlp(x)
    # Whichever kernels win, the tuned forward is at least as close to the exact product.
    assert rms_error(tuned) <= 1.05 * rms_error(plain)
    assert rms_error(tuned) <= 2e-2 * reference.square().mean().sqrt().item()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mlp(x)
    x.copy_(torch.randn_like(x))
    expected = mlp(x)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)
