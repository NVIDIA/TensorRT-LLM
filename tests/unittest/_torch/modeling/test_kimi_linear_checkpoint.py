# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint naming, weight sharding, and FP8 loading for the Kimi Linear model."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

pytest.importorskip("fla")

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_linear import (  # noqa: E402
    KimiLinearForCausalLM,
    KimiMLARuntime,
    _Fp8BlockScaleWeightReadLinear,
    _helix_cp_v_b_shard,
    _shard_head_major_param,
    resolve_attention_quant_config,
)
from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


def test_checkpoint_plan_preserves_external_attention_names():
    class _PlanHarness:
        checkpoint_name_plan = KimiLinearForCausalLM.checkpoint_name_plan
        _attention_fp8_linears = KimiLinearForCausalLM._attention_fp8_linears
        model = SimpleNamespace(layers=[])

        def _trunk_parameters(self):
            return {
                "model.layers.0.linear_attn.q_proj.weight": torch.empty(0),
                "model.layers.1.self_attn.mixer.q_a_proj.weight": torch.empty(0),
                "lm_head.weight": torch.empty(0),
            }

    name_map, expected_keys, expert_jobs = _PlanHarness().checkpoint_name_plan("language_model.")

    assert name_map == {
        "model.layers.0.linear_attn.q_proj.weight": (
            "language_model.model.layers.0.self_attn.q_proj.weight"
        ),
        "model.layers.1.self_attn.mixer.q_a_proj.weight": (
            "language_model.model.layers.1.self_attn.q_a_proj.weight"
        ),
        "lm_head.weight": "language_model.lm_head.weight",
    }
    assert expected_keys == set(name_map.values())
    assert expert_jobs == []


def _distinct(*shape: int) -> torch.Tensor:
    """A contiguous tensor with a distinct non-zero value at every position,
    so a wrong-rank slice is never equal to the right one (a no-op shard is
    also distinguishable from a correct slice)."""
    n = 1
    for d in shape:
        n *= d
    return torch.arange(1, n + 1, dtype=torch.float32).reshape(shape)


@pytest.mark.parametrize("kda_tp_size,kda_tp_rank", [(2, 0), (2, 1), (4, 3)])
def test_shard_kda_column_projection(kda_tp_size, kda_tp_rank):
    # Every head-major KDA projection except o_proj (q/k/v/g/f_b/b, conv,
    # dt_bias) is COLUMN-sharded on its output rows (dim 0) by kda_tp_size.
    local = 4
    src = _distinct(local * kda_tp_size, 8)
    param = torch.nn.Parameter(torch.empty(local, 8))
    out = _shard_head_major_param(
        "model.layers.0.linear_attn.q_proj.weight",
        src,
        param,
        kda_tp_size=kda_tp_size,
        kda_tp_rank=kda_tp_rank,
        model_tp_rank=0,
    )
    expected = src[kda_tp_rank * local : (kda_tp_rank + 1) * local]
    assert out.shape == param.shape
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("kda_tp_size,kda_tp_rank", [(2, 0), (2, 1), (4, 2)])
def test_shard_kda_o_proj_row(kda_tp_size, kda_tp_rank):
    # o_proj alone is ROW-sharded on its input columns (dim 1) by kda_tp_size.
    local = 4
    src = _distinct(6, local * kda_tp_size)
    param = torch.nn.Parameter(torch.empty(6, local))
    out = _shard_head_major_param(
        "model.layers.0.linear_attn.o_proj.weight",
        src,
        param,
        kda_tp_size=kda_tp_size,
        kda_tp_rank=kda_tp_rank,
        model_tp_rank=0,
    )
    expected = src[:, kda_tp_rank * local : (kda_tp_rank + 1) * local]
    assert out.shape == param.shape
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("scope", [".shared_experts.", ".mlp."])
@pytest.mark.parametrize("model_tp_rank", [0, 1, 3])
def test_shard_mlp_down_proj_row(scope, model_tp_rank):
    # down_proj is ROW-sharded on its input columns; tp comes from the shapes
    # and the shard index repeats modulo the parameter's shard count.
    local, tp = 4, 2
    src = _distinct(6, local * tp)
    param = torch.nn.Parameter(torch.empty(6, local))
    out = _shard_head_major_param(
        f"model.layers.3{scope}down_proj.weight",
        src,
        param,
        kda_tp_size=1,
        kda_tp_rank=0,
        model_tp_rank=model_tp_rank,
    )
    rank = model_tp_rank % tp
    expected = src[:, rank * local : (rank + 1) * local]
    assert out.shape == param.shape
    torch.testing.assert_close(out, expected)


def test_shard_passthrough_when_shape_matches():
    # Replicated KDA projections (f_a/g_a, o_norm) already match the param and
    # must be returned untouched.
    src = _distinct(4, 8)
    param = torch.nn.Parameter(torch.empty(4, 8))
    out = _shard_head_major_param(
        "model.layers.0.linear_attn.f_a_proj.weight",
        src,
        param,
        kda_tp_size=2,
        kda_tp_rank=1,
        model_tp_rank=0,
    )
    assert out is src


def test_shard_passthrough_for_mla_names():
    # MLA (.self_attn.) tensors are head-sharded by their own Linear modules, so
    # a shape mismatch here must pass through (not be treated as KDA/down_proj).
    src = _distinct(8, 4)
    param = torch.nn.Parameter(torch.empty(4, 4))
    out = _shard_head_major_param(
        "model.layers.1.self_attn.q_b_proj.weight",
        src,
        param,
        kda_tp_size=1,
        kda_tp_rank=0,
        model_tp_rank=0,
    )
    assert out is src


def test_shard_down_proj_non_divisible_raises():
    # The divisibility guard turns a misaligned checkpoint into a clear error
    # instead of a floor-tp silent misshard.
    src = _distinct(6, 6)
    param = torch.nn.Parameter(torch.empty(6, 4))  # 6 % 4 != 0
    with pytest.raises(AssertionError):
        _shard_head_major_param(
            "model.layers.3.mlp.down_proj.weight",
            src,
            param,
            kda_tp_size=1,
            kda_tp_rank=0,
            model_tp_rank=0,
        )


@pytest.mark.parametrize("cp_rank", [0, 1, 3])
def test_helix_cp_v_b_shard_slices_this_rank(cp_rank):
    # Helix cp_size=4: v_b_proj keeps only this rank's 1/cp head chunk, while
    # kv_b_proj / k_b_proj_trans (built from the un-sliced v_weight) keep every
    # tp-local head. This slice only fires when cp_size > 1, so no cp_size=1
    # smoke test exercises it.
    num_heads_tp, num_heads_tp_cp = 8, 2  # cp_size == 8 / 2 == 4
    v_weight = _distinct(num_heads_tp, 3, 5)  # [heads, v_head_dim, kv_lora_rank]
    out = _helix_cp_v_b_shard(v_weight, num_heads_tp_cp=num_heads_tp_cp, cp_rank=cp_rank)
    expected = v_weight[cp_rank * num_heads_tp_cp : (cp_rank + 1) * num_heads_tp_cp]
    assert out.shape[0] == num_heads_tp_cp
    torch.testing.assert_close(out, expected)


def test_helix_cp_v_b_shard_noop_without_cp():
    # cp_size == 1: num_heads_tp_cp equals the full tp-local head count, so the
    # tensor is returned untouched (no accidental slicing).
    v_weight = _distinct(8, 3, 5)
    out = _helix_cp_v_b_shard(v_weight, num_heads_tp_cp=8, cp_rank=0)
    assert out is v_weight


_requires_fp8_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Kimi checkpoint FP8 tests require SM100/SM103",
)

_KDA_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "g_proj",
    "o_proj",
    "f_a_proj",
    "f_b_proj",
    "b_proj",
)
_MLA_PROJECTIONS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "g_proj", "o_proj")


class _LocalAllReduce(nn.Identity):
    def uses_nccl_symmetric_memory_window(self):
        return False


def _model(
    *,
    kda: bool,
    tp_size: int = 1,
    tp_rank: int = 0,
    cp_size: int = 1,
    quantized=None,
    alias="language_model.model.",
    excluded=None,
    cute_mm=True,
    q_lora_rank=128,
):
    cfg = SimpleNamespace(
        hidden_size=256,
        rms_norm_eps=1e-6,
        num_attention_heads=8,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=256,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_output_gate=True,
        max_position_embeddings=128,
        num_hidden_layers=1,
        linear_attn_config={
            "num_heads": 8,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    names = _KDA_PROJECTIONS if kda else _MLA_PROJECTIONS
    if quantized is None:
        quantized = names
    config = ModelConfig(
        pretrained_config=cfg,
        mapping=Mapping(world_size=tp_size * cp_size, tp_size=tp_size * cp_size, rank=tp_rank),
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION, exclude_modules=excluded),
        quant_config_dict={
            f"{alias}layers.0.self_attn.{name}": QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
            for name in quantized
        },
        skip_create_weights_in_init=not kda,
        use_cute_dsl_blockscaling_mm=cute_mm,
        use_cute_dsl_blockscaling_bmm=True,
    )
    layer = nn.Module()
    layer.is_kda, layer.is_moe = kda, False
    # These tests inspect rank-local loading and GEMMs on one GPU. No
    # collective is executed, so avoid allocating multi-GPU IPC workspaces.
    with (
        patch(
            "tensorrt_llm._torch.modules.kimi_kda.kimi_kda_mixer.AllReduce",
            return_value=_LocalAllReduce(),
        ),
        patch(
            "tensorrt_llm._torch.models.modeling_kimi_linear.AllReduce",
            return_value=_LocalAllReduce(),
        ),
        patch("tensorrt_llm._torch.distributed.AllReduce", return_value=_LocalAllReduce()),
    ):
        if kda:
            attention_config = copy.copy(config)
            attention_config.quant_config_dict = {
                name: resolve_attention_quant_config(config, 0, name) for name in _KDA_PROJECTIONS
            }
            layer.linear_attn = KimiKDALinearAttention(
                cfg,
                0,
                mapping=config.mapping,
                model_config=attention_config,
            )
        else:
            mapping_with_cp = (
                Mapping(
                    world_size=tp_size * cp_size,
                    tp_size=tp_size,
                    cp_size=cp_size,
                    rank=tp_rank,
                    cp_config={"cp_type": "HELIX"},
                )
                if cp_size > 1
                else None
            )
            layer.self_attn = KimiMLARuntime(cfg, 0, config, {}, mapping_with_cp=mapping_with_cp)
    model = KimiLinearForCausalLM.__new__(KimiLinearForCausalLM)
    nn.Module.__init__(model)
    model._fp8_weight_read_moe_mlp = False
    model.model_config = config
    model._repurposed_tp_mapping = config.mapping if cp_size > 1 else None
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([layer])
    if not kda:
        model.__post_init__()
    return model.cuda()


def _checkpoint(model):
    names, _, _ = model.checkpoint_name_plan("")
    params = dict(model.named_parameters())
    result = {key: torch.randn_like(params[name]) * 0.01 for name, key in names.items()}
    for name, module, _ in model._attention_fp8_linears():
        key = name.replace(".linear_attn.", ".self_attn.").replace(
            ".self_attn.mixer.", ".self_attn."
        )
        # Independently chosen codes/scales, never produced by a BF16 quantizer.
        result[key + ".weight"] = torch.randint(
            -32, 33, (module.out_features, module.in_features), device="cuda"
        ).to(torch.float8_e4m3fn)
        scale = torch.rand(module.weight_scale.shape, device="cuda") * 0.002 + 0.001
        result[key + ".weight_scale"] = scale[:, None, :, None]
    layer = model.model.layers[0]
    if not layer.is_kda and layer.self_attn.mixer.fuse_qkv_a_proj:
        attention = layer.self_attn.mixer
        key = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
        q_key = "model.layers.0.self_attn.q_a_proj.weight"
        result[q_key] = result[key][: attention.q_lora_rank].clone()
        result[key] = result[key][attention.q_lora_rank :].clone()
        if key + "_scale" in result:
            split = attention.q_lora_rank // 128
            result[q_key + "_scale"] = result[key + "_scale"][:split].clone()
            result[key + "_scale"] = result[key + "_scale"][split:].clone()
    return result


def _projection(attention, name):
    if getattr(attention, "fuse_qkv_a_proj", False):
        if name == "q_a_proj":
            return attention.kv_a_proj_with_mqa, slice(0, attention.q_lora_rank)
        if name == "kv_a_proj_with_mqa":
            return attention.kv_a_proj_with_mqa, slice(attention.q_lora_rank, None)
    return getattr(attention, name), slice(None)


@_requires_fp8_gpu
@pytest.mark.parametrize("quantized", [(), _KDA_PROJECTIONS])
def test_kda_construction_follows_each_projection_config(quantized):
    attention = _model(kda=True, quantized=quantized).model.layers[0].linear_attn
    for name in _KDA_PROJECTIONS:
        dtype = (
            torch.float8_e4m3fn
            if name in quantized and name in _KDA_PROJECTIONS[:5]
            else torch.bfloat16
        )
        assert getattr(attention, name).weight.dtype == dtype


@_requires_fp8_gpu
@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 1), (4, 3)])
def test_kda_load_preserves_checkpoint_shards_and_fused_outputs(tp_size, tp_rank):
    checkpoint = _checkpoint(_model(kda=True))
    model = _model(kda=True, tp_size=tp_size, tp_rank=tp_rank)
    model.load_weights(checkpoint)
    attention = model.model.layers[0].linear_attn
    for name in _KDA_PROJECTIONS[:5]:
        module = getattr(attention, name)
        assert isinstance(module, _Fp8BlockScaleWeightReadLinear)
        expected = checkpoint[f"model.layers.0.self_attn.{name}.weight"]
        expected = expected.chunk(tp_size, dim=1 if name == "o_proj" else 0)[tp_rank]
        assert torch.equal(
            module.weight[: module.out_features].view(torch.uint8), expected.view(torch.uint8)
        )
        assert module.weight.dtype == torch.float8_e4m3fn
        scales = checkpoint[f"model.layers.0.self_attn.{name}.weight_scale"][:, 0, :, 0]
        axis = 1 if name == "o_proj" else 0
        width = module.in_features if axis == 1 else module.out_features
        start, end = tp_rank * width, (tp_rank + 1) * width
        indices = [slice(None), slice(None)]
        indices[axis] = slice(start // 128, (end + 127) // 128)
        scales = scales[tuple(indices)]
        torch.testing.assert_close(module.weight_scale, scales, rtol=0, atol=0)
    hidden = torch.randn(7, 256, dtype=torch.bfloat16, device="cuda")
    expected = torch.cat(
        [getattr(attention, name)(hidden) for name in _KDA_PROJECTIONS[:4]], dim=-1
    )
    torch.testing.assert_close(attention.qkvg_proj(hidden), expected, rtol=0.02, atol=0.02)
    expected_bfa = torch.cat([attention.f_a_proj(hidden), attention.b_proj(hidden)], dim=-1)
    actual_bfa = torch.nn.functional.linear(hidden, attention._bfa_proj_weight)
    torch.testing.assert_close(
        actual_bfa[:, : expected_bfa.shape[1]], expected_bfa, rtol=0.02, atol=0.02
    )
    assert attention._bfa_proj_weight.dtype == torch.bfloat16
    for name in _KDA_PROJECTIONS[:4]:
        assert (
            getattr(attention, name).weight.untyped_storage().data_ptr()
            == attention.qkvg_proj.weight.untyped_storage().data_ptr()
        )


@_requires_fp8_gpu
@pytest.mark.parametrize("failure", ["missing_scale", "wrong_dtype", "wrong_scale_shape"])
def test_kda_rejects_checkpoint_inconsistent_with_quant_config(failure):
    model = _model(kda=True)
    checkpoint = _checkpoint(model)
    key = "model.layers.0.self_attn.q_proj.weight"
    if failure == "missing_scale":
        del checkpoint[key + "_scale"]
    elif failure == "wrong_dtype":
        checkpoint[key] = checkpoint[key].to(torch.bfloat16)
    else:
        checkpoint[key + "_scale"] = torch.ones(1, 1, device="cuda")
    with pytest.raises((KeyError, ValueError)):
        model.load_weights(checkpoint)


@_requires_fp8_gpu
@pytest.mark.parametrize("tp_size,tp_rank,cp_size", [(1, 0, 1), (2, 1, 1), (2, 3, 2)])
def test_mla_load_preserves_checkpoint_and_absorption_pairs(tp_size, tp_rank, cp_size):
    checkpoint = _checkpoint(_model(kda=False))
    model = _model(kda=False, tp_size=tp_size, tp_rank=tp_rank, cp_size=cp_size)
    model.load_weights(checkpoint)
    attention = model.model.layers[0].self_attn.mixer
    for name in _MLA_PROJECTIONS:
        module, rows = _projection(attention, name)
        assert isinstance(module, _Fp8BlockScaleWeightReadLinear)
        assert module.weight.dtype == torch.float8_e4m3fn
        if name != "kv_b_proj":
            expected = checkpoint[f"model.layers.0.self_attn.{name}.weight"]
            if name in ("q_b_proj", "g_proj", "o_proj"):
                expected = expected.chunk(module.tp_size, dim=1 if name == "o_proj" else 0)[
                    module.tp_rank
                ]
            assert torch.equal(module.weight[rows].view(torch.uint8), expected.view(torch.uint8))
    weight = checkpoint["model.layers.0.self_attn.kv_b_proj.weight"].reshape(8, 256, 256)
    weight = weight.chunk(tp_size, dim=0)[attention.mapping.tp_rank]
    key, value = weight.split(128, dim=1)
    value = value.chunk(cp_size, dim=0)[attention.mapping.cp_rank]
    assert torch.equal(
        attention.k_b_proj_trans.view(torch.uint8), key.transpose(1, 2).view(torch.uint8)
    )
    assert torch.equal(attention.v_b_proj.view(torch.uint8), value.view(torch.uint8))
    scale = checkpoint["model.layers.0.self_attn.kv_b_proj.weight_scale"].reshape(8, 2, 2)
    scale = scale.chunk(tp_size, dim=0)[attention.mapping.tp_rank]
    torch.testing.assert_close(attention.k_b_proj_trans_scale, scale[:, :1].transpose(1, 2))
    torch.testing.assert_close(
        attention.v_b_proj_scale, scale[:, 1:].chunk(cp_size, dim=0)[attention.mapping.cp_rank]
    )
    hidden = torch.randn(7, 256, dtype=torch.bfloat16, device="cuda")
    assert torch.isfinite(attention.kv_a_proj_with_mqa(hidden)).all()
    assert torch.isfinite(attention.kv_b_proj(hidden)).all()


@_requires_fp8_gpu
@pytest.mark.parametrize("kda", [True, False])
@pytest.mark.parametrize("alias", ["language_model.model.", "model.", ""])
def test_attention_config_aliases_and_exclusions(kda, alias):
    names = _KDA_PROJECTIONS if kda else _MLA_PROJECTIONS
    excluded = names[0]
    if kda:
        with pytest.raises(ValueError, match="all q/k/v/g/o"):
            _model(kda=True, alias=alias, excluded=[f"*.self_attn.{excluded}"])
        return
    model = _model(kda=kda, alias=alias, excluded=[f"*.self_attn.{excluded}"])
    layer = model.model.layers[0]
    attention = layer.linear_attn if kda else layer.self_attn.mixer
    for name in names:
        expected = torch.bfloat16 if name == excluded else torch.float8_e4m3fn
        assert getattr(attention, name).weight.dtype == expected


@_requires_fp8_gpu
@pytest.mark.parametrize("kda", [True, False])
@pytest.mark.parametrize("cute_mm", [False, True])
@torch.no_grad()
def test_checkpoint_projection_gemms_use_checkpoint_scales(kda, cute_mm):
    torch.manual_seed(0)
    model = _model(kda=kda, cute_mm=cute_mm)
    checkpoint = _checkpoint(model)
    model.load_weights(checkpoint)
    for module in model.modules():
        if isinstance(module, (Linear, _Fp8BlockScaleWeightReadLinear)):
            module.post_load_weights()
    layer = model.model.layers[0]
    attention = layer.linear_attn if kda else layer.self_attn.mixer
    for name in _KDA_PROJECTIONS if kda else _MLA_PROJECTIONS:
        # MLA A projections are covered by the datatype/fusion matrix below.
        if not kda and name in ("q_a_proj", "kv_a_proj_with_mqa"):
            continue
        module, rows = _projection(attention, name)
        key = f"model.layers.0.self_attn.{name}.weight"
        weight = checkpoint[key].float()
        if key + "_scale" in checkpoint:
            scale = checkpoint[key + "_scale"][:, 0, :, 0]
            weight = (
                weight
                * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[
                    : weight.shape[0], : weight.shape[1]
                ]
            )
        if name == "kv_b_proj":
            head_weights = weight.reshape(8, 256, 256)
            weight = torch.cat(
                (head_weights[:, :128].reshape(-1, 256), head_weights[:, 128:].reshape(-1, 256))
            )
        hidden = torch.randn(7, module.in_features, device="cuda", dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(hidden.float(), weight)
        actual = module(hidden)[:, rows]
        # Activation quantization and DeepGEMM's UE8M0 weight resmoothing
        # add FP8 rounding error. Compare the forward error relative to the
        # full signal; elementwise relative error is unstable near zero.
        relative_l2 = (actual.float() - expected).norm() / expected.norm()
        assert relative_l2 < 0.05, f"{name}: relative L2 error {relative_l2.item()}"


@_requires_fp8_gpu
@pytest.mark.parametrize(
    "quantized", [(), ("q_a_proj",), ("kv_a_proj_with_mqa",), ("q_a_proj", "kv_a_proj_with_mqa")]
)
@torch.no_grad()
def test_mla_a_fusion_follows_checkpoint_datatypes(quantized):
    model = _model(kda=False, quantized=quantized)
    attention = model.model.layers[0].self_attn.mixer
    assert attention.fuse_qkv_a_proj == (len(quantized) in (0, 2))
    assert hasattr(attention, "q_a_proj") == (len(quantized) == 1)
    checkpoint = _checkpoint(model)
    model.load_weights(checkpoint)
    hidden = torch.randn(7, 256, dtype=torch.bfloat16, device="cuda")
    references = []
    for name in ("q_a_proj", "kv_a_proj_with_mqa"):
        module, rows = _projection(attention, name)
        key = f"model.layers.0.self_attn.{name}.weight"
        expected_dtype = torch.float8_e4m3fn if name in quantized else torch.bfloat16
        assert module.weight.dtype == expected_dtype
        assert torch.equal(module.weight[rows].view(torch.uint8), checkpoint[key].view(torch.uint8))
        weight = checkpoint[key].float()
        if name in quantized:
            scales = checkpoint[key + "_scale"][:, 0, :, 0]
            scale_rows = slice(0, 1) if name == "q_a_proj" else slice(1, None)
            if not attention.fuse_qkv_a_proj:
                scale_rows = slice(None)
            torch.testing.assert_close(module.weight_scale[scale_rows], scales, rtol=0, atol=0)
            weight *= scales.repeat_interleave(128, 0).repeat_interleave(128, 1)[: weight.shape[0]]
        references.append((module, rows, torch.nn.functional.linear(hidden.float(), weight)))
    for module, rows, expected in references:
        actual = module(hidden)[:, rows].float()
        assert (actual - expected).norm() / expected.norm() < 0.05


@_requires_fp8_gpu
@pytest.mark.parametrize("quantized", [("q_proj",), _KDA_PROJECTIONS[1:5]])
def test_kda_rejects_partial_fp8_projection_group(quantized):
    with pytest.raises(ValueError, match="all q/k/v/g/o"):
        _model(kda=True, quantized=quantized)


@_requires_fp8_gpu
@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (4, 3)])
@torch.no_grad()
def test_kda_small_checkpoint_fp8_projections_load_as_bf16(tp_size, tp_rank):
    checkpoint = _checkpoint(_model(kda=True))
    for name in ("b_proj", "f_a_proj", "f_b_proj"):
        key = f"model.layers.0.self_attn.{name}.weight"
        checkpoint[key] = torch.randint(-32, 33, checkpoint[key].shape, device="cuda").to(
            torch.float8_e4m3fn
        )
        shape = tuple((dim + 127) // 128 for dim in checkpoint[key].shape)
        checkpoint[key + "_scale"] = (torch.rand(shape, device="cuda") * 0.002 + 0.001)[
            :, None, :, None
        ]
    model = _model(kda=True, tp_size=tp_size, tp_rank=tp_rank)
    model.load_weights(checkpoint)
    attention = model.model.layers[0].linear_attn
    for name in ("b_proj", "f_a_proj", "f_b_proj"):
        module = getattr(attention, name)
        key = f"model.layers.0.self_attn.{name}.weight"
        weight = checkpoint[key].float()
        scale = checkpoint[key + "_scale"][:, 0, :, 0]
        expected = (
            weight
            * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[
                : weight.shape[0], : weight.shape[1]
            ]
        ).bfloat16()
        if name != "f_a_proj":
            expected = expected.chunk(tp_size, dim=0)[tp_rank]
        assert isinstance(module, nn.Linear)
        assert module.weight.dtype == torch.bfloat16
        torch.testing.assert_close(module.weight, expected, rtol=0, atol=0)
