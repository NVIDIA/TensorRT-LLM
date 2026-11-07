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

import pytest
import torch

from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

pytestmark = pytest.mark.cpu_only

# Two llm-compressor FP8 per-channel-per-token checkpoint variants are tested:
#
# - rowwise (FP8-dynamic):
#     https://huggingface.co/RedHatAI/Qwen3.5-4B-FP8-dynamic
#     FP8 quantization applied to most linear layers; linear-attention in_proj_b
#     and in_proj_a are excluded and remain BF16.
#
# - rowwise-full (FP8-dynamic-full):
#     A more aggressive variant where all linear layers (except lm_head) are
#     FP8 quantized, including in_proj_b and in_proj_a.
#
# Both use per-channel [out, 1] weight scales (FP8_PER_CHANNEL_PER_TOKEN).
# FP8_BLOCK_SCALES (ModelOpt) and MIXED_PRECISION/NVFP4 are covered by e2e tests.

# Small model dimensions matching Qwen3.5 linear-attention config.
_HIDDEN = 8
_LINEAR_NUM_KEY_HEADS = 2
_LINEAR_KEY_HEAD_DIM = 4
_LINEAR_NUM_VALUE_HEADS = 4
_LINEAR_VALUE_HEAD_DIM = 4


def _make_mapper(
    quant_algo: QuantAlgo, exclude_modules: list[str] | None = None
) -> Qwen3_5MoeHfWeightMapper:
    mapper = Qwen3_5MoeHfWeightMapper()
    mapper._config = SimpleNamespace(
        quant_config=QuantConfig(quant_algo=quant_algo, exclude_modules=exclude_modules),
        pretrained_config=SimpleNamespace(
            linear_num_key_heads=_LINEAR_NUM_KEY_HEADS,
            linear_key_head_dim=_LINEAR_KEY_HEAD_DIM,
            linear_num_value_heads=_LINEAR_NUM_VALUE_HEADS,
            linear_value_head_dim=_LINEAR_VALUE_HEAD_DIM,
            torch_dtype=torch.bfloat16,
            num_experts=0,
            num_hidden_layers=1,
        ),
        mapping=SimpleNamespace(tp_size=1, tp_rank=0, enable_attention_dp=False),
        quant_config_dict=None,
    )
    return mapper


def _fp8(rows: int, cols: int = _HIDDEN) -> torch.Tensor:
    return torch.zeros(rows, cols, dtype=torch.float8_e4m3fn)


def _scale(rows: int) -> torch.Tensor:
    return torch.ones(rows, 1, dtype=torch.bfloat16)


def _bf16(rows: int, cols: int = _HIDDEN) -> torch.Tensor:
    return torch.zeros(rows, cols, dtype=torch.bfloat16)


_Q_ROWS = _LINEAR_NUM_KEY_HEADS * _LINEAR_KEY_HEAD_DIM  # 8
_V_ROWS = _LINEAR_NUM_VALUE_HEADS * _LINEAR_VALUE_HEAD_DIM  # 16
_BA_ROWS = _LINEAR_NUM_VALUE_HEADS  # 4
_PACKED_QKVZ_ROWS = _Q_ROWS + _Q_ROWS + _V_ROWS + _V_ROWS  # 48 (q+k+v+z)
_PACKED_BA_ROWS = _BA_ROWS + _BA_ROWS  # 8 (b+a)

_ATTN_PREFIX = "model.layers.0.linear_attn"
_MLP_PREFIX = "model.layers.0.mlp"


def _make_weights(full: bool = False) -> dict[str, torch.Tensor]:
    weights = {}
    for name, rows in [
        ("in_proj_q", _Q_ROWS),
        ("in_proj_k", _Q_ROWS),
        ("in_proj_v", _V_ROWS),
        ("in_proj_z", _V_ROWS),
    ]:
        weights[f"{_ATTN_PREFIX}.{name}.weight"] = _fp8(rows)
        weights[f"{_ATTN_PREFIX}.{name}.weight_scale"] = _scale(rows)
    for name in ("gate_proj", "up_proj", "down_proj"):
        weights[f"{_MLP_PREFIX}.{name}.weight"] = _fp8(_HIDDEN)
        weights[f"{_MLP_PREFIX}.{name}.weight_scale"] = _scale(_HIDDEN)

    # rowwise: b/a remain BF16; rowwise-full: b/a are also FP8 with per-channel scales.
    for name in ("in_proj_b", "in_proj_a"):
        if full:
            weights[f"{_ATTN_PREFIX}.{name}.weight"] = _fp8(_BA_ROWS)
            weights[f"{_ATTN_PREFIX}.{name}.weight_scale"] = _scale(_BA_ROWS)
        else:
            weights[f"{_ATTN_PREFIX}.{name}.weight"] = _bf16(_BA_ROWS)
    return weights


def _assert_common(out: dict[str, torch.Tensor]) -> None:
    # qkvz packing preserves FP8 dtype and [out, 1] scale shape.
    assert out[f"{_ATTN_PREFIX}.in_proj_qkvz.weight"].dtype == torch.float8_e4m3fn
    assert out[f"{_ATTN_PREFIX}.in_proj_qkvz.weight_scale"].shape == (_PACKED_QKVZ_ROWS, 1)

    # MLP keys are remapped to mlp.mlp.* and FP8 dtype + [out, 1] scale shape are preserved.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        assert out[f"{_MLP_PREFIX}.mlp.{proj}.weight"].dtype == torch.float8_e4m3fn
        assert out[f"{_MLP_PREFIX}.mlp.{proj}.weight_scale"].shape == (_HIDDEN, 1)


def test_fp8_rowwise() -> None:
    # RedHatAI/Qwen3.5-4B-FP8-dynamic: q/k/v/z are FP8 with per-channel scales; b/a remain BF16.
    mapper = _make_mapper(QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN)
    out = mapper.preprocess_weights(_make_weights(full=False))

    _assert_common(out)
    assert out[f"{_ATTN_PREFIX}.in_proj_ba.weight"].dtype == torch.bfloat16
    assert f"{_ATTN_PREFIX}.in_proj_ba.weight_scale" not in out


def test_fp8_rowwise_full() -> None:
    # FP8-dynamic-full: all in_proj projections including b/a are FP8 with per-channel scales.
    mapper = _make_mapper(QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN)
    out = mapper.preprocess_weights(_make_weights(full=True))

    _assert_common(out)
    assert out[f"{_ATTN_PREFIX}.in_proj_ba.weight"].dtype == torch.float8_e4m3fn
    assert out[f"{_ATTN_PREFIX}.in_proj_ba.weight_scale"].shape == (_PACKED_BA_ROWS, 1)


def _pack_int4_rows(weight: torch.Tensor) -> torch.Tensor:
    return ((weight[0::2] & 15) | ((weight[1::2] & 15) << 4)).to(torch.uint8)


def _unpack_int4_rows(weight: torch.Tensor) -> torch.Tensor:
    packed = weight.view(torch.uint8)
    values = torch.stack((packed & 15, packed >> 4), dim=1).flatten(0, 1).to(torch.int8)
    return torch.where(values >= 8, values - 16, values)


def _make_int4_weights(
    split_qkv: bool = False, quantize_ba: bool = True, head_dim: int = 4
) -> tuple[Qwen3_5MoeHfWeightMapper, dict, dict, dict]:
    mapper = _make_mapper(QuantAlgo.W4A16_AWQ)
    config = mapper.config.pretrained_config
    config.linear_num_key_heads = 16
    config.linear_num_value_heads = 48
    config.linear_key_head_dim = head_dim
    config.linear_value_head_dim = head_dim
    generator = torch.Generator().manual_seed(6562942)
    logical_weights, scales, weights = {}, {}, {}
    for name, rows in (
        ("q", 16 * head_dim),
        ("k", 16 * head_dim),
        ("v", 48 * head_dim),
        ("z", 48 * head_dim),
        ("b", 48),
        ("a", 48),
    ):
        logical_weights[name] = torch.randint(
            -8, 8, (rows, 256), generator=generator, dtype=torch.int8
        )
        scales[name] = torch.rand(rows, 2, generator=generator) + 0.1
        if name in ("b", "a") and not quantize_ba:
            logical_weights[name] = logical_weights[name].to(torch.bfloat16)
            weights[f"{_ATTN_PREFIX}.in_proj_{name}.weight"] = logical_weights[name]
        else:
            weights[f"{_ATTN_PREFIX}.in_proj_{name}.weight"] = _pack_int4_rows(
                logical_weights[name]
            )
            weights[f"{_ATTN_PREFIX}.in_proj_{name}.weight_scale"] = scales[name]
    if not split_qkv:
        for suffix in ("weight", "weight_scale"):
            weights[f"{_ATTN_PREFIX}.in_proj_qkv.{suffix}"] = torch.cat(
                [weights.pop(f"{_ATTN_PREFIX}.in_proj_{name}.{suffix}") for name in ("q", "k", "v")]
            )
    return mapper, weights, logical_weights, scales


@pytest.mark.parametrize(
    "tp_size,split_qkv,quantize_ba,storage_dtype,head_dim,has_pre_quant_scale",
    [
        # Cover both checkpoint layouts and gate formats in the TP1 and TP2 paths.
        (1, False, False, torch.uint8, 4, False),
        pytest.param(1, False, True, torch.uint8, 128, False, id="qwen35-27b"),
        (1, True, False, torch.uint8, 4, False),
        (1, True, True, torch.uint8, 4, False),
        (2, False, False, torch.uint8, 4, False),
        (2, False, True, torch.uint8, 4, False),
        (2, True, False, torch.uint8, 4, False),
        (2, True, True, torch.uint8, 4, False),
        # Shared activation scales in both checkpoint layouts.
        (2, False, True, torch.uint8, 4, True),
        (2, True, True, torch.uint8, 4, True),
        # Signed storage in both layouts; TP8 gives three packed gate rows per rank.
        (2, True, True, torch.int8, 4, False),
        (8, False, True, torch.int8, 4, False),
    ],
)
def test_int4_projections_preserve_values_and_scales(
    tp_size: int,
    split_qkv: bool,
    quantize_ba: bool,
    storage_dtype: torch.dtype,
    head_dim: int,
    has_pre_quant_scale: bool,
) -> None:
    mapper, weights, logical_weights, scales = _make_int4_weights(split_qkv, quantize_ba, head_dim)
    if head_dim == 128:
        assert weights[f"{_ATTN_PREFIX}.in_proj_qkv.weight"].shape[0] == 5120
    weights = {
        name: value.view(storage_dtype) if value.dtype == torch.uint8 else value
        for name, value in weights.items()
    }
    mapper.config.mapping.tp_size = tp_size
    input_scales = {"qkvz": torch.linspace(0.5, 1.5, 256), "ba": torch.linspace(1.0, 2.0, 256)}
    if has_pre_quant_scale:
        mapper.config.pretrained_config.hidden_size = 256
        for name in list(weights):
            if name.endswith(".weight"):
                projection = "ba" if ".in_proj_b." in name or ".in_proj_a." in name else "qkvz"
                weights[name.removesuffix("weight") + "pre_quant_scale"] = input_scales[
                    projection
                ].clone()
    out = mapper.preprocess_weights(weights)
    activation = torch.randn(3, 256, generator=torch.Generator().manual_seed(17))

    for projection, components in (("qkvz", ("q", "k", "v", "z")), ("ba", ("b", "a"))):
        quantized = projection == "qkvz" or quantize_ba
        scale_key = f"{_ATTN_PREFIX}.in_proj_{projection}.pre_quant_scale"
        if has_pre_quant_scale:
            torch.testing.assert_close(out[scale_key], input_scales[projection])
            fused_input = activation * out[scale_key]
            reference_input = activation * input_scales[projection]
        else:
            assert scale_key not in out
            fused_input = reference_input = activation
        packed_weight = out[f"{_ATTN_PREFIX}.in_proj_{projection}.weight"]
        if quantized:
            assert packed_weight.dtype == storage_dtype
            assert packed_weight.shape[0] * 2 == sum(
                logical_weights[name].shape[0] for name in components
            )
            fused_weight = _unpack_int4_rows(packed_weight)
            fused_scale = out[f"{_ATTN_PREFIX}.in_proj_{projection}.weight_scale"]
            assert fused_scale.shape == (fused_weight.shape[0], 2)
            fused_dequantized = fused_weight.float() * fused_scale.repeat_interleave(128, dim=1)
        else:
            assert packed_weight.dtype == torch.bfloat16
            fused_weight = packed_weight
            fused_dequantized = fused_weight.float()

        # Each rank's fused projection must compute the same outputs as the
        # individual checkpoint projections, with every scale attached to its row.
        for rank, shard in enumerate(fused_dequantized.chunk(tp_size, dim=0)):
            references = []
            row_offset = rank * (fused_weight.shape[0] // tp_size)
            for name in components:
                original = logical_weights[name].chunk(tp_size, dim=0)[rank]
                torch.testing.assert_close(
                    fused_weight[row_offset : row_offset + original.shape[0]], original
                )
                row_offset += original.shape[0]
                dequantized = logical_weights[name].float()
                if quantized:
                    dequantized = dequantized * scales[name].repeat_interleave(128, dim=1)
                references.append((reference_input @ dequantized.T).chunk(tp_size, dim=1)[rank])
            torch.testing.assert_close(fused_input @ shard.T, torch.cat(references, dim=1))


@pytest.mark.parametrize("projection", ["z", "a"])
@pytest.mark.parametrize("invalid", ["different", "missing", "shape"])
def test_int4_rejects_incompatible_pre_quant_scales(projection: str, invalid: str) -> None:
    """A fused Linear cannot apply different input scales to its components."""
    mapper, weights, _, _ = _make_int4_weights()
    mapper.config.pretrained_config.hidden_size = 256
    for name in ("qkv", "z", "b", "a"):
        weights[f"{_ATTN_PREFIX}.in_proj_{name}.pre_quant_scale"] = torch.ones(256)
    key = f"{_ATTN_PREFIX}.in_proj_{projection}.pre_quant_scale"
    if invalid == "missing":
        del weights[key]
    elif invalid == "shape":
        weights[key] = torch.ones(1, 256)
    else:
        weights[key][0] = 2
    with pytest.raises(ValueError, match="pre_quant_scale"):
        mapper.preprocess_weights(weights)


def test_int4_rejects_non_byte_aligned_tp_shards() -> None:
    mapper, weights, _, _ = _make_int4_weights()
    mapper.config.mapping.tp_size = 16
    with pytest.raises(ValueError, match="projection rows 24 are not divisible by tp_size=16"):
        mapper.preprocess_weights(weights)


@pytest.mark.parametrize("projection", ["z", "a"])
@pytest.mark.parametrize("other_dtype", [torch.int8, torch.bfloat16])
def test_int4_rejects_mixed_storage_dtypes(projection: str, other_dtype: torch.dtype) -> None:
    mapper, weights, _, _ = _make_int4_weights()
    name = f"{_ATTN_PREFIX}.in_proj_{projection}.weight"
    weights[name] = weights[name].to(other_dtype)
    message = "same storage dtype" if other_dtype == torch.int8 else "packed INT4 and unpacked"
    with pytest.raises(ValueError, match=message):
        mapper.preprocess_weights(weights)


@pytest.mark.parametrize("projection", ["q", "v", "ba"])
def test_int4_rejects_odd_logical_dimensions(projection: str) -> None:
    mapper, weights, _, _ = _make_int4_weights()
    config = mapper.config.pretrained_config
    config.linear_num_key_heads = 1
    if projection == "ba":
        config.linear_num_value_heads = 47
        weights = {
            key: value
            for key, value in weights.items()
            if ".in_proj_b." in key or ".in_proj_a." in key
        }
    elif projection == "q":
        config.linear_key_head_dim = 3
    else:
        config.linear_num_value_heads = 1
        config.linear_value_head_dim = 3
    with pytest.raises(ValueError, match="projection dimensions must be even"):
        mapper.preprocess_weights(weights)


def test_int4_excluded_qkvz_keeps_bf16_layout() -> None:
    mapper, weights, logical_weights, _ = _make_int4_weights(split_qkv=True)
    mapper.config.mapping.tp_size = 2
    for name in ("q", "k", "v", "z"):
        weights[f"{_ATTN_PREFIX}.in_proj_{name}.weight"] = logical_weights[name].to(torch.bfloat16)
        del weights[f"{_ATTN_PREFIX}.in_proj_{name}.weight_scale"]
    out = mapper.preprocess_weights(weights)
    fused = out[f"{_ATTN_PREFIX}.in_proj_qkvz.weight"]
    assert fused.dtype == torch.bfloat16
    assert f"{_ATTN_PREFIX}.in_proj_qkvz.weight_scale" not in out
    for rank, shard in enumerate(fused.chunk(2)):
        torch.testing.assert_close(
            shard,
            torch.cat([logical_weights[name].chunk(2)[rank] for name in ("q", "k", "v", "z")]).to(
                torch.bfloat16
            ),
        )


def test_int4_attention_dp_uses_unsplit_projections() -> None:
    mapper, weights, _, _ = _make_int4_weights()
    reference = mapper.preprocess_weights(weights)
    mapper.config.mapping.tp_size = 16
    mapper.config.mapping.enable_attention_dp = True
    out = mapper.preprocess_weights(weights)
    for name in reference:
        torch.testing.assert_close(out[name], reference[name])


def test_int4_partial_loading_and_mtp_names() -> None:
    mapper, weights, _, _ = _make_int4_weights()
    weights = {
        name.replace("model.layers.0", "mtp.layers.0"): value for name, value in weights.items()
    }
    reference = mapper.preprocess_weights(weights)
    assert "model.layers.1.linear_attn.in_proj_qkvz.weight" in reference
    out = {}
    for name, value in weights.items():
        out.update(mapper.preprocess_weights({name: value}, allow_partial_loading=True))
    mapper.finalize_update_weights()
    assert out.keys() == reference.keys()
    for name in reference:
        torch.testing.assert_close(out[name], reference[name])


def test_modelopt_fp8_per_tensor_linear_attention() -> None:
    # ModelOpt stores QKV and Z as separate per-tensor FP8 projections with independent scales.
    checkpoint_prefix = "model.language_model.layers.0.linear_attn"
    weights = {}
    for name, rows, weight_scale, input_scale in [
        ("in_proj_qkv", _Q_ROWS * 2 + _V_ROWS, 2.0, 3.0),
        ("in_proj_z", _V_ROWS, 4.0, 5.0),
    ]:
        weights[f"{checkpoint_prefix}.{name}.weight"] = torch.ones(
            rows, _HIDDEN, dtype=torch.float8_e4m3fn
        )
        weights[f"{checkpoint_prefix}.{name}.weight_scale"] = torch.tensor(weight_scale)
        weights[f"{checkpoint_prefix}.{name}.input_scale"] = torch.tensor(input_scale)
    for name in ("in_proj_b", "in_proj_a"):
        weights[f"{checkpoint_prefix}.{name}.weight"] = _bf16(_BA_ROWS)

    # Global FP8 maps both checkpoint projections into one fused FP8 module, so the mapper must
    # requantize them onto a shared scale before packing the weights.
    mapper = _make_mapper(QuantAlgo.FP8)
    out = mapper.preprocess_weights(weights)

    assert out[f"{_ATTN_PREFIX}.in_proj_qkvz.weight"].shape == (_PACKED_QKVZ_ROWS, _HIDDEN)
    assert out[f"{_ATTN_PREFIX}.in_proj_qkvz.weight"].dtype == torch.float8_e4m3fn
    packed_weight = out[f"{_ATTN_PREFIX}.in_proj_qkvz.weight"]
    qkv_rows = _Q_ROWS * 2 + _V_ROWS
    # QKV moves from scale 2 to the fused scale 4, halving its stored FP8 values. Z already uses
    # scale 4, so its packed values remain unchanged.
    torch.testing.assert_close(
        packed_weight[:qkv_rows].float(), torch.full((qkv_rows, _HIDDEN), 0.5)
    )
    torch.testing.assert_close(packed_weight[qkv_rows:].float(), torch.ones((_V_ROWS, _HIDDEN)))
    torch.testing.assert_close(out[f"{_ATTN_PREFIX}.in_proj_qkvz.weight_scale"], torch.tensor(4.0))
    torch.testing.assert_close(out[f"{_ATTN_PREFIX}.in_proj_qkvz.input_scale"], torch.tensor(5.0))
    assert out[f"{_ATTN_PREFIX}.in_proj_ba.weight"].dtype == torch.bfloat16


def test_modelopt_fp8_excluded_linear_attention_falls_back_to_bf16() -> None:
    # Small but valid Qwen3.5 linear-attention dimensions: q/k each have 2 heads * 4 dims, while
    # v/z each have 4 heads * 4 dims. These sizes exercise the qkv+z packing path without
    # constructing a full model.
    checkpoint_prefix = "model.language_model.layers.0.linear_attn"
    weights = {}
    for name, rows, weight_scale, input_scale in [
        ("in_proj_qkv", _Q_ROWS * 2 + _V_ROWS, 2.0, 3.0),
        ("in_proj_z", _V_ROWS, 4.0, 5.0),
    ]:
        weights[f"{checkpoint_prefix}.{name}.weight"] = torch.ones(
            rows, _HIDDEN, dtype=torch.float8_e4m3fn
        )
        weights[f"{checkpoint_prefix}.{name}.weight_scale"] = torch.tensor(weight_scale)
        weights[f"{checkpoint_prefix}.{name}.input_scale"] = torch.tensor(input_scale)

    # Use the real `QuantConfig` exclusion check instead of stubbing it: the global FP8 must respect
    # an excluded fused in_proj_qkvz module and therefore fall back to bf16.
    mapper = _make_mapper(QuantAlgo.FP8, exclude_modules=[f"{_ATTN_PREFIX}.in_proj_qkvz"])
    # `preprocess_weights` only needs the mapper config for this path; binding a full model would
    # add unrelated construction cost and GPU-facing setup.
    out = mapper.preprocess_weights(weights)

    assert out[f"{_ATTN_PREFIX}.in_proj_qkvz.weight"].shape == (
        _PACKED_QKVZ_ROWS,
        _HIDDEN,
    )
    assert out[f"{_ATTN_PREFIX}.in_proj_qkvz.weight"].dtype == torch.bfloat16
    packed_weight = out[f"{_ATTN_PREFIX}.in_proj_qkvz.weight"]
    qkv_rows = _Q_ROWS * 2 + _V_ROWS
    torch.testing.assert_close(
        packed_weight[:qkv_rows].float(),
        torch.full((qkv_rows, _HIDDEN), 2.0),
    )
    torch.testing.assert_close(
        packed_weight[qkv_rows:].float(),
        torch.full((_V_ROWS, _HIDDEN), 4.0),
    )
    # The scalar per-projection scales are consumed by the bf16 fallback before
    # split projections are packed; no fused FP8 scales should be synthesized.
    for name in (
        f"{_ATTN_PREFIX}.in_proj_qkvz.weight_scale",
        f"{_ATTN_PREFIX}.in_proj_qkvz.input_scale",
        f"{_ATTN_PREFIX}.in_proj_qkv.weight_scale",
        f"{_ATTN_PREFIX}.in_proj_qkv.input_scale",
        f"{_ATTN_PREFIX}.in_proj_z.weight_scale",
        f"{_ATTN_PREFIX}.in_proj_z.input_scale",
    ):
        assert name not in out
