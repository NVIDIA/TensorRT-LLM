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
