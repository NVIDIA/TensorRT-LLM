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


def _make_mapper(quant_algo: QuantAlgo) -> Qwen3_5MoeHfWeightMapper:
    mapper = Qwen3_5MoeHfWeightMapper()
    mapper._config = SimpleNamespace(
        quant_config=SimpleNamespace(quant_algo=quant_algo),
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


# --- mixed FP8 / NVFP4 compressed-tensors ("mixed-precision") ----------------
#
# unsloth/Qwen3.8-27B-NVFP4: one config group is rowwise FP8 (attention,
# linear-attention in_proj/out_proj, lm_head, the last MLP blocks), the other is
# NVFP4 for the remaining MLP blocks. The NVFP4 modules are the only ones
# carrying weight_packed / weight_global_scale / input_global_scale, and *both*
# groups carry a weight_scale -- per-channel [out, 1] for FP8, per-group
# [out, in/16] for NVFP4.

_NVFP4_GROUP_SIZE = 16
_NVFP4_HIDDEN = 32  # divisible by the group size


def _nvfp4_module(weights: dict, prefix: str) -> None:
    weights[f"{prefix}.weight_packed"] = torch.zeros(
        _NVFP4_HIDDEN, _NVFP4_HIDDEN // 2, dtype=torch.uint8
    )
    weights[f"{prefix}.weight_scale"] = torch.zeros(
        _NVFP4_HIDDEN, _NVFP4_HIDDEN // _NVFP4_GROUP_SIZE, dtype=torch.float8_e4m3fn
    )
    weights[f"{prefix}.weight_global_scale"] = torch.tensor([4.0])
    weights[f"{prefix}.input_global_scale"] = torch.tensor([8.0])


def _make_mixed_weights() -> dict[str, torch.Tensor]:
    """Rowwise-FP8 linear attention + NVFP4 dense MLP, in checkpoint naming."""
    weights = {}
    for name, rows in [
        ("in_proj_qkv", _Q_ROWS * 2 + _V_ROWS),
        ("in_proj_z", _V_ROWS),
    ]:
        weights[f"{_ATTN_PREFIX}.{name}.weight"] = _fp8(rows)
        # Per-channel [out, 1]: broadcasts against the [out, in] weight.
        weights[f"{_ATTN_PREFIX}.{name}.weight_scale"] = _scale(rows) * 2.0
    for name in ("in_proj_b", "in_proj_a"):
        weights[f"{_ATTN_PREFIX}.{name}.weight"] = _bf16(_BA_ROWS)
    weights[f"{_ATTN_PREFIX}.out_proj.weight"] = _fp8(_HIDDEN)
    weights[f"{_ATTN_PREFIX}.out_proj.weight_scale"] = _scale(_HIDDEN)
    for name in ("gate_proj", "up_proj", "down_proj"):
        _nvfp4_module(weights, f"{_MLP_PREFIX}.{name}")
    return weights


def test_mixed_precision_nvfp4_dense_mlp_and_rowwise_gdn() -> None:
    mapper = _make_mapper(QuantAlgo.MIXED_PRECISION)
    out = mapper.preprocess_weights(_make_mixed_weights())

    # NVFP4 MLP tensors are renamed to the ModelOpt names the NVFP4 Linear
    # loader reads, and re-pathed onto the _DenseMlpAdapter's doubled .mlp.mlp.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        module = f"{_MLP_PREFIX}.mlp.{proj}"
        assert out[f"{module}.weight"].dtype == torch.uint8
        assert out[f"{module}.weight_scale"].dtype == torch.float8_e4m3fn
        # global scales are reciprocals of the compressed-tensors form
        assert out[f"{module}.weight_scale_2"].item() == pytest.approx(0.25)
        assert out[f"{module}.input_scale"].item() == pytest.approx(0.125)
        assert f"{module}.weight_packed" not in out
        assert f"{module}.weight_global_scale" not in out
        assert f"{module}.input_global_scale" not in out

    # No fused in_proj_qkvz FP8 entry exists for rowwise FP8, so the split
    # projections are dequantized to bf16 before packing -- exactly, because the
    # [out, 1] scale broadcasts over the [out, in] weight -- and their scales
    # are dropped so they never reach the packer.
    assert out[f"{_ATTN_PREFIX}.in_proj_qkvz.weight"].dtype == torch.bfloat16
    assert f"{_ATTN_PREFIX}.in_proj_qkvz.weight_scale" not in out
    assert out[f"{_ATTN_PREFIX}.in_proj_ba.weight"].dtype == torch.bfloat16

    # out_proj keeps its name and its rowwise FP8 tensors: it matches its own
    # per-layer quant entry and loads through FP8RowwiseLinearMethod.
    assert out[f"{_ATTN_PREFIX}.out_proj.weight"].dtype == torch.float8_e4m3fn
    assert out[f"{_ATTN_PREFIX}.out_proj.weight_scale"].shape == (_HIDDEN, 1)


def test_mixed_precision_rowwise_gdn_dequant_is_exact() -> None:
    mapper = _make_mapper(QuantAlgo.MIXED_PRECISION)
    rows = _V_ROWS
    weight = torch.full((rows, _HIDDEN), 2.0, dtype=torch.float8_e4m3fn)
    scale = torch.arange(1, rows + 1, dtype=torch.bfloat16).reshape(rows, 1)
    weights = {
        f"{_ATTN_PREFIX}.in_proj_z.weight": weight,
        f"{_ATTN_PREFIX}.in_proj_z.weight_scale": scale,
    }

    out = mapper._dequantize_linear_attn_fp8_per_tensor(weights)

    expected = weight.to(torch.float32) * scale.to(torch.float32)
    torch.testing.assert_close(
        out[f"{_ATTN_PREFIX}.in_proj_z.weight"].to(torch.float32),
        expected.to(torch.bfloat16).to(torch.float32),
    )
    assert f"{_ATTN_PREFIX}.in_proj_z.weight_scale" not in out
