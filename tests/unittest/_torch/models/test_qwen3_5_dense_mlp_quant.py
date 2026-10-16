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
"""Per-layer quant-config key translation for dense Qwen3.5/3.8 MLP blocks.

``_DenseMlpAdapter`` wraps ``GatedMLP`` as ``self.mlp``, so a dense MLP
projection lives at ``model.layers.N.mlp.mlp.*`` at runtime while the checkpoint
stores it at ``model.layers.N.mlp.*``.  The weight mapper moves every dense MLP
*tensor* onto the doubled path unconditionally, so any per-layer *quant* entry
left on the checkpoint path is dead: the module is built from the global
(unquantized) MIXED_PRECISION config and its quantized weights fail to load, or
load with their scales silently dropped.

CPU only: no weights are loaded and no module is constructed.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.models.modeling_qwen3_5 import _normalize_qwen35_quant_config_dict
from tensorrt_llm._torch.modules.linear import (
    FP8RowwiseLinearMethod,
    NVFP4LinearMethod,
    W4A16NVFP4LinearMethod,
    get_quant_method,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

NUM_HIDDEN_LAYERS = 64

BLACKWELL_SMS = [100, 103]
# SM121 (DGX Spark) is where the Qwen3.8-27B-NVFP4 recipe is served; SM120/89/90
# stand in for the other non-promoting architectures.
OTHER_SMS = [89, 90, 120, 121]
ALL_SMS = BLACKWELL_SMS + OTHER_SMS

# What the compressed-tensors Qwen3.8-27B recipe resolves to per dense MLP
# block: NVFP4 (W4A4) for the early blocks, rowwise FP8 for the tail.
CHECKPOINT_ALGOS = [QuantAlgo.NVFP4, QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN, QuantAlgo.FP8]

DENSE_PROJECTIONS = ["gate_proj", "up_proj", "down_proj"]


def _model_config(quant_config_dict) -> SimpleNamespace:
    return SimpleNamespace(
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.MIXED_PRECISION,
            kv_cache_quant_algo=QuantAlgo.FP8,
            exclude_modules=[],
        ),
        quant_config_dict=quant_config_dict,
        pretrained_config=SimpleNamespace(num_hidden_layers=NUM_HIDDEN_LAYERS),
        mapping=SimpleNamespace(tp_size=1, enable_attention_dp=False),
    )


def _normalize(quant_config_dict, sm_version: int) -> dict:
    model_config = _model_config(quant_config_dict)
    with patch(
        "tensorrt_llm._torch.models.modeling_qwen3_5.get_sm_version", return_value=sm_version
    ):
        _normalize_qwen35_quant_config_dict(model_config)
    return model_config.quant_config_dict


@pytest.mark.parametrize("sm_version", ALL_SMS)
@pytest.mark.parametrize("algo", CHECKPOINT_ALGOS)
@pytest.mark.parametrize("proj", DENSE_PROJECTIONS)
def test_dense_mlp_repathed_for_every_algorithm(algo, proj, sm_version) -> None:
    """The re-path is a module-tree fact, not an algorithm-specific one."""
    key = f"model.layers.7.mlp.{proj}"
    normalized = _normalize({key: QuantConfig(quant_algo=algo)}, sm_version)

    assert key not in normalized
    assert f"model.layers.7.mlp.mlp.{proj}" in normalized
    # An algorithm the checkpoint states outright is never rewritten.
    assert normalized[f"model.layers.7.mlp.mlp.{proj}"].quant_algo == algo


@pytest.mark.parametrize("sm_version", BLACKWELL_SMS)
@pytest.mark.parametrize("proj", DENSE_PROJECTIONS)
def test_dense_mlp_w4a16_nvfp4_promoted_on_blackwell(proj, sm_version) -> None:
    normalized = _normalize(
        {f"model.layers.7.mlp.{proj}": QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)},
        sm_version,
    )

    assert normalized[f"model.layers.7.mlp.mlp.{proj}"].quant_algo == QuantAlgo.NVFP4


@pytest.mark.parametrize("sm_version", OTHER_SMS)
@pytest.mark.parametrize("proj", DENSE_PROJECTIONS)
def test_dense_mlp_w4a16_nvfp4_repathed_without_promotion(proj, sm_version) -> None:
    """Off SM100/103 the entry is still re-pathed -- only the promotion is gated."""
    normalized = _normalize(
        {f"model.layers.7.mlp.{proj}": QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)},
        sm_version,
    )

    assert normalized[f"model.layers.7.mlp.mlp.{proj}"].quant_algo == QuantAlgo.W4A16_NVFP4


def test_dense_mlp_repathed_from_vlm_namespace() -> None:
    """Qwen3.8-27B-NVFP4 is a VLM checkpoint: keys arrive language_model-prefixed."""
    normalized = _normalize(
        {
            f"model.language_model.layers.0.mlp.{proj}": QuantConfig(quant_algo=QuantAlgo.NVFP4)
            for proj in DENSE_PROJECTIONS
        },
        121,
    )

    assert set(normalized) == {f"model.layers.0.mlp.mlp.{proj}" for proj in DENSE_PROJECTIONS}


def test_dense_mlp_repathed_from_mtp_namespace() -> None:
    normalized = _normalize(
        {"mtp.layers.0.mlp.down_proj": QuantConfig(quant_algo=QuantAlgo.NVFP4)}, 121
    )

    assert set(normalized) == {f"model.layers.{NUM_HIDDEN_LAYERS}.mlp.mlp.down_proj"}


@pytest.mark.parametrize("sm_version", ALL_SMS)
def test_non_dense_mlp_entries_are_untouched(sm_version) -> None:
    """Attention, linear-attention and MoE-expert keys keep their paths."""
    entries = {
        "model.layers.3.self_attn.q_proj": QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
        "model.layers.3.self_attn.o_proj": QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
        "model.layers.0.linear_attn.out_proj": QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
        "model.layers.0.mlp.shared_expert.gate_proj": QuantAlgo.NVFP4,
    }
    normalized = _normalize({k: QuantConfig(quant_algo=v) for k, v in entries.items()}, sm_version)

    assert set(normalized) == set(entries)
    for key, algo in entries.items():
        assert normalized[key].quant_algo == algo


@pytest.mark.parametrize("sm_version", BLACKWELL_SMS)
def test_moe_experts_promotion_is_unchanged(sm_version) -> None:
    """Regression guard for the ModelOpt Qwen3.5/3.6 MoE path."""
    normalized = _normalize(
        {"model.layers.0.mlp.experts": QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)}, sm_version
    )

    assert normalized["model.layers.0.mlp.experts"].quant_algo == QuantAlgo.NVFP4


@pytest.mark.parametrize("sm_version", OTHER_SMS)
def test_moe_experts_not_promoted_off_blackwell(sm_version) -> None:
    normalized = _normalize(
        {"model.layers.0.mlp.experts": QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)}, sm_version
    )

    assert normalized["model.layers.0.mlp.experts"].quant_algo == QuantAlgo.W4A16_NVFP4


@pytest.mark.parametrize(
    "algo, expected_method",
    [
        # nvfp4-pack-quantized with FP4 input_activations parses to NVFP4
        # (W4A4). SM120/121 has CUTLASS FP4 GEMM tiles, so this is the intended
        # path there -- Marlin (W4A16) is only substituted for W4A16_NVFP4.
        (QuantAlgo.NVFP4, NVFP4LinearMethod),
        (QuantAlgo.W4A16_NVFP4, W4A16NVFP4LinearMethod),
        # float-quantized channel/token: e4m3 weight + per-channel [out, 1]
        # weight_scale, flattened onto the 1-D buffer by load_weights_vanilla.
        (QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN, FP8RowwiseLinearMethod),
    ],
)
def test_preserved_algorithm_resolves_to_the_loading_method(algo, expected_method) -> None:
    """The algorithms kept above must select a method that reads the stored tensors."""
    normalized = _normalize({"model.layers.7.mlp.down_proj": QuantConfig(quant_algo=algo)}, 121)
    cfg = normalized["model.layers.7.mlp.mlp.down_proj"]

    assert type(get_quant_method(cfg)) is expected_method


def test_split_linear_attn_fp8_fusion_is_unchanged() -> None:
    """Per-tensor FP8 in_proj still fuses; rowwise FP8 still does not."""
    per_tensor = _normalize(
        {
            "model.layers.0.linear_attn.in_proj_qkv": QuantConfig(quant_algo=QuantAlgo.FP8),
            "model.layers.0.linear_attn.in_proj_z": QuantConfig(quant_algo=QuantAlgo.FP8),
        },
        121,
    )
    assert set(per_tensor) == {"model.layers.0.linear_attn.in_proj_qkvz"}

    rowwise = _normalize(
        {
            "model.layers.0.linear_attn.in_proj_qkv": QuantConfig(
                quant_algo=QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
            ),
            "model.layers.0.linear_attn.in_proj_z": QuantConfig(
                quant_algo=QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
            ),
        },
        121,
    )
    # No fused entry: the fused Linear stays unquantized and the weight mapper
    # dequantizes the split projections to bf16 (exactly, via the [out, 1]
    # per-channel scale). The split keys match no runtime module.
    assert "model.layers.0.linear_attn.in_proj_qkvz" not in rowwise
