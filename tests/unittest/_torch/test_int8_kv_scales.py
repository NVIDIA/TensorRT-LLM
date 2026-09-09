# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._torch.modules.linear import Linear, WeightMode, WeightsLoadingConfig
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def _qkv() -> Linear:
    with torch.device("cuda"):
        return Linear(
            4,
            12,
            bias=False,
            dtype=torch.float16,
            reduce_output=False,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.INT8),
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_QKV_LINEAR),
            fused_weight_shard_indices_mapping={"q": (0, 4), "k": (4, 4), "v": (8, 4)},
        )


def _weights(k_scale=0.03125, v_scale=0.0625) -> list[dict]:
    weights = [{"weight": torch.full((4, 4), float(i + 1), dtype=torch.float16)} for i in range(3)]
    if k_scale is not None:
        weights[1]["k_scale"] = torch.as_tensor(k_scale, dtype=torch.float64)
    if v_scale is not None:
        weights[2]["v_scale"] = torch.as_tensor(v_scale, dtype=torch.float64)
    return weights


def test_int8_kv_load_and_partial_reload() -> None:
    qkv = _qkv()
    pointers = (qkv.kv_cache_scaling_factor.data_ptr(), qkv.inv_kv_cache_scaling_factor.data_ptr())
    qkv.load_weights(_weights())
    assert qkv.kv_cache_scaling_factor.dtype == torch.float32
    assert qkv.kv_cache_scaling_factor.item() == 0.0625
    assert qkv.inv_kv_cache_scaling_factor.item() == 16
    torch.testing.assert_close(qkv.weight, torch.cat([w["weight"] for w in _weights()]).cuda())
    qkv.load_weights(
        [{}, {"weight": torch.full((4, 4), 7, dtype=torch.float16)}, {}], allow_partial_loading=True
    )
    assert qkv.kv_cache_scaling_factor.item() == 0.0625
    assert qkv.inv_kv_cache_scaling_factor.item() == 16
    assert (qkv.weight[4:8] == 7).all()
    qkv.load_weights(
        [{}, {"k_scale": torch.tensor(0.125)}, {"v_scale": torch.tensor(0.25)}],
        allow_partial_loading=True,
    )
    assert qkv.kv_cache_scaling_factor.item() == 0.25
    assert qkv.inv_kv_cache_scaling_factor.item() == 4
    assert pointers == (
        qkv.kv_cache_scaling_factor.data_ptr(),
        qkv.inv_kv_cache_scaling_factor.data_ptr(),
    )


@pytest.mark.parametrize(
    "bad_scale", [0.0, -1.0, float("nan"), float("inf"), 1e-45, 1e40, [0.1, 0.2]]
)
def test_int8_kv_rejects_invalid_scale_without_changing_loaded_scales(bad_scale) -> None:
    qkv = _qkv()
    qkv.load_weights(_weights())
    with pytest.raises(ValueError, match="INT8 KV cache"):
        qkv.load_weights(_weights(bad_scale, bad_scale))
    assert qkv.kv_cache_scaling_factor.item() == 0.0625
    assert qkv.inv_kv_cache_scaling_factor.item() == 16


@pytest.mark.parametrize("k_scale,v_scale", [(None, None), (None, 0.1), (0.1, None)])
def test_int8_kv_requires_paired_checkpoint_scales(k_scale, v_scale) -> None:
    with pytest.raises(ValueError, match="both scales must be loaded together"):
        _qkv().load_weights(_weights(k_scale, v_scale))


def test_int8_kv_partial_reload_rejects_unpaired_scale() -> None:
    qkv = _qkv()
    qkv.load_weights(_weights())
    with pytest.raises(ValueError, match="both scales must be loaded together"):
        qkv.load_weights([{}, {"k_scale": torch.tensor(0.125)}, {}], allow_partial_loading=True)
    assert qkv.kv_cache_scaling_factor.item() == 0.0625


@pytest.mark.parametrize("backend", ["VANILLA", "FLASHINFER"])
def test_int8_kv_rejects_unsupported_backend(backend: str) -> None:
    with pytest.raises(ValueError, match="INT8 KV cache requires the TRTLLM"):
        create_attention(
            backend,
            layer_idx=0,
            num_heads=4,
            num_kv_heads=2,
            head_dim=128,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.INT8),
        )


def test_int8_kv_rejects_quantized_projections() -> None:
    with pytest.raises(ValueError, match="unquantized FP16/BF16 projections"):
        create_attention(
            "TRTLLM",
            layer_idx=0,
            num_heads=4,
            num_kv_heads=2,
            head_dim=128,
            quant_config=QuantConfig(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.INT8),
        )
