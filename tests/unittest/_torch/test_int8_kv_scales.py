# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.attention.attention import Attention
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.linear import Linear, WeightMode, WeightsLoadingConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _qkv(kv_quant: QuantAlgo = QuantAlgo.INT8) -> Linear:
    """Construct an unquantized fused QKV projection with quantized KV storage."""
    with torch.device("cuda"):
        return Linear(
            4,
            12,
            bias=False,
            dtype=torch.float16,
            reduce_output=False,
            quant_config=QuantConfig(kv_cache_quant_algo=kv_quant),
            weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.FUSED_QKV_LINEAR),
            fused_weight_shard_indices_mapping={"q": (0, 4), "k": (4, 4), "v": (8, 4)},
        )


def _weights(k_scale=0.03125, v_scale=0.0625) -> list[dict]:
    """Create separate Q/K/V checkpoint shards with optional calibration tensors."""
    weights = [{"weight": torch.full((4, 4), float(i + 1), dtype=torch.float16)} for i in range(3)]
    if k_scale is not None:
        weights[1]["k_scale"] = torch.as_tensor(k_scale, dtype=torch.float64)
    if v_scale is not None:
        weights[2]["v_scale"] = torch.as_tensor(v_scale, dtype=torch.float64)
    return weights


def test_int8_kv_load_and_partial_reload() -> None:
    """Calibration and weights reload independently without moving scale pointers."""
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
    """Invalid calibration must leave previously loaded scale values intact."""
    qkv = _qkv()
    qkv.load_weights(_weights())
    with pytest.raises(ValueError, match="INT8 KV cache"):
        qkv.load_weights(_weights(bad_scale, bad_scale))
    assert qkv.kv_cache_scaling_factor.item() == 0.0625
    assert qkv.inv_kv_cache_scaling_factor.item() == 16


@pytest.mark.parametrize("k_scale,v_scale", [(None, None), (None, 0.1), (0.1, None)])
def test_int8_kv_requires_paired_checkpoint_scales(k_scale, v_scale) -> None:
    """A full checkpoint must include both calibrated K and V scales."""
    with pytest.raises(ValueError, match="both scales must be loaded together"):
        _qkv().load_weights(_weights(k_scale, v_scale))


def test_int8_kv_partial_reload_rejects_unpaired_scale() -> None:
    """Partial scale updates must supply K and V together."""
    qkv = _qkv()
    qkv.load_weights(_weights())
    with pytest.raises(ValueError, match="both scales must be loaded together"):
        qkv.load_weights([{}, {"k_scale": torch.tensor(0.125)}, {}], allow_partial_loading=True)
    assert qkv.kv_cache_scaling_factor.item() == 0.0625


@pytest.mark.parametrize("backend", ["VANILLA", "FLASHINFER"])
def test_int8_kv_rejects_unsupported_backend(backend: str) -> None:
    """INT8 cache configuration must fail before selecting another attention backend."""
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
    """INT8 KV cache must not silently enable untested quantized-weight combinations."""
    with pytest.raises(ValueError, match="unquantized FP16/BF16 projections"):
        create_attention(
            "TRTLLM",
            layer_idx=0,
            num_heads=4,
            num_kv_heads=2,
            head_dim=128,
            quant_config=QuantConfig(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.INT8),
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_int8_kv_scale_loading_environment(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """INT8 must not silently use unity calibration when scale loading is disabled."""
    monkeypatch.setenv("TRTLLM_LOAD_KV_SCALES", str(int(enabled)))
    qkv = _qkv()
    if enabled:
        qkv.load_weights(_weights())
        assert qkv.kv_cache_scaling_factor.item() == 0.0625
    else:
        with pytest.raises(ValueError, match="TRTLLM_LOAD_KV_SCALES=1"):
            qkv.load_weights(_weights())


@pytest.mark.parametrize("enabled", [False, True])
def test_fp4_kv_scale_hook_preserves_optional_loading(
    enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared checkpoint hook retains FP4 opt-out and stable reload buffers."""
    monkeypatch.setenv("TRTLLM_LOAD_KV_SCALES", str(int(enabled)))
    qkv = _qkv(QuantAlgo.NVFP4)
    pointers = (qkv.kv_scales.data_ptr(), qkv.inv_kv_scales.data_ptr())
    qkv.load_weights(_weights(None, None))
    torch.testing.assert_close(qkv.kv_scales, torch.ones_like(qkv.kv_scales))
    qkv.load_weights(_weights())
    expected = qkv.kv_scales.new_tensor([1.0, 0.03125, 0.0625] if enabled else [1.0] * 3)
    torch.testing.assert_close(qkv.kv_scales, expected)
    torch.testing.assert_close(qkv.inv_kv_scales, expected.reciprocal())
    qkv.load_weights([{}, {"weight": torch.zeros(4, 4)}, {}], allow_partial_loading=True)
    torch.testing.assert_close(qkv.kv_scales, expected)
    qkv.load_weights(
        [{}, {"k_scale": torch.tensor(0.125)}, {"v_scale": torch.tensor(0.25)}],
        allow_partial_loading=True,
    )
    expected = qkv.kv_scales.new_tensor([1.0, 0.125, 0.25] if enabled else [1.0] * 3)
    torch.testing.assert_close(qkv.kv_scales, expected)
    torch.testing.assert_close(qkv.inv_kv_scales, expected.reciprocal())
    assert pointers == (qkv.kv_scales.data_ptr(), qkv.inv_kv_scales.data_ptr())


def test_int8_kv_attention_construction_rejects_context_parallelism() -> None:
    """Reject Helix CP from the actual Attention constructor before any forward."""
    config = ModelConfig(
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.INT8),
        mapping=Mapping(world_size=2, cp_size=2, cp_config={"cp_type": "HELIX"}),
        attn_backend="TRTLLM",
    )
    with torch.device("cuda"), pytest.raises(ValueError, match="without context parallelism"):
        Attention(
            hidden_size=512,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            bias=False,
            dtype=torch.float16,
            config=config,
            layer_idx=0,
            reduce_output=False,
        )
