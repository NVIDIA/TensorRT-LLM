# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor._util import CacheCost, _create_kv_cache_manager
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.model_loader import initialize_dummy_weights
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = pytest.mark.cpu_only


def _model_config(kv_quant: QuantAlgo | None = QuantAlgo.INT8) -> ModelConfig:
    """Build a dense two-layer configuration for CPU cache-sizing checks."""
    return ModelConfig(
        pretrained_config=LlamaConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_hidden_layers=2,
            vocab_size=512,
        ),
        quant_config=QuantConfig(kv_cache_quant_algo=kv_quant),
    )


@pytest.mark.parametrize("sm", [100, 103, 107])
def test_int8_kv_rejects_trtllm_gen_devices(sm: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject an unsupported decode path before native attention is initialized."""
    from tensorrt_llm._torch.attention.backends import trtllm as backend

    monkeypatch.setattr(backend, "get_sm_version", lambda: sm)
    attention = backend.TrtllmAttention.__new__(backend.TrtllmAttention)
    attention.is_mla_enable = False
    attention.sparse_params = None
    with pytest.raises(ValueError, match=f"INT8 KV cache.*TRTLLM-Gen.*SM{sm}"):
        attention.update_quant_config(QuantConfig(kv_cache_quant_algo=QuantAlgo.INT8))


@pytest.mark.parametrize(("tp_size", "kv_heads_per_rank"), [(1, 2), (2, 1), (4, 1)])
@pytest.mark.parametrize("manager_type", [KVCacheManager, KVCacheManagerV2])
def test_int8_kv_estimation_uses_half_the_bf16_bytes(
    tp_size: int, kv_heads_per_rank: int, manager_type: type
) -> None:
    """Both managers must budget one byte per INT8 element before allocation."""
    mapping = Mapping(world_size=tp_size, tp_size=tp_size)
    expected_bytes = 2 * 2 * kv_heads_per_rank * 64
    options = {"tokens_per_block": 32} if manager_type is KVCacheManagerV2 else {}
    int8_bytes = CacheCost.from_raw(
        manager_type.get_cache_size_per_token(_model_config(), mapping, **options)
    ).slope
    bf16_bytes = CacheCost.from_raw(
        manager_type.get_cache_size_per_token(_model_config(None), mapping, **options)
    ).slope
    assert int8_bytes == expected_bytes
    assert bf16_bytes == 2 * int8_bytes


@pytest.mark.parametrize("manager_type", [KVCacheManager, KVCacheManagerV2])
def test_int8_kv_pool_accounting(manager_type: type) -> None:
    # Sizing is CPU-only; constructor allocation is covered by GPU integration tests.
    """Runtime pool accounting must match static INT8/BF16 element sizes."""
    manager = manager_type.__new__(manager_type)
    manager.dtype = DataType.INT8
    manager.kv_factor = 2
    manager.kv_cache_type = CacheType.SELF
    manager.num_local_layers = 2
    manager.num_kv_heads_per_layer = [2, 2]
    manager.total_num_kv_heads_per_layer = [2, 2]
    manager.head_dim = 64
    manager.head_dim_per_layer = [64, 64]
    if manager_type is KVCacheManagerV2:
        manager.layer_specs = None
        manager.pp_layers = [0, 1]
        manager.mixed_precision_low_dtype = None
    assert manager.get_cache_bytes_per_token() == 512
    manager.dtype = DataType.BF16
    assert manager.get_cache_bytes_per_token() == 1024


class _CaptureCacheManager:
    def __init__(self, *args, **kwargs) -> None:
        self.dtype = kwargs["dtype"]


def _create_test_manager(
    config: ModelConfig,
    *,
    reuse: bool = False,
    speculative: bool = False,
    chunked_prefill: bool = False,
    is_disagg: bool = False,
    kv_connector: bool = False,
) -> _CaptureCacheManager:
    """Exercise the real cache factory without allocating GPU cache pages."""
    engine = SimpleNamespace(
        model=SimpleNamespace(model_config=config),
        dtype=torch.bfloat16,
        is_draft_model=False,
        attn_runtime_features=SimpleNamespace(chunked_prefill=chunked_prefill),
    )
    return _create_kv_cache_manager(
        model_engine=engine,
        dtype=torch.bfloat16,
        kv_cache_manager_cls=_CaptureCacheManager,
        mapping=Mapping(),
        kv_cache_config=KvCacheConfig(enable_block_reuse=reuse),
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=2,
        spec_config=SimpleNamespace() if speculative else None,
        sparse_attention_config=None,
        max_num_tokens=128,
        max_beam_width=1,
        kv_connector_manager=SimpleNamespace() if kv_connector else None,
        is_disagg=is_disagg,
    )


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"reuse": True}, "enable_block_reuse=False"),
        ({"speculative": True}, "speculative decoding"),
        ({"chunked_prefill": True}, "enable_chunked_prefill=False"),
        ({"is_disagg": True}, "disaggregated serving"),
        ({"kv_connector": True}, "KV connectors"),
    ],
)
def test_int8_kv_rejects_paged_context_features(options: dict, error: str) -> None:
    """Unsupported cache population paths must fail before pool allocation."""
    with pytest.raises(ValueError, match=error):
        _create_test_manager(_model_config(), **options)


@pytest.mark.parametrize("model_kind", ["hybrid", "encoder_decoder"])
def test_int8_kv_rejects_unsupported_model_families(model_kind: str) -> None:
    """Hybrid and encoder-decoder models must not enter the dense INT8 path."""
    config = _model_config()
    if model_kind == "hybrid":
        config.pretrained_config.hybrid_override_pattern = "M*"
    else:
        config.is_encoder_decoder = True
    with pytest.raises(ValueError, match="dense decoder-only"):
        _create_test_manager(config)


def test_dummy_weights_preserve_int8_kv_calibration() -> None:
    """Dummy-weight initialization must not randomize cache calibration parameters."""
    projection = torch.nn.Module()
    projection.register_parameter(
        "kv_cache_scaling_factor", torch.nn.Parameter(torch.ones(1), requires_grad=False)
    )
    projection.register_parameter(
        "inv_kv_cache_scaling_factor", torch.nn.Parameter(torch.ones(1), requires_grad=False)
    )
    projection.register_parameter("weight", torch.nn.Parameter(torch.zeros(8)))
    model = torch.nn.Module()
    model.add_module("qkv_proj", projection)
    initialize_dummy_weights(model)
    torch.testing.assert_close(projection.kv_cache_scaling_factor, torch.ones(1))
    torch.testing.assert_close(projection.inv_kv_cache_scaling_factor, torch.ones(1))
    assert torch.count_nonzero(projection.weight) > 0
