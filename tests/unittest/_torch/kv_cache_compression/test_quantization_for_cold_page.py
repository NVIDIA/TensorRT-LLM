# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Control-plane tests for NVFP4 cold-page compression."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page import (
    ColdPageQuantizationCompression,
)
from tensorrt_llm._torch.pyexecutor import _util as util_mod
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._torch.speculative.utils import update_spec_config_from_model_config
from tensorrt_llm.llmapi.llm_args import (
    ColdPageQuantizationCompressionConfig,
    MTPDecodingConfig,
)
from tensorrt_llm.runtime import kv_cache_manager_v2 as runtime_v2_mod
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    SsmLayerConfig,
)


def _manager(scale_checkpoint_path=None):
    config = ColdPageQuantizationCompressionConfig(
        scale_checkpoint_path=(
            str(scale_checkpoint_path) if scale_checkpoint_path is not None else None
        )
    )
    return ColdPageQuantizationCompression(config)


def _cache_config(*layers):
    configs = []
    for layer_id, kind in layers:
        layer_type = SsmLayerConfig if kind == "ssm" else AttentionLayerConfig
        roles = ("ssm_state", "conv_state") if kind == "ssm" else ("key", "value")
        configs.append(
            layer_type(
                layer_id=layer_id,
                buffers=[BufferConfig(role=role, size=128) for role in roles],
            )
        )
    return SimpleNamespace(tokens_per_block=64, layers=tuple(configs))


def _native():
    def layer_config():
        return SimpleNamespace(
            fp8_scale_orig_quant=(1.0, 1.0),
            fp8_scale_quant_orig=(1.0, 1.0),
        )

    codec = MagicMock()
    module = SimpleNamespace(
        Nvfp4BoundaryRuntimeType=SimpleNamespace(
            FLOAT16="native-fp16",
            BFLOAT16="native-bf16",
            FP8_E4M3="native-fp8",
        ),
        Nvfp4ColdPageLayerConfig=layer_config,
        create_nvfp4_cold_page_codec=MagicMock(return_value=codec),
    )
    return module, codec


def _write_quant_metadata(directory, algorithm="NVFP4"):
    metadata = {
        "producer": {"name": "modelopt"},
        "quantization": {"kv_cache_quant_algo": algorithm},
    }
    (directory / "hf_quant_config.json").write_text(json.dumps(metadata))


def _write_scales(directory, scales_by_layer, *, filename="model.safetensors", prefix="model"):
    _write_quant_metadata(directory)
    tensors = {}
    for layer_id, (k_scale, v_scale) in scales_by_layer.items():
        base = f"{prefix}.layers.{layer_id}.self_attn"
        tensors[f"{base}.k_proj.k_scale"] = torch.as_tensor(k_scale, dtype=torch.float32)
        tensors[f"{base}.v_proj.v_scale"] = torch.as_tensor(v_scale, dtype=torch.float32)
    save_file(tensors, str(directory / filename))


def _validate_compression(mode=None):
    spec_config = None if mode is None else SimpleNamespace(spec_dec_mode=mode)
    util_mod.validate_kv_cache_compression_compatibility(
        ColdPageQuantizationCompressionConfig(),
        SimpleNamespace(enable_block_reuse=False),
        spec_config,
    )


def test_optional_modelopt_scales_map_pp_layers_and_ignore_local_draft_id(tmp_path):
    native, codec = _native()
    _write_scales(
        tmp_path,
        {10: (0.5, 0.25)},
        filename="model-00001-of-00002.safetensors",
    )
    _write_scales(
        tmp_path,
        {4: (0.125, 0.0625), 2: (0.75, 0.5)},
        filename="model-00002-of-00002.safetensors",
        prefix="model.language_model",
    )

    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        result = _manager(tmp_path).create_cold_page_codec(
            _cache_config((0, "attention"), (1, "attention"), (2, "attention")),
            runtime_dtype=DataType.BF16,
            pp_layers=(10, 4, 32),
            num_kv_heads_per_layer=(8, 8, 8),
            head_dim_per_layer=(128, 128, 128),
        )

    assert result is codec
    configs = native.create_nvfp4_cold_page_codec.call_args.args[0]
    assert [config.layer_id for config in configs] == [0, 1, 2]
    assert [config.runtime_type for config in configs] == ["native-bf16"] * 3
    assert [
        (config.nvfp4_scale_orig_quant, config.nvfp4_scale_quant_orig)
        for config in configs
    ] == [
        ((2.0, 4.0), (0.5, 0.25)),
        ((8.0, 16.0), (0.125, 0.0625)),
        ((1.0, 1.0), (1.0, 1.0)),
    ]


def test_omitted_scale_checkpoint_uses_identity_and_keeps_kv_geometry():
    native, _ = _native()
    cache_config = _cache_config((0, "attention"))
    cache_config.tokens_per_block = 5

    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        _manager().create_cold_page_codec(
            cache_config,
            runtime_dtype=DataType.HALF,
            pp_layers=(10,),
            num_kv_heads_per_layer=(4,),
            head_dim_per_layer=(128,),
        )

    config = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert config.runtime_type == "native-fp16"
    assert config.num_kv_heads == 4
    assert config.tokens_per_page == 5
    assert config.head_dim == 128
    assert config.nvfp4_scale_orig_quant == config.nvfp4_scale_quant_orig == (1.0, 1.0)


def test_provider_creates_one_native_codec_per_kv_cache_manager():
    native, _ = _native()
    codecs = (object(), object())
    native.create_nvfp4_cold_page_codec.side_effect = codecs
    provider = _manager()

    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        results = tuple(
            provider.create_cold_page_codec(
                _cache_config((0, "attention")),
                runtime_dtype=DataType.BF16,
                pp_layers=(layer_id,),
                num_kv_heads_per_layer=(8,),
                head_dim_per_layer=(128,),
            )
            for layer_id in (0, 32)
        )

    assert results == codecs
    assert native.create_nvfp4_cold_page_codec.call_count == 2


def test_scale_loader_matches_hf_shard_and_consolidated_policy(tmp_path):
    _write_scales(tmp_path, {7: (0.5, 0.25)}, filename="model.safetensors")
    _write_scales(
        tmp_path,
        {7: (0.125, 0.0625)},
        filename="consolidated.00.safetensors",
    )
    assert _manager(tmp_path)._model_nvfp4_scales[7] == (
        (2.0, 4.0),
        (0.5, 0.25),
    )

    consolidated_only = tmp_path / "consolidated-only"
    consolidated_only.mkdir()
    _write_scales(
        consolidated_only,
        {9: (0.125, 0.0625)},
        filename="consolidated.00.safetensors",
    )
    assert _manager(consolidated_only)._model_nvfp4_scales[9] == (
        (8.0, 16.0),
        (0.125, 0.0625),
    )


def test_scale_loader_reduces_duplicate_shards_like_native_qkv_loader(tmp_path):
    _write_scales(tmp_path, {7: (0.25, 0.125)}, filename="model-00001.safetensors")
    _write_scales(
        tmp_path,
        {7: (0.5, 0.25)},
        filename="model-00002.safetensors",
        prefix="model.language_model",
    )
    assert _manager(tmp_path)._model_nvfp4_scales[7] == (
        (2.0, 4.0),
        (0.5, 0.25),
    )


def test_trtllm_load_kv_scales_zero_uses_identity(tmp_path, monkeypatch):
    _write_scales(tmp_path, {7: (0.5, 0.25)})
    monkeypatch.setenv("TRTLLM_LOAD_KV_SCALES", "0")
    assert _manager(tmp_path)._model_nvfp4_scales == {}


def test_non_nvfp4_checkpoint_scales_are_not_reused(tmp_path):
    _write_scales(tmp_path, {7: (0.5, 0.25)})
    _write_quant_metadata(tmp_path, "FP8")
    assert _manager(tmp_path)._model_nvfp4_scales == {}


def test_unquantized_checkpoint_uses_identity_scales(tmp_path):
    save_file({"model.weight": torch.ones(1)}, str(tmp_path / "model.safetensors"))
    assert _manager(tmp_path)._model_nvfp4_scales == {}


def test_explicit_scale_checkpoint_requires_safetensors(tmp_path):
    with pytest.raises(FileNotFoundError, match="No safetensors files"):
        _manager(tmp_path)


@pytest.mark.parametrize("present_kind", ["k", "v"])
def test_scale_checkpoint_requires_kv_pair(tmp_path, present_kind):
    _write_scales(tmp_path, {7: (0.5, 0.5)})
    base = "model.layers.7.self_attn"
    name = f"{base}.{present_kind}_proj.{present_kind}_scale"
    save_file({name: torch.tensor(0.5)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="both K and V"):
        _manager(tmp_path)


def test_hybrid_codec_skips_ssm_layers_and_ssm_only_rank_is_lossless(tmp_path):
    native, codec = _native()
    _write_scales(tmp_path, {4: (0.5, 0.25)})
    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        _manager(tmp_path).create_cold_page_codec(
            _cache_config((0, "ssm"), (1, "attention")),
            runtime_dtype=DataType.BF16,
            pp_layers=(10, 4),
            num_kv_heads_per_layer=(0, 8),
            head_dim_per_layer=(128, 128),
        )
        config = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
        assert config.layer_id == 1
        assert config.nvfp4_scale_orig_quant == (2.0, 4.0)

        result = _manager().create_cold_page_codec(
            _cache_config((0, "ssm")),
            runtime_dtype=DataType.INT8,
            pp_layers=(10,),
            num_kv_heads_per_layer=(0,),
            head_dim_per_layer=(128,),
        )

    assert result is codec
    native.create_nvfp4_cold_page_codec.assert_called_with([])


def test_mla_key_only_layout_with_index_key_uses_identity_scales(tmp_path):
    native, codec = _native()
    _write_scales(tmp_path, {10: (0.5, 0.25)})
    cache_config = SimpleNamespace(
        tokens_per_block=64,
        layers=(
            AttentionLayerConfig(
                layer_id=0,
                buffers=[
                    BufferConfig(role="key", size=64 * 576 * 2),
                    BufferConfig(role="index_key", size=64 * 132),
                ],
            ),
        ),
    )
    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        result = _manager(tmp_path).create_cold_page_codec(
            cache_config,
            runtime_dtype=DataType.BF16,
            pp_layers=(10,),
            num_kv_heads_per_layer=(1,),
            head_dim_per_layer=(576,),
        )

    assert result is codec
    config = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert config.layer_id == 0
    assert config.runtime_type == "native-bf16"
    assert config.num_kv_heads == 1
    assert config.tokens_per_page == 64
    assert config.head_dim == 576
    assert config.nvfp4_scale_orig_quant == config.nvfp4_scale_quant_orig == (1.0, 1.0)


def test_fp8_runtime_uses_native_unit_source_scale_default(tmp_path):
    native, _ = _native()
    _write_scales(tmp_path, {10: (0.5, 0.25)})
    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        _manager(tmp_path).create_cold_page_codec(
            _cache_config((0, "attention")),
            runtime_dtype=DataType.FP8,
            pp_layers=(10,),
            num_kv_heads_per_layer=(8,),
            head_dim_per_layer=(128,),
        )

    config = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert config.fp8_scale_orig_quant == config.fp8_scale_quant_orig == (1.0, 1.0)
    assert config.nvfp4_scale_quant_orig == (0.5, 0.25)


def test_runtime_admission_is_checked_in_utils_before_manager_creation(monkeypatch):
    monkeypatch.setattr(runtime_v2_mod, "_BACKEND", "python")
    with pytest.raises(ValueError, match=r"require.*C\+\+ KVCacheManagerV2"):
        _validate_compression()

    monkeypatch.setattr(runtime_v2_mod, "_BACKEND", "cpp")
    monkeypatch.setattr(util_mod, "is_sm_100f", lambda: False)
    with pytest.raises(RuntimeError, match="requires an SM100-family device"):
        _validate_compression()

    monkeypatch.setattr(util_mod, "is_sm_100f", lambda: True)
    _validate_compression()


def test_speculative_admission_accepts_verified_one_model_modes(monkeypatch):
    monkeypatch.setattr(runtime_v2_mod, "_BACKEND", "cpp")
    monkeypatch.setattr(util_mod, "is_sm_100f", lambda: True)

    _validate_compression(SpeculativeDecodingMode.EAGLE3_ONE_MODEL)
    _validate_compression(SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL)
    for mode in (
        SpeculativeDecodingMode.MTP,
        SpeculativeDecodingMode.MTP_EAGLE,
        SpeculativeDecodingMode.EAGLE3,
        SpeculativeDecodingMode.DFLASH,
    ):
        with pytest.raises(ValueError, match="one-model MTP-EAGLE or EAGLE3"):
            _validate_compression(mode)


def test_qwen35_mtp3_resolves_to_supported_one_model_mode(monkeypatch):
    monkeypatch.setattr(runtime_v2_mod, "_BACKEND", "cpp")
    monkeypatch.setattr(util_mod, "is_sm_100f", lambda: True)

    spec_config = MTPDecodingConfig(max_draft_len=3)
    update_spec_config_from_model_config(
        spec_config,
        SimpleNamespace(mtp_num_hidden_layers=1),
    )

    assert spec_config.spec_dec_mode is SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL
    assert spec_config.max_draft_len == 3
    util_mod.validate_kv_cache_compression_compatibility(
        ColdPageQuantizationCompressionConfig(),
        SimpleNamespace(enable_block_reuse=False),
        spec_config,
    )


def test_cold_manager_is_disabled_for_estimation_and_active_nvfp4():
    def build(*, estimating=False, active_kv_quant=None):
        creator = object.__new__(util_mod.KvCacheCreator)
        creator._skip_est = False
        creator._max_seq_len = 1024
        creator._kv_cache_config = SimpleNamespace()
        creator._llm_args = SimpleNamespace(
            kv_cache_compression_config=ColdPageQuantizationCompressionConfig()
        )
        model_config = SimpleNamespace(
            quant_config=active_kv_quant, pretrained_config=object()
        )
        creator._model_engine = SimpleNamespace(
            model=SimpleNamespace(model_config=model_config)
        )
        creator._draft_model_engine = None
        creator._kv_connector_manager = None
        creator._fp8_ctx_mla_kv_len_cap = None
        creator._is_encoder_decoder = MagicMock(return_value=False)
        creator._should_create_separate_draft_kv_cache = MagicMock(return_value=False)
        creator._create_kv_cache_manager = MagicMock(return_value=SimpleNamespace())
        resources = {}
        creator.build_managers(resources, estimating_kv_cache=estimating)
        return resources

    resources = build()
    manager = resources[util_mod.ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER]
    assert isinstance(manager, ColdPageQuantizationCompression)
    assert manager.provides_cold_page_codec
    assert not manager.uses_iteration_lifecycle
    for kwargs in (
        {"estimating": True},
        {"active_kv_quant": SimpleNamespace(kv_cache_quant_algo="NVFP4")},
    ):
        assert util_mod.ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER not in build(
            **kwargs
        )
