# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Control-plane tests for NVFP4 cold-page compression."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page.nvfp4 import (
    _load_modelopt_nvfp4_scales,
)
from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page.quantization_for_cold_page import (
    ColdPageQuantizationCompression,
)
from tensorrt_llm._torch.pyexecutor import _util as util_mod
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._torch.speculative.utils import update_spec_config_from_model_config
from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig, MTPDecodingConfig
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.runtime import kv_cache_manager_v2 as runtime_v2_mod
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    SsmLayerConfig,
)

pytestmark = pytest.mark.cpu_only


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
    def config():
        return SimpleNamespace()

    def buffer_layout():
        return SimpleNamespace(scales=None, cold_scale_offset=0)

    codec = MagicMock()
    module = SimpleNamespace(
        Nvfp4ColdPageRuntimeType=SimpleNamespace(
            FLOAT16="native-fp16",
            BFLOAT16="native-bf16",
            FP8_E4M3="native-fp8",
        ),
        Nvfp4ColdPageScales=config,
        Nvfp4ColdPageBufferLayout=buffer_layout,
        Nvfp4ColdPageLayerLayout=config,
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


def test_optional_modelopt_scales_map_pp_layers_and_default_missing_layers(tmp_path):
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
    plans = native.create_nvfp4_cold_page_codec.call_args.args[0]
    assert [plan.layer_id for plan in plans] == [0, 1, 2]
    assert [plan.runtime_type for plan in plans] == ["native-bf16"] * 3
    assert [
        (
            tuple(buffer.scales.nvfp4_scale_orig_quant for buffer in plan.buffers),
            tuple(buffer.scales.nvfp4_scale_quant_orig for buffer in plan.buffers),
        )
        for plan in plans
    ] == [
        ((2.0, 4.0), (0.5, 0.25)),
        ((8.0, 16.0), (0.125, 0.0625)),
        ((1.0, 1.0), (1.0, 1.0)),
    ]


def test_draft_codec_does_not_reuse_target_modelopt_scales(tmp_path) -> None:
    native, _ = _native()
    _write_scales(tmp_path, {10: (0.5, 0.25)})
    manager = _manager(tmp_path)

    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        manager.create_cold_page_codec(
            _cache_config((0, "attention")),
            runtime_dtype=DataType.BF16,
            pp_layers=(10,),
            num_kv_heads_per_layer=(8,),
            head_dim_per_layer=(128,),
        )
        target_plan = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
        manager.create_cold_page_codec(
            _cache_config((0, "attention")),
            runtime_dtype=DataType.BF16,
            pp_layers=(10,),
            num_kv_heads_per_layer=(8,),
            head_dim_per_layer=(128,),
            is_draft=True,
        )

    draft_plan = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert [buffer.scales.nvfp4_scale_orig_quant for buffer in target_plan.buffers] == [
        2.0,
        4.0,
    ]
    assert [buffer.scales.nvfp4_scale_orig_quant for buffer in draft_plan.buffers] == [
        1.0,
        1.0,
    ]
    assert [buffer.scales.nvfp4_scale_quant_orig for buffer in draft_plan.buffers] == [
        1.0,
        1.0,
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

    plan = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert [buffer.role for buffer in plan.buffers] == ["key", "value"]
    assert [buffer.cold_data_offset for buffer in plan.buffers] == [0, 1280]
    assert [buffer.cold_scale_offset for buffer in plan.buffers] == [2560, 2720]
    assert plan.cold_page_bytes == 2880
    assert plan.cold_padding_offset == plan.cold_page_bytes
    assert plan.runtime_type == "native-fp16"
    assert plan.num_kv_heads == 4
    assert plan.tokens_per_page == 5
    assert plan.head_dim == 128
    assert [buffer.scales.nvfp4_scale_orig_quant for buffer in plan.buffers] == [
        1.0,
        1.0,
    ]
    assert [buffer.scales.nvfp4_scale_quant_orig for buffer in plan.buffers] == [
        1.0,
        1.0,
    ]


def test_mha_layout_is_k_v_then_scales_and_layer_padding() -> None:
    native, _ = _native()
    cache_config = SimpleNamespace(
        tokens_per_block=5,
        layers=(
            AttentionLayerConfig(
                layer_id=0,
                buffers=[
                    BufferConfig(role="key", size=320),
                    BufferConfig(role="value", size=320),
                ],
            ),
        ),
    )

    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        _manager().create_cold_page_codec(
            cache_config,
            runtime_dtype=DataType.HALF,
            pp_layers=(0,),
            num_kv_heads_per_layer=(1,),
            head_dim_per_layer=(32,),
        )

    plan = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert [buffer.role for buffer in plan.buffers] == ["key", "value"]
    assert [buffer.cold_data_offset for buffer in plan.buffers] == [0, 80]
    assert [buffer.cold_scale_offset for buffer in plan.buffers] == [160, 170]
    assert plan.cold_padding_offset == 180
    assert plan.cold_page_bytes == 192


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


def test_unsupported_quant_does_not_construct_nvfp4_policy() -> None:
    config = SimpleNamespace(
        quant="future-format",
        scale_checkpoint_path="/not/a/checkpoint",
    )
    with patch(
        "tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page."
        "quantization_for_cold_page.Nvfp4ColdPagePolicy"
    ) as policy:
        with pytest.raises(NotImplementedError, match="future-format"):
            ColdPageQuantizationCompression(config)
    policy.assert_not_called()


def test_scale_loader_matches_hf_shard_and_consolidated_policy(tmp_path):
    _write_scales(tmp_path, {7: (0.5, 0.25)}, filename="model.safetensors")
    _write_scales(
        tmp_path,
        {7: (0.125, 0.0625)},
        filename="consolidated.00.safetensors",
    )
    assert _load_modelopt_nvfp4_scales(str(tmp_path))[7] == (
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
    assert _load_modelopt_nvfp4_scales(str(consolidated_only))[9] == (
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
    assert _load_modelopt_nvfp4_scales(str(tmp_path))[7] == (
        (2.0, 4.0),
        (0.5, 0.25),
    )


def test_scale_loader_ignores_multimodal_towers_with_the_same_layer_id(tmp_path):
    _write_quant_metadata(tmp_path)
    tensors = {
        "model.language_model.layers.7.self_attn.k_proj.k_scale": torch.tensor(0.5),
        "model.language_model.layers.7.self_attn.v_proj.v_scale": torch.tensor(0.25),
        "model.vision_tower.encoder.layers.7.self_attn.k_proj.k_scale": torch.tensor(4.0),
        "model.vision_tower.encoder.layers.7.self_attn.v_proj.v_scale": torch.tensor(2.0),
        "model.audio_tower.layers.7.self_attn.k_proj.k_scale": torch.tensor(8.0),
        "model.audio_tower.layers.7.self_attn.v_proj.v_scale": torch.tensor(4.0),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))

    assert _load_modelopt_nvfp4_scales(str(tmp_path))[7] == (
        (2.0, 4.0),
        (0.5, 0.25),
    )


def test_trtllm_load_kv_scales_zero_uses_identity(tmp_path, monkeypatch):
    _write_scales(tmp_path, {7: (0.5, 0.25)})
    monkeypatch.setenv("TRTLLM_LOAD_KV_SCALES", "0")
    assert _load_modelopt_nvfp4_scales(str(tmp_path)) == {}


def test_non_nvfp4_checkpoint_scales_are_not_reused(tmp_path):
    _write_scales(tmp_path, {7: (0.5, 0.25)})
    _write_quant_metadata(tmp_path, "FP8")
    assert _load_modelopt_nvfp4_scales(str(tmp_path)) == {}


def test_unquantized_checkpoint_uses_identity_scales(tmp_path):
    save_file({"model.weight": torch.ones(1)}, str(tmp_path / "model.safetensors"))
    assert _load_modelopt_nvfp4_scales(str(tmp_path)) == {}


def test_explicit_scale_checkpoint_requires_safetensors(tmp_path):
    with pytest.raises(FileNotFoundError, match="No safetensors files"):
        _load_modelopt_nvfp4_scales(str(tmp_path))


@pytest.mark.parametrize("present_kind", ["k", "v"])
def test_scale_checkpoint_requires_kv_pair(tmp_path, present_kind):
    _write_scales(tmp_path, {7: (0.5, 0.5)})
    base = "model.layers.7.self_attn"
    name = f"{base}.{present_kind}_proj.{present_kind}_scale"
    save_file({name: torch.tensor(0.5)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="both K and V"):
        _load_modelopt_nvfp4_scales(str(tmp_path))


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
        plan = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
        assert plan.layer_id == 1
        assert [buffer.scales.nvfp4_scale_orig_quant for buffer in plan.buffers] == [
            2.0,
            4.0,
        ]

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
    plan = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert plan.layer_id == 0
    assert plan.cold_page_bytes == 29184
    assert plan.cold_padding_offset == 29184
    assert [buffer.role for buffer in plan.buffers] == ["key", "index_key"]
    assert [buffer.scales is not None for buffer in plan.buffers] == [True, False]
    assert [buffer.cold_data_offset for buffer in plan.buffers] == [0, 20736]
    assert [buffer.cold_scale_offset for buffer in plan.buffers] == [18432, 0]
    assert plan.runtime_type == "native-bf16"
    assert plan.num_kv_heads == 1
    assert plan.tokens_per_page == 64
    assert plan.head_dim == 576
    scales = plan.buffers[0].scales
    assert scales.nvfp4_scale_orig_quant == scales.nvfp4_scale_quant_orig == 1.0
    assert plan.buffers[1].scales is None


def test_mla_all_non_latent_roles_are_explicit_lossless_spans() -> None:
    native, _ = _native()
    cache_config = SimpleNamespace(
        tokens_per_block=5,
        layers=(
            AttentionLayerConfig(
                layer_id=0,
                buffers=[
                    BufferConfig(role="key", size=320),
                    BufferConfig(role="index_key", size=68),
                    BufferConfig(role="rope_state", size=7),
                ],
            ),
        ),
    )

    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        _manager().create_cold_page_codec(
            cache_config,
            runtime_dtype=DataType.BF16,
            pp_layers=(0,),
            num_kv_heads_per_layer=(1,),
            head_dim_per_layer=(32,),
        )

    plan = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert [buffer.role for buffer in plan.buffers] == [
        "key",
        "index_key",
        "rope_state",
    ]
    assert [buffer.scales is not None for buffer in plan.buffers] == [True, False, False]
    assert [buffer.cold_data_offset for buffer in plan.buffers] == [0, 90, 158]
    assert plan.cold_padding_offset == 165
    assert plan.cold_page_bytes == 176


def test_tokens_per_block_override_expands_raw_and_lossless_bytes() -> None:
    native, _ = _native()
    cache_config = SimpleNamespace(
        tokens_per_block=4,
        layers=(
            AttentionLayerConfig(
                layer_id=0,
                buffers=[
                    BufferConfig(role="key", size=128),
                    BufferConfig(
                        role="index_key",
                        size=3,
                        tokens_per_block_override=2,
                    ),
                ],
            ),
        ),
    )

    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        _manager().create_cold_page_codec(
            cache_config,
            runtime_dtype=DataType.BF16,
            pp_layers=(0,),
            num_kv_heads_per_layer=(1,),
            head_dim_per_layer=(16,),
        )

    plan = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert [buffer.cold_data_offset for buffer in plan.buffers] == [0, 36]
    assert plan.cold_padding_offset == 42
    assert plan.cold_page_bytes == 48


def test_tokens_per_block_override_must_divide_page_size() -> None:
    native, _ = _native()
    cache_config = SimpleNamespace(
        tokens_per_block=5,
        layers=(
            AttentionLayerConfig(
                layer_id=0,
                buffers=[
                    BufferConfig(
                        role="key",
                        size=64,
                        tokens_per_block_override=2,
                    ),
                ],
            ),
        ),
    )

    with (
        patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native),
        pytest.raises(ValueError, match="positive divisor"),
    ):
        _manager().create_cold_page_codec(
            cache_config,
            runtime_dtype=DataType.BF16,
            pp_layers=(0,),
            num_kv_heads_per_layer=(1,),
            head_dim_per_layer=(16,),
        )


@pytest.mark.parametrize(
    ("owns_index", "expected_layers", "expected_buffers", "expected_bytes"),
    [
        ([True] * 61, 61, 122, 1_780_224),
        (
            [layer < 3 or (layer >= 6 and layer % 4 == 2) for layer in range(78)],
            78,
            99,
            1_794_816,
        ),
    ],
    ids=("deepseek-v3.2", "glm-5.2"),
)
def test_mla_model_layouts_are_built_in_python(
    owns_index: list[bool],
    expected_layers: int,
    expected_buffers: int,
    expected_bytes: int,
) -> None:
    native, _ = _native()
    layers = []
    for layer_id, has_index in enumerate(owns_index):
        buffers = [BufferConfig(role="key", size=64 * 576 * 2)]
        if has_index:
            buffers.append(BufferConfig(role="index_key", size=64 * (128 + 4)))
        layers.append(AttentionLayerConfig(layer_id=layer_id, buffers=buffers))
    cache_config = SimpleNamespace(tokens_per_block=64, layers=tuple(layers))

    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        _manager().create_cold_page_codec(
            cache_config,
            runtime_dtype=DataType.BF16,
            pp_layers=tuple(range(expected_layers)),
            num_kv_heads_per_layer=(1,) * expected_layers,
            head_dim_per_layer=(576,) * expected_layers,
        )

    plans = native.create_nvfp4_cold_page_codec.call_args.args[0]
    assert len(plans) == expected_layers
    assert sum(len(plan.buffers) for plan in plans) == expected_buffers
    assert sum(plan.cold_page_bytes for plan in plans) == expected_bytes


def test_fp8_runtime_uses_modelopt_nvfp4_scales(tmp_path):
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

    plan = native.create_nvfp4_cold_page_codec.call_args.args[0][0]
    assert plan.runtime_type == "native-fp8"
    assert [buffer.scales.nvfp4_scale_orig_quant for buffer in plan.buffers] == [
        2.0,
        4.0,
    ]
    assert [buffer.scales.nvfp4_scale_quant_orig for buffer in plan.buffers] == [
        0.5,
        0.25,
    ]
    assert all(
        buffer.scales.fp8_scale_orig_quant == buffer.scales.fp8_scale_quant_orig == 1.0
        for buffer in plan.buffers
    )


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
        creator._kv_cache_config = SimpleNamespace(host_cache_size=None, disk_cache_size=None)
        creator._llm_args = SimpleNamespace(
            kv_cache_compression_config=ColdPageQuantizationCompressionConfig()
        )
        model_config = SimpleNamespace(quant_config=active_kv_quant, pretrained_config=object())
        creator._model_engine = SimpleNamespace(model=SimpleNamespace(model_config=model_config))
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
        {"active_kv_quant": QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4)},
    ):
        assert util_mod.ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER not in build(**kwargs)
