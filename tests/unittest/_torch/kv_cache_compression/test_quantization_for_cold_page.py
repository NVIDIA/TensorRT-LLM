# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Control-plane tests for NVFP4 cold-page compression."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page.nvfp4_quantization import (
    Nvfp4ColdPageQuantizationCompression,
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
    return Nvfp4ColdPageQuantizationCompression(config)


def _factory_model_engine(
    *, active_kv_quant: object | None = None, helix: bool = False
) -> SimpleNamespace:
    return SimpleNamespace(
        mapping=SimpleNamespace(has_cp_helix=lambda: helix),
        spec_config=None,
        model=SimpleNamespace(
            model_config=SimpleNamespace(
                quant_config=active_kv_quant,
                pretrained_config=object(),
            )
        ),
    )


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


def _native() -> tuple[SimpleNamespace, MagicMock]:
    codec = MagicMock()
    module = SimpleNamespace(
        ColdPageLifecycleProperties=lambda: SimpleNamespace(),
        ColdPageIndexLocation=SimpleNamespace(HOST="host"),
        create_python_cold_page_codec=MagicMock(return_value=codec),
        nvfp4_cold_page_encode=MagicMock(),
        nvfp4_cold_page_decode=MagicMock(),
    )
    return module, codec


def _provider(native: SimpleNamespace) -> object:
    return native.create_python_cold_page_codec.call_args.args[0]


def _codec_state(native: SimpleNamespace) -> object:
    return native.create_python_cold_page_codec.call_args.args[1]


def _layouts(native: SimpleNamespace) -> list[object]:
    return list(_codec_state(native).layer_layouts.values())


def _configure_lifecycle(native: SimpleNamespace, layer_bytes: dict[int, dict[str, int]]) -> object:
    address = 0x10000
    layers = {}
    for layer_id, roles in layer_bytes.items():
        hot = {}
        for role, raw_bytes in roles.items():
            hot[role] = SimpleNamespace(
                raw_base=address,
                raw_slot_bytes=(raw_bytes + 15) // 16 * 16,
                raw_bytes=raw_bytes,
            )
            address += 0x10000
        layers[layer_id] = hot
    provider = _provider(native)
    codec_state = _codec_state(native)
    provider.configure(codec_state, [SimpleNamespace(layers=layers)])
    return codec_state.lifecycle_metadata[0]


def _configure_default_lifecycle(native: SimpleNamespace, raw_bytes: int) -> object:
    return _configure_lifecycle(native, {0: {"key": raw_bytes, "value": raw_bytes}})


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


def _validate_compression(mode: object | None = None) -> None:
    spec_config = None if mode is None else SimpleNamespace(spec_dec_mode=mode)
    with patch.object(util_mod, "is_sm_100f", return_value=True):
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
    layouts = _layouts(native)
    assert [layout.layer_id for layout in layouts] == [0, 1, 2]
    assert _codec_state(native).runtime_type == 1
    assert [
        (
            tuple(buffer.scales.nvfp4_orig_quant for buffer in layout.buffers),
            tuple(buffer.scales.nvfp4_quant_orig for buffer in layout.buffers),
        )
        for layout in layouts
    ] == [
        ((2.0, 4.0), (0.5, 0.25)),
        ((8.0, 16.0), (0.125, 0.0625)),
        ((1.0, 1.0), (1.0, 1.0)),
    ]
    metadata = _configure_lifecycle(
        native,
        {layer_id: {"key": 131072, "value": 131072} for layer_id in range(3)},
    )
    assert metadata.scales[:6].tolist() == [
        [2.0, 0.5, 1.0, 1.0],
        [4.0, 0.25, 1.0, 1.0],
        [8.0, 0.125, 1.0, 1.0],
        [16.0, 0.0625, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
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
        target_layout = _layouts(native)[0]
        manager.create_cold_page_codec(
            _cache_config((0, "attention")),
            runtime_dtype=DataType.BF16,
            pp_layers=(10,),
            num_kv_heads_per_layer=(8,),
            head_dim_per_layer=(128,),
            is_draft=True,
        )

    draft_layout = _layouts(native)[0]
    assert [buffer.scales.nvfp4_orig_quant for buffer in target_layout.buffers] == [
        2.0,
        4.0,
    ]
    assert [buffer.scales.nvfp4_orig_quant for buffer in draft_layout.buffers] == [
        1.0,
        1.0,
    ]
    assert [buffer.scales.nvfp4_quant_orig for buffer in draft_layout.buffers] == [
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

    layout = _layouts(native)[0]
    assert [buffer.role for buffer in layout.buffers] == ["key", "value"]
    assert layout.num_kv_heads == 4
    assert layout.tokens_per_page == 5
    assert layout.head_dim == 128
    assert [buffer.scales.nvfp4_orig_quant for buffer in layout.buffers] == [
        1.0,
        1.0,
    ]
    assert [buffer.scales.nvfp4_quant_orig for buffer in layout.buffers] == [
        1.0,
        1.0,
    ]
    metadata = _configure_default_lifecycle(native, raw_bytes=5120)
    assert metadata.cold_page_bytes == 2880
    assert metadata.wide[:2, 3].tolist() == [0, 1280]
    assert metadata.wide[:2, 4].tolist() == [2560, 2720]
    assert metadata.integers[:2, 0].tolist() == [0, 0]


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

    metadata = _configure_default_lifecycle(native, raw_bytes=320)
    assert metadata.cold_page_bytes == 192
    assert metadata.wide[:2, 3].tolist() == [0, 80]
    assert metadata.wide[:2, 4].tolist() == [160, 170]
    assert metadata.wide[1, 5].item() == 180
    assert metadata.integers[:2, 0].tolist() == [0, 12]


def test_provider_creates_independent_state_per_kv_cache_manager() -> None:
    native, _ = _native()
    codecs = (object(), object())
    native.create_python_cold_page_codec.side_effect = codecs
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
    assert native.create_python_cold_page_codec.call_count == 2
    calls = native.create_python_cold_page_codec.call_args_list
    assert all(call.args[0] is provider for call in calls)
    target_state, draft_state = (call.args[1] for call in calls)
    assert target_state is not draft_state
    assert target_state.layer_ids == draft_state.layer_ids == (0,)


def test_provider_forwards_a_4096_page_batch_through_one_native_call() -> None:
    native, _ = _native()
    encode = native.nvfp4_cold_page_encode
    decode = native.nvfp4_cold_page_decode

    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        _manager().create_cold_page_codec(
            _cache_config((0, "attention")),
            runtime_dtype=DataType.BF16,
            pp_layers=(0,),
            num_kv_heads_per_layer=(1,),
            head_dim_per_layer=(16,),
        )
        provider = _provider(native)
        codec_state = _codec_state(native)
        hot = {
            role: SimpleNamespace(
                raw_base=0x1000 + index * 0x1000,
                raw_slot_bytes=4096,
                raw_bytes=2048,
            )
            for index, role in enumerate(("key", "value"))
        }
        properties = provider.configure(codec_state, [SimpleNamespace(layers={0: hot})])
        provider.encode_cold_pages(codec_state, 0, 0x3000, 0x4000, 4096, 0x5000)
        provider.decode_cold_pages(codec_state, 0, 0x3000, 0x4000, 4096, 0x5000)

    assert properties[0].cold_page_bytes == 1152
    assert properties[0].page_index_location == "host"
    metadata = codec_state.lifecycle_metadata[0]
    for operation in (encode, decode):
        operation.assert_called_once()
        arguments = operation.call_args.args
        assert arguments == (
            0x4000,
            4096,
            metadata.wide.data_ptr(),
            metadata.integers.data_ptr(),
            metadata.scales.data_ptr(),
            2,
            128,
            1152,
            1,
            0x3000,
            0x5000,
        )


def test_codec_state_metadata_stays_on_cpu_with_non_cpu_default_device() -> None:
    native, _ = _native()
    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        _manager().create_cold_page_codec(
            _cache_config((0, "attention")),
            runtime_dtype=DataType.BF16,
            pp_layers=(0,),
            num_kv_heads_per_layer=(1,),
            head_dim_per_layer=(16,),
        )

    with torch.device("meta"):
        metadata = _configure_default_lifecycle(native, raw_bytes=2048)
    for tensor, dtype, shape in (
        (metadata.wide, torch.int64, (256, 6)),
        (metadata.integers, torch.int32, (256, 5)),
        (metadata.scales, torch.float32, (256, 4)),
    ):
        assert tensor.device.type == "cpu"
        assert tensor.dtype == dtype
        assert tensor.shape == shape
        assert tensor.is_contiguous()


def test_provider_rejects_invalid_resolved_hot_buffers() -> None:
    native, _ = _native()
    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        _manager().create_cold_page_codec(
            _cache_config((0, "attention")),
            runtime_dtype=DataType.BF16,
            pp_layers=(0,),
            num_kv_heads_per_layer=(1,),
            head_dim_per_layer=(16,),
        )

    provider = _provider(native)
    codec_state = _codec_state(native)

    def hot(raw_base: int = 0x1000, raw_bytes: int = 2048) -> SimpleNamespace:
        return SimpleNamespace(
            raw_base=raw_base,
            raw_slot_bytes=2048,
            raw_bytes=raw_bytes,
        )

    with pytest.raises(ValueError, match="roles do not match"):
        provider.configure(
            codec_state,
            [SimpleNamespace(layers={0: {"key": hot(), "value": hot(), "extra": hot()}})],
        )
    with pytest.raises(ValueError, match="size does not match"):
        provider.configure(
            codec_state, [SimpleNamespace(layers={0: {"key": hot(raw_bytes=32), "value": hot()}})]
        )
    with pytest.raises(ValueError, match="16-byte aligned"):
        provider.configure(
            codec_state,
            [SimpleNamespace(layers={0: {"key": hot(raw_base=0x1001), "value": hot()}})],
        )


def test_provider_rejects_more_than_256_lifecycle_buffers() -> None:
    native, _ = _native()
    layers = tuple(
        AttentionLayerConfig(
            layer_id=layer_id,
            buffers=[
                BufferConfig(role="key", size=32),
                BufferConfig(role="value", size=32),
            ],
        )
        for layer_id in range(129)
    )
    cache_config = SimpleNamespace(tokens_per_block=1, layers=layers)
    with patch("tensorrt_llm.bindings.internal.kv_cache_compression", new=native):
        _manager().create_cold_page_codec(
            cache_config,
            runtime_dtype=DataType.BF16,
            pp_layers=tuple(range(129)),
            num_kv_heads_per_layer=(1,) * 129,
            head_dim_per_layer=(16,) * 129,
        )

    with pytest.raises(ValueError, match="maximum is 256"):
        _configure_lifecycle(
            native,
            {layer_id: {"key": 32, "value": 32} for layer_id in range(129)},
        )


def test_unsupported_quant_is_rejected_before_manager_construction() -> None:
    config = SimpleNamespace(
        algorithm="quantization_for_cold_page",
        quant="future-format",
        scale_checkpoint_path="/not/a/checkpoint",
    )
    with pytest.raises(NotImplementedError, match="future-format"):
        util_mod.create_kv_cache_compression_manager(
            config,
            model_engine=_factory_model_engine(),
            kv_cache_config=SimpleNamespace(enable_block_reuse=False),
        )


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


def test_scale_checkpoint_requires_float32_reciprocals(tmp_path) -> None:
    smallest_subnormal = torch.tensor(1e-45, dtype=torch.float32).item()
    _write_scales(tmp_path, {7: (smallest_subnormal, smallest_subnormal)})
    with pytest.raises(ValueError, match="representable as float32"):
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
        layout = _layouts(native)[0]
        assert layout.layer_id == 1
        assert [buffer.scales.nvfp4_orig_quant for buffer in layout.buffers] == [
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
    assert _codec_state(native).layer_ids == ()


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
    layout = _layouts(native)[0]
    assert layout.layer_id == 0
    assert [buffer.role for buffer in layout.buffers] == ["key", "index_key"]
    assert [buffer.scales is not None for buffer in layout.buffers] == [True, False]
    assert _codec_state(native).runtime_type == 1
    assert layout.num_kv_heads == 1
    assert layout.tokens_per_page == 64
    assert layout.head_dim == 576
    scales = layout.buffers[0].scales
    assert scales.nvfp4_orig_quant == scales.nvfp4_quant_orig == 1.0
    assert layout.buffers[1].scales is None
    metadata = _configure_lifecycle(native, {0: {"key": 64 * 576 * 2, "index_key": 64 * 132}})
    assert metadata.cold_page_bytes == 29184
    assert metadata.wide[:2, 3].tolist() == [0, 20736]
    assert metadata.wide[:2, 4].tolist() == [18432, 0]


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

    layout = _layouts(native)[0]
    assert [buffer.role for buffer in layout.buffers] == [
        "key",
        "index_key",
        "rope_state",
    ]
    assert [buffer.scales is not None for buffer in layout.buffers] == [True, False, False]
    metadata = _configure_lifecycle(native, {0: {"key": 320, "index_key": 68, "rope_state": 7}})
    assert metadata.wide[:3, 3].tolist() == [0, 90, 158]
    assert metadata.wide[2, 5].item() == 165
    assert metadata.integers[2, 0].item() == 11
    assert metadata.cold_page_bytes == 176


def test_lossless_layout_uses_resolved_hot_buffer_bytes() -> None:
    native, _ = _native()
    cache_config = SimpleNamespace(
        tokens_per_block=4,
        layers=(
            AttentionLayerConfig(
                layer_id=0,
                buffers=[
                    BufferConfig(role="key", size=128),
                    BufferConfig(role="index_key", size=3),
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

    metadata = _configure_lifecycle(native, {0: {"key": 128, "index_key": 6}})
    assert metadata.wide[:2, 3].tolist() == [0, 36]
    assert metadata.wide[1, 5].item() == 42
    assert metadata.integers[1, 0].item() == 6
    assert metadata.cold_page_bytes == 48


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

    layouts = _layouts(native)
    assert len(layouts) == expected_layers
    assert sum(len(layout.buffers) for layout in layouts) == expected_buffers
    layer_bytes = {
        layer_id: {
            "key": 64 * 576 * 2,
            **({"index_key": 64 * (128 + 4)} if has_index else {}),
        }
        for layer_id, has_index in enumerate(owns_index)
    }
    assert _configure_lifecycle(native, layer_bytes).cold_page_bytes == expected_bytes


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

    layout = _layouts(native)[0]
    assert _codec_state(native).runtime_type == 2
    assert [buffer.scales.nvfp4_orig_quant for buffer in layout.buffers] == [
        2.0,
        4.0,
    ]
    assert [buffer.scales.nvfp4_quant_orig for buffer in layout.buffers] == [
        0.5,
        0.25,
    ]
    assert all(
        buffer.scales.fp8_orig_quant == buffer.scales.fp8_quant_orig == 1.0
        for buffer in layout.buffers
    )


def test_runtime_admission_is_checked_before_manager_creation(monkeypatch) -> None:
    monkeypatch.setattr(runtime_v2_mod, "_BACKEND", "python")
    with pytest.raises(ValueError, match=r"require.*C\+\+ KVCacheManagerV2"):
        _validate_compression()

    monkeypatch.setattr(runtime_v2_mod, "_BACKEND", "cpp")
    monkeypatch.setattr(util_mod, "is_sm_100f", lambda: False)
    with pytest.raises(RuntimeError, match="requires an SM100-family device"):
        util_mod.create_kv_cache_compression_manager(
            ColdPageQuantizationCompressionConfig(),
            model_engine=_factory_model_engine(),
            kv_cache_config=SimpleNamespace(enable_block_reuse=False),
        )

    monkeypatch.setattr(util_mod, "is_sm_100f", lambda: True)
    assert isinstance(
        util_mod.create_kv_cache_compression_manager(
            ColdPageQuantizationCompressionConfig(),
            model_engine=_factory_model_engine(),
            kv_cache_config=SimpleNamespace(enable_block_reuse=False),
        ),
        Nvfp4ColdPageQuantizationCompression,
    )


def test_speculative_admission_accepts_verified_one_model_modes(monkeypatch) -> None:
    monkeypatch.setattr(runtime_v2_mod, "_BACKEND", "cpp")

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


def test_qwen35_mtp3_resolves_to_supported_one_model_mode(monkeypatch) -> None:
    monkeypatch.setattr(runtime_v2_mod, "_BACKEND", "cpp")

    spec_config = MTPDecodingConfig(max_draft_len=3)
    update_spec_config_from_model_config(
        spec_config,
        SimpleNamespace(mtp_num_hidden_layers=1),
    )

    assert spec_config.spec_dec_mode is SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL
    assert spec_config.max_draft_len == 3
    with patch.object(util_mod, "is_sm_100f", return_value=True):
        util_mod.validate_kv_cache_compression_compatibility(
            ColdPageQuantizationCompressionConfig(),
            SimpleNamespace(enable_block_reuse=False),
            spec_config,
        )


def test_cold_manager_is_disabled_for_estimation_and_active_nvfp4(monkeypatch) -> None:
    monkeypatch.setattr(runtime_v2_mod, "_BACKEND", "cpp")
    monkeypatch.setattr(util_mod, "is_sm_100f", lambda: True)

    def build(
        *,
        estimating: bool = False,
        skip_est: bool = False,
        active_kv_quant: object | None = None,
    ) -> tuple[dict, util_mod.KvCacheCreator]:
        creator = object.__new__(util_mod.KvCacheCreator)
        creator._skip_est = skip_est
        creator._max_seq_len = 1024
        creator._kv_cache_config = SimpleNamespace(
            host_cache_size=None,
            disk_cache_size=None,
            enable_block_reuse=False,
        )
        creator._llm_args = SimpleNamespace(
            kv_cache_compression_config=ColdPageQuantizationCompressionConfig()
        )
        creator._model_engine = _factory_model_engine(active_kv_quant=active_kv_quant)
        creator._draft_model_engine = None
        creator._kv_connector_manager = None
        creator._fp8_ctx_mla_kv_len_cap = None
        creator._is_encoder_decoder = MagicMock(return_value=False)
        creator._should_create_separate_draft_kv_cache = MagicMock(return_value=False)
        creator._create_kv_cache_manager = MagicMock(return_value=SimpleNamespace())
        creator.configure_kv_cache_capacity = MagicMock()
        resources = {}
        creator.build_managers(resources, estimating_kv_cache=estimating)
        return resources, creator

    resources, creator = build()
    manager = creator._create_kv_cache_manager.call_args.kwargs["cold_page_codec_provider"]
    assert isinstance(manager, Nvfp4ColdPageQuantizationCompression)
    assert isinstance(manager, ColdPageQuantizationCompression)
    assert manager.provides_cold_page_codec
    assert not manager.uses_iteration_lifecycle
    assert util_mod.ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER not in resources
    _, estimation_creator = build(estimating=True)
    assert (
        estimation_creator._create_kv_cache_manager.call_args.kwargs["cold_page_codec_provider"]
        is None
    )
    _, skip_est_creator = build(estimating=True, skip_est=True)
    assert isinstance(
        skip_est_creator._create_kv_cache_manager.call_args.kwargs["cold_page_codec_provider"],
        Nvfp4ColdPageQuantizationCompression,
    )
    with patch.object(util_mod.logger, "info") as log:
        active_resources, active_creator = build(
            active_kv_quant=QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4)
        )
    assert util_mod.ResourceManagerType.KV_CACHE_COMPRESSION_MANAGER not in active_resources
    assert (
        active_creator._create_kv_cache_manager.call_args.kwargs["cold_page_codec_provider"] is None
    )
    log.assert_called_once()
