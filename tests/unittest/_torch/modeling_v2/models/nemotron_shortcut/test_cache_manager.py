# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.configs.nemotron_shortcut import NemotronShortcutTextConfig
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.nemotron_shortcut.cache_manager import NemotronShortcutCacheManager
from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager, get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.config_utils import extract_mamba_kv_cache_params
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig, KvCacheConfig, MTPDecodingConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo


def _config(layer_types=None):
    return NemotronShortcutTextConfig(
        layer_types=layer_types
        or ["linear_attention", "mlp", "shortcut_moe_attn", "shortcut_moe_gdp"],
        architectures=[NemotronShortcutTextConfig.causal_lm_architecture],
        dtype=torch.bfloat16,
        head_dim=64,
        index_topk=32,
        index_head_dim=32,
        gdp_num_heads=8,
        gdp_num_groups=8,
        gdp_num_householder=5,
        gdp_head_dim=16,
        gdp_key_head_dim=32,
        gdp_conv_kernel=3,
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "tp, attention_dp, local_heads, kv_algo, kv_bytes",
    [(4, False, 2, None, 256), (2, True, 8, None, 256), (4, False, 2, QuantAlgo.NVFP4, 72)],
)
def test_sparse_kv_replication_and_gdp_sharding(tp, attention_dp, local_heads, kv_algo, kv_bytes):
    mapping = Mapping(world_size=tp, tp_size=tp, enable_attention_dp=attention_dp)
    model = ModelConfig(
        pretrained_config=_config(),
        mapping=mapping,
        quant_config=QuantConfig(kv_cache_quant_algo=kv_algo),
    )
    params = extract_mamba_kv_cache_params(model.pretrained_config)
    assert model.get_num_attention_layers() == 1
    assert model.get_num_mamba_layers() == 2
    # Five (V, K) updates and a final query keep two BF16 convolution taps.
    conv_bytes = local_heads * (5 * (16 + 32) + 32) * 2 * torch.bfloat16.itemsize
    temporal_bytes = local_heads * 16 * 32 * torch.float32.itemsize
    assert params.get_states_bytes_per_layer(mapping) == conv_bytes + temporal_bytes
    assert NemotronShortcutCacheManager._estimate_attention_cache_cost(model, mapping) == (
        kv_bytes + 32 * torch.bfloat16.itemsize,
        0,
        0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA-backed cache pages")
def test_repeated_mtp_layer_owns_one_index_page_per_block():
    config = _config(["linear_attention", "shortcut_moe_attn", "shortcut_moe_attn"])
    mapping = Mapping()
    model = ModelConfig(pretrained_config=config, mapping=mapping)
    spec = MTPDecodingConfig(max_draft_len=2)
    spec.num_nextn_predict_layers = 1
    estimate = NemotronShortcutCacheManager._estimate_attention_cache_cost
    layer_bytes = (2 * config.head_dim + config.index_head_dim) * torch.bfloat16.itemsize
    assert (
        estimate(model, mapping, spec_config=spec, use_separate_draft_kv_cache=True)[0]
        == 2 * layer_bytes
    )
    assert estimate(model, mapping, spec_config=spec, is_draft=True)[0] == layer_bytes
    cache = KvCacheConfig(max_tokens=128, enable_block_reuse=False, use_kv_cache_manager_v2=True)
    manager = _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=NemotronShortcutCacheManager,
        mapping=mapping,
        kv_cache_config=cache,
        tokens_per_block=32,
        max_seq_len=64,
        max_batch_size=1,
        spec_config=spec,
        sparse_attention_config=None,
        max_num_tokens=64,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model,
        dtype=torch.bfloat16,
        is_draft=True,
        num_layers=1,
    )
    try:
        layer = config.num_hidden_layers
        assert manager.pp_layers == [layer]
        assert manager.mamba_pp_layers == []
        index = manager.get_index_k_buffer(layer, num_heads=1, head_dim=config.index_head_dim)
        assert index.dtype == torch.bfloat16
        assert index.shape[1:] == (32, 1, config.index_head_dim)
        assert manager.get_cache_bytes_per_token() == layer_bytes
    finally:
        manager.shutdown()


def _disagg_manager(model, cache, manager_cls, spec):
    return _create_kv_cache_manager(
        model_engine=None,
        kv_cache_manager_cls=manager_cls,
        mapping=model.mapping,
        kv_cache_config=cache,
        tokens_per_block=32,
        max_seq_len=64,
        max_batch_size=1,
        spec_config=spec,
        sparse_attention_config=None,
        max_num_tokens=64,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model,
        dtype=torch.bfloat16,
        is_disagg=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA-backed cache pages")
def test_nvfp4_data_and_scale_views_share_slot_indices_with_bf16_index():
    config = _config(["linear_attention", "shortcut_moe_attn", "shortcut_moe_attn"])
    model = ModelConfig(
        pretrained_config=config,
        mapping=Mapping(),
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4),
    )
    cache = KvCacheConfig(max_tokens=128, enable_block_reuse=False, use_kv_cache_manager_v2=True)
    manager = _disagg_manager(model, cache, NemotronShortcutCacheManager, None)
    try:
        planes = []
        for layer in (1, 2):
            data, scales = manager.get_nvfp4_kv_buffers(layer)
            index = manager.get_index_k_buffer(layer, head_dim=config.index_head_dim)
            assert data.shape[1:] == (2, 1, 32, 32)
            assert scales.shape == (*data.shape[:-1], 4)
            assert data.dtype == scales.dtype == torch.uint8
            assert index.shape == (data.shape[0], 32, 1, 32)
            assert index.dtype == torch.bfloat16
            planes.extend((data[:, 0], data[:, 1], scales[:, 0], scales[:, 1], index))

        # Two adjacent lifecycle slots must address distinct data, scales and index keys
        # even when both layers' pages are coalesced into the same physical pools.
        for plane_id, plane in enumerate(planes):
            for slot in range(2):
                plane[slot].fill_(2 * plane_id + slot)
        for plane_id, plane in enumerate(planes):
            for slot in range(2):
                assert (plane[slot] == 2 * plane_id + slot).all()

        # Each attention layer stores two packed 64-element rows, two sets of four
        # FP8 block scales, and a 32-element BF16 index key.
        expected_bytes = 2 * (2 * 32 + 2 * 4 + 32 * 2)
        assert manager.get_cache_bytes_per_token() == expected_bytes
        assert manager._estimate_attention_cache_cost(model, model.mapping) == (
            expected_bytes,
            0,
            0,
        )
        roles = {
            role
            for group in build_page_table_from_manager(manager).layer_groups
            for view in group.pool_views
            for role in view.pool_role
        }
        assert {"key", "value", "key_block_scale", "value_block_scale", "index_key"} <= roles
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA-backed cache pages")
@pytest.mark.parametrize("mtp", [False, True], ids=["plain", "mtp"])
def test_gdp_and_index_page_views(mtp):
    config = _config()
    model = ModelConfig(pretrained_config=config, mapping=Mapping())
    cache = KvCacheConfig(max_tokens=128, enable_block_reuse=False, use_kv_cache_manager_v2=True)
    manager_cls = get_kv_cache_manager_cls(
        model,
        cache,
        is_disagg=True,
        cache_transceiver_config=CacheTransceiverConfig(
            backend="NIXL", transceiver_runtime="PYTHON"
        ),
    )
    assert manager_cls is NemotronShortcutCacheManager
    spec = MTPDecodingConfig(max_draft_len=2) if mtp else None
    manager = _disagg_manager(model, cache, manager_cls, spec)
    try:
        assert manager.pp_layers == [0, 2, 3]
        assert manager.conv_section_dims == [128] * 5 + [256] * 6
        state = manager.mamba_layer_cache(3)
        assert state.conv.shape[1:] == (2176, 2)
        assert state.conv.dtype == torch.bfloat16
        assert state.temporal.shape[1:] == (8, 16, 32)
        assert state.temporal.dtype == torch.float32
        assert (getattr(state, "intermediate_ssm", None) is not None) == mtp
        index = manager.get_index_k_buffer(2, num_heads=1, head_dim=config.index_head_dim)
        assert index.shape[1:] == (32, 1, 32)
        assert index.dtype == torch.bfloat16
        assert manager.get_index_k_buffer(3, head_dim=config.index_head_dim) is None
        assert manager.get_cache_bytes_per_token() == (2 * 64 + 32) * torch.bfloat16.itemsize

        page_table = build_page_table_from_manager(manager)
        views = {
            role: view
            for group in page_table.layer_groups
            for view in group.pool_views
            for role in view.pool_role
        }
        assert views["key"].mapper_kind == views["value"].mapper_kind == MapperKind.INDEXED
        assert views["index_key"].mapper_kind == MapperKind.REPLICATED
        assert (
            views["index_key"].bytes_per_layer == 32 * config.index_head_dim * index.element_size()
        )
        assert views["conv_state"].section_bytes == [512] * 5 + [1024] * 6
        assert views["ssm_state"].bytes_per_head == 16 * 32 * state.temporal.element_size()
    finally:
        manager.shutdown()


@pytest.mark.cpu_only
def test_disagg_rejects_cpp_transport():
    model = ModelConfig(pretrained_config=_config(), mapping=Mapping())
    cache = KvCacheConfig(use_kv_cache_manager_v2=True)
    transceiver = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="CPP")
    with pytest.raises(ValueError, match="transceiver_runtime='PYTHON'"):
        get_kv_cache_manager_cls(model, cache, is_disagg=True, cache_transceiver_config=transceiver)
