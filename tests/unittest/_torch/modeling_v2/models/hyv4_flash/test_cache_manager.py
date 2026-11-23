# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV cache manager this target declares, at tp4, tep4, dep4 and tep8.

HY4-Flash runs gated MLA over one 576-wide latent (``kv_lora_rank + qk_rope_head_dim``) on all 42
layers and the DSA indexer on 12 of them, so the engine resolves ``DSACacheManagerV2``; the target
declares ``Hyv4FlashCacheManager``, its subclass, through ``RuntimeHooks.kv_cache_manager_cls``, and
the engine builds that instead. These tests take the engine's own path --
``get_kv_cache_manager_cls``, ``model_declared_kv_cache_manager_cls`` and
``_create_kv_cache_manager`` over the checkpoint's ``ModelConfig`` -- on the first rank of each
member, one GPU, with a bf16 and an FP8 KV cache and block reuse off and on, and hold every pool to
the geometry the checkpoint and the engine arguments derive:

* the latent: one page of ``tokens_per_block`` tokens per layer, ``num_kv_heads`` = 1 of
  ``kv_lora_rank + qk_rope_head_dim`` wide, so a page is ``tokens_per_block * 576`` elements of the
  KV dtype and a layer's pages step by that; the engine coalesces the layers that share a window
  into one slot space, so a page index is ``slot * group_size + position_in_group`` and the group
  size is the number of layers with that window -- 12 sparse, 30 windowed;
* the indexer: on the 12 sparse layers only, one record per ``index_kpool`` tokens -- 128
  quantized bytes and a fp32 scale -- so a page holds ``tokens_per_block // index_kpool`` records
  of 132 bytes, on the same page index as the latent;
* the windows: full context (``None``) on the sparse layers and ``sliding_window`` on the 30
  windowed ones, which the engine derives from the checkpoint's ``layer_types``.
"""

import contextlib
import json

import pytest
import torch
from _hyv4_flash_config import SPARSE_LAYERS, config_dict  # noqa: E402

from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.cache_manager import (  # noqa: E501
    Hyv4FlashCacheManager,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.routing import (
    layout,
)
from tensorrt_llm._torch.pyexecutor._util import (
    _create_kv_cache_manager,
    get_kv_cache_manager_cls,
    model_declared_kv_cache_manager_cls,
)
from tensorrt_llm._torch.pyexecutor.config_utils import unwrap_hyv4_flash_text_config
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2, Role
from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_kv_cache_quant
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo

TOKENS_PER_BLOCK = 32  # `kv_cache_config.tokens_per_block`'s default, which the runtime keeps
MAX_TOKENS = 4096
MAX_SEQ_LEN = 2048
MAX_BATCH_SIZE = 4
MAX_NUM_TOKENS = 2048

#: The DSA indexer quantizes its key in blocks of this many elements, one fp32 scale each.
INDEX_SCALE_BLOCK = 128

#: Each member's mapping: attention TP4 with the experts split by intermediate (tp4) or by expert
#: (tep4), attention DP with expert parallelism (dep4), and attention TP8 across two nodes (tep8).
MEMBERS = {
    "tp4": dict(world_size=4, tp_size=4, moe_tp_size=4, moe_ep_size=1, enable_attention_dp=False),
    "tep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=False),
    "dep4": dict(world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4, enable_attention_dp=True),
    "tep8": dict(world_size=8, tp_size=8, moe_tp_size=1, moe_ep_size=8, enable_attention_dp=False),
}

#: The latent pool's storage per resolved KV-cache algorithm: the torch dtype the pool view holds
#: and the manager's binding dtype.
KV_STORAGE = {
    None: ("kv_bf16", torch.bfloat16, DataType.BF16),
    QuantAlgo.FP8: ("kv_fp8", torch.float8_e4m3fn, DataType.FP8),
}

REUSE_IDS = {False: "reuse_off", True: "reuse_on"}

_needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the manager allocates its pools on the GPU"
)


def _mapping(member: str, rank: int = 0) -> Mapping:
    return Mapping(rank=rank, gpus_per_node=4, pp_size=1, **MEMBERS[member])


def _write_checkpoint(directory, config: dict) -> str:
    (directory / "config.json").write_text(json.dumps(config))
    return str(directory)


def _model_config(checkpoint: str, member: str, kv_cache_quant_algo=None) -> ModelConfig:
    """The checkpoint's model config, with the KV-cache algorithm the deployment asks for."""
    model_config = ModelConfig.from_pretrained(checkpoint, mapping=_mapping(member))
    validate_and_set_kv_cache_quant(model_config, "auto" if kv_cache_quant_algo is None else "fp8")
    assert model_config.quant_config.kv_cache_quant_algo == kv_cache_quant_algo
    return model_config


def _kv_cache_config(reuse: bool) -> KvCacheConfig:
    return KvCacheConfig(
        enable_block_reuse=reuse, max_tokens=MAX_TOKENS, use_kv_cache_manager_v2=True
    )


class _Declares:
    """A model that declares the family's manager, as the target's outer class does."""

    runtime_hooks = RuntimeHooks(kv_cache_manager_cls=Hyv4FlashCacheManager)


@contextlib.contextmanager
def _manager(checkpoint: str, member: str, kv_cache_quant_algo=None, reuse: bool = False):
    """The trunk manager the engine builds on rank 0 of `member`, through the engine's own class
    choice, the model's declaration and the engine's factory."""
    mapping = _mapping(member)
    model_config = _model_config(checkpoint, member, kv_cache_quant_algo)
    kv_cache_config = _kv_cache_config(reuse)
    resolved = get_kv_cache_manager_cls(model_config, kv_cache_config)
    declared = model_declared_kv_cache_manager_cls(_Declares(), resolved)
    cm = _create_kv_cache_manager(
        None,
        declared,
        mapping,
        kv_cache_config,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        spec_config=None,
        sparse_attention_config=model_config.sparse_attention_config,
        max_num_tokens=MAX_NUM_TOKENS,
        max_beam_width=1,
        kv_connector_manager=None,
        model_config=model_config,
        dtype=torch.bfloat16,
        max_cuda_graph_batch_size=MAX_BATCH_SIZE,
    )
    try:
        yield cm
    finally:
        cm.shutdown()


def _groups(text) -> dict:
    """The layers the engine coalesces into one slot space, by window, in layer order.

    A layer's life cycle is its window, so the sparse layers (full context) and the windowed ones
    are two pools, and the engine gives every layer of a pool the same page scale: the pool's layer
    count, with the layer's position in the pool as its page offset.
    """
    groups = {}
    for layer in range(text.num_hidden_layers):
        window = None if layer in SPARSE_LAYERS else text.sliding_window
        groups.setdefault(window, []).append(layer)
    return groups


@pytest.mark.parametrize("member", list(MEMBERS), ids=list(MEMBERS))
def test_the_engine_resolves_the_dsa_v2_manager_and_accepts_the_declaration(
    tmp_path, member
) -> None:
    """The engine resolves the DSA V2 manager for this checkpoint, and the target's subclass of
    it is what `RuntimeHooks.kv_cache_manager_cls` hands back; anything else is refused."""
    mapping = _mapping(member)
    assert layout(mapping) == member
    model_config = _model_config(_write_checkpoint(tmp_path, config_dict()), member)
    resolved = get_kv_cache_manager_cls(model_config, _kv_cache_config(reuse=False))

    assert resolved is DSACacheManagerV2
    assert issubclass(Hyv4FlashCacheManager, resolved)
    assert issubclass(Hyv4FlashCacheManager, KVCacheManagerV2), "a V2 manager"
    assert model_declared_kv_cache_manager_cls(_Declares(), resolved) is Hyv4FlashCacheManager

    class _DeclaresAnother:
        runtime_hooks = RuntimeHooks(kv_cache_manager_cls=object)

    with pytest.raises(ValueError, match="not a subclass"):
        model_declared_kv_cache_manager_cls(_DeclaresAnother(), resolved)
    # A model that declares nothing keeps the class the engine resolved.
    assert model_declared_kv_cache_manager_cls(object(), resolved) is resolved


@pytest.mark.parametrize("reuse", list(REUSE_IDS), ids=list(REUSE_IDS.values()))
@pytest.mark.parametrize(
    "kv_cache_quant_algo", list(KV_STORAGE), ids=[v[0] for v in KV_STORAGE.values()]
)
@pytest.mark.parametrize("member", list(MEMBERS), ids=list(MEMBERS))
@_needs_gpu
def test_every_pool_is_the_derived_geometry(tmp_path, member, kv_cache_quant_algo, reuse) -> None:
    """The latent, the indexer pool and the windows are what the checkpoint and the engine
    arguments derive, on rank 0 of every member, under both KV dtypes and with block reuse off
    and on."""
    _, kv_dtype, binding_dtype = KV_STORAGE[kv_cache_quant_algo]
    element_size = torch.empty(0, dtype=kv_dtype).element_size()
    checkpoint = _write_checkpoint(tmp_path, config_dict())
    text = unwrap_hyv4_flash_text_config(
        ModelConfig.from_pretrained(checkpoint, mapping=_mapping(member)).pretrained_config
    )
    num_layers = text.num_hidden_layers
    latent_width = text.kv_lora_rank + text.qk_rope_head_dim
    page_elements = TOKENS_PER_BLOCK * latent_width
    page_bytes = page_elements * element_size
    indexer_record = text.index_head_dim + text.index_head_dim // INDEX_SCALE_BLOCK * 4
    groups_per_page = TOKENS_PER_BLOCK // text.index_kpool
    indexer_bytes_per_token = indexer_record // text.index_kpool
    groups = _groups(text)

    with _manager(checkpoint, member, kv_cache_quant_algo, reuse) as cm:
        assert type(cm) is Hyv4FlashCacheManager
        assert cm.dtype == binding_dtype
        assert cm.kv_factor == 1, "SELFKONLY: one latent per token, no separate value"
        # The engine derives the per-layer windows from the checkpoint's layer_types; block
        # reuse adds no backoff without speculative decoding, so it leaves them alone.
        assert cm.max_attention_window_vec == [
            None if layer in SPARSE_LAYERS else text.sliding_window for layer in range(num_layers)
        ]
        # One latent head of `latent_width` on every layer, on every member: the MLA latent is
        # never head-split, and attention DP uses the whole TP group as its cache rank.
        assert set(cm.num_kv_heads_per_layer) == {1}
        assert set(cm.head_dim_per_layer) == {latent_width}
        assert cm.num_local_layers == num_layers

        # The indexer geometry, read from the checkpoint.
        assert cm.index_kpool == text.index_kpool
        assert cm.indexer_record_bytes == indexer_record
        assert cm.indexer_data_bytes == text.index_head_dim
        assert cm.indexer_bytes_per_token == indexer_bytes_per_token
        assert cm.indexer_k_cache_local_layer_mask == [
            layer in SPARSE_LAYERS for layer in range(num_layers)
        ]

        # The pools' page counts: the engine's per-rank number for the primary (full-context)
        # pool, and the same slot space for every layer of a window group.
        slots = {None: cm.num_blocks}
        latent = {}
        for layer in range(num_layers):
            offset = cm.layer_offsets[layer]
            window = None if layer in SPARSE_LAYERS else text.sliding_window
            scale = len(groups[window])
            position = groups[window].index(layer)
            assert cm.get_layer_page_index_scale(layer) == scale, (
                "a layer's page index steps by the pool's layer count"
            )
            view = cm.get_buffers(layer)
            latent[layer] = view
            if window not in slots:
                # The first layer of a pool sees every slot of the engine's allocation for that
                # window; every later layer of the group must agree on that slot count.
                assert position == 0
                assert view.shape[0] % scale == 0
                slots[window] = view.shape[0] // scale
            assert view.shape == (
                scale * slots[window] - position,
                1,
                TOKENS_PER_BLOCK,
                1,
                latent_width,
            )
            assert view.dtype == kv_dtype
            assert view.is_contiguous()
            assert view.stride() == (page_elements, page_elements, latent_width, latent_width, 1)
            assert cm.get_layer_bytes_per_token(offset, Role.KEY) == latent_width * element_size
            assert cm.get_layer_bytes_per_token(offset, Role.ALL) == latent_width * element_size + (
                indexer_bytes_per_token if layer in SPARSE_LAYERS else 0
            )

        # Within a pool the layers' pages are contiguous in layer order, and a slot holds one
        # page per layer.
        for members_of_window in groups.values():
            for before, after in zip(members_of_window, members_of_window[1:]):
                assert latent[after].data_ptr() - latent[before].data_ptr() == page_bytes, (
                    "a pool's pages follow layer order"
                )

        # The indexer K cache: one record per `index_kpool` tokens on the same page index as the
        # latent, and only on the sparse layers.
        assert cm.indexer_k_cache_page_scale == len(groups[None])
        for layer in range(num_layers):
            offset = cm.layer_offsets[layer]
            assert cm.get_layer_bytes_per_token(offset, Role.INDEX_KEY) == indexer_bytes_per_token
            if layer not in SPARSE_LAYERS:
                assert cm.indexer_k_cache_pool_per_layer[offset] is None, (
                    "a windowed layer runs no indexer"
                )
                continue
            records = cm.get_indexer_k_cache_buffers(layer)
            assert records.shape == (latent[layer].shape[0], groups_per_page, 1, indexer_record)
            assert records.dtype == torch.uint8
            assert records.is_contiguous()
            assert records.shape[1] * records.shape[3] == TOKENS_PER_BLOCK * (
                indexer_bytes_per_token
            )
            plane = cm.get_indexer_k_cache_index_plane(layer)
            assert plane.shape == (records.shape[0], groups_per_page, 1, text.index_head_dim)
            assert plane.dtype == torch.uint8
            assert plane.data_ptr() == records.data_ptr()
            assert plane.stride() == (
                groups_per_page * indexer_record,
                indexer_record,
                text.index_head_dim,
                1,
            )
            assert plane[0, 1, 0, 0].data_ptr() - plane[0, 0, 0, 0].data_ptr() == indexer_record, (
                "a group key starts after the previous record's data and scale"
            )
            scales = cm.get_indexer_k_cache_scale_plane(layer)
            assert scales.shape == (records.shape[0], groups_per_page, 1, 1)
            assert scales.dtype == torch.float32
            assert scales.data_ptr() - records.data_ptr() == text.index_head_dim
            assert scales.stride() == (
                groups_per_page * (indexer_record // 4),
                indexer_record // 4,
                indexer_record // 4,
                1,
            )


@pytest.mark.parametrize("member", list(MEMBERS), ids=list(MEMBERS))
def test_a_page_that_does_not_hold_whole_groups_is_refused(tmp_path, member) -> None:
    """The grouped pool cannot address a page that cuts a group key in half, so the manager
    refuses one before it allocates anything."""
    checkpoint = _write_checkpoint(tmp_path, config_dict())
    model_config = _model_config(checkpoint, member)
    mapping = _mapping(member)
    kv_cache_config = _kv_cache_config(reuse=False)
    declared = model_declared_kv_cache_manager_cls(
        _Declares(), get_kv_cache_manager_cls(model_config, kv_cache_config)
    )
    with pytest.raises(ValueError, match="whole groups"):
        _create_kv_cache_manager(
            None,
            declared,
            mapping,
            kv_cache_config,
            tokens_per_block=text_index_kpool(config_dict()) + 2,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=MAX_BATCH_SIZE,
            spec_config=None,
            sparse_attention_config=model_config.sparse_attention_config,
            max_num_tokens=MAX_NUM_TOKENS,
            max_beam_width=1,
            kv_connector_manager=None,
            model_config=model_config,
            dtype=torch.bfloat16,
            max_cuda_graph_batch_size=MAX_BATCH_SIZE,
        )


def text_index_kpool(config: dict) -> int:
    """The checkpoint's indexer group width, read the way the config class reads it."""
    return config["text_config"]["compressed_indexer_compress_ratio"]


def test_the_manager_needs_the_checkpoints_indexer_group_width() -> None:
    """Without `index_kpool` (or with a non-positive one) the group geometry is unknown, and the
    manager says so instead of sizing the pool from a guess."""
    from transformers import PretrainedConfig

    with pytest.raises(ValueError, match="index_kpool"):
        Hyv4FlashCacheManager(None, None, pretrained_config=PretrainedConfig())
    with pytest.raises(ValueError, match="index_kpool"):
        Hyv4FlashCacheManager(None, None, pretrained_config=PretrainedConfig(index_kpool=0))
