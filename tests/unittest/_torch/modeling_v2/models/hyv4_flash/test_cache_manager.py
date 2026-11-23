# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``Hyv4FlashCacheManager``'s pools, built through the engine's own factory on one GPU."""

import json

import pytest
import torch
from _hyv4_flash_config import SPARSE_LAYERS, config_dict  # noqa: E402

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.cache_manager import Hyv4FlashCacheManager
from tensorrt_llm._torch.pyexecutor._util import (
    _create_kv_cache_manager,
    get_kv_cache_manager_cls,
    model_declared_kv_cache_manager_cls,
)
from tensorrt_llm._torch.pyexecutor.config_utils import unwrap_hyv4_flash_text_config
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.model_loader import validate_and_set_kv_cache_quant
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo

TOKENS_PER_BLOCK = 32
MAX_SEQ_LEN = 2048
MAX_BATCH_SIZE = 4
MAX_NUM_TOKENS = 2048
#: The DSA indexer's fp32 scale covers this many key elements.
INDEX_SCALE_BLOCK = 128


class _Declares:
    runtime_hooks = RuntimeHooks(kv_cache_manager_cls=Hyv4FlashCacheManager)


@pytest.mark.parametrize(
    "kv_cache_quant_algo, kv_dtype",
    [(None, torch.bfloat16), (QuantAlgo.FP8, torch.float8_e4m3fn)],
    ids=["kv_bf16", "kv_fp8"],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the manager allocates on the GPU")
def test_pool_geometry(tmp_path, kv_cache_quant_algo, kv_dtype) -> None:
    """One 576-wide latent row per token on every layer; grouped indexer records on sparse ones."""
    (tmp_path / "config.json").write_text(json.dumps(config_dict()))
    mapping = Mapping(
        rank=0, gpus_per_node=4, world_size=4, tp_size=4, moe_tp_size=1, moe_ep_size=4
    )
    model_config = ModelConfig.from_pretrained(str(tmp_path), mapping=mapping)
    validate_and_set_kv_cache_quant(model_config, "auto" if kv_cache_quant_algo is None else "fp8")
    text = unwrap_hyv4_flash_text_config(model_config.pretrained_config)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False, max_tokens=4096, use_kv_cache_manager_v2=True
    )
    declared = model_declared_kv_cache_manager_cls(
        _Declares(), get_kv_cache_manager_cls(model_config, kv_cache_config)
    )
    latent_width = text.kv_lora_rank + text.qk_rope_head_dim
    element_size = torch.empty(0, dtype=kv_dtype).element_size()
    record = text.index_head_dim + text.index_head_dim // INDEX_SCALE_BLOCK * 4
    groups_per_page = TOKENS_PER_BLOCK // text.index_kpool

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
        assert type(cm) is Hyv4FlashCacheManager
        assert cm.max_attention_window_vec == [
            None if layer in SPARSE_LAYERS else text.sliding_window
            for layer in range(text.num_hidden_layers)
        ]
        for layer in range(text.num_hidden_layers):
            offset = cm.layer_offsets[layer]
            latent = cm.get_buffers(layer)
            assert latent.shape[1:] == (1, TOKENS_PER_BLOCK, 1, latent_width)
            assert latent.dtype == kv_dtype
            index_bytes = record // text.index_kpool if layer in SPARSE_LAYERS else 0
            assert cm.get_layer_bytes_per_token(offset, Role.ALL) == (
                latent_width * element_size + index_bytes
            )
            if layer not in SPARSE_LAYERS:
                assert cm.indexer_k_cache_pool_per_layer[offset] is None
                continue
            records = cm.get_indexer_k_cache_buffers(layer)
            assert records.shape == (latent.shape[0], groups_per_page, 1, record)
            assert records.dtype == torch.uint8
    finally:
        cm.shutdown()
