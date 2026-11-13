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
"""Layout of the NVFP4 DSA KV cache: ``fp8rope`` records where PrimTS reads them in place."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import prims_ts_dsa_sparse_mla
from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import (
    NVFP4_MLA_KV_CACHE_LAYOUT_ENV,
    NVFP4_MLA_RECORD_LAYOUT,
    NVFP4_MLA_RESIDUAL_LAYOUT,
    DSACacheManagerV2,
    _get_nvfp4_mla_record_unsupported_reason,
    resolve_nvfp4_mla_kv_cache_layout,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 12),
    reason="NVFP4 KV caches require Blackwell",
)

_NUM_LAYERS = 2
_TOKENS_PER_BLOCK = 64
_RECORD_BYTES = 352
_INDEXER_BYTES = 128 + 4


def _make_manager(sparse_config):
    pretrained_config = SimpleNamespace(
        num_hidden_layers=_NUM_LAYERS, kv_lora_rank=512, qk_rope_head_dim=64
    )
    return DSACacheManagerV2(
        KvCacheConfig(max_tokens=4096, enable_block_reuse=False),
        CacheType.SELFKONLY,
        num_layers=_NUM_LAYERS,
        num_kv_heads=1,
        head_dim=576,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=1024,
        max_batch_size=2,
        mapping=Mapping(),
        dtype=DataType.NVFP4,
        sparse_attention_config=sparse_config,
        pretrained_config=pretrained_config,
    )


def _request_layout(monkeypatch, layout):
    """Set ``TRTLLM_NVFP4_MLA_KV_CACHE_LAYOUT``; None leaves it unset, which asks for ``auto``."""
    if layout is None:
        monkeypatch.delenv(NVFP4_MLA_KV_CACHE_LAYOUT_ENV, raising=False)
    else:
        monkeypatch.setenv(NVFP4_MLA_KV_CACHE_LAYOUT_ENV, layout)


def _set_record_reader(monkeypatch, unsupported_reason):
    """Make the PrimTS record reader of this process available (None) or unavailable."""
    monkeypatch.setattr(
        prims_ts_dsa_sparse_mla,
        "get_nvfp4_mla_record_reader_unsupported_reason",
        lambda mapping=None: unsupported_reason,
    )


@pytest.mark.parametrize(
    "layout, reader_unsupported_reason, expected",
    [
        (None, None, NVFP4_MLA_RECORD_LAYOUT),
        ("auto", None, NVFP4_MLA_RECORD_LAYOUT),
        (None, "no reader.", NVFP4_MLA_RESIDUAL_LAYOUT),
        ("auto", "no reader.", NVFP4_MLA_RESIDUAL_LAYOUT),
        ("residual", None, NVFP4_MLA_RESIDUAL_LAYOUT),
        ("fp8rope", None, NVFP4_MLA_RECORD_LAYOUT),
        # An explicit request for records is kept; generation then gathers them.
        ("fp8rope", "no reader.", NVFP4_MLA_RECORD_LAYOUT),
    ],
)
def test_record_layout_has_no_scale_pool(monkeypatch, layout, reader_unsupported_reason, expected):
    _request_layout(monkeypatch, layout)
    _set_record_reader(monkeypatch, reader_unsupported_reason)
    sparse_config = DeepSeekSparseAttentionConfig(index_head_dim=128)
    manager = _make_manager(sparse_config)
    try:
        roles = manager._get_layer_roles_by_local_id(0)
        pointers = manager.kv_cache_pool_pointers
        if expected == NVFP4_MLA_RESIDUAL_LAYOUT:
            assert manager.mla_kv_cache_layout == NVFP4_MLA_RESIDUAL_LAYOUT
            assert manager.mla_kv_cache_residual_dim == 64
            assert Role.KEY_BLOCK_SCALE in roles
            assert manager.get_layer_bytes_per_token(0, Role.KEY) == 320
            assert manager.get_layer_bytes_per_token(0, Role.KEY_BLOCK_SCALE) == 40
            assert int(pointers[0, 0, 1]) != 0
            latent_bytes = 360
        else:
            assert manager.mla_kv_cache_layout == NVFP4_MLA_RECORD_LAYOUT
            assert manager.mla_kv_cache_residual_dim == 128
            assert Role.KEY_BLOCK_SCALE not in roles
            assert manager.get_layer_bytes_per_token(0, Role.KEY) == _RECORD_BYTES
            assert int(pointers[0, 0, 1]) == 0
            latent_bytes = _RECORD_BYTES
            pool = manager.get_unique_primary_pool()
            assert pool.numel() == pool.shape[0] * _NUM_LAYERS * _TOKENS_PER_BLOCK * _RECORD_BYTES
        assert manager.get_layer_bytes_per_token(0, Role.ALL) == latent_bytes + _INDEXER_BYTES
        assert manager.get_cache_bytes_per_token() == _NUM_LAYERS * (latent_bytes + _INDEXER_BYTES)

        model_config = SimpleNamespace(
            pretrained_config=SimpleNamespace(
                num_hidden_layers=_NUM_LAYERS, kv_lora_rank=512, qk_rope_head_dim=64
            ),
            sparse_attention_config=sparse_config,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4),
        )
        estimate = DSACacheManagerV2.get_cache_size_per_token(
            model_config, Mapping(), num_layers=_NUM_LAYERS
        )
        assert estimate == manager.get_cache_bytes_per_token()
    finally:
        manager.shutdown()


def test_record_layout_rejects_kv_cache_offload(monkeypatch):
    """The offload arena reads NVFP4 latents with a block-scale pool, which records lack."""
    monkeypatch.setenv(NVFP4_MLA_KV_CACHE_LAYOUT_ENV, "fp8rope")
    sparse_config = DeepSeekSparseAttentionConfig(index_head_dim=128, enable_kv_cache_offload=True)
    with pytest.raises(ValueError, match="KV cache offload"):
        _make_manager(sparse_config)


@pytest.mark.parametrize(
    "head_dim, rope_head_dim, kv_cache_config, kv_cache_offload, unsupported",
    [
        (576, 64, KvCacheConfig(), False, None),
        (320, 64, KvCacheConfig(), False, "512 \\+ 64 latent"),
        (576, 64, SimpleNamespace(mixed_precision_kv=object()), False, "mixed-precision"),
        (576, 64, KvCacheConfig(), True, "KV cache offload"),
    ],
)
def test_cache_that_cannot_hold_records_keeps_the_residual_layout(
    monkeypatch, head_dim, rope_head_dim, kv_cache_config, kv_cache_offload, unsupported
):
    reason = _get_nvfp4_mla_record_unsupported_reason(
        head_dim, rope_head_dim, kv_cache_config, kv_cache_offload
    )
    assert (reason is None) == (unsupported is None)
    _set_record_reader(monkeypatch, None)

    for layout in (None, "auto"):
        _request_layout(monkeypatch, layout)
        expected = NVFP4_MLA_RECORD_LAYOUT if unsupported is None else NVFP4_MLA_RESIDUAL_LAYOUT
        assert resolve_nvfp4_mla_kv_cache_layout(reason) == expected

    _request_layout(monkeypatch, "residual")
    assert resolve_nvfp4_mla_kv_cache_layout(reason) == NVFP4_MLA_RESIDUAL_LAYOUT

    # An explicit request for records is not overridden.
    _request_layout(monkeypatch, "fp8rope")
    if unsupported is None:
        assert resolve_nvfp4_mla_kv_cache_layout(reason) == NVFP4_MLA_RECORD_LAYOUT
    else:
        with pytest.raises(ValueError, match=unsupported):
            resolve_nvfp4_mla_kv_cache_layout(reason)


@pytest.mark.parametrize("layout", [None, "auto"])
def test_auto_stores_records_only_where_they_are_read_in_place(monkeypatch, layout):
    _request_layout(monkeypatch, layout)
    seen = []

    def reader_unsupported_reason(mapping=None):
        seen.append(mapping)
        return "requires SM100, SM103 or SM107, got SM120."

    monkeypatch.setattr(
        prims_ts_dsa_sparse_mla,
        "get_nvfp4_mla_record_reader_unsupported_reason",
        reader_unsupported_reason,
    )
    mapping = Mapping()
    assert resolve_nvfp4_mla_kv_cache_layout(None, mapping) == NVFP4_MLA_RESIDUAL_LAYOUT
    assert seen == [mapping]
    _set_record_reader(monkeypatch, None)
    assert resolve_nvfp4_mla_kv_cache_layout(None, mapping) == NVFP4_MLA_RECORD_LAYOUT


@pytest.mark.parametrize("layout", ["residual", "fp8rope"])
def test_explicit_layout_does_not_ask_for_the_record_reader(monkeypatch, layout):
    _request_layout(monkeypatch, layout)

    def reader_unsupported_reason(mapping=None):
        raise AssertionError("an explicit layout does not depend on the record reader")

    monkeypatch.setattr(
        prims_ts_dsa_sparse_mla,
        "get_nvfp4_mla_record_reader_unsupported_reason",
        reader_unsupported_reason,
    )
    assert resolve_nvfp4_mla_kv_cache_layout(None, Mapping()) == layout


def test_unknown_layout_request_is_rejected(monkeypatch):
    _request_layout(monkeypatch, "records")
    with pytest.raises(ValueError, match=NVFP4_MLA_KV_CACHE_LAYOUT_ENV):
        resolve_nvfp4_mla_kv_cache_layout(None)
