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
"""The family KDA + DSA cache manager under DKV (CPU): the latent keeps one page per DKV
block on each rank, the index keys every page of it.

* ``_extra_buffers_per_layer`` gives INDEX_KEY a per-buffer page size of one physical page, which
  under DKV's blocks of ``size`` pages is a page expansion of ``size``.
* ``get_pool_block_indices`` lists every expanded index page of each DKV block,
  ``slot * scale * size + j``, as PageIndexConverter numbers them; ``indexer_k_cache_page_scale`` is 1.
* the index block table has ``size`` times the columns; a block-table read covers capacity in DKV
  blocks; a rank's per-token cost counts the whole DKV block's index keys.
* ``dkv_config`` (``TorchLlmArgs``, carried on the model config) reaches the manager's arguments.
"""

from types import SimpleNamespace

import pytest
import torch
from _kda_ops_util import bare

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm import (
    rubin_kda_dsa_cache_manager as dsa_cache_manager,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_dsa_cache_manager import (
    RubinKdaDsaHybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm.llmapi.llm_args import DkvConfig

pytestmark = pytest.mark.cpu_only

TPB = 64


def _manager(**attributes):
    defaults = dict(
        _dkv=object(),
        _kv_shard_size=4,
        _ledger_tokens_per_block=4 * TPB,
        tokens_per_block=TPB,
        index_head_dim=128,
        quant_block_size=128,
        indexer_k_dtype="fp8",
        _indexer_layer_ids=frozenset({1, 3}),
        layer_offsets={1: 1, 3: 3},
    )
    defaults.update(attributes)
    return bare(RubinKdaDsaHybridCacheManager, **defaults)


def test_index_keys_keep_every_page_of_a_dkv_block():
    buffers = _manager()._extra_buffers_per_layer(tokens_per_block=TPB)
    assert sorted(buffers) == [1, 3]
    (index,) = buffers[1]
    assert index.role == Role.INDEX_KEY and index.tokens_per_block_override == TPB
    (plain,) = _manager(_dkv=None)._extra_buffers_per_layer(tokens_per_block=TPB)[1]
    assert plain.tokens_per_block_override is None and plain.size == index.size


def test_block_indices_list_every_expanded_index_page():
    """DKV blocks of 4 pages, page scale 2: slot s holds index pages 8s .. 8s + 3."""
    host = torch.full((1, 3, 2, 8), -1, dtype=torch.int32)
    host[0, 0, 0, :3] = torch.tensor([5, 0, 7])
    host[0, 1, 0, :1] = torch.tensor([2])
    manager = _manager(
        host_kv_cache_block_offsets=host,
        _attention_pool_id=0,
        _num_attention_slots=16,
        _indexer_page_expansion=4,
        _indexer_slot_page_scale=2,
        index_mapper=SimpleNamespace(get_copy_index=lambda ids, nc, bw: torch.tensor([0, 1])),
        kv_cache_map={7: SimpleNamespace(capacity=3 * 4 * TPB), 8: SimpleNamespace(capacity=10)},
        max_blocks_per_seq=8,
    )
    pages = manager.get_pool_block_indices(2, request_ids=[7, 8])
    assert pages.dtype == torch.int32 and pages.shape == (2, 12)
    assert pages[0].tolist() == [40, 41, 42, 43, 0, 1, 2, 3, 56, 57, 58, 59]
    assert pages[1, :4].tolist() == [16, 17, 18, 19]
    assert manager.indexer_max_blocks_per_seq == 32
    # The block-offset table may be wider than max_blocks_per_seq (draft slots): every column of
    # it can come back expanded.
    manager.max_blocks_per_seq = 7
    assert manager.indexer_max_blocks_per_seq == 32


def test_a_rank_pays_for_every_index_key_of_its_dkv_blocks():
    manager = _manager(pp_layers=[0, 1, 2, 3], _fp8_mla_global_layer_ids=frozenset())
    plain = _manager(_dkv=None, pp_layers=[0, 1, 2, 3])
    assert manager.get_layer_bytes_per_token(
        1, Role.INDEX_KEY
    ) == 4 * plain.get_layer_bytes_per_token(1, Role.INDEX_KEY)
    assert manager.get_layer_bytes_per_token(0, Role.INDEX_KEY) == 0


def test_dkv_config_reaches_the_manager_arguments():
    """``page_split`` turns DKV on with its index layout; unset or ``replicated`` adds nothing."""
    model_config = SimpleNamespace(
        extra_attrs={},
        sparse_attention_config=None,
        pretrained_config=SimpleNamespace(),
        dkv_config=None,
    )
    dsa_cache_manager.bind_kv_cache_args(
        model_config,
        zero_gpu_pools=True,
        kv_dtype="fp8",
        indexer_k_dtype="fp8",
        state_dtype=torch.float32,
        fp8_kv_layers=None,
    )
    assert "dkv" not in RubinKdaDsaHybridCacheManager.extra_init_kwargs(model_config)
    model_config.dkv_config = DkvConfig(kv_layout="replicated")
    assert "dkv" not in RubinKdaDsaHybridCacheManager.extra_init_kwargs(model_config)
    model_config.dkv_config = DkvConfig()
    kwargs = RubinKdaDsaHybridCacheManager.extra_init_kwargs(model_config)
    assert (kwargs["dkv"], kwargs["dkv_index"], kwargs["dkv_draft"]) == (True, "sharded", True)
    # model_kwargs: {text_config: {kimi_k31_dkv_draft: false}} keeps the drafter history replicated.
    model_config.pretrained_config.kimi_k31_dkv_draft = False
    assert RubinKdaDsaHybridCacheManager.extra_init_kwargs(model_config)["dkv_draft"] is False
    model_config.dkv_config = DkvConfig(index_layout="whole")
    assert RubinKdaDsaHybridCacheManager.extra_init_kwargs(model_config)["dkv_index"] == "whole"


def test_sharded_index_keys_keep_one_page_per_dkv_block():
    """With sharded index keys the INDEX_KEY buffer has no page expansion and a rank pays for its
    own pages only, as for the latent."""
    sharded = _manager(_dkv_index_whole=False, pp_layers=[0, 1, 2, 3])
    (index,) = sharded._extra_buffers_per_layer(tokens_per_block=TPB)[1]
    assert index.tokens_per_block_override is None
    plain = _manager(_dkv=None, pp_layers=[0, 1, 2, 3])
    assert sharded.get_layer_bytes_per_token(1, Role.INDEX_KEY) == plain.get_layer_bytes_per_token(
        1, Role.INDEX_KEY
    )


def test_an_end_of_turn_snapshot_stays_exact_with_partial_reuse():
    """D-202's ``additional_snapshot_offsets_from_end: [0]`` asks for a snapshot at the prompt's end.
    With partial reuse (the default) it stays there under DKV: the match may end inside a
    DKV block. Without partial reuse it rounds down to whole DKV blocks, and a prompt shorter
    than one DKV block keeps none."""
    from tensorrt_llm.bindings import DataType

    unit = 4 * TPB
    exact = _manager(
        dtype=DataType.FP8, enable_partial_reuse=True, _attention_cache_layout_policy=None
    )
    request = SimpleNamespace(expect_snapshot_points=[3 * unit + 17, 5 * unit])
    exact._align_fp4_mla_snapshot_points(request)
    assert request.expect_snapshot_points == [3 * unit + 17, 5 * unit]

    whole = _manager(dtype=DataType.FP8, enable_partial_reuse=False)
    request = SimpleNamespace(expect_snapshot_points=[3 * unit + 17, 5 * unit])
    whole._align_fp4_mla_snapshot_points(request)
    assert request.expect_snapshot_points == [3 * unit, 5 * unit]
    short = SimpleNamespace(expect_snapshot_points=[unit - 1])
    whole._align_fp4_mla_snapshot_points(short)
    assert short.expect_snapshot_points == []
