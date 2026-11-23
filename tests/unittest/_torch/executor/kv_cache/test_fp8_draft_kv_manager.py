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
"""KVCacheManagerV2 with FP8 draft KV layers (``TRTLLM_DFLASH_FP8_DRAFT_KV``), on one GPU.

The drafter's FP8 K/V buffers are exactly the size of the target's FP8 ones, on the last stage
of a PP4 context server (32 KiB) and on a TP4 generation rank (8 KiB). Their coalescing group
keeps them in a pool of their own. Built from the same layer specs with BF16 and with FP8 draft
layers, the two managers must differ only in the draft pool's page size:

* the bindings take ``BufferConfig.coalescing_group`` (first: nothing below builds without it);
* the target layers' page strides, life cycles, page-table choice and attention-op pool
  mapping are the same;
* the draft layers keep their own pool, at half the BF16 page, with FP8 page views;
* the transfer page table gives the draft layers views of their own, with their head count.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.disaggregation import pp_dflash_context
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2 as kv_cache_v2_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig

_FP8_KNOB = "TRTLLM_DFLASH_FP8_DRAFT_KV"
TARGET_LAYERS = 62
DRAFT_LAYERS = 5
DRAFT_IDS = list(range(TARGET_LAYERS, TARGET_LAYERS + DRAFT_LAYERS))
PARTITION = [15, 16, 16, 15]
WINDOW = 2048
MAX_SEQ_LEN = 4096
TOKENS_PER_BLOCK = 32
MAX_DRAFT_LEN = 7
QUOTA_BYTES = 256 << 20
KIB = 1 << 10


@pytest.fixture(autouse=True)
def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.cuda.init()


class _OneProcessGroup:
    """Answers the managers' collectives as if every rank reported the same value."""

    local_world_size = 1

    @staticmethod
    def allreduce(value, op=None):
        return value

    @staticmethod
    def allgather(value):
        return [value]


@pytest.fixture(autouse=True)
def _fp8_draft_kv_requested(monkeypatch):
    # The FP8 draft layers keep a pool of their own only under the knob.
    monkeypatch.setenv(_FP8_KNOB, "1")


@pytest.fixture
def one_process(monkeypatch):
    monkeypatch.setattr(
        kv_cache_v2_module, "Distributed", SimpleNamespace(get=lambda mapping: _OneProcessGroup())
    )


def _is_full(layer: int) -> bool:
    return layer < TARGET_LAYERS and layer % 4 == 1


def _specs(target_layers, tp, draft_dtype):
    specs = [
        KVCacheLayerSpec(
            global_layer_id=layer,
            owner=KVCacheLayerOwner.TARGET,
            dtype=DataType.FP8,
            num_kv_heads=16 // tp,
            head_dim=64,
            buffer_roles=(Role.KEY, Role.VALUE),
            sliding_window_size=None if _is_full(layer) else WINDOW,
        )
        for layer in target_layers
    ]
    specs += [
        KVCacheLayerSpec(
            global_layer_id=layer,
            owner=KVCacheLayerOwner.DRAFT,
            dtype=draft_dtype,
            num_kv_heads=8 // tp,
            head_dim=128,
            buffer_roles=(Role.KEY, Role.VALUE),
            sliding_window_size=WINDOW,
        )
        for layer in DRAFT_IDS
    ]
    return specs


# (target layers, mapping, tp) of the context server's last stage and a generation rank.
_LAYOUTS = {
    "ctx_stage3": (
        list(range(sum(PARTITION[:3]), TARGET_LAYERS)),
        lambda: Mapping(world_size=4, rank=3, gpus_per_node=4, pp_size=4, pp_partition=PARTITION),
        1,
    ),
    "gen_tp4": (
        list(range(TARGET_LAYERS)),
        lambda: Mapping(world_size=4, rank=0, gpus_per_node=4, tp_size=4),
        4,
    ),
}
# Page bytes per slot of the sliding life cycle's pools: (target, BF16 draft, FP8 draft).
_SLIDING_PAGES = {
    "ctx_stage3": (704 * KIB, 640 * KIB, 320 * KIB),
    "gen_tp4": (736 * KIB, 160 * KIB, 80 * KIB),
}


def _manager(layout: str, draft_dtype) -> KVCacheManagerV2:
    target_layers, mapping, tp = _LAYOUTS[layout]
    return KVCacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=False,
            max_gpu_total_bytes=QUOTA_BYTES,
            host_cache_size=0,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=TARGET_LAYERS,
        num_kv_heads=16 // tp,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=2,
        mapping=mapping(),
        dtype=DataType.FP8,
        layer_specs=_specs(target_layers, tp, draft_dtype),
        spec_config=DFlashDecodingConfig(max_draft_len=MAX_DRAFT_LEN, speculative_model="draft"),
        vocab_size=4096,
        enable_stats=False,
    )


def test_bindings_take_a_coalescing_group():
    assert hasattr(BufferConfig, "coalescing_group")
    config = BufferConfig(role=Role.KEY, size=32 * KIB, coalescing_group=3)
    assert config.coalescing_group == 3


def _page_stride(manager, layer: int) -> int:
    return manager.impl.get_page_stride(manager.layer_offsets[layer], Role.KEY)


def _group(manager, layer: int):
    return manager.impl.get_layer_group_id(manager.layer_offsets[layer])


@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_fp8_draft_layers_change_only_the_draft_pool(monkeypatch, one_process, layout):
    # The context stage is a stage of a pipeline-parallel DFlash context server
    # (what its model construction records); the other layouts are not.
    monkeypatch.setattr(pp_dflash_context, "_pp_dflash_context_server", layout == "ctx_stage3")
    target_layers, _, tp = _LAYOUTS[layout]
    bf16 = _manager(layout, DataType.BF16)
    fp8 = _manager(layout, DataType.FP8)
    try:
        assert fp8._use_per_layer_page_tables == bf16._use_per_layer_page_tables
        assert fp8.num_attention_op_pools == bf16.num_attention_op_pools
        target_local = [bf16.layer_offsets[layer] for layer in target_layers]
        assert [fp8.layer_offsets[layer] for layer in target_layers] == target_local
        assert torch.equal(
            fp8.kv_cache_pool_mapping[target_local], bf16.kv_cache_pool_mapping[target_local]
        )
        for layer in target_layers + DRAFT_IDS:
            assert _group(fp8, layer) == _group(bf16, layer), layer
        for layer in target_layers:
            assert _page_stride(fp8, layer) == _page_stride(bf16, layer), layer
        target_page, bf16_draft_page, fp8_draft_page = _SLIDING_PAGES[layout]
        sliding = next(layer for layer in target_layers if not _is_full(layer))
        assert _page_stride(fp8, sliding) == target_page
        for layer in DRAFT_IDS:
            assert _page_stride(bf16, layer) == bf16_draft_page
            assert _page_stride(fp8, layer) == fp8_draft_page
            pages = fp8.get_buffers(layer, kv_layout="HND")
            assert pages.dtype == torch.float8_e4m3fn
            assert tuple(pages.shape[1:]) == (2, 8 // tp, TOKENS_PER_BLOCK, 128)
    finally:
        bf16.shutdown()
        fp8.shutdown()


@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_fp8_draft_layers_get_transfer_views_of_their_own(monkeypatch, one_process, layout):
    # The context stage is a stage of a pipeline-parallel DFlash context server
    # (what its model construction records); the other layouts are not.
    monkeypatch.setattr(pp_dflash_context, "_pp_dflash_context_server", layout == "ctx_stage3")
    _, _, tp = _LAYOUTS[layout]
    manager = _manager(layout, DataType.FP8)
    try:
        page_table = build_page_table_from_manager(manager)
        sliding = [group for group in page_table.layer_groups if group.sliding_window_size]
        assert len(sliding) == 1
        views = sliding[0].pool_views
        assert sorted(view.num_kv_heads for view in views) == sorted([16 // tp, 8 // tp])
        draft_view = next(view for view in views if view.num_kv_heads == 8 // tp)
        # K and V of one layer: 128 B per head per token each.
        assert draft_view.bytes_per_layer == 2 * (8 // tp) * 128 * TOKENS_PER_BLOCK
    finally:
        manager.shutdown()
