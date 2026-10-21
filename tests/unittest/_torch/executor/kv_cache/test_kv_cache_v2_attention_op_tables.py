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
"""Attention-op page tables of a KV cache whose life cycles mix page sizes.

A unified target/draft cache (FP8 target, BF16 draft sharing a sliding window)
cannot use one table per life cycle. The batch's rows of its per-layer tables
must match the old full-capacity rebuild bit for bit, and be the host rows at
enqueue time, however the host table changes before the asynchronous copy runs.
"""

from dataclasses import replace

import pytest
import torch

import tensorrt_llm._torch.utils as torch_utils
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
    _copy_swa_block_offsets_with_scratch_compiled,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

TOKENS_PER_BLOCK = 4
WINDOW = 2 * TOKENS_PER_BLOCK
MAX_SEQ_LEN = 64
SENTINEL = -7
# (request id, capacity, history length, dummy): one request far past the
# window, one inside it, and a CUDA-graph padding dummy.
REQUESTS = ((11, 40, 36, False), (12, 8, 0, False), (13, 4, 0, True))
UNSCHEDULED = (14, 24, 20, False)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _layer_specs():
    # Muse Spark-like: FP8 target layers (windowed and full) plus BF16 draft
    # layers with the target's window and twice its page size.
    target = KVCacheLayerSpec(
        global_layer_id=0,
        owner=KVCacheLayerOwner.TARGET,
        dtype=DataType.FP8,
        num_kv_heads=2,
        head_dim=16,
        buffer_roles=(Role.KEY, Role.VALUE),
        sliding_window_size=WINDOW,
    )
    specs = [
        replace(target, global_layer_id=i, sliding_window_size=None if i % 3 == 2 else WINDOW)
        for i in range(6)
    ]
    draft = replace(target, owner=KVCacheLayerOwner.DRAFT, dtype=DataType.BF16)
    return specs + [replace(draft, global_layer_id=6 + i) for i in range(2)]


def _make_manager() -> KVCacheManagerV2:
    torch.cuda.init()
    manager = KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=6,
        num_kv_heads=2,
        head_dim=16,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        layer_specs=_layer_specs(),
    )
    for request_id, capacity, history, dummy in REQUESTS + (UNSCHEDULED,):
        cache = manager._create_kv_cache(request_id, None, None, is_dummy=dummy)
        assert cache.resume(manager._stream.cuda_stream)
        assert cache.resize(capacity, history)
    return manager


def _full_rebuild(manager: KVCacheManagerV2, copy_idx: torch.Tensor) -> torch.Tensor:
    """The pre-change per-layer path: convert every IndexMapper row, keep the batch's."""
    params = []
    for layer_id in range(manager.num_local_layers):
        pool_id = manager.impl.get_layer_group_id(layer_id)
        converters = [
            manager.impl.get_page_index_converter(layer_id, role) for role in (Role.KEY, Role.VALUE)
        ]
        params.append(
            [[pool_id] * 2, [c.scale for c in converters], [c.layer_offset for c in converters]]
        )
    params = torch.tensor(params, dtype=torch.int32, device="cuda")  # [layers, 3, 2]
    host = manager.host_kv_cache_block_offsets
    num_pools, capacity, _, num_blocks = host.shape
    padded_idx = torch.zeros(capacity, dtype=torch.long, device="cuda")
    padded_idx[: copy_idx.numel()] = copy_idx.to(device="cuda", dtype=torch.long)
    no_scratch = torch.zeros(num_pools, capacity, dtype=torch.int32, device="cuda")
    out = torch.empty(params.shape[0], capacity, 2, num_blocks, dtype=torch.int32, device="cuda")
    _copy_swa_block_offsets_with_scratch_compiled(
        host.cuda(),
        padded_idx,
        params[:, 0].long(),
        params[:, 1],
        params[:, 2],
        torch.ones_like(params[:, 1]),
        torch.arange(num_blocks, dtype=torch.int32, device="cuda"),
        no_scratch,
        no_scratch,
        no_scratch[:, :, None],
        torch.zeros((), dtype=torch.int32, device="cuda"),
        out,
    )
    return out[:, : copy_idx.numel()]


def _copy_batch(manager: KVCacheManagerV2):
    request_ids = [request_id for request_id, *_ in REQUESTS]
    num_seqs = len(request_ids)
    dst = torch.full(
        (manager.num_attention_op_pools, num_seqs + 2, 2, manager.max_blocks_per_seq),
        SENTINEL,
        dtype=torch.int32,
        device="cuda",
    )
    manager.copy_batch_block_offsets(dst, request_ids, 1, 0, num_seqs)
    copy_idx = manager.index_mapper.get_copy_index(request_ids, 0, 1, False)
    torch.cuda.synchronize()
    # The other rows of the destination are not the batch's and stay untouched.
    assert (dst[:, num_seqs:] == SENTINEL).all()
    return dst[:, :num_seqs], copy_idx


def _check_windowed_out_pages(manager: KVCacheManagerV2, copy_idx: torch.Tensor) -> None:
    """The long request's windowed rows have BAD_PAGE_INDEX before their live tail."""
    window_pool = manager.impl.get_layer_group_id(0)
    row = manager.host_kv_cache_block_offsets[window_pool, int(copy_idx[0]), 0].tolist()
    last_valid = max(i for i, page in enumerate(row) if page != BAD_PAGE_INDEX)
    assert BAD_PAGE_INDEX in row[:last_valid]


@pytest.fixture(autouse=True)
def eager_full_rebuild(monkeypatch):
    # Run the reference rebuild uncompiled (maybe_compile honors this switch).
    monkeypatch.setattr(torch_utils, "is_piecewise_running", lambda: True)


def test_per_layer_tables_match_full_rebuild() -> None:
    manager = _make_manager()
    try:
        assert manager._use_per_layer_page_tables
        assert manager.num_attention_op_pools == manager.num_local_layers
        # No full-capacity staging without SWA scratch slots.
        assert not hasattr(manager, "_device_attention_op_block_offsets_staging")
        tables, copy_idx = _copy_batch(manager)
        _check_windowed_out_pages(manager, copy_idx)
        assert torch.equal(tables, _full_rebuild(manager, copy_idx))
    finally:
        manager.shutdown()


def test_copy_is_a_snapshot_of_the_host_table() -> None:
    """The next step's host-table writes must not reach this step's pending copy."""
    manager = _make_manager()
    try:
        expected, _ = _copy_batch(manager)
        request_ids = [request_id for request_id, *_ in REQUESTS]
        dst = torch.full_like(expected, SENTINEL)
        host = manager.host_kv_cache_block_offsets
        saved = host.clone()
        torch.cuda.synchronize()
        # Hold the stream so the copy is still pending when the host table changes.
        torch.cuda._sleep(1 << 30)
        manager.copy_batch_block_offsets(dst, request_ids, 1, 0, len(request_ids))
        host.fill_(7)
        torch.cuda.synchronize()
        host.copy_(saved)
        assert torch.equal(dst, expected)
    finally:
        manager.shutdown()
