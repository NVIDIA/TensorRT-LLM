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
"""The step-input stager's block-offset copy over a target/draft ``KVCacheManagerV2``.

A unified cache whose life cycles mix page sizes (FP8 target, BF16 draft) has one
attention-op table per pool group. The stager's copy of the batch's rows, full or
incremental, must leave the device table bit for bit as the manager's own copy leaves
it: the rows' every column, since a drafter in the manager reads a whole row for its
extent, and nothing outside the batch's rows.
"""

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep import (  # noqa: E501
    runtime,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.runtime import (  # noqa: E501
    step_commit,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
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
# (request id, capacity, history length, dummy): one request far past the window, two
# inside it, a CUDA-graph padding dummy, and a mid-length one.
REQUESTS = ((11, 40, 36, False), (12, 8, 0, False), (13, 4, 0, True), (14, 24, 20, False))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _layer_specs():
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


@pytest.fixture
def manager():
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
    for request_id, capacity, history, dummy in REQUESTS:
        cache = manager._create_kv_cache(request_id, None, None, is_dummy=dummy)
        assert cache.resume(manager._stream.cuda_stream)
        assert cache.resize(capacity, history)
    # (window, FP8 pages), (window, BF16 pages), (full, FP8 pages).
    assert manager._attention_op_pool_groups is not None
    assert manager.num_attention_op_pools == 3
    try:
        yield manager
    finally:
        manager.shutdown()


def _table(manager):
    return torch.full(
        (manager.num_attention_op_pools, 6, 2, manager.max_blocks_per_seq),
        SENTINEL,
        dtype=torch.int32,
        device="cuda",
    )


def _native(manager, request_ids):
    dst = _table(manager)
    manager.copy_batch_block_offsets(dst, request_ids, 1, 0, len(request_ids))
    torch.cuda.synchronize()
    return dst


def _staged(manager, dst, request_ids, memo, attn_metadata):
    stager = runtime._PrepareStager(
        step_commit.StepStaging(),
        Counter(),
        SimpleNamespace(),
        attn_metadata,
        None,
        blocks_memo=memo,
    )
    assert stager.block_offsets(manager, dst, request_ids, 1, 0, len(request_ids)), stager.declined
    # A step whose rows all match what the memo recorded has nothing to launch.
    stager.staging.launch()
    torch.cuda.synchronize()
    return stager


@pytest.mark.parametrize("incremental", [False, True], ids=["full", "incremental"])
def test_staged_copy_is_the_native_copy(manager, incremental):
    request_ids = [11, 12, 13]
    want = _native(manager, request_ids)
    dst = _table(manager)
    memo = runtime._BlockOffsetsMemo() if incremental else None
    stager = _staged(manager, dst, request_ids, memo, SimpleNamespace())
    cols = stager.block_cols
    assert cols < manager.max_blocks_per_seq
    # The long request's windowed tables carry BAD_PAGE_INDEX inside the copied columns.
    assert (want[:, 0, :, :cols] == BAD_PAGE_INDEX).any()
    assert torch.equal(dst, want)


def test_incremental_steps_track_the_native_copy(manager):
    memo, attn_metadata = runtime._BlockOffsetsMemo(), SimpleNamespace()
    dst = _table(manager)
    # A batch, the same batch, a narrower batch in the same rows, then a request that grows.
    steps = ([11, 12, 13], [11, 12, 13], [14, 12, 13], [14, 12, 13])
    kept, copied = [], []
    for step, request_ids in enumerate(steps):
        if step == 3:
            assert manager.kv_cache_map[14].resize(32, 28)
        stager = _staged(manager, dst, request_ids, memo, attn_metadata)
        kept.append(stager.counts["block columns kept"])
        copied.append(stager.counts["block columns copied"])
        assert torch.equal(dst, _native(manager, request_ids)), step
    # The repeated batch copies nothing; the narrower one re-copies only what changed.
    assert copied[1] == 0 and kept[1] > 0
    assert kept[2] > 0 and copied[3] > 0


def test_a_new_attention_metadata_copies_every_column_again(manager):
    memo = runtime._BlockOffsetsMemo()
    dst = _table(manager)
    _staged(manager, dst, [11, 12, 13], memo, SimpleNamespace())
    dst.fill_(SENTINEL)
    _staged(manager, dst, [11, 12, 13], memo, SimpleNamespace())
    assert torch.equal(dst, _native(manager, [11, 12, 13]))


def test_blocks_descriptor_bad_page_word():
    pages = torch.tensor([3, BAD_PAGE_INDEX, 5, BAD_PAGE_INDEX], dtype=torch.int32).pin_memory()
    for bad_page in (0, BAD_PAGE_INDEX):
        k = torch.full((4,), SENTINEL, dtype=torch.int32, device="cuda")
        v = torch.full((4,), SENTINEL, dtype=torch.int32, device="cuda")
        staging = step_commit.StepStaging()
        staging.blocks(k.data_ptr(), v.data_ptr(), pages.data_ptr(), 4, 2, 1, bad_page)
        assert staging.launch()
        torch.cuda.synchronize()
        assert k.tolist() == [6, bad_page, 10, bad_page]
        assert v.tolist() == [7, bad_page, 11, bad_page]
