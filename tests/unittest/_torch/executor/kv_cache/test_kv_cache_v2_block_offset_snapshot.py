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
"""Opt-in snapshot block-offset copy (TLLM_KV_BLOCK_OFFSETS_SNAPSHOT=1) of KVCacheManagerV2.

The default dense copy (``copy_batch_block_offsets_to_device``) reads the host
page-index table and the IndexMapper copy index -- one persistent pinned buffer
that every ``get_copy_index`` rewrites -- when the kernel runs. When the stream
is held up (here by a GPU sleep; in serving by page onboarding before a PP
stage's step, or a host that runs ahead), preparing the next copy rewrites the
copy index the pending copy will read. The snapshot copy reads the batch's rows
as they were when it was enqueued, and writes the same tables otherwise.
"""

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

SNAPSHOT_ENV = "TLLM_KV_BLOCK_OFFSETS_SNAPSHOT"
TOKENS_PER_BLOCK = 8
MAX_SEQ_LEN = 1024
# Long enough that the host reliably prepares the next copy before the first runs.
SLEEP_CYCLES = 2_000_000_000


def _make_manager(monkeypatch, *, snapshot: bool) -> KVCacheManagerV2:
    torch.cuda.init()
    monkeypatch.setenv(SNAPSHOT_ENV, "1" if snapshot else "0")
    return KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=2,
        num_kv_heads=8,
        head_dim=128,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=16,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
    )


def _active_cache(manager: KVCacheManagerV2, request_id: int, tokens, capacity: int):
    cache = manager._create_kv_cache(request_id, None, tokens)
    assert cache is not None
    assert manager._resume_and_restore(request_id, cache)
    assert cache.resize(capacity)
    return cache


@pytest.mark.parametrize(
    "snapshot",
    [
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                strict=True,
                reason="the default dense copy reads the IndexMapper copy index when it runs",
            ),
            id="default",
        ),
        pytest.param(True, id="snapshot"),
    ],
)
def test_block_offset_copy_reads_enqueue_time_rows(monkeypatch, snapshot: bool) -> None:
    """Batch A's copy waits behind a GPU stall while the host prepares batch B's copy."""
    manager = _make_manager(monkeypatch, snapshot=snapshot)
    try:
        ids_a, ids_b = [11, 12], [13, 14]
        for i, request_id in enumerate(ids_a + ids_b):
            tokens = [1000 * (i + 1) + t for t in range(4 * TOKENS_PER_BLOCK)]
            _active_cache(manager, request_id, tokens, len(tokens))

        def empty():
            return torch.zeros(
                manager.num_pools,
                2,
                2,
                manager.max_blocks_per_seq,
                dtype=torch.int32,
                device="cuda",
            )

        reference = empty()
        manager.copy_batch_block_offsets(reference, ids_a, 1, 2, 2)
        torch.cuda.synchronize()

        dst_a, dst_b = empty(), empty()
        with torch.cuda.stream(manager._stream):
            torch.cuda._sleep(SLEEP_CYCLES)
        manager.copy_batch_block_offsets(dst_a, ids_a, 1, 2, 2)
        manager.copy_batch_block_offsets(dst_b, ids_b, 1, 2, 2)
        torch.cuda.synchronize()
        assert not torch.equal(dst_a, dst_b)
        assert torch.equal(dst_a, reference)
    finally:
        manager.shutdown()


def test_snapshot_copy_matches_default_copy(monkeypatch) -> None:
    """With no host race the snapshot copy writes the same block tables."""
    tables = []
    for snapshot in (False, True):
        manager = _make_manager(monkeypatch, snapshot=snapshot)
        try:
            ids = [21, 22, 23]
            for i, request_id in enumerate(ids):
                tokens = [500 * (i + 1) + t for t in range((i + 1) * 3 * TOKENS_PER_BLOCK)]
                _active_cache(manager, request_id, tokens, len(tokens))
            dst = torch.full(
                (manager.num_pools, 4, 2, manager.max_blocks_per_seq),
                -7,
                dtype=torch.int32,
                device="cuda",
            )
            manager.copy_batch_block_offsets(dst, ids, 1, 3, 3)
            torch.cuda.synchronize()
            tables.append(dst.cpu().numpy())
        finally:
            manager.shutdown()
    np.testing.assert_array_equal(tables[0], tables[1])
