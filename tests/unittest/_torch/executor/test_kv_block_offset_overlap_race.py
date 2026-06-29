# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression test for nvbug 6293536.

``KVCacheManager.copy_batch_block_offsets`` ships each iteration's block-offset
page tables to the device with an *asynchronous* H2D. The async copy reads its
source at execution time, not enqueue time, so the host staging buffer must not
be overwritten before the copy drains. Under the overlap scheduler the CPU runs
an iteration ahead, so a single persistently-reused staging buffer let the next
iteration's in-place overwrite clobber the source of the previous iteration's
still-pending H2D -> the attention kernel indexed another batch's blocks
(memory corruption). The fix stages through a freshly-allocated pinned buffer
each call.

This test drives that exact window deterministically: it stalls the stream with
``torch.cuda._sleep`` (standing in for the production ``syncTransfers`` offload
stall that delays the H2D), enqueues batch A's async H2D behind the stall, then
lets the host immediately issue batch B before A's copy can drain. With the bug
present, batch A's device tensor ends up holding batch B's offsets.
"""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType, LayerType
from tensorrt_llm.bindings import ModelConfig as ModelConfigCpp
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

# Small geometry so each sequence spans several blocks and the two batches get
# visibly different physical block indices.
_NUM_LAYERS = 2
_NUM_KV_HEADS = 2
_HEAD_DIM = 16
_TOKENS_PER_BLOCK = 16
_MAX_SEQ_LEN = 256
_BATCH = 6
_TOKENS_PER_SEQ = _TOKENS_PER_BLOCK * 5
# Long GPU stall so the host reliably wins the race and overwrites the staging
# buffer before batch A's H2D drains. Generous to stay deterministic on fast GPUs.
_SLEEP_CYCLES = 2_000_000_000


def _build_manager():
    model_config = ModelConfigCpp(
        vocab_size=32000,
        num_layers=_NUM_LAYERS,
        num_attention_layers=_NUM_LAYERS,
        num_rnn_layers=0,
        num_heads=_NUM_KV_HEADS,
        hidden_size=_NUM_KV_HEADS * _HEAD_DIM,
        data_type=DataType.HALF,
    )
    model_config.layer_types = [LayerType.ATTENTION] * _NUM_LAYERS
    model_config.set_num_kv_heads(_NUM_KV_HEADS)

    kv_cache_config = KvCacheConfig(
        max_tokens=4096, free_gpu_memory_fraction=0.2, enable_block_reuse=False
    )
    mapping = Mapping(world_size=1, tp_size=1, pp_size=1)

    return KVCacheManager(
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheType.SELF,
        num_layers=_NUM_LAYERS,
        num_kv_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=_MAX_SEQ_LEN,
        max_batch_size=2 * _BATCH,
        mapping=mapping,
        dtype=DataType.HALF,
        model_config=model_config,
        max_beam_width=1,
    )


def _empty_device_offsets(mgr):
    return torch.zeros(
        mgr.num_pools, 2 * _BATCH, 2, mgr.max_blocks_per_seq, dtype=torch.int32, device="cuda"
    )


def _reference_offsets(mgr, ids):
    """Non-racy ground truth: copy followed by an immediate synchronize."""
    dst = _empty_device_offsets(mgr)
    mgr.copy_batch_block_offsets(dst, ids, 1, len(ids), len(ids))
    torch.cuda.synchronize()
    return dst[:, : len(ids)].clone()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_copy_batch_block_offsets_survives_overlap_overwrite():
    mgr = _build_manager()

    # Two disjoint batches -> different physical block indices.
    ids_a = list(range(1, 1 + _BATCH))
    ids_b = list(range(101, 101 + _BATCH))
    toks = [_TOKENS_PER_SEQ] * _BATCH
    mgr.add_dummy_requests(request_ids=ids_a, token_nums=toks, prepare_resource=True)
    mgr.add_dummy_requests(request_ids=ids_b, token_nums=toks, prepare_resource=True)

    ref_a = _reference_offsets(mgr, ids_a)
    ref_b = _reference_offsets(mgr, ids_b)
    assert not torch.equal(ref_a, ref_b), (
        "test setup invalid: batches A and B produced identical offsets"
    )

    # Drive the overlap window: stall the stream, enqueue batch A's async H2D
    # behind the stall, then let the host run ahead and issue batch B (whose
    # host-side fill would overwrite a shared staging buffer in place) before
    # A's copy can drain.
    dst_a = _empty_device_offsets(mgr)
    dst_b = _empty_device_offsets(mgr)
    torch.cuda.synchronize()

    torch.cuda._sleep(_SLEEP_CYCLES)
    mgr.copy_batch_block_offsets(dst_a, ids_a, 1, _BATCH, _BATCH)
    mgr.copy_batch_block_offsets(dst_b, ids_b, 1, _BATCH, _BATCH)
    torch.cuda.synchronize()

    # Batch A's device offsets must still be batch A's, not clobbered by B.
    assert torch.equal(dst_a[:, :_BATCH], ref_a), (
        "nvbug 6293536: batch A's async H2D read another batch's host staging "
        "buffer (overlap-scheduler data race on host_kv_cache_block_offsets)"
    )
    # And batch B's copy is independently correct.
    assert torch.equal(dst_b[:, :_BATCH], ref_b)
