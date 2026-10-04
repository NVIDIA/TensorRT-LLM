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
"""GPU unit tests for DFlash worker backend-specific cache setup.

The tests exercise framework-side TRTLLM-Gen cache plumbing without loading a
draft model. The full generated-FMHA path is covered by the Qwen3.6 NVFP4
DFlash accuracy test in ``integration/defs/accuracy/test_llm_api_pytorch.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.standalone_draft_cache import StandaloneDraftHistory
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.llmapi import DFlashDecodingConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DFlash worker allocates CUDA context-cache buffers"
)


class _FakeDraftModel:
    block_size = 8
    config = SimpleNamespace(max_position_embeddings=128)

    def __init__(self, attention_backend="TRTLLM"):
        self.dflash_attention_backend = attention_backend
        self.fc = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16, device="cuda"))
        self._num_attn_layers = 0
        self._num_heads = 0
        self._num_kv_heads = 0
        self._head_dim = 0

    def _build_fused_kv_buffers(self):
        self._num_attn_layers = 2
        self._num_heads = 8
        self._num_kv_heads = 1
        self._head_dim = 64

    def _get_attention_mask_args(self, layer_idx):
        return True, (-1, -1)

    def validate_block_attention_windows(self):
        # Every layer is causal and unwindowed, so the TRTLLM check passes.
        pass


def test_trtllm_backend_builds_private_paged_context_cache(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.dflash.validate_dflash_trtllm_gen_runtime",
        lambda **kwargs: None,
    )
    config = DFlashDecodingConfig(max_draft_len=7, attention_backend="TRTLLM")
    worker = DFlashWorker(config, Mapping())
    draft_model = _FakeDraftModel()
    worker.set_draft_model(draft_model)
    spec_metadata = SimpleNamespace(max_num_requests=2)
    attn_metadata = SimpleNamespace(max_seq_len=64)

    worker._lazy_init_ctx_buffers(draft_model, spec_metadata, attn_metadata)

    # max_ctx=64 plus an 8-token draft block requires three 32-token pages
    # per slot. Two request slots plus one scratch slot therefore use 9 pages.
    assert worker._ctx_pages_per_slot == 3
    assert worker._ctx_kv_buf.shape == (2, 9, 2, 1, 32, 64)
    assert worker._ctx_k_buf is None
    assert worker._ctx_v_buf is None
    assert worker._ctx_page_table.tolist() == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]


def test_managed_metadata_replay_preserves_overlapping_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queued replays retain their own history across reorder, removal and slot reuse."""
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.dflash.validate_dflash_trtllm_gen_runtime",
        lambda **kwargs: None,
    )
    worker = DFlashWorker(
        DFlashDecodingConfig(max_draft_len=7, attention_backend="TRTLLM"), Mapping()
    )
    model = _FakeDraftModel()
    pool = torch.zeros((9, 2, 1, 32, 64), dtype=torch.bfloat16, device="cuda")
    histories = {11: StandaloneDraftHistory(20, 30), 22: StandaloneDraftHistory(40, 40)}
    caches = {rid: SimpleNamespace(is_active=True) for rid in (11, 22, 33, 44, 999)}
    published = {}
    manager = SimpleNamespace(
        draft_layer_ids=(0, 1),
        tokens_per_block=32,
        draft_max_blocks_per_seq=3,
        _draft_dummy_request_ids={999},
        kv_cache_map=caches,
        get_draft_buffers=lambda layer, kv_layout: pool,
        get_draft_history=lambda rid: histories.get(rid),
        get_draft_block_table=lambda ids: torch.tensor(
            [[1, 2, 3] if rid == 11 else [4, 5, 6] for rid in ids], dtype=torch.int32
        ),
        get_draft_num_blocks=lambda rid: 2 if rid == 11 else 3,
        set_draft_history=lambda rid, length, position: published.update({rid: (length, position)}),
    )
    resource = SimpleNamespace(get_resource_manager=lambda kind: manager)
    metadata = SimpleNamespace(max_num_requests=3, request_ids=[11, 999, 22])
    attention = SimpleNamespace(max_seq_len=64, num_seqs=3, num_contexts=0)
    worker.prepare_managed_draft_cache(model, metadata, attention, resource)
    increments = torch.ones(3, dtype=torch.long, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        worker._ctx_len.index_add_(0, worker._batch_to_slot, increments)
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        worker._ctx_len.index_add_(0, worker._batch_to_slot, increments)
    addresses = [
        value.data_ptr()
        for value in (
            worker._batch_to_slot,
            worker._ctx_block_counts,
            worker._managed_snapshot_slots,
        )
    ]
    tensor = torch.tensor

    def host_tensor(*args, **kwargs) -> torch.Tensor:
        assert torch.device(kwargs.get("device", "cpu")).type == "cpu"
        return tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", host_tensor)
    snapshots = []
    counts = []
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        for iteration in range(8):
            metadata.request_ids = [11, 999, 22] if iteration % 2 == 0 else [22, 11, 999]
            worker.prepare_managed_draft_cache(model, metadata, attention, resource)
            counts.append(worker._ctx_block_counts[:3].clone())
            graph.replay()
            snapshots.append(worker.snapshot_managed_draft_history())
        # Replace one cache and remove another before publishing older readbacks.
        caches.pop(11)
        caches[22] = SimpleNamespace(is_active=True)
        histories[44] = StandaloneDraftHistory(7, 17)
        histories[33] = StandaloneDraftHistory(9, 19)
        metadata.request_ids = [33, 999, 44]
        worker.prepare_managed_draft_cache(model, metadata, attention, resource)
        graph.replay()
        latest = worker.snapshot_managed_draft_history()
        completed = torch.cuda.Event()
        completed.record()
    completed.synchronize()
    for iteration, update in enumerate(snapshots):
        expected = {11: [22 + iteration, 32 + iteration], 22: [42 + iteration, 42 + iteration]}
        assert update.values_host.tolist() == [expected[rid] for rid in update.request_ids]
        assert counts[iteration].tolist() == ([2, 3, 3] if iteration % 2 == 0 else [3, 2, 3])
        update.publish()
    assert published == {}
    latest.publish()
    assert published == {33: (10, 20), 44: (8, 18)}
    assert worker._batch_to_slot.tolist() == [2, worker._dummy_slot, 0]
    assert worker._ctx_block_counts[:3].tolist() == [3, 3, 3]
    assert addresses == [
        value.data_ptr()
        for value in (
            worker._batch_to_slot,
            worker._ctx_block_counts,
            worker._managed_snapshot_slots,
        )
    ]
