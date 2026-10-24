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


def _make_worker(monkeypatch):
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.dflash.validate_dflash_trtllm_gen_runtime",
        lambda **kwargs: None,
    )
    config = DFlashDecodingConfig(max_draft_len=7, attention_backend="TRTLLM")
    return DFlashWorker(config, Mapping())


class _FakeDraftCacheManager:
    """Unified manager whose draft layer 61 lives in pool 1 with two KV planes."""

    draft_layer_ids = [61]
    layer_offsets = {61: 7}
    layer_to_pool_mapping_dict = {7: 1}
    _index_scale_ints = [1, 1]
    enable_swa_scratch_reuse = False

    def __init__(self, pages, helix_cp_size=1):
        self.pages = pages
        self._helix_cp_size = helix_cp_size
        self.kv_cache_map = {
            request_id: SimpleNamespace(num_blocks=len(request_pages), is_active=True)
            for request_id, request_pages in pages.items()
        }

    def get_layer_page_index_scale(self, layer_idx):
        return 1

    def get_layer_kv_factor(self, layer_idx):
        return 2

    def get_batch_cache_indices(self, request_ids, layer_idx):
        return [list(self.pages[request_id]) for request_id in request_ids]

    def block_offsets(self, request_ids, width):
        """The target's [pool, batch, plane, block] offsets: K at page * 2, V at page * 2 + 1."""
        offsets = torch.full((2, len(request_ids), 2, width), -1, dtype=torch.int32)
        for row, request_id in enumerate(request_ids):
            pages = torch.tensor(self.pages[request_id], dtype=torch.int32)
            offsets[0, row, :, : len(pages)] = 1000  # the target pool's own pages
            offsets[1, row, 0, : len(pages)] = pages * 2
            offsets[1, row, 1, : len(pages)] = pages * 2 + 1
        return offsets.cuda()


def _bind_device_tables(worker, manager):
    worker._ctx_kv_manager = manager
    worker._init_ctx_block_tables(num_slots=3, max_blocks=4)
    worker._init_unified_draft_tables(manager)


def test_unified_draft_tables_are_decoded_from_target_block_offsets(monkeypatch):
    worker = _make_worker(monkeypatch)
    manager = _FakeDraftCacheManager({11: [5, 9, 2], 12: [7]})
    _bind_device_tables(worker, manager)
    assert worker._device_draft_tables
    assert (worker._ctx_pool_idx, worker._ctx_block_divisor) == (1, 2)

    attn_metadata = SimpleNamespace(kv_cache_block_offsets=manager.block_offsets([11, 12], 4))
    assert worker._refresh_unified_block_tables(attn_metadata, 2, [11, 12])

    tables = worker._ctx_block_tables.cpu()
    assert tables[0, :3].tolist() == [5, 9, 2]
    assert tables[1, :1].tolist() == [7]
    assert worker._ctx_block_counts[:2].tolist() == [3, 1]
    # The startup check compared the rows with the manager's page lists.
    assert worker._device_draft_tables
    assert worker._draft_table_checks_left == 7


def test_unified_draft_tables_fall_back_to_host_on_mismatch(monkeypatch):
    worker = _make_worker(monkeypatch)
    manager = _FakeDraftCacheManager({11: [5, 9, 2], 12: [7]})
    _bind_device_tables(worker, manager)
    attn_metadata = SimpleNamespace(kv_cache_block_offsets=manager.block_offsets([11, 12], 4))
    manager.pages[12] = [8]

    assert not worker._refresh_unified_block_tables(attn_metadata, 2, [11, 12])
    assert not worker._device_draft_tables


def test_unified_draft_tables_stay_on_host_for_helix_managers(monkeypatch):
    worker = _make_worker(monkeypatch)
    _bind_device_tables(worker, _FakeDraftCacheManager({}, helix_cp_size=2))

    assert not worker._device_draft_tables


def test_fp8_draft_cache_stores_context_kv_by_direct_page_writes(monkeypatch):
    worker = _make_worker(monkeypatch)
    num_layers, nkv, page_size, hd = 2, 2, 32, 64
    worker._ctx_kv_buf = [
        torch.zeros(4, 2, nkv, page_size, hd, dtype=torch.float8_e4m3fn, device="cuda")
        for _ in range(num_layers)
    ]
    worker._ctx_block_tables = torch.tensor([[3, 1], [0, 2]], dtype=torch.int32, device="cuda")
    worker._ctx_page_size = page_size
    rows = torch.tensor([0, 0, 1], device="cuda")
    positions = torch.tensor([5, 40, 33], device="cuda")
    k = torch.randn(3, num_layers, nkv, hd, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)

    worker._store_context_kv_paged(k, v, rows, positions)

    pages = torch.tensor([3, 1, 2], device="cuda")
    offsets = positions % page_size
    for layer, pool in enumerate(worker._ctx_kv_buf):
        stored = pool.view(torch.uint8)
        expected_k = k[:, layer].to(pool.dtype).view(torch.uint8)
        expected_v = v[:, layer].to(pool.dtype).view(torch.uint8)
        assert torch.equal(stored[pages, 0, :, offsets], expected_k)
        assert torch.equal(stored[pages, 1, :, offsets], expected_v)
        stored[pages, :, :, offsets] = 0
        assert not stored.any(), "the store wrote outside the addressed slots"
