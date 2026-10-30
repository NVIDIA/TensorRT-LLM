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

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_dspark import build_dspark_mla_yarn_freqs_cis
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.standalone_draft_cache import (
    DraftContextWindow,
    DraftContextWindowMove,
    StandaloneDraftHistory,
)
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
    spec_metadata = SimpleNamespace(max_num_requests=2, num_seq_slots=2)
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


def test_context_cache_spans_full_seq_slot_pool_on_graph_bucket(monkeypatch):
    """The ctx cache is sized by num_seq_slots, not a narrowed graph bucket."""
    worker = _make_worker(monkeypatch)
    draft_model = _FakeDraftModel()
    worker.set_draft_model(draft_model)
    spec_metadata = SimpleNamespace(max_num_requests=1, num_seq_slots=4)
    attn_metadata = SimpleNamespace(max_seq_len=64)

    worker._lazy_init_ctx_buffers(draft_model, spec_metadata, attn_metadata)

    # Four request slots plus one scratch slot, three pages each.
    assert worker._dummy_slot == 4
    assert list(worker._free_slots) == [0, 1, 2, 3]
    assert worker._ctx_kv_buf.shape == (2, 15, 2, 1, 32, 64)
    assert worker._ctx_len.shape == (5,)


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


def test_drafter_without_dense_staging_has_nothing_to_refresh_from_the_pages(monkeypatch):
    worker = _make_worker(monkeypatch)
    worker._dflash_attention_backend = "VANILLA"
    worker._ctx_k_buf = None  # the MLA drafter reads the manager pages itself
    worker._req_to_slot = {11: 0}
    worker._ctx_len_host = [3]

    worker._gather_managed_context([11], [0])


def test_vanilla_drafter_refreshes_its_dense_staging_from_the_pages(monkeypatch):
    worker = _make_worker(monkeypatch)
    page_size, nkv, hd = 4, 2, 8
    pool = torch.arange(3 * 2 * nkv * page_size * hd, dtype=torch.float32, device="cuda").view(
        3, 2, nkv, page_size, hd
    )
    worker._dflash_attention_backend = "VANILLA"
    worker._ctx_kv_buf = [pool]
    worker._ctx_page_size = page_size
    # Request 11 sits in batch row 0 and owns pages 2 and 0; its slot is 1.
    worker._ctx_block_tables = torch.tensor([[2, 0]], dtype=torch.int32, device="cuda")
    worker._req_to_slot = {11: 1}
    worker._ctx_len_host = [0, 5]
    worker._ctx_k_buf = torch.zeros(2, 1, 8, nkv, hd, device="cuda")
    worker._ctx_v_buf = torch.zeros_like(worker._ctx_k_buf)

    worker._gather_managed_context([11], [0])

    for position, (page, offset) in enumerate([(2, 0), (2, 1), (2, 2), (2, 3), (0, 0)]):
        assert torch.equal(worker._ctx_k_buf[1, 0, position], pool[page, 0, :, offset])
        assert torch.equal(worker._ctx_v_buf[1, 0, position], pool[page, 1, :, offset])
    assert not worker._ctx_k_buf[1, :, 5:].any()
    assert not worker._ctx_k_buf[0].any()


class _RecordingBuffer:
    """Stands in for a device buffer and keeps what was uploaded into it."""

    def __init__(self):
        self.uploads = []

    def __getitem__(self, key):
        return self

    def copy_(self, src, non_blocking=False):
        self.uploads.append(src)


class _HostTableManager(_FakeDraftCacheManager):
    def get_draft_block_table(self, request_ids):
        table = torch.zeros(len(request_ids), 4, dtype=torch.int32)
        for row, request_id in enumerate(request_ids):
            pages = self.pages[request_id]
            table[row, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
        return table

    def get_draft_num_blocks(self, request_id):
        return len(self.pages[request_id])


@pytest.mark.parametrize("pinned", [False, True])
def test_unified_host_tables_upload_from_pinned_memory_when_preferred(monkeypatch, pinned):
    monkeypatch.setattr("tensorrt_llm._torch.speculative.dflash.prefer_pinned", lambda: pinned)
    worker = _make_worker(monkeypatch)
    worker._ctx_kv_manager = _HostTableManager({11: [5, 9, 2], 12: [7]})
    worker._ctx_block_tables = _RecordingBuffer()
    worker._ctx_block_counts = _RecordingBuffer()

    assert worker._refresh_ctx_block_tables(SimpleNamespace(), 2, [11, 12])

    (table,) = worker._ctx_block_tables.uploads
    assert table.tolist() == [[5, 9, 2, 0], [7, 0, 0, 0]]
    assert table.is_pinned() == pinned
    (counts,) = worker._ctx_block_counts.uploads
    assert counts.tolist() == [3, 1]


class _HistoryManager:
    """A draft-owning KVCacheManagerV2 without pages; its real draft-history bookkeeping runs."""

    draft_layer_ids = (61,)
    draft_attention_backend = "VANILLA"
    draft_window_size = None
    enable_block_reuse = False
    _draft_dummy_request_ids = frozenset()
    layout = {"num_layers": 5, "window_size": None}
    _draft_history_uses_kv_history = KVCacheManagerV2._draft_history_uses_kv_history
    get_draft_history = KVCacheManagerV2.get_draft_history
    set_draft_history = KVCacheManagerV2.set_draft_history
    restore_draft_history = KVCacheManagerV2.restore_draft_history
    take_draft_context_move = KVCacheManagerV2.take_draft_context_move

    def __init__(self, draft_context_limit, request_id, capacity, draft_context_window=None):
        self.draft_context_limit = draft_context_limit
        self.draft_context_window = draft_context_window
        self._draft_context_moves = {}
        self.draft_history = {}
        self.kv_cache_map = {request_id: SimpleNamespace(is_active=True, capacity=capacity)}

    def get_draft_transfer_identity(self):
        return self.layout

    def get_draft_block_table(self, request_ids, histories=None):
        return None


def _bind_managed_history(worker, manager, max_batch=2):
    worker._ctx_kv_manager = manager
    worker._dummy_slot = max_batch
    worker._graph_dummy_id_floor = 1 << 40
    worker._ctx_len = torch.zeros(max_batch + 1, dtype=torch.long, device="cuda")
    worker._ctx_position_offset = torch.zeros_like(worker._ctx_len)
    worker._ctx_len_host = [0] * (max_batch + 1)
    worker._batch_to_slot = torch.zeros(max_batch, dtype=torch.long, device="cuda")
    worker._managed_snapshot_slots = torch.empty_like(worker._batch_to_slot)
    worker._free_slots = deque(range(max_batch))
    worker._req_to_slot = {}
    worker._req_ctx_pos = {}
    worker._managed_cache_bindings = {}
    worker._managed_request_ids = ()


@pytest.mark.parametrize(
    ("limit", "ctx_len", "offset"),
    [(None, 120_000, 0), (120_000, 120_000, 0), (119_999, 0, 120_000)],
    ids=["no_limit", "prompt_at_the_limit", "prompt_above_the_limit"],
)
def test_worker_binds_the_draft_history_the_manager_restored(monkeypatch, limit, ctx_len, offset):
    """A prompt above the draft context limit starts the drafter from an empty context, and the
    history it publishes afterwards still counts the prompt."""
    prompt_len, accepted = 120_000, 4
    manager = _HistoryManager(limit, 11, capacity=prompt_len + 16)
    manager.restore_draft_history(
        11, {"layout": manager.layout, "valid_length": prompt_len, "position": prompt_len}
    )
    worker = _make_worker(monkeypatch)
    _bind_managed_history(worker, manager)

    assert worker._prepare_managed_history([11]) == [0]

    slot = worker._req_to_slot[11]
    assert worker._ctx_len[slot].item() == ctx_len
    assert worker._ctx_position_offset[slot].item() == offset
    # A decode step stores the accepted tokens after the context the drafter holds.
    worker._ctx_len[slot] += accepted
    update = worker.snapshot_managed_draft_history()
    torch.cuda.synchronize()
    update.publish()
    assert manager.get_draft_history(11) == StandaloneDraftHistory(
        ctx_len + accepted, prompt_len + accepted
    )


def test_worker_binds_the_window_the_manager_restored(monkeypatch):
    """A prompt above the limit with TLLM_DSPARK_DRAFT_CONTEXT_WINDOW binds the sink plus the
    window, and the history it publishes afterwards still counts the prompt."""
    prompt_len, accepted = 120_000, 4
    window = DraftContextWindow(sink=64, recent=65536)
    manager = _HistoryManager(119_999, 11, capacity=prompt_len + 16, draft_context_window=window)
    manager.restore_draft_history(
        11, {"layout": manager.layout, "valid_length": prompt_len, "position": prompt_len}
    )
    worker = _make_worker(monkeypatch)
    _bind_managed_history(worker, manager)

    assert worker._prepare_managed_history([11]) == [0]

    slot = worker._req_to_slot[11]
    assert worker._ctx_len[slot].item() == window.length
    assert worker._ctx_position_offset[slot].item() == prompt_len - window.length
    worker._ctx_len[slot] += accepted
    update = worker.snapshot_managed_draft_history()
    torch.cuda.synchronize()
    update.publish()
    assert manager.get_draft_history(11) == StandaloneDraftHistory(
        window.length + accepted, prompt_len + accepted
    )


_WINDOW_LATENT, _WINDOW_ROPE, _WINDOW_PAGE = 512, 64, 32
_YARN = dict(
    dim=_WINDOW_ROPE,
    base=50000.0,
    scaling_factor=4.0,
    original_max_position_embeddings=64,
    max_position_embeddings=1024,
    beta_fast=32.0,
    beta_slow=1.0,
    mscale=1.0,
    mscale_all_dim=1.0,
)


class _WindowDraftModel:
    """The two things a window move reads from the MLA drafter: the latent width and its table."""

    kv_lora_rank = _WINDOW_LATENT

    def __init__(self, **yarn):
        self.table = build_dspark_mla_yarn_freqs_cis(device="cuda", **{**_YARN, **yarn})

    def _mla_freqs_cis(self, device):
        return self.table


def _rotate(latents, freqs, positions):
    """``latents`` [N, latent + rope] with the RoPE slice rotated to ``positions``, in fp32."""
    rope = torch.view_as_complex(latents[:, _WINDOW_LATENT:].float().reshape(len(latents), -1, 2))
    out = latents.float().clone()
    out[:, _WINDOW_LATENT:] = torch.view_as_real(rope * freqs[positions]).reshape(len(latents), -1)
    return out


def _window_pools(worker, num_layers, num_pages, dtype):
    torch.manual_seed(5)
    width = _WINDOW_LATENT + _WINDOW_ROPE
    worker._ctx_kv_buf = [
        torch.randn(num_pages, 1, 1, _WINDOW_PAGE, width, device="cuda").to(dtype)
        for _ in range(num_layers)
    ]
    worker._ctx_page_size = _WINDOW_PAGE
    return [pool.clone() for pool in worker._ctx_kv_buf]


def _token_rows(pool, pages, positions):
    """The latents of ``positions`` in the pages ``pages`` (byte views for FP8 indexing)."""
    fp8 = pool.dtype == torch.float8_e4m3fn
    view = pool.view(torch.uint8) if fp8 else pool
    rows = view[pages[positions // _WINDOW_PAGE].long(), 0, 0, positions % _WINDOW_PAGE]
    return rows.view(torch.float8_e4m3fn) if fp8 else rows


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
@pytest.mark.parametrize(
    ("prompt_len", "sink", "recent", "mscale_all_dim"),
    [
        (150, 0, 64, 1.0),
        (150, 8, 64, 1.0),
        (100, 4, 60, 1.0),
        (900, 32, 300, 1.0),
        (200, 0, 180, 1.0),
        (150, 8, 64, 0.0),
    ],
    ids=[
        "window",
        "sink_and_window",
        "overlapping",
        "several_chunks",
        "overlapping_chunks",
        "scaled_table",
    ],
)
def test_window_move_copies_the_last_latents_behind_the_sink_rebased(
    monkeypatch, dtype, prompt_len, sink, recent, mscale_all_dim
):
    """The kept latents land right after the sink with their RoPE slice at the new positions, the
    latent part copied as it is; the sink and every other request's pages are untouched. A table
    whose modulus carries a YaRN mscale ratio other than 1 rotates by its phase alone."""
    monkeypatch.setattr("tensorrt_llm._torch.speculative.dflash._DRAFT_WINDOW_MOVE_CHUNK", 128)
    worker = _make_worker(monkeypatch)
    num_pages = 2 * -(-prompt_len // _WINDOW_PAGE) + 2
    before = _window_pools(worker, num_layers=2, num_pages=num_pages, dtype=dtype)
    # Request 11 in batch row 1 owns every odd page, in a scrambled order; row 0 owns the rest.
    own = torch.randperm(num_pages // 2, device="cuda") * 2 + 1
    other = torch.arange(0, num_pages, 2, device="cuda")
    worker._ctx_block_tables = torch.stack([other, own]).to(torch.int32)
    model = _WindowDraftModel(mscale_all_dim=mscale_all_dim)
    move = DraftContextWindowMove(source_start=prompt_len - recent, dest_start=sink, length=recent)

    worker._move_draft_context_window(model, 1, move)

    source = torch.arange(prompt_len - recent, prompt_len, device="cuda")
    dest = source - move.shift
    for layer, pool in enumerate(worker._ctx_kv_buf):
        old = before[layer]
        moved = _token_rows(pool, own, dest).float()
        original = _token_rows(old, own, source).float()
        # The latent part is the very same values.
        assert torch.equal(moved[:, :_WINDOW_LATENT], original[:, :_WINDOW_LATENT])
        # The RoPE slice is the stored one rotated back by the shift: what storing the same
        # latent at the new position gives, up to one rounding to the pool's dtype.
        phase = model.table / model.table.abs()
        expected = _rotate(_rotate(original, phase.conj(), source), phase, dest)
        tolerance = 0.07 if dtype == torch.float8_e4m3fn else 0.01
        torch.testing.assert_close(
            moved[:, _WINDOW_LATENT:], expected[:, _WINDOW_LATENT:], rtol=tolerance, atol=tolerance
        )
        # Nothing else changed: the sink, the rest of this request's pages up to the window
        # start, and every page of the other request.
        changed = torch.zeros(num_pages, _WINDOW_PAGE, dtype=torch.bool, device="cuda")
        changed[own[dest // _WINDOW_PAGE], dest % _WINDOW_PAGE] = True
        same = (pool.view(torch.uint8) == old.view(torch.uint8)).all(-1)[:, 0, 0]
        assert same[~changed].all()
        assert same[other].all()


def test_window_move_keeps_every_distance_the_drafter_attends_over(monkeypatch):
    """Attention scores of a query at the bound context's next position over the moved latents
    are those of the same query at the prompt's next position over the original latents."""
    worker = _make_worker(monkeypatch)
    prompt_len, recent = 200, 96
    before = _window_pools(worker, num_layers=1, num_pages=8, dtype=torch.bfloat16)
    pages = torch.arange(8, device="cuda")
    worker._ctx_block_tables = pages.view(1, -1).to(torch.int32)
    model = _WindowDraftModel()
    move = DraftContextWindowMove(source_start=prompt_len - recent, dest_start=0, length=recent)

    worker._move_draft_context_window(model, 0, move)

    query = torch.randn(1, _WINDOW_LATENT + _WINDOW_ROPE, device="cuda")
    true_positions = torch.arange(prompt_len - recent, prompt_len, device="cuda")
    true_keys = _token_rows(before[0], pages, true_positions)
    bound_keys = _token_rows(worker._ctx_kv_buf[0], pages, torch.arange(recent, device="cuda"))
    for step in range(3):
        at_prompt = _rotate(query, model.table, torch.tensor([prompt_len + step], device="cuda"))
        at_bound = _rotate(query, model.table, torch.tensor([recent + step], device="cuda"))
        torch.testing.assert_close(
            at_bound @ bound_keys.float().T, at_prompt @ true_keys.float().T, rtol=0.02, atol=0.3
        )


def test_window_moves_run_once_and_only_for_the_rows_bound_this_step(monkeypatch):
    worker = _make_worker(monkeypatch)
    window = DraftContextWindow(sink=0, recent=64)
    manager = SimpleNamespace(draft_context_window=window, moves={12: "move-12"}, asked=[])

    def take_draft_context_move(request_id):
        manager.asked.append(request_id)
        return manager.moves.pop(request_id, None)

    manager.take_draft_context_move = take_draft_context_move
    worker._ctx_kv_manager = manager
    moved = []
    monkeypatch.setattr(
        worker, "_move_draft_context_window", lambda model, row, move: moved.append((row, move))
    )

    worker._move_draft_context_windows(None, [11, 12, 13], [1, 2])
    worker._move_draft_context_windows(None, [11, 12, 13], [1])

    assert moved == [(1, "move-12")]
    # Rows bound in an earlier step (request 11 here) are not asked for a move.
    assert manager.asked == [12, 13, 12]
    # Without the switch nothing is asked of the manager.
    manager.draft_context_window = None
    manager.moves[13] = "move-13"
    worker._move_draft_context_windows(None, [11, 12, 13], [2])
    assert moved == [(1, "move-12")] and manager.moves == {13: "move-13"}
    assert manager.asked == [12, 13, 12]


def test_fp8_window_move_saturates_like_the_store(monkeypatch):
    """A rotated RoPE pair can leave the FP8 range its stored pair was clamped to: it saturates as
    the drafter's own store does instead of turning into NaN."""
    worker = _make_worker(monkeypatch)
    _window_pools(worker, num_layers=1, num_pages=2, dtype=torch.float8_e4m3fn)
    pool = worker._ctx_kv_buf[0]
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    pool.view(torch.uint8)[1, 0, 0, :, _WINDOW_LATENT:] = (
        torch.tensor(fp8_max).to(torch.float8_e4m3fn).view(torch.uint8)
    )
    pages = torch.arange(2, device="cuda")
    worker._ctx_block_tables = pages.view(1, -1).to(torch.int32)
    model = _WindowDraftModel()
    move = DraftContextWindowMove(source_start=_WINDOW_PAGE, dest_start=0, length=_WINDOW_PAGE)

    worker._move_draft_context_window(model, 0, move)

    rope = _token_rows(pool, pages, torch.arange(_WINDOW_PAGE, device="cuda")).float()
    rope = rope[:, _WINDOW_LATENT:]
    assert torch.isfinite(rope).all()
    assert rope.abs().max() == fp8_max
