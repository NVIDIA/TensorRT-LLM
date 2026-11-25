# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate-draft-KV policy follows the DFlash backend's storage ownership.

TRTLLM borrows the managed paged pool. VANILLA and FA4 own private context
buffers and must not allocate a redundant manager or be gated on its capacity.
"""

from types import SimpleNamespace
from typing import Optional

import pytest

from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache
from tensorrt_llm.llmapi import DFlashDecodingConfig, EagleDecodingConfig

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("backend", "uses_managed_pool"),
    [("VANILLA", False), ("FA4", False), ("TRTLLM", True)],
)
def test_dflash_draft_cache_matches_backend_storage(backend: str, uses_managed_pool: bool) -> None:
    config = DFlashDecodingConfig(max_draft_len=7, attention_backend=backend)
    assert config.spec_dec_mode.use_one_engine()
    assert not config._use_shared_kv_cache
    assert config._allow_separate_draft_kv_cache
    assert should_use_separate_draft_kv_cache(config) is uses_managed_pool


def test_default_dflash_has_no_redundant_mirror_or_target_draft_layers() -> None:
    from tensorrt_llm._torch.speculative.utils import get_num_spec_layers

    config = DFlashDecodingConfig(max_draft_len=7)
    assert config.attention_backend == "VANILLA"
    assert should_use_separate_draft_kv_cache(config) is False
    assert get_num_spec_layers(config) == 0


@pytest.mark.parametrize("backend", ["VANILLA", "FA4", "TRTLLM"])
def test_dflash_explicit_cache_opt_out_is_preserved(backend: str) -> None:
    config = DFlashDecodingConfig(max_draft_len=7, attention_backend=backend)
    config._allow_separate_draft_kv_cache = False
    assert should_use_separate_draft_kv_cache(config) is False


def test_eagle3_one_model_still_gets_a_separate_draft_kv_cache():
    """The DFlash exemption must not leak into other one-engine drafters."""
    config = EagleDecodingConfig(max_draft_len=3, speculative_model="/tmp/eagle3-draft")
    assert config.spec_dec_mode.is_eagle3_one_model()
    assert should_use_separate_draft_kv_cache(config) is True


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        pytest.param({}, "managed", id="contiguous-cursor"),
        pytest.param(None, "private", id="absent-record"),
        pytest.param({"draft": None}, "private", id="skipped-store"),
        pytest.param({"version": 0}, "reject", id="wrong-version"),
        pytest.param({"identity": "0" * 32}, "reject", id="wrong-protocol"),
        pytest.param({"transfer_id": 43}, "reject", id="wrong-transfer"),
        pytest.param({"prompt_len": 76}, "reject", id="wrong-prompt"),
        pytest.param(
            {"draft": {"draft_ctx_len": 76, "next_position": 77}}, "reject", id="partial-store"
        ),
    ],
)
def test_generic_disagg_cursor_validation_and_binding(change: dict | None, expected: str) -> None:
    from tensorrt_llm._torch.disaggregation.native.auxiliary import (
        decode_handoff_record,
        encode_handoff_record,
    )
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._ctx_buf_inited = worker._ctx_unified_cache = True
    worker._ctx_has_position_cap = False
    worker._max_ctx = 4096
    worker.generic_disagg_handoff_enabled = True
    worker._req_to_slot = {42: 0}
    worker._req_ctx_pos = {42: 77}
    worker._ctx_len_host = [77]
    worker._private_tail_requests = set()
    worker._untrusted_disagg_contexts = set()
    worker._pending_disagg_imports = {}
    worker.mapping = SimpleNamespace(enable_attention_dp=False, tp_size=1)
    source = SimpleNamespace(
        py_request_id=42, request_id=42, prompt_len=77, py_disaggregated_params=None
    )
    exported = worker.export_generic_disagg_context(source)
    assert exported["draft"] == {"draft_ctx_len": 77, "next_position": 77}
    worker._ctx_len_host[0] = 76
    assert worker.export_generic_disagg_context(source)["draft"] is None
    worker._req_to_slot.clear()
    worker._req_ctx_pos.clear()

    record = None if change is None else {**exported, **change}
    request = SimpleNamespace(
        py_request_id=123,
        prompt_len=77,
        context_phase_params=SimpleNamespace(req_id=42),
        py_disagg_handoff=decode_handoff_record(encode_handoff_record(record)),
    )
    error = worker.validate_generic_disagg_context(request)
    if expected == "reject":
        assert error is not None
        assert not worker._pending_disagg_imports and not worker._untrusted_disagg_contexts
        return
    assert error is None
    worker.prepare_generic_disagg_generation(request)
    if expected == "managed":
        assert worker._pending_disagg_imports == {123: (77, 77)}
        assert not worker._untrusted_disagg_contexts
    else:
        assert not worker._pending_disagg_imports
        assert worker._untrusted_disagg_contexts == {123}
    worker.release_request_state(123)
    assert not worker._pending_disagg_imports and not worker._untrusted_disagg_contexts


@pytest.mark.parametrize(
    "generic_handoff",
    [pytest.param(True, id="composed"), pytest.param(False, id="model-owned")],
)
def test_disagg_generation_keeps_model_and_worker_lifecycle_hooks(
    monkeypatch: pytest.MonkeyPatch, generic_handoff: bool
) -> None:
    from unittest.mock import Mock

    from tensorrt_llm._torch.pyexecutor import py_executor
    from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType

    request = SimpleNamespace(
        py_request_id=123,
        py_beam_width=1,
        context_phase_params=SimpleNamespace(first_gen_tokens=[2], draft_tokens=[3]),
    )
    worker = SimpleNamespace(
        generic_disagg_handoff_enabled=generic_handoff,
        prepare_generic_disagg_generation=Mock(),
        release_request_state=Mock(),
        release_private_tail=Mock(),
    )
    model = SimpleNamespace(
        spec_worker=worker, prepare_disagg_generation_request=Mock(), release_request_state=Mock()
    )
    disagg = Mock()
    disagg.completed_gen_receives.return_value = [request]
    disagg.try_finish_gen_receive.return_value = True
    resource_manager = Mock()
    resource_manager.resource_managers = {ResourceManagerType.SEQ_SLOT_MANAGER: Mock()}
    executor = SimpleNamespace(
        disagg=disagg,
        resource_manager=resource_manager,
        model_engine=SimpleNamespace(model=model, max_total_draft_tokens=1),
        draft_model_engine=None,
        sampler=Mock(),
        enable_attention_dp=False,
        enable_spec_decode=True,
        _prefetched_request_ids=set(),
        _setup_sampler_step=Mock(),
        _update_sampler_state_for_disagg_gen_request=Mock(return_value=True),
        _maybe_prepend_logprobs_and_logits=Mock(),
    )
    monkeypatch.setattr(py_executor, "_rebuild_disagg_imported_cache", Mock())
    monkeypatch.setattr(py_executor, "_import_first_gen_tokens", Mock())
    monkeypatch.setattr(
        py_executor.disagg_diagnostics, "DISAGG_TRANSFER_DIAGNOSTICS_ENABLED", False
    )

    py_executor.PyExecutor._prepare_disagg_gen_transmission_complete(executor, object())
    py_executor.PyExecutor._free_request_resources(executor, request)

    model.prepare_disagg_generation_request.assert_called_once_with(request)
    model.release_request_state.assert_called_once_with(123)
    if generic_handoff:
        worker.prepare_generic_disagg_generation.assert_called_once_with(request)
        worker.release_request_state.assert_called_once_with(123)
    else:
        worker.prepare_generic_disagg_generation.assert_not_called()
        worker.release_request_state.assert_not_called()
    worker.release_private_tail.assert_not_called()


@pytest.mark.parametrize(
    ("bound_to_pool", "reuse_addressable", "reads_rebased_view", "cur", "first_pos", "expected"),
    [
        # Managed pool: the block table is indexed by target position, so a
        # tail that reuse let the target start at first_pos is stored there.
        (True, False, True, 0, 8192, 8192),
        (True, True, True, 0, 8192, 8192),
        # A pool paired with the target's reuse holds the prefix below the
        # tail, so the tail goes on top of it even for a worker that reads
        # from position 0 (the Kimi K3 workers).
        (True, True, False, 0, 8192, 8192),
        # An unpaired pool read from position 0 numbers the tail from 0.
        (True, False, False, 0, 8192, 0),
        # A continuing chunk keeps the slot's stored length.
        (True, False, True, 4096, 4096, 4096),
        # Private arena: every request numbers from 0.
        (False, False, True, 0, 8192, 0),
        (False, False, True, 0, 0, 0),
    ],
)
def test_dflash_ctx_store_start_follows_the_pool_numbering(
    bound_to_pool: bool,
    reuse_addressable: bool,
    reads_rebased_view: bool,
    cur: int,
    first_pos: int,
    expected: int,
) -> None:
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._ctx_block_tables = object() if bound_to_pool else None
    worker._ctx_reuse_addressable = reuse_addressable
    worker._ctx_reads_rebased_view = reads_rebased_view
    assert worker._ctx_store_start(cur, first_pos) == expected


@pytest.mark.parametrize(
    ("cur", "first_pos", "expected"),
    [(0, 8192, None), (0, 65, None), (0, 64, 0), (0, 0, 0), (4096, 4096, 4096)],
)
def test_slot_from_zero_cannot_store_a_chunk_past_the_trail_on_a_windowed_pool(
    cur: int, first_pos: int, expected: Optional[int]
) -> None:
    """A worker that numbers slots from 0 (the Kimi K3 workers) on an unpaired
    sliding-window pool with a 64-token trail has no store start for a chunk
    whose slot starts more than the trail past 0: the manager releases the
    pages that slot would address. Within the trail every entry stays on a page
    it holds."""
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._ctx_block_tables = object()
    worker._ctx_reuse_addressable = False
    worker._ctx_reads_rebased_view = False
    worker._ctx_pool_windowed = True
    worker._ctx_pool_trail = 64
    assert worker._ctx_store_start(cur, first_pos) == expected


@pytest.mark.parametrize(
    ("capped_unified", "first_pos", "slen"),
    [
        pytest.param(False, 8192, 4, id="unaddressable-window"),
        pytest.param(True, 0, 8192, id="oversized-prompt"),
        pytest.param(True, 2047, 1, id="unverified-reused-prefix"),
    ],
)
def test_skipped_prefill_does_not_publish_draft_history(
    monkeypatch: pytest.MonkeyPatch,
    capped_unified: bool,
    first_pos: int,
    slen: int,
) -> None:
    import collections

    import torch

    from tensorrt_llm._torch.speculative import dflash
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._ctx_block_tables = object()
    worker._ctx_block_counts = torch.tensor([300])
    worker._ctx_page_size = 64
    worker._max_ctx = 4096 if capped_unified else 1 << 17
    worker._ctx_unified_cache = capped_unified
    worker._ctx_has_position_cap = capped_unified
    worker._ctx_buf_inited = True
    worker._private_tail_requests = set()
    worker._ctx_reuse_addressable = capped_unified
    worker._ctx_reads_rebased_view = capped_unified
    worker._ctx_pool_windowed = True
    worker._ctx_pool_trail = 64
    worker._req_ctx_pos = {}
    worker._req_to_slot = {}
    worker._free_slots = collections.deque([2])
    worker._ctx_len_host = [0, 0, 0]
    worker._ctx_base_host = [0, 0, 0]
    worker._ctx_len = torch.zeros(3, dtype=torch.long)
    worker._ctx_base_pages = torch.zeros(3, dtype=torch.long)
    warnings = []
    monkeypatch.setattr(dflash.logger, "warning", warnings.append)

    draft_model = SimpleNamespace(
        fc=object(), hidden_norm=object(), project_target_hidden=lambda hs: hs
    )
    spec_metadata = SimpleNamespace(
        get_hidden_states=lambda n: torch.zeros(n, 8),
        host_position_ids=torch.arange(first_pos, first_pos + slen),
        request_ids=[7],
    )
    attn_metadata = SimpleNamespace(
        num_ctx_tokens=slen, num_contexts=1, _seq_lens=torch.tensor([slen])
    )
    worker._store_prefill_context(
        draft_model,
        spec_metadata,
        attn_metadata,
        torch.arange(first_pos, first_pos + slen),
        slen,
    )

    assert worker._ctx_len_host[2] == 0 and worker._ctx_len.tolist()[2] == 0
    assert worker.export_disagg_context(7) is None
    if capped_unified:
        assert worker._private_tail_requests == {7}
        assert worker._req_ctx_pos[7] == first_pos + slen
        return

    assert 7 not in worker._req_to_slot and 7 not in worker._req_ctx_pos
    assert list(worker._free_slots) == [2]
    assert len(warnings) == 1
    assert "skipping its context store" in warnings[0]
    assert "drafts nothing" not in warnings[0]


@pytest.mark.parametrize(
    ("cur", "start", "reuse_addressable", "expected"),
    [
        # Unpaired pool, tail stored at a reused prefix's end: the pages below
        # it hold no drafter K/V and are hidden from the block decode.
        (0, 8192, False, 128),
        # A start inside a page rounds the base up; the tail tokens in that
        # page stay stored but unread.
        (0, 8200, False, 129),
        # A continuing chunk numbers from the stored length: no base change.
        (4096, 4096, False, None),
        # A pool paired with the target's reuse holds the prefix's drafter K/V.
        (0, 8192, True, None),
    ],
)
def test_dflash_ctx_base_for_store(
    cur: int, start: int, reuse_addressable: bool, expected: int | None
) -> None:
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._ctx_page_size = 64
    worker._ctx_reuse_addressable = reuse_addressable
    assert worker._ctx_base_for_store(cur, start) == expected


def test_dflash_rebased_ctx_rows_hide_the_reused_prefix() -> None:
    """A tail stored after a reused prefix on an unpaired pool. The block decode
    reads the slot's row rebased to the tail and a context length shortened by
    the prefix, so the never-written prefix pages (poisoned here) are not within
    reach, and every tail position still resolves to the page and offset the
    store wrote it to."""
    import torch

    from tensorrt_llm._torch.speculative.dflash import dflash_rebased_ctx_rows

    page_size, first_pos, tail = 4, 24, 5
    row = torch.tensor([[7, 3, 11, 0, 5, 9, 13, 2, 6, 1, 15, 4, 8, 10, 12, 14]], dtype=torch.int32)
    pool = torch.zeros(16, page_size)
    pool[row[0, : first_pos // page_size].long()] = float("nan")
    for pos in range(first_pos, first_pos + tail):
        pool[row[0, pos // page_size], pos % page_size] = pos

    base_pages = torch.tensor([first_pos // page_size])
    view = dflash_rebased_ctx_rows(row, base_pages)
    num_ctx = first_pos + tail - base_pages.item() * page_size
    assert view.shape == row.shape
    readable = view[0, : -(-num_ctx // page_size)].long()
    assert torch.isfinite(pool[readable]).all()
    for pos in range(num_ctx):
        assert pool[view[0, pos // page_size], pos % page_size] == first_pos + pos
    # A slot without a base reads its row unchanged.
    assert torch.equal(dflash_rebased_ctx_rows(row, torch.zeros(1, dtype=torch.long)), row)


@pytest.mark.parametrize(
    ("first_pos", "tail"),
    [
        # A reused prefix ending inside a 64-token page with a one-token tail:
        # the base rounds up to the next page and the context stays shorter
        # than it for 55 steps.
        (8200, 1),
        # One token into a page: the first write offsets overlap the stored
        # tail, which sits below the base and is never read.
        (8193, 1),
        # A page-aligned prefix end is never short of its base.
        (8192, 1),
    ],
)
def test_dflash_rebased_ctx_view_writes_the_block_into_an_owned_page(
    first_pos: int, tail: int
) -> None:
    """A tail stored after a reused prefix on an unpaired pool, with the mirror
    funded the way the manager funds it: pages up to the one holding the
    context plus the draft tokens, the window below it, nothing past it. The
    block decode writes the block's own K/V at rebased positions
    num_ctx .. num_ctx + block_size through the view it reads, so at every
    context length those positions must resolve to a funded entry, including
    the steps on which the context is still shorter than the rounded-up base
    and the decode reads nothing. Entry ``base`` is unfunded on those steps and
    holds 0, the pool page another request owns."""
    import torch

    from tensorrt_llm._torch.speculative.dflash import DFlashWorker, dflash_rebased_ctx_view

    page_size, block_size, draft_len, window_pages = 64, 3, 3, 64
    worker = object.__new__(DFlashWorker)
    worker._ctx_page_size = page_size
    worker._ctx_reuse_addressable = False
    base = worker._ctx_base_for_store(0, first_pos)
    assert base == -(-first_pos // page_size)
    base_pages = torch.tensor([base])

    def funded_row(ctx_len: int) -> torch.Tensor:
        last = (ctx_len + draft_len - 1) // page_size
        row = torch.zeros(1, base + 8, dtype=torch.int32)
        row[0, last + 1 - window_pages : last + 1] = (
            torch.arange(last + 1 - window_pages, last + 1, dtype=torch.int32) + 1000
        )
        return row

    blind_steps = 0
    for ctx_len in range(first_pos + tail, (base + 1) * page_size):
        row = funded_row(ctx_len)
        view, num_ctx = dflash_rebased_ctx_view(row, base_pages, torch.tensor([ctx_len]), page_size)
        assert view.shape == row.shape
        blind = ctx_len < base * page_size
        blind_steps += blind
        assert num_ctx.item() == max(0, ctx_len - base * page_size)
        if blind:
            assert view[0, 0] == row[0, ctx_len // page_size]
            if (ctx_len + draft_len - 1) // page_size < base:
                # Entry ``base`` is funded only once the draft tokens reach it.
                assert row[0, base] == 0
        for pos in range(num_ctx.item() + block_size):
            entry = view[0, pos // page_size]
            assert entry >= 1000, (ctx_len, pos)
            if pos < num_ctx.item():
                # A read position p is absolute position base * page_size + p.
                assert entry == row[0, (base * page_size + pos) // page_size]
    assert blind_steps == max(0, base * page_size - first_pos - tail)


def test_dflash_write_ctx_len_zero_length_reset_wins_over_a_pending_base() -> None:
    """_store_prefill_context records a chunk's base pages before the overflow
    check that may reset the same slot to length 0 in the same call. The reset
    takes precedence: the slot ends with length 0 and base 0 on the host
    mirrors and the device tensors alike, while a slot that was not reset keeps
    its new length and base."""
    import torch

    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._ctx_len_host = [0, 512, 64]
    worker._ctx_base_host = [0, 8, 1]
    worker._ctx_len = torch.tensor(worker._ctx_len_host)
    worker._ctx_base_pages = torch.tensor(worker._ctx_base_host)

    # Slot 1: base recorded, then reset by the overflow check. Slot 2: a
    # continuing chunk with a new base.
    worker._write_ctx_len({1: 0, 2: 96}, {1: 128, 2: 2})

    assert worker._ctx_len_host == [0, 0, 96]
    assert worker._ctx_base_host == [0, 0, 2]
    assert worker._ctx_len.tolist() == [0, 0, 96]
    assert worker._ctx_base_pages.tolist() == [0, 0, 2]


@pytest.mark.parametrize(
    ("bound_to_pool", "unified", "page_aligned", "reuse_addressable", "position", "expected"),
    [
        # A disaggregated generation row: the slot starts empty at the first
        # position of the page holding its position and reads from there,
        # paired with reuse or not (the mirror of such a request never looked
        # a prefix up).
        (True, False, True, False, 8200, (8192, 128)),
        (True, False, True, True, 8200, (8192, 128)),
        (True, False, True, False, 8192, (8192, 128)),
        # Clamped to the drafter's context ceiling, base included.
        (True, False, True, False, 1 << 20, (65536, 1024)),
        # A pool window shorter than a page may already have released that
        # page: the slot starts at the position, reading from the next page.
        (True, False, False, False, 8200, (8200, 129)),
        # A unified manager holds the drafter K/V below the position (the
        # draft layers travel with the target's prompt KV): the slot starts at
        # the position and reads from base 0, clamped like any other.
        (True, True, True, True, 8200, (8200, 0)),
        (True, True, True, True, 1 << 20, (65536, 0)),
        # The private arena is slot-addressed and starts at 0.
        (False, False, True, False, 8200, (0, None)),
    ],
)
def test_dflash_generation_slot_seed_is_absolute_with_a_base(
    bound_to_pool: bool,
    unified: bool,
    page_aligned: bool,
    reuse_addressable: bool,
    position: int,
    expected: tuple,
) -> None:
    """A generation row that gets a fresh slot (a disaggregated generation
    worker's first step, or a request whose slot was evicted) starts in the
    coordinates of its draft mirror, at a page boundary no further than a page
    below its absolute position, with every page below the base hidden from
    the block decode -- or, on a unified manager, at the position itself with
    nothing hidden."""
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._ctx_block_tables = object() if bound_to_pool else None
    worker._ctx_unified_cache = unified
    worker._ctx_reuse_addressable = reuse_addressable
    worker._ctx_seed_page_aligned = page_aligned
    worker._ctx_page_size = 64
    worker._max_ctx = 65536
    updates, bases = {}, {}
    worker._seed_generation_slot(3, position, updates, bases)
    assert (updates[3], bases.get(3)) == expected


def test_dflash_assign_slot_resets_the_base_of_a_reused_slot() -> None:
    """A slot freed and handed out again inside one batched update starts
    with base 0, not the previous occupant's base."""
    from collections import deque

    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._req_to_slot = {7: 0}
    worker._free_slots = deque()
    updates, bases = {}, {}
    assert worker._assign_slot(8, reset=True, updates=updates, bases=bases) is None
    assert worker._assign_slot(7, reset=True, updates=updates, bases=bases) == 0
    assert (updates, bases) == ({0: 0}, {0: 0})


def test_dflash_rollback_restores_the_base_pages() -> None:
    """A failed or warmup forward rolls the slot lengths back to the
    pre-forward snapshot, and the base pages with them, on the device and the
    host mirror; a base seeded during the forward does not survive."""
    import torch

    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._ctx_len_host = [0, 512, 64]
    worker._ctx_base_host = [0, 8, 0]
    worker._ctx_len = torch.tensor(worker._ctx_len_host)
    worker._ctx_base_pages = torch.tensor(worker._ctx_base_host)
    worker._req_ctx_pos = {}
    worker._saved_ctx_len = worker._ctx_len.clone()
    worker._saved_ctx_len_host = list(worker._ctx_len_host)
    worker._saved_ctx_base = worker._ctx_base_pages.clone()
    worker._saved_ctx_base_host = list(worker._ctx_base_host)
    worker._saved_req_ctx_pos = {}

    worker._write_ctx_len({1: 0, 2: 8200}, {2: 129})
    worker._restore_ctx_len_device()
    worker._restore_ctx_len_host()

    assert worker._ctx_len.tolist() == [0, 512, 64]
    assert worker._ctx_base_pages.tolist() == [0, 8, 0]
    assert worker._ctx_len_host == [0, 512, 64]
    assert worker._ctx_base_host == [0, 8, 0]


@pytest.mark.parametrize(
    ("windows", "expected"),
    [
        (None, True),
        ([None, None], True),
        ([None, 4096], True),
        ([128, 4096], True),
        ([None, 64], False),
    ],
)
def test_dflash_windows_span_a_page(windows: list | None, expected: bool) -> None:
    """A full-attention layer (window None) keeps every page; only a window
    shorter than a page rules out the page-aligned generation seed."""
    from tensorrt_llm._torch.speculative.dflash import dflash_windows_span_a_page

    assert dflash_windows_span_a_page(windows, 128) is expected


def test_warmup_reseeds_generation_slots_from_the_batch_positions() -> None:
    """CUDA-graph warmup reuses dummy request ids across batches. A generation
    row's slot that kept a longer batch's length is seeded from this batch's
    position (capped at the context capacity); context rows, rows without a
    slot and the attention-DP dummy keep their slots as they are."""
    import torch

    from tensorrt_llm._torch.pyexecutor.llm_request import ATTENTION_DP_DUMMY_REQUEST_ID
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._max_ctx = 1000
    worker._req_to_slot = {5: 3, 7: 0, 8: 1, ATTENTION_DP_DUMMY_REQUEST_ID: 2}
    worker._ctx_len = torch.tensor([900, 900, 900, 900, 0])
    worker._ctx_len_host = [900, 900, 900, 900, 0]
    worker._ctx_base_pages = torch.zeros(5, dtype=torch.long)
    worker._ctx_base_host = [0] * 5
    spec_metadata = SimpleNamespace(
        request_ids=[5, 7, 8, ATTENTION_DP_DUMMY_REQUEST_ID, 9],
        num_generations=4,
        seq_lens=[3, 2, 2, 2, 2],
        host_position_ids=torch.tensor([0, 1, 2, 40, 41, 2000, 2001, 10, 11, 60, 61]),
    )
    worker._reseed_warmup_generation_slots(spec_metadata)
    assert worker._ctx_len.tolist() == [40, 1000, 900, 900, 0]
    assert worker._ctx_len_host == [40, 1000, 900, 900, 0]
