# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SpecSampler's per-slot store with and without a model's SpecStepFold."""

from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.runtime_hooks import SpecStepFold
from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler
from tensorrt_llm.bindings import SamplingConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

MAX_DRAFT_LEN = 3
MAX_NUM_SEQUENCES = 6
SEQ_SLOTS = [4, 1]


class _RecordingStepFold:
    """A SpecStepFold whose store decisions are fixed per test; it records what the sampler asks."""

    def __init__(self, rows_stored: bool = False, scatter_taken: bool = False) -> None:
        self._rows_stored = rows_stored
        self._scatter_taken = scatter_taken
        self.store: Optional[object] = None
        self.rows_stored_calls: list[tuple[int, list[int]]] = []
        self.scatter_calls: list[dict[str, torch.Tensor]] = []

    def bind_engine_buffers(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        position_id_offsets: torch.Tensor,
        kv_lens_offsets: torch.Tensor,
        max_batch_size: int,
    ) -> None:
        raise AssertionError("the sampler never binds engine buffers")

    def bind_sampler_store(self, store: object) -> None:
        self.store = store

    def begin_step(self, seq_slots: Optional[list[int]], num_rows: int) -> bool:
        raise AssertionError("the sampler never begins a step")

    def gather(self, *args: object) -> bool:
        raise AssertionError("the sampler never gathers")

    def sampler_rows_stored(self, outputs: dict, num_skip: int, seq_slots: list[int]) -> bool:
        self.rows_stored_calls.append((num_skip, list(seq_slots)))
        return self._rows_stored

    def scatter(
        self,
        seq_slots: torch.Tensor,
        new_tokens: torch.Tensor,
        next_new_tokens: torch.Tensor,
        new_tokens_lens: torch.Tensor,
        next_draft_tokens: torch.Tensor,
    ) -> bool:
        self.scatter_calls.append(
            dict(
                seq_slots=seq_slots.clone(),
                new_tokens=new_tokens.clone(),
                next_new_tokens=next_new_tokens.clone(),
                new_tokens_lens=new_tokens_lens.clone(),
                next_draft_tokens=next_draft_tokens.clone(),
            )
        )
        return self._scatter_taken


def _fold(**kwargs: bool) -> SpecStepFold:
    return _RecordingStepFold(**kwargs)


def _sampler() -> SpecSampler:
    sampler = SpecSampler(
        TorchSampler.Args(
            max_seq_len=64,
            max_draft_len=MAX_DRAFT_LEN,
            max_num_sequences=MAX_NUM_SEQUENCES,
            max_beam_width=1,
            max_total_draft_tokens=MAX_DRAFT_LEN,
            disable_overlap_scheduler=True,
        )
    )
    # The store is allocated uninitialized; give it a recognizable baseline.
    for index, tensor in enumerate(
        (
            sampler.store.new_tokens,
            sampler.store.next_new_tokens,
            sampler.store.next_draft_tokens,
            sampler.store.new_tokens_lens,
        )
    ):
        tensor.fill_(-100 - index)
    return sampler


def _store_snapshot(sampler: SpecSampler) -> dict[str, torch.Tensor]:
    store = sampler.store
    return dict(
        new_tokens=store.new_tokens.clone(),
        next_new_tokens=store.next_new_tokens.clone(),
        next_draft_tokens=store.next_draft_tokens.clone(),
        new_tokens_lens=store.new_tokens_lens.clone(),
    )


def _assert_store_equals(sampler: SpecSampler, expected: dict[str, torch.Tensor]) -> None:
    torch.cuda.synchronize()
    for name, tensor in _store_snapshot(sampler).items():
        assert torch.equal(tensor, expected[name]), name


def _step(draft_width: int):
    requests = []
    for request_id, seq_slot in enumerate(SEQ_SLOTS):
        request = LlmRequest(
            request_id=request_id,
            max_new_tokens=8,
            input_tokens=[1, 2, 3],
            sampling_config=SamplingConfig(),
            seq_slot=seq_slot,
            is_streaming=False,
        )
        request.py_draft_tokens = [7] * MAX_DRAFT_LEN
        requests.append(request)
    scheduled = ScheduledRequests()
    scheduled.generation_requests = requests
    rows = len(requests) + 1  # plus a CUDA-graph padding row the sampler must not store
    outputs = dict(
        new_tokens=torch.arange(rows * (draft_width + 1), dtype=torch.int32, device="cuda").view(
            rows, draft_width + 1
        ),
        new_tokens_lens=torch.arange(1, rows + 1, dtype=torch.int32, device="cuda"),
        next_draft_tokens=(
            100 + torch.arange(rows * draft_width, dtype=torch.int32, device="cuda")
        ).view(rows, draft_width),
        next_new_tokens=(
            200 + torch.arange(rows * (draft_width + 1), dtype=torch.int32, device="cuda")
        ).view(rows, draft_width + 1),
    )
    return scheduled, outputs


def _reference_store(before: dict[str, torch.Tensor], outputs: dict) -> dict[str, torch.Tensor]:
    """What the sampler has always stored: each live row's outputs at its slot, zero-padded."""
    expected = {name: tensor.clone() for name, tensor in before.items()}
    count = len(SEQ_SLOTS)
    slots = torch.tensor(SEQ_SLOTS, dtype=torch.long, device="cuda")

    def padded(tensor: torch.Tensor, width: int) -> torch.Tensor:
        tensor = tensor[:count, :width]
        return torch.nn.functional.pad(tensor, (0, width - tensor.shape[1]))

    accepted_width = expected["new_tokens"].shape[0]
    wire_width = expected["next_new_tokens"].shape[0]
    draft_width = expected["next_draft_tokens"].shape[1]
    expected["new_tokens"].squeeze(-1).T.index_copy_(
        0, slots, padded(outputs["new_tokens"], accepted_width)
    )
    expected["next_new_tokens"].squeeze(-1).T.index_copy_(
        0, slots, padded(outputs["next_new_tokens"], wire_width)
    )
    expected["new_tokens_lens"].index_copy_(0, slots, outputs["new_tokens_lens"][:count])
    expected["next_draft_tokens"].index_copy_(
        0, slots, padded(outputs["next_draft_tokens"], draft_width)
    )
    return expected


@pytest.mark.parametrize("draft_width", [MAX_DRAFT_LEN, MAX_DRAFT_LEN - 1])
def test_without_a_fold_the_store_is_written_as_before(draft_width):
    sampler = _sampler()
    scheduled, outputs = _step(draft_width)
    before = _store_snapshot(sampler)

    state = sampler.sample_async(scheduled, outputs, [])

    assert sampler.spec_step_fold is None
    expected = _reference_store(before, outputs)
    _assert_store_equals(sampler, expected)
    state.sampler_event.synchronize()
    assert torch.equal(state.host.new_tokens, expected["new_tokens"].cpu())
    assert torch.equal(state.host.next_draft_tokens, expected["next_draft_tokens"].cpu())


def test_binding_a_fold_hands_it_the_sampler_store():
    sampler = _sampler()
    fold = _fold()

    sampler.bind_spec_step_fold(fold)

    assert sampler.spec_step_fold is fold
    assert fold.store is sampler.store


def test_rows_the_forward_already_stored_are_left_alone():
    sampler = _sampler()
    fold = _fold(rows_stored=True)
    sampler.bind_spec_step_fold(fold)
    scheduled, outputs = _step(MAX_DRAFT_LEN)
    before = _store_snapshot(sampler)

    state = sampler.sample_async(scheduled, outputs, [])

    assert fold.rows_stored_calls == [(0, SEQ_SLOTS)]
    assert fold.scatter_calls == []
    _assert_store_equals(sampler, before)
    # The host copies still read the (fold-owned) store.
    state.sampler_event.synchronize()
    assert torch.equal(state.host.new_tokens, before["new_tokens"].cpu())


def test_a_fold_scatter_replaces_the_index_copies():
    sampler = _sampler()
    fold = _fold(scatter_taken=True)
    sampler.bind_spec_step_fold(fold)
    scheduled, outputs = _step(MAX_DRAFT_LEN)
    before = _store_snapshot(sampler)

    sampler.sample_async(scheduled, outputs, [])

    _assert_store_equals(sampler, before)
    assert len(fold.scatter_calls) == 1
    call = fold.scatter_calls[0]
    count = len(SEQ_SLOTS)
    assert call["seq_slots"].tolist() == SEQ_SLOTS
    assert call["seq_slots"].dtype == torch.long and call["seq_slots"].is_cuda
    assert torch.equal(call["new_tokens"], outputs["new_tokens"][:count])
    assert torch.equal(call["next_new_tokens"], outputs["next_new_tokens"][:count])
    assert torch.equal(call["new_tokens_lens"], outputs["new_tokens_lens"][:count])
    assert torch.equal(call["next_draft_tokens"], outputs["next_draft_tokens"][:count])


@pytest.mark.parametrize("draft_width", [MAX_DRAFT_LEN, MAX_DRAFT_LEN - 1])
def test_a_fold_that_declines_leaves_the_sampler_store_path(draft_width):
    sampler = _sampler()
    fold = _fold(rows_stored=False, scatter_taken=False)
    sampler.bind_spec_step_fold(fold)
    scheduled, outputs = _step(draft_width)
    before = _store_snapshot(sampler)

    sampler.sample_async(scheduled, outputs, [])

    assert fold.rows_stored_calls == [(0, SEQ_SLOTS)]
    assert len(fold.scatter_calls) == 1
    _assert_store_equals(sampler, _reference_store(before, outputs))


def _host_outputs(state) -> dict[str, torch.Tensor]:
    state.sampler_event.synchronize()
    return dict(
        new_tokens=state.host.new_tokens.clone(),
        new_tokens_lens=state.host.new_tokens_lens.clone(),
        next_draft_tokens=state.host.next_draft_tokens.clone(),
    )


def test_store_copies_stay_on_the_main_stream_by_default():
    sampler = _sampler()
    scheduled, outputs = _step(MAX_DRAFT_LEN)

    state = sampler.sample_async(scheduled, outputs, [])

    assert not sampler.d2h_side_stream_copies
    assert state.sampler_event.side_stream_event is None


@pytest.mark.parametrize("draft_width", [MAX_DRAFT_LEN, MAX_DRAFT_LEN - 1])
def test_side_stream_store_copies_match_the_main_stream_copies(draft_width):
    scheduled, outputs = _step(draft_width)
    on_main = _host_outputs(_sampler().sample_async(scheduled, outputs, []))
    sampler = _sampler()
    sampler.d2h_side_stream_copies = True

    state = sampler.sample_async(scheduled, outputs, [])

    assert state.sampler_event.side_stream_event is not None
    on_side = _host_outputs(state)
    for name, tensor in on_main.items():
        assert torch.equal(on_side[name], tensor), name


def test_a_fold_that_stored_the_rows_keeps_the_copies_on_the_main_stream():
    sampler = _sampler()
    sampler.bind_spec_step_fold(_fold(rows_stored=True))
    sampler.d2h_side_stream_copies = True
    scheduled, outputs = _step(MAX_DRAFT_LEN)

    state = sampler.sample_async(scheduled, outputs, [])

    assert state.sampler_event.side_stream_event is None


def test_a_fold_that_awaits_the_store_copies_moves_them_to_the_side_stream():
    """A fold with store_copies_done: the copies go on the side stream although its forward stored the rows, and
    the event is recorded after them."""
    scheduled, outputs = _step(MAX_DRAFT_LEN)
    fold = _fold(rows_stored=True)
    fold.store_copies_done = torch.cuda.Event(external=True)
    fold.store_copies_done.record()
    sampler = _sampler()
    sampler.bind_spec_step_fold(fold)
    sampler.d2h_side_stream_copies = True
    expected = _store_snapshot(sampler)

    state = sampler.sample_async(scheduled, outputs, [])

    assert state.sampler_event.side_stream_event is not None
    state.sampler_event.synchronize()
    fold.store_copies_done.synchronize()
    host = _host_outputs(state)
    for name in ("new_tokens", "new_tokens_lens", "next_draft_tokens"):
        assert torch.equal(host[name], expected[name].cpu()), name


def test_a_fold_that_awaits_the_store_copies_keeps_them_on_the_main_stream_when_side_copies_are_off():
    scheduled, outputs = _step(MAX_DRAFT_LEN)
    fold = _fold(rows_stored=True)
    fold.store_copies_done = torch.cuda.Event(external=True)
    sampler = _sampler()
    sampler.bind_spec_step_fold(fold)

    state = sampler.sample_async(scheduled, outputs, [])

    assert state.sampler_event.side_stream_event is None


# HO-04 on GB: the folds the M3.1 sm_103 targets bind never store the rows in the forward (sampler_rows_stored is
# always False) and carry no store_copies_done.


class _ScatteringStepFold(_RecordingStepFold):
    """The shape of the targets' M3SpecStepFold (Eagle3): the forward stores nothing, and ``scatter`` writes the
    sampler store itself (as m3_spec_tail.slot_scatter does) and takes the step."""

    def __init__(self) -> None:
        super().__init__(rows_stored=False, scatter_taken=True)

    def scatter(self, seq_slots, new_tokens, next_new_tokens, new_tokens_lens, next_draft_tokens):
        taken = super().scatter(
            seq_slots, new_tokens, next_new_tokens, new_tokens_lens, next_draft_tokens
        )
        store = self.store
        count = seq_slots.shape[0]

        def padded(tensor: torch.Tensor, width: int) -> torch.Tensor:
            tensor = tensor[:count, :width]
            return torch.nn.functional.pad(tensor, (0, width - tensor.shape[1]))

        store.new_tokens.squeeze(-1).T.index_copy_(
            0, seq_slots, padded(new_tokens, store.new_tokens.shape[0])
        )
        store.next_new_tokens.squeeze(-1).T.index_copy_(
            0, seq_slots, padded(next_new_tokens, store.next_new_tokens.shape[0])
        )
        store.new_tokens_lens.index_copy_(0, seq_slots, new_tokens_lens[:count])
        store.next_draft_tokens.index_copy_(
            0, seq_slots, padded(next_draft_tokens, store.next_draft_tokens.shape[1])
        )
        return taken


class _AwaitedEvent:
    """Stands in for the side-stream event and records that SamplerEvent.synchronize waited on it."""

    def __init__(self, event: torch.cuda.Event) -> None:
        self._event = event
        self.waits = 0

    def synchronize(self) -> None:
        self.waits += 1
        self._event.synchronize()


@pytest.mark.parametrize("draft_width", [MAX_DRAFT_LEN, MAX_DRAFT_LEN - 1])
def test_side_stream_copies_engage_under_the_gb_eagle3_fold(draft_width):
    scheduled, outputs = _step(draft_width)
    stock = _sampler()
    stock.bind_spec_step_fold(_ScatteringStepFold())
    on_main = _host_outputs(stock.sample_async(scheduled, outputs, []))
    sampler = _sampler()
    fold = _ScatteringStepFold()
    sampler.bind_spec_step_fold(fold)
    sampler.d2h_side_stream_copies = True
    expected = _reference_store(_store_snapshot(sampler), outputs)

    state = sampler.sample_async(scheduled, outputs, [])

    assert not hasattr(fold, "store_copies_done")
    assert fold.rows_stored_calls == [(0, SEQ_SLOTS)] and len(fold.scatter_calls) == 1
    side_event = state.sampler_event.side_stream_event
    assert side_event is not None
    awaited = _AwaitedEvent(side_event)
    state.sampler_event.side_stream_event = awaited
    host = _host_outputs(state)
    assert awaited.waits == 1
    _assert_store_equals(sampler, expected)
    for name, tensor in on_main.items():
        assert torch.equal(host[name], tensor), name
        assert torch.equal(host[name], expected[name].cpu()), name


@pytest.mark.parametrize("draft_width", [MAX_DRAFT_LEN, MAX_DRAFT_LEN - 1])
def test_side_stream_copies_engage_under_the_gb_dspark_fold(draft_width):
    """The targets' M3DSparkStepFold (standalone DSpark): scatter declines every step, so the store is written only by
    the sampler's own _store_outputs, and the copies go on the side stream."""
    scheduled, outputs = _step(draft_width)
    sampler = _sampler()
    fold = _fold(rows_stored=False, scatter_taken=False)
    sampler.bind_spec_step_fold(fold)
    sampler.d2h_side_stream_copies = True
    before = _store_snapshot(sampler)
    expected = _reference_store(before, outputs)
    store_writes = []
    store_outputs = sampler._store_outputs

    def recording_store_outputs(*args):
        store_writes.append(args[0])
        return store_outputs(*args)

    sampler._store_outputs = recording_store_outputs

    state = sampler.sample_async(scheduled, outputs, [])

    assert not hasattr(fold, "store_copies_done")
    assert store_writes == [SEQ_SLOTS]
    assert len(fold.scatter_calls) == 1  # offered, declined
    assert state.sampler_event.side_stream_event is not None
    host = _host_outputs(state)
    _assert_store_equals(sampler, expected)
    for name, tensor in host.items():
        assert torch.equal(tensor, expected[name].cpu()), name


def test_flags_off_issues_the_stock_copies_in_the_stock_order():
    """Flags off, sample_async issues exactly the stock calls: four _copy_to_host (new_tokens, new_tokens_lens,
    next_draft_tokens, verify_lens) and one argument-less _record_sampler_event."""
    sampler = _sampler()
    scheduled, outputs = _step(MAX_DRAFT_LEN)
    rows = outputs["new_tokens_lens"].shape[0]
    outputs["verify_lens"] = torch.full(
        (rows,), MAX_DRAFT_LEN + 1, dtype=torch.int32, device="cuda"
    )
    outputs["verify_lens_in_output_order"] = True
    copies = []
    events = []
    copy_to_host = sampler._copy_to_host
    record_sampler_event = sampler._record_sampler_event

    def recording_copy(src):
        copies.append(src)
        return copy_to_host(src)

    def recording_event(*args, **kwargs):
        events.append((args, kwargs))
        return record_sampler_event(*args, **kwargs)

    sampler._copy_to_host = recording_copy
    sampler._record_sampler_event = recording_event

    state = sampler.sample_async(scheduled, outputs, [])

    assert len(copies) == 4
    assert copies[0] is sampler.store.new_tokens
    assert copies[1] is sampler.store.new_tokens_lens
    assert copies[2] is sampler.store.next_draft_tokens
    assert copies[3].data_ptr() == outputs["verify_lens"].data_ptr()
    assert copies[3].shape[0] == len(SEQ_SLOTS)
    assert events == [((), {})]
    assert state.sampler_event.side_stream_event is None


def test_side_stream_copies_fall_back_to_the_main_stream_with_the_async_worker():
    sampler = _sampler()
    sampler.d2h_side_stream_copies = True
    scheduled, outputs = _step(MAX_DRAFT_LEN)
    copies = []

    def plain_copy(src):
        # What _copy_to_host does without a worker thread; the worker itself is not started here.
        copies.append(src)
        dest = torch.empty_like(src, device="cpu", pin_memory=True)
        dest.copy_(src, non_blocking=True)
        return dest

    sampler._async_worker_active = lambda: True
    sampler._copy_to_host = plain_copy

    state = sampler.sample_async(scheduled, outputs, [])

    assert len(copies) == 3
    assert state.sampler_event.side_stream_event is None


def test_a_sampler_built_without_init_reads_both_hooks_off():
    sampler = object.__new__(SpecSampler)

    assert sampler.d2h_side_stream_copies is False
    assert sampler.static_slot_ids is False
    assert sampler._slot_ids_cuda is None and sampler._slot_ids_host is None


# HO-12a: static slot ids.


def test_static_slot_ids_are_off_by_default():
    assert not _sampler().static_slot_ids


@pytest.mark.parametrize("draft_width", [MAX_DRAFT_LEN, MAX_DRAFT_LEN - 1])
def test_static_slot_ids_store_the_same_rows(draft_width):
    scheduled, outputs = _step(draft_width)
    plain = _sampler()
    expected = _reference_store(_store_snapshot(plain), outputs)
    sampler = _sampler()
    sampler.static_slot_ids = True

    sampler.sample_async(scheduled, outputs, [])
    sampler.sample_async(scheduled, outputs, [])  # second step reuses the uploaded ids

    _assert_store_equals(sampler, expected)


def test_static_slot_ids_follow_a_changed_batch():
    sampler = _sampler()
    sampler.static_slot_ids = True

    first_view = sampler._device_slot_ids([4, 1])
    first = first_view.clone()
    # Unchanged ids reuse the sampler's buffer without a new upload. The returned tensor is a view of that buffer,
    # which the next call rewrites, so snapshot it before changing the batch.
    reused_view = sampler._device_slot_ids([4, 1])
    assert reused_view.data_ptr() == first_view.data_ptr()
    reused = reused_view.clone()
    changed = sampler._device_slot_ids([2, 5, 0]).clone()

    torch.cuda.synchronize()
    assert first.tolist() == [4, 1]
    assert reused.tolist() == [4, 1]
    assert changed.tolist() == [2, 5, 0]


@pytest.fixture
def slot_id_uploads(monkeypatch):
    """Counts the copies into the sampler's slot-id buffer (bound with ``.watch(sampler)``)."""

    class _Uploads:
        sampler: Optional[SpecSampler] = None
        count = 0

        def watch(self, sampler: SpecSampler) -> "_Uploads":
            self.sampler = sampler
            return self

    uploads = _Uploads()
    copy_ = torch.Tensor.copy_

    def counting_copy_(dst, src, *args, **kwargs):
        buffer = uploads.sampler._slot_ids_cuda if uploads.sampler is not None else None
        if (
            buffer is not None
            and dst.is_cuda
            and dst.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
        ):
            uploads.count += 1
        return copy_(dst, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", counting_copy_)
    return uploads


def test_steady_slot_ids_upload_once_over_many_steps(slot_id_uploads):
    sampler = _sampler()
    sampler.static_slot_ids = True
    uploads = slot_id_uploads.watch(sampler)
    scheduled, outputs = _step(MAX_DRAFT_LEN)

    for _ in range(5):
        sampler.sample_async(scheduled, outputs, [])

    assert uploads.count == 1
    torch.cuda.synchronize()
    assert sampler._slot_ids_cuda[: len(SEQ_SLOTS)].tolist() == SEQ_SLOTS


def test_a_changed_slot_list_re_uploads(slot_id_uploads):
    sampler = _sampler()
    uploads = slot_id_uploads.watch(sampler)

    sampler._device_slot_ids([4, 1])
    sampler._device_slot_ids([4, 1])
    assert uploads.count == 1
    sampler._device_slot_ids([1, 4])
    assert uploads.count == 2
    sampler._device_slot_ids([1])  # a prefix of the buffer still differs from the last list
    assert uploads.count == 3
    view = sampler._device_slot_ids([1])
    assert uploads.count == 3
    torch.cuda.synchronize()
    assert view.tolist() == [1]


def test_a_longer_slot_list_reallocates_and_uploads(slot_id_uploads):
    sampler = _sampler()
    uploads = slot_id_uploads.watch(sampler)

    sampler._device_slot_ids([4, 1])
    small = sampler._slot_ids_cuda
    assert small.numel() == 8  # max(n, 8)
    long_ids = list(range(10))
    view = sampler._device_slot_ids(long_ids)

    assert sampler._slot_ids_cuda is not small and sampler._slot_ids_cuda.numel() == 10
    assert uploads.count == 2
    torch.cuda.synchronize()
    assert view.tolist() == long_ids
    # The same ids again: no upload into the new buffer.
    sampler._device_slot_ids(long_ids)
    assert uploads.count == 2


def test_both_hooks_on_store_and_copy_what_the_stock_sampler_does_over_a_changing_batch():
    """The targets run HO-04 and HO-12a together: over steady, reordered and repeated steps the store and the host
    copies are bitwise the stock ones."""
    stock = _sampler()
    sampler = _sampler()
    sampler.d2h_side_stream_copies = True
    sampler.static_slot_ids = True
    scheduled, outputs = _step(MAX_DRAFT_LEN)
    for step in range(4):
        if step == 2:  # the batch's slot order changes
            scheduled.generation_requests = scheduled.generation_requests[::-1]
        step_outputs = {name: tensor + step for name, tensor in outputs.items()}
        on_main = _host_outputs(stock.sample_async(scheduled, step_outputs, []))
        state = sampler.sample_async(scheduled, step_outputs, [])
        assert state.sampler_event.side_stream_event is not None
        on_side = _host_outputs(state)
        for name, tensor in on_main.items():
            assert torch.equal(on_side[name], tensor), (step, name)
        _assert_store_equals(sampler, _store_snapshot(stock))
