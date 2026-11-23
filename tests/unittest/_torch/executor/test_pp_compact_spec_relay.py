# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The compact sample-state relay of SpecSampler host states (disaggregated PP server).

Codec checks of ``pp_spec_relay.py``: every value ``SpecSampler.update_requests``
reads is equal for the full and the compact form (and ``update_requests`` itself
gives the same request updates), the wire is small and tensor-free, unsupported
host states fall back to the full form, and an inconsistent receiver fails loudly.

Ring checks: the real ``PyExecutor._executor_loop_pp`` on pipeline-depth threads
(``test_pp_ring_depth.py`` stand-ins) with an MPI stand-in whose ``isend`` is eager
up to a byte limit and rendezvous above it, as a real transport is. With the
compact relay (``PyExecutor._pp_compact_spec_relay``, on where a disaggregated
server runs at ``pipeline_parallel_size > 1``), every relay message is compact
and eager, non-last ranks forward the received wire unchanged, every rank reads
the sender's rows, and nothing hangs; without it, the full store goes by
rendezvous as before.
"""

import pickle
import random
import sys
import threading
import types

import numpy as np
import pytest
import test_pp_ring_depth as ring
import torch

from tensorrt_llm._torch.pyexecutor import pp_spec_relay
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.pp_ctx_diag import PPCtxDiag
from tensorrt_llm._torch.pyexecutor.pp_spec_relay import (
    WIRE_TAG,
    encode_compact_spec_relay,
    expand_compact_spec_relay,
    is_compact_spec_relay,
    relay_payload_bytes,
)
from tensorrt_llm._torch.pyexecutor.pp_utils import PPCommTag
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.sampler import SampleStateTensors
from tensorrt_llm._torch.speculative.spec_sampler_base import (
    SampleStateSpec,
    SampleStateTensorsSpec,
    SpecSampler,
)

pytestmark = pytest.mark.cpu_only

# The PP CTX store: max_draft_len 7 -> 8 accepted-path rows, max_batch_size 64 x pp 4 slots.
PATH_LEN = 8
NUM_SLOTS = 256
DRAFT_WIDTH = 7


class _Req:
    """The request fields the codec and ``SpecSampler.update_requests`` read and write."""

    def __init__(self, request_id, slot, state=LlmRequestState.GENERATION_IN_PROGRESS):
        self.py_request_id = request_id
        self.py_seq_slot = slot
        self.state = state
        self.tokens = [11, 12, 13]
        self.py_orig_prompt_len = 3
        self.py_max_new_tokens = 1 << 20
        self.py_end_id = -1
        self.py_stop_words_list = None
        self.py_draft_tokens = []
        self.py_decoding_iter = 0
        self.py_num_accepted_draft_tokens = None
        self.py_num_draft_tokens_verified = None
        self.py_rewind_len = None

    def add_new_token(self, token, beam_idx):
        self.tokens.append(token)

    def get_num_tokens(self, beam_idx):
        return len(self.tokens)

    def finish_by(self, reason, beam_idx):
        self.state = LlmRequestState.GENERATION_COMPLETE

    def snapshot(self):
        return (
            self.state,
            list(self.tokens),
            list(self.py_draft_tokens),
            self.py_decoding_iter,
            self.py_num_accepted_draft_tokens,
            self.py_num_draft_tokens_verified,
            self.py_rewind_len,
        )


def _host(requests, seed, with_verify=False, num_slots=NUM_SLOTS):
    """A SpecSampler host state: random (stale) slots, valid rows at the requests' slots."""
    g = torch.Generator().manual_seed(seed)
    new_tokens = torch.randint(0, 1 << 30, (PATH_LEN, num_slots, 1), dtype=torch.int32, generator=g)
    lens = torch.randint(0, 1 << 30, (num_slots,), dtype=torch.int32, generator=g)
    drafts = torch.randint(0, 1 << 30, (num_slots, DRAFT_WIDTH), dtype=torch.int32, generator=g)
    rng = random.Random(seed)
    windows = []
    for request in requests:
        window = rng.randint(1, DRAFT_WIDTH + 1)
        windows.append(window)
        if request.py_seq_slot is not None:
            lens[request.py_seq_slot] = rng.randint(1, window)
    verify = torch.tensor(windows, dtype=torch.int32) if with_verify else None
    return SampleStateTensorsSpec(
        new_tokens=new_tokens, new_tokens_lens=lens, next_draft_tokens=drafts, verify_lens=verify
    )


def _requests(num, seed, num_slots=NUM_SLOTS):
    rng = random.Random(seed)
    slots = rng.sample(range(num_slots), num)
    ids = rng.sample(range(1, 1 << 40), num)
    return [_Req(request_id, slot) for request_id, slot in zip(ids, slots)]


def _read_values(host, requests):
    """Everything ``SpecSampler.update_requests`` reads, per request."""
    out = []
    for i, request in enumerate(requests):
        slot = request.py_seq_slot
        out.append(
            (
                host.new_tokens[:, slot, 0].tolist(),
                int(host.new_tokens_lens[slot]),
                host.next_draft_tokens[slot].tolist(),
                None if host.verify_lens is None else int(host.verify_lens[i]),
            )
        )
    return out


def _round_trip(host, requests):
    wire = encode_compact_spec_relay(host, requests)
    assert wire is not None
    # The wire crosses a process boundary pickled.
    wire = pickle.loads(pickle.dumps(wire, protocol=5))
    assert is_compact_spec_relay(wire)
    return wire, expand_compact_spec_relay(wire, requests)


# --------------------------------------------------------------------------- wiring


def test_executor_default_is_off():
    assert PyExecutor._pp_compact_spec_relay is False


# --------------------------------------------------------------------------- codec


@pytest.mark.parametrize("num", [0, 1, 3, 8, 64, NUM_SLOTS])
@pytest.mark.parametrize("with_verify", [False, True])
def test_round_trip_gives_every_value_the_reader_reads(num, with_verify):
    requests = _requests(num, seed=num)
    host = _host(requests, seed=num + 1, with_verify=with_verify)
    _, expanded = _round_trip(host, requests)
    assert type(expanded) is SampleStateTensorsSpec
    for name in ("new_tokens", "new_tokens_lens", "next_draft_tokens"):
        full, compact = getattr(host, name), getattr(expanded, name)
        assert compact.shape == full.shape and compact.dtype == full.dtype == torch.int32, name
        assert compact.device.type == "cpu"
    assert (expanded.verify_lens is None) == (not with_verify)
    assert _read_values(expanded, requests) == _read_values(host, requests)
    # Slots of no request of the micro-batch are zero (stale in the full store).
    others = sorted(set(range(NUM_SLOTS)) - {r.py_seq_slot for r in requests})
    assert not expanded.new_tokens[:, others].any()
    assert not expanded.new_tokens_lens[others].any()
    assert not expanded.next_draft_tokens[others].any()


def test_wire_is_small_and_holds_no_tensors():
    sizes = {}
    for num in (0, 3, 8, 32):
        requests = _requests(num, seed=7)
        # Disaggregated request ids can use the full unsigned 64-bit range.
        if requests:
            requests[0].py_request_id = (1 << 64) - 2
        host = _host(requests, seed=8, with_verify=True)
        wire = encode_compact_spec_relay(host, requests)
        assert wire[0] == WIRE_TAG
        assert all(isinstance(item, (str, int, bool, tuple, bytes)) for item in wire)
        assert all(type(request_id) is int for request_id in wire[5])
        sizes[num] = len(pickle.dumps((wire, [[] for _ in requests]), protocol=5))
        _, expanded = _round_trip(host, requests)
        assert _read_values(expanded, requests) == _read_values(host, requests)
    full = len(pickle.dumps((_host([], seed=9), []), protocol=5))
    # The full 256-slot store pickles to about 17.5 KB, the compact rows to well under 1 KB.
    assert full > 16000
    assert sizes[0] <= 256
    assert sizes[3] <= 512
    assert sizes[8] <= 1024
    assert sizes[32] <= 3072


def _fallback_cases():
    requests = _requests(3, seed=3)
    base = _host(requests, seed=4)

    def variant(**changes):
        fields = dict(
            new_tokens=base.new_tokens,
            new_tokens_lens=base.new_tokens_lens,
            next_draft_tokens=base.next_draft_tokens,
            verify_lens=base.verify_lens,
        )
        fields.update(changes)
        return SampleStateTensorsSpec(**fields)

    with_log_probs = variant()
    with_log_probs.log_probs = torch.zeros(3)
    # An attribute set after construction (as the two-model path does with logits)
    # would be dropped by the wire.
    with_extra = variant()
    with_extra.logits = torch.zeros(3)
    return requests, {
        "other_sampler": SampleStateTensors(new_tokens=base.new_tokens),
        "namespace": types.SimpleNamespace(**vars(base)),
        "log_probs": with_log_probs,
        "extra_attribute": with_extra,
        "int64_tokens": variant(new_tokens=base.new_tokens.long()),
        "beam_2": variant(new_tokens=base.new_tokens.repeat(1, 1, 2)),
        "lens_shape": variant(new_tokens_lens=base.new_tokens_lens[:-1]),
        "drafts_shape": variant(next_draft_tokens=base.next_draft_tokens[:-1]),
        "drafts_rank": variant(next_draft_tokens=base.next_draft_tokens.flatten()),
        "verify_len": variant(verify_lens=torch.ones(2, dtype=torch.int32)),
        "verify_dtype": variant(verify_lens=torch.ones(3, dtype=torch.int64)),
    }


@pytest.mark.parametrize("case", list(_fallback_cases()[1]))
def test_unsupported_host_states_fall_back_to_the_full_form(case):
    requests, cases = _fallback_cases()
    assert encode_compact_spec_relay(cases[case], requests) is None


@pytest.mark.parametrize("slots", [[1, None, 3], [1, 2, NUM_SLOTS], [1, -1, 3], [4, 5, 4]])
def test_requests_without_a_unique_valid_slot_fall_back(slots):
    requests = [_Req(10 + i, slot) for i, slot in enumerate(slots)]
    host = _host([_Req(0, 0)], seed=1)
    assert encode_compact_spec_relay(host, requests) is None


# --------------------------------------------------------------------------- receiver


def test_finished_or_slotless_receiver_requests_are_skipped():
    requests = _requests(4, seed=11)
    host = _host(requests, seed=12)
    wire = encode_compact_spec_relay(host, requests)
    # The receiver also holds a finished request the sender did not sample, and a
    # request whose slot was freed (both skipped by update_requests).
    receiver = list(requests) + [
        _Req(99, 77, LlmRequestState.GENERATION_COMPLETE),
        _Req(98, None),
    ]
    expanded = expand_compact_spec_relay(wire, receiver)
    assert _read_values(expanded, requests) == _read_values(host, requests)


def test_receiver_slot_mismatch_raises():
    requests = _requests(3, seed=13)
    wire = encode_compact_spec_relay(_host(requests, seed=14), requests)
    receiver = [_Req(r.py_request_id, r.py_seq_slot) for r in requests]
    receiver[1].py_seq_slot = (receiver[1].py_seq_slot + 1) % NUM_SLOTS
    with pytest.raises(RuntimeError, match="no row of its own"):
        expand_compact_spec_relay(wire, receiver)
    # A finished request may differ: update_requests does not read it.
    receiver[1].state = LlmRequestState.GENERATION_COMPLETE
    expand_compact_spec_relay(wire, receiver)


def test_receiver_request_missing_from_the_wire_raises():
    requests = _requests(3, seed=15)
    wire = encode_compact_spec_relay(_host(requests[:2], seed=16), requests[:2])
    with pytest.raises(RuntimeError, match="no row of its own"):
        expand_compact_spec_relay(wire, requests)


def test_order_mismatch_maps_rows_by_request_id():
    requests = _requests(5, seed=17)
    host = _host(requests, seed=18, with_verify=True)
    wire = encode_compact_spec_relay(host, requests)
    receiver = [requests[i] for i in (3, 0, 4, 2, 1)]
    expanded = expand_compact_spec_relay(wire, receiver)
    expected = _read_values(host, requests)
    assert _read_values(expanded, receiver) == [expected[i] for i in (3, 0, 4, 2, 1)]


def test_order_mismatch_with_a_repeated_request_id_raises():
    requests = _requests(3, seed=19)
    requests[2].py_request_id = requests[0].py_request_id
    wire = encode_compact_spec_relay(_host(requests, seed=20), requests)
    with pytest.raises(RuntimeError, match="not unique"):
        expand_compact_spec_relay(wire, [requests[1], requests[0], requests[2]])


def test_malformed_wire_raises():
    requests = _requests(2, seed=21)
    wire = encode_compact_spec_relay(_host(requests, seed=22), requests)
    with pytest.raises(ValueError):
        expand_compact_spec_relay(wire[:6] + (wire[6][:-4],), requests)
    with pytest.raises(ValueError):
        expand_compact_spec_relay(("other",) + wire[1:], requests)


# --------------------------------------------------------------------------- update_requests


class _Event:
    def synchronize(self):
        pass


def _sampler():
    sampler = object.__new__(SpecSampler)
    sampler.max_seq_len = 1 << 20
    sampler.draft_len = DRAFT_WIDTH
    sampler.max_accepted_path_len = PATH_LEN
    return sampler


@pytest.mark.parametrize("with_verify", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_update_requests_is_identical_for_full_and_compact(with_verify, seed):
    def batch():
        requests = _requests(9, seed=seed)
        requests[4].state = LlmRequestState.GENERATION_COMPLETE
        # An end-id hit inside the accepted tokens stops that request early.
        requests[6].py_end_id = -2
        return requests

    host = _host(batch(), seed=100 + seed, with_verify=with_verify)
    full_requests, compact_requests = batch(), batch()
    sender = batch()
    slot = sender[6].py_seq_slot
    host.new_tokens[0, slot, 0] = -2
    wire = pickle.loads(pickle.dumps(encode_compact_spec_relay(host, sender), protocol=5))

    for requests, state_host in (
        (full_requests, pickle.loads(pickle.dumps(host, protocol=5))),
        (compact_requests, expand_compact_spec_relay(wire, compact_requests)),
    ):
        state = SampleStateSpec(
            requests=requests,
            device=None,
            host=state_host,
            sampler_event=_Event(),
            runtime_draft_len=DRAFT_WIDTH,
            draft_lens=[DRAFT_WIDTH] * len(requests),
        )
        _sampler().update_requests(state)
    assert [r.snapshot() for r in compact_requests] == [r.snapshot() for r in full_requests]
    assert full_requests[6].state == LlmRequestState.GENERATION_COMPLETE
    assert any(r.py_num_accepted_draft_tokens for r in full_requests)


# --------------------------------------------------------------------------- diagnostics


def test_is_compact_spec_relay_rejects_other_tuples():
    assert not is_compact_spec_relay((torch.zeros(2),) + (0,) * 6)
    assert not is_compact_spec_relay([WIRE_TAG, 0, 0, 0, False, (), b""])
    assert not is_compact_spec_relay(_host([], seed=1))


def test_relay_payload_bytes_counts_buffers_only():
    requests = _requests(3, seed=27)
    host = _host(requests, seed=28)
    full = (PATH_LEN * NUM_SLOTS + NUM_SLOTS + NUM_SLOTS * DRAFT_WIDTH) * 4
    assert relay_payload_bytes(host) == full
    wire = encode_compact_spec_relay(host, requests)
    assert relay_payload_bytes(wire) == 3 * (2 + PATH_LEN + DRAFT_WIDTH) * 4 + 3 * 8
    with_verify = _host(requests, seed=28, with_verify=True)
    assert relay_payload_bytes(with_verify) == full + 3 * 4
    assert relay_payload_bytes(None) == 0


def test_diag_counts_relay_messages_and_waits():
    diag = PPCtxDiag(1, 4)
    requests = _requests(3, seed=23)
    host = _host(requests, seed=24)
    diag.relay_sent(False, relay_payload_bytes(host))
    diag.relay_sent(True, relay_payload_bytes(encode_compact_spec_relay(host, requests)))
    diag.relay_drain_waited(0.004)
    diag.relay_drain_waited(0.001)
    diag.relay_recv_waited(0.002)
    diag.sampler_sync_waited(0.0005)
    fields = dict(token.split("=", 1) for token in diag.line("periodic").split()[1:])
    assert fields["relay_full_sends"] == "1"
    assert fields["relay_compact_sends"] == "1"
    assert int(fields["relay_bytes_max"]) == relay_payload_bytes(host) > 16000
    assert fields["drain_waits"] == "2"
    assert fields["drain_wait_ms"] == "5.000"
    assert fields["drain_wait_ms_max"] == "4.000"
    assert fields["relay_recvs"] == "1"
    assert fields["relay_recv_wait_ms"] == "2.000"
    assert fields["sync_prevs"] == "1"
    assert fields["sync_prev_ms"] == "0.500"


# --------------------------------------------------------------------------- ring

EAGER_LIMIT = 4096


class _SizedWorld(ring._World):
    """MPI stand-in: an ``isend`` is eager up to ``eager_limit`` pickled bytes, else
    rendezvous (completes when the peer receives). Records every relay message."""

    def __init__(self, world_size, eager_limit):
        super().__init__(world_size, eager=False)
        self.eager_limit = eager_limit
        self.relays = []

    def isend(self, obj, src, dst, tag):
        payload = pickle.dumps(obj)
        eager = len(payload) <= self.eager_limit
        if int(tag) == int(PPCommTag.SAMPLE_STATE):
            with self._lock:
                self.relays.append((src, len(payload), eager, pickle.loads(payload)[0]))
        consumed = threading.Event()
        self._queue(src, dst, tag).put((payload, consumed))
        return ring._Handle(consumed, eager)


def _slot(request_id):
    return (request_id * 37 + 5) % NUM_SLOTS


def _row(request_id, microbatch):
    base = request_id * 100003 + microbatch * 1009
    return (
        [base + k for k in range(PATH_LEN)],
        1 + (request_id + microbatch) % PATH_LEN,
        [base - k for k in range(1, DRAFT_WIDTH + 1)],
    )


class _SpecRelayExecutor(ring._Executor):
    """Ring stand-in whose sampled micro-batches carry a real SpecSampler host state."""

    def _sample_state(self, scheduled_batch, done):
        requests = scheduled_batch.context_requests_last_chunk + scheduled_batch.generation_requests
        microbatch = self._launched - 1
        # Stale noise everywhere, the micro-batch's rows at their slots.
        host = _host([], seed=microbatch)
        for request in requests:
            tokens, length, drafts = _row(request.py_request_id, microbatch)
            host.new_tokens[:, request.py_seq_slot, 0] = torch.tensor(tokens, dtype=torch.int32)
            host.new_tokens_lens[request.py_seq_slot] = length
            host.next_draft_tokens[request.py_seq_slot] = torch.tensor(drafts, dtype=torch.int32)
        return types.SimpleNamespace(
            requests=requests,
            sampler_event=ring._SampleEvent(done),
            host=host,
            microbatch=microbatch,
        )

    def _ring_broadcast_sample_state(self, executed_batch):
        if executed_batch is not None:
            self._log.relayed[self.dist.rank].append(executed_batch.sample_state.microbatch)
        PyExecutor._ring_broadcast_sample_state(self, executed_batch)

    def _handle_executed_batch(self, executed_batch):
        if executed_batch is None:
            return
        state = executed_batch.sample_state
        for request in state.requests:
            tokens, length, drafts = _row(request.py_request_id, state.microbatch)
            slot = request.py_seq_slot
            assert state.host.new_tokens[:, slot, 0].tolist() == tokens
            assert int(state.host.new_tokens_lens[slot]) == length
            assert state.host.next_draft_tokens[slot].tolist() == drafts
        self._log.handled[self.dist.rank].append(state.microbatch)
        scheduled = executed_batch.scheduled_requests
        self._remove_inflight_ids(scheduled)
        for request in scheduled.context_requests_last_chunk:
            self.active_requests.remove(request)


def _assert_same_wire(first, second):
    if is_compact_spec_relay(first):
        # Plain Python values: equal tuples carry byte-identical rows.
        assert first == second
        return
    # A pickled tensor's bytes also carry storage ids, so compare the values.
    for name in ("new_tokens", "new_tokens_lens", "next_draft_tokens"):
        assert torch.equal(getattr(first, name), getattr(second, name)), name
    assert first.verify_lens is None and second.verify_lens is None


def _run_spec_ring(monkeypatch, compact_ranks, ring_depth, buffered, eager_limit, prompts, budget):
    pp_size = 4
    worlds = []

    def make_world(world_size, eager):
        world = _SizedWorld(world_size, eager_limit)
        worlds.append(world)
        return world

    original = ring._make_executor

    def make(world, rank, *args, **kwargs):
        exe = original(world, rank, *args, **kwargs)
        exe.__class__ = _SpecRelayExecutor
        exe._pp_compact_spec_relay = rank in compact_ranks
        for request in exe.active_requests:
            request.py_seq_slot = _slot(request.py_request_id)
            request.state = LlmRequestState.CONTEXT_INIT
        return exe

    monkeypatch.setattr(ring, "_World", make_world)
    monkeypatch.setattr(sys.modules[ring.__name__], "_make_executor", make)
    rng = random.Random(len(prompts))
    costs = {}

    def cost(microbatch):
        if microbatch not in costs:
            costs[microbatch] = 0.002 * rng.uniform(0.5, 1.5)
        return costs[microbatch]

    executors, log, _, _ = ring._run_pipeline(
        pp_size, ring_depth, prompts, budget, buffered, 2, cost, monkeypatch
    )
    return executors, log, worlds[0]


# The stand-in links' delivery threads (test_pp_ring_depth._Link) exit only after the
# test returns, which pytest-threadleak reports racily; the PP loop itself joins.
@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("compact", ["off", "on", "last_rank_only", "others_only"])
@pytest.mark.parametrize("ring_depth", [None, 8])
@pytest.mark.parametrize("buffered", [False, True])
def test_real_loop_relays_compact_eager_rows(compact, ring_depth, buffered, monkeypatch):
    compact_ranks = {
        "off": set(),
        "on": {0, 1, 2, 3},
        "last_rank_only": {3},
        "others_only": {0, 1, 2},
    }[compact]
    prompts = [64] * 6 + [200, 130, 64, 96, 64, 300]
    executors, log, world = _run_spec_ring(
        monkeypatch, compact_ranks, ring_depth, buffered, EAGER_LIMIT, prompts, 256
    )
    num_microbatches = len(log.launched[0])
    for rank in range(4):
        # Every micro-batch relayed and handled once, in launch order, with the sender's rows.
        assert log.handled[rank] == list(range(num_microbatches)), rank
        assert log.relayed[rank] == list(range(num_microbatches)), rank
        assert executors[rank].active_requests == []
        assert executors[rank].unhandled_batch_counter == 0
    compact_sent = 3 in compact_ranks
    relays = world.relays
    # Last, first and second rank send one relay message per micro-batch.
    assert len(relays) == 3 * num_microbatches
    for src, size, eager, wire in relays:
        assert is_compact_spec_relay(wire) is compact_sent, src
        assert eager is compact_sent, (src, size)
    # Non-last ranks forward the wire they received unchanged.
    by_src = {src: [wire for s, _, _, wire in relays if s == src] for src in (3, 0, 1)}
    for first, second in zip(by_src[3], by_src[0]):
        _assert_same_wire(first, second)
    for first, second in zip(by_src[0], by_src[1]):
        _assert_same_wire(first, second)


@pytest.mark.threadleak(enabled=False)
def test_real_loop_mixed_eager_and_rendezvous_compact_messages(monkeypatch):
    """A low eager limit sends the larger compact messages by rendezvous: still no hang."""
    prompts = [16] * 24 + [500, 40, 300]
    executors, log, world = _run_spec_ring(monkeypatch, {3}, 8, False, 600, prompts, 512)
    num_microbatches = len(log.launched[0])
    for rank in range(4):
        assert log.handled[rank] == list(range(num_microbatches)), rank
    eager = {e for _, _, e, _ in world.relays}
    assert eager == {True, False}
    assert all(is_compact_spec_relay(wire) for _, _, _, wire in world.relays)


def test_numpy_rows_do_not_alias_the_host_state():
    requests = _requests(3, seed=25)
    host = _host(requests, seed=26)
    wire = encode_compact_spec_relay(host, requests)
    before = expand_compact_spec_relay(wire, requests)
    # A later sample overwriting the pinned host buffers cannot change the wire.
    host.new_tokens.zero_()
    host.new_tokens_lens.zero_()
    host.next_draft_tokens.zero_()
    after = expand_compact_spec_relay(wire, requests)
    for name in ("new_tokens", "new_tokens_lens", "next_draft_tokens"):
        assert torch.equal(getattr(before, name), getattr(after, name))
    assert np.frombuffer(wire[6], dtype=np.int32).any()
    assert pp_spec_relay.WIRE_TAG == wire[0]
