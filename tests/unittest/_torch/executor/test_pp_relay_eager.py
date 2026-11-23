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
"""The pipeline-parallel sample-state relay under an eager vs a rendezvous MPI protocol.

A pipeline-parallel context server whose last rank samples with ``SpecSampler``
relays the sampler's full slot store (``new_tokens``, ``new_tokens_lens``,
``next_draft_tokens``) on every iteration. That message is larger than the small
schedule and report messages, so a transport may complete it by rendezvous
(the send finishes only once the receiver posted its receive) while the rest
goes eager. ``PyExecutor._drain_relay_sends_before_forward`` waits on every relay
send of a non-last rank before its next forward, so under rendezvous rank r
cannot launch micro-batch k before rank r + 1 has finished micro-batch k - 2 on
its device and reached its relay receive.

Checked, on the real ``PyExecutor._executor_loop_pp`` and the threaded harness of
``test_pp_ring_depth.py`` (an MPI stand-in whose protocol is chosen per tag):

- the relay payload a 256-slot, 7-draft-token ``SpecSampler`` produces fits in
  one eager message under a 64 KiB rendezvous threshold, including one result
  diff per request of a full 64-request micro-batch;
- with a rendezvous relay, every launch of the first rank happens after the
  second rank's device finished the micro-batch two before it (the wait-for
  edge); with an eager relay that edge is gone and nothing else changes: every
  stage runs the same chunks in the same order, every sample state is relayed
  and handled once in launch order, and the ring closure still bounds the first
  rank's lead to the ring depth;
- with host work before each launch on the first rank, the eager relay lets that
  work overlap the rank's own device instead of adding to the period.
"""

import pickle
import threading
import time
from typing import Dict, List, Tuple

import pytest
import test_pp_ring_depth as ring
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import PyResult
from tensorrt_llm._torch.pyexecutor.pp_utils import PPCommTag
from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler, sampler_features
from tensorrt_llm._torch.speculative import spec_sampler_base
from tensorrt_llm._torch.speculative.spec_sampler_base import SampleStateTensorsSpec, SpecSampler

pytestmark = pytest.mark.cpu_only

# Rendezvous threshold the context server's transport is configured with
# (bytes); the relay payload must fit in one eager message below it.
EAGER_THRESHOLD_BYTES = 64 * 1024
# Room left for the transport's own message header.
HEADER_ALLOWANCE_BYTES = 1024
# A context micro-batch of the pipeline-parallel context server: 64 requests
# per micro-batch, 4 stages, 7 draft tokens.
MAX_BATCH_SIZE = 64
PP_SIZE = 4
MAX_DRAFT_LEN = 7

# --------------------------------------------------------------------------- payload size


def _spec_sampler_on_cpu(monkeypatch, max_num_sequences: int) -> SpecSampler:
    """A ``SpecSampler`` whose slot store lives on the host (no device needed)."""
    monkeypatch.setattr(
        spec_sampler_base,
        "int_tensor",
        lambda shape, device="cuda": torch.empty(shape, dtype=torch.int, device="cpu"),
    )
    monkeypatch.setattr(sampler_features, "prefer_pinned", lambda: False)

    def _no_async_worker(self, enable_async_worker):
        self._enable_async_worker = False
        self._async_worker = None
        self._async_worker_futures = []

    monkeypatch.setattr(SpecSampler, "_async_worker_init", _no_async_worker)
    args = TorchSampler.Args(
        max_seq_len=1024,
        max_draft_len=MAX_DRAFT_LEN,
        max_num_sequences=max_num_sequences,
        max_beam_width=1,
        max_total_draft_tokens=MAX_DRAFT_LEN,
    )
    return SpecSampler(args)


def _relay_payload(sampler: SpecSampler, num_requests: int):
    """The ``(sample_state.host, py_result_diffs)`` tuple a PP rank relays, with the
    host tensors made as ``SpecSampler.sample_async`` makes them without a store
    writer (fresh host copies of the whole slot store)."""
    store = sampler.store
    host = SampleStateTensorsSpec(
        new_tokens=sampler._copy_to_host(store.new_tokens),
        new_tokens_lens=sampler._copy_to_host(store.new_tokens_lens),
        next_draft_tokens=sampler._copy_to_host(store.next_draft_tokens),
        verify_lens=None,
    )
    diffs = [PyResult.Diff() for _ in range(num_requests)]
    return host, diffs


@pytest.mark.parametrize("num_requests", [0, 1, 16, MAX_BATCH_SIZE])
def test_spec_sampler_relay_payload_fits_one_eager_message(num_requests, monkeypatch):
    sampler = _spec_sampler_on_cpu(monkeypatch, MAX_BATCH_SIZE * PP_SIZE)
    host, diffs = _relay_payload(sampler, num_requests)

    # The host copies are fresh tensors, not views of a larger buffer, so
    # pickling serialises only their own storage: 16 KiB of int32 data.
    tensors = (host.new_tokens, host.new_tokens_lens, host.next_draft_tokens)
    for tensor in tensors:
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    assert host.new_tokens.shape == (MAX_DRAFT_LEN + 1, MAX_BATCH_SIZE * PP_SIZE, 1)
    assert host.new_tokens_lens.shape == (MAX_BATCH_SIZE * PP_SIZE,)
    assert host.next_draft_tokens.shape == (MAX_BATCH_SIZE * PP_SIZE, MAX_DRAFT_LEN)
    assert sum(t.numel() * t.element_size() for t in tensors) == 16 * 1024

    # mpi4py's pkl5 pickles with protocol 5 when the send is posted; CPU
    # tensors pickle in-band, so this is the message size.
    size = len(pickle.dumps((host, diffs), protocol=5))
    assert size <= EAGER_THRESHOLD_BYTES - HEADER_ALLOWANCE_BYTES, size


def test_relay_payload_round_trips_bitwise(monkeypatch):
    sampler = _spec_sampler_on_cpu(monkeypatch, MAX_BATCH_SIZE * PP_SIZE)
    generator = torch.Generator().manual_seed(0)
    for tensor in (
        sampler.store.new_tokens,
        sampler.store.new_tokens_lens,
        sampler.store.next_draft_tokens,
    ):
        tensor.copy_(torch.randint(0, 2**31 - 1, tensor.shape, generator=generator))
    host, diffs = _relay_payload(sampler, 2)
    received, received_diffs = pickle.loads(pickle.dumps((host, diffs), protocol=5))
    for name in ("new_tokens", "new_tokens_lens", "next_draft_tokens"):
        assert torch.equal(getattr(received, name), getattr(host, name)), name
        assert getattr(received, name).dtype == torch.int32
    assert received.verify_lens is None
    assert len(received_diffs) == 2


# --------------------------------------------------------------------------- loop harness


def _world_class(rendezvous_tags):
    """MPI stand-in completing an ``isend`` eagerly unless its tag is listed."""

    class _TagWorld(ring._World):
        def isend(self, obj, src, dst, tag):
            consumed = threading.Event()
            # Pickle when the send is posted, as pkl5 does.
            self._queue(src, dst, tag).put((pickle.dumps(obj), consumed))
            return ring._Handle(consumed, eager=int(tag) not in rendezvous_tags)

    return _TagWorld


class _StampedEvent(threading.Event):
    """A device completion event that records when it fired."""

    def __init__(self, stamps: Dict[Tuple[int, int], float], key: Tuple[int, int]):
        super().__init__()
        self._stamps = stamps
        self._key = key

    def set(self):
        self._stamps[self._key] = time.perf_counter()
        super().set()


def _gpu_class(stage_cost_s: List[float], done_at: Dict[Tuple[int, int], float]):
    """Device stand-in with a per-stage cost that stamps each finished micro-batch."""

    class _TimedGpu(ring._Gpu):
        def __init__(self, stage, pp_size, inbound, outbound, cost_s, log):
            super().__init__(
                stage, pp_size, inbound, outbound, lambda microbatch: stage_cost_s[stage], log
            )

        def launch(self, microbatch):
            done = _StampedEvent(done_at, (self.stage, microbatch))
            self._jobs.put((microbatch, done))
            return done

    return _TimedGpu


class _TimedExecutor(ring._Executor):
    """The first rank does ``host_cost_s`` of host work before each launch (the
    resource preparation and launch prologue a real stage runs after the relay
    drain) and stamps every launch."""

    def _launch(self, scheduled_batch):
        if self.dist.rank == 0:
            if self._host_cost_s:
                time.sleep(self._host_cost_s)
            self._launch_at.append(time.perf_counter())
        return super()._launch(scheduled_batch)


_MAKE_EXECUTOR = ring._make_executor


def _run(monkeypatch, relay_rendezvous: bool, stage_cost_s, host_cost_s, num_prompts=40):
    """One run of the real loop: pp 4, ring depth 8, buffered transport (2 slots),
    one single-chunk prompt per micro-batch so every iteration launches."""
    rendezvous_tags = {int(PPCommTag.SAMPLE_STATE)} if relay_rendezvous else set()
    done_at: Dict[Tuple[int, int], float] = {}
    launch_at: List[float] = []
    monkeypatch.setattr(ring, "_World", _world_class(rendezvous_tags))
    monkeypatch.setattr(ring, "_Gpu", _gpu_class(stage_cost_s, done_at))

    def make_timed(*args, **kwargs):
        exe = _MAKE_EXECUTOR(*args, **kwargs)
        exe.__class__ = _TimedExecutor
        exe._host_cost_s = host_cost_s
        exe._launch_at = launch_at
        return exe

    monkeypatch.setattr(ring, "_make_executor", make_timed)
    prompts = [64] * num_prompts
    executors, log, _, n = ring._run_pipeline(
        PP_SIZE,
        8,
        prompts,
        64,
        True,
        2,
        None,
        monkeypatch,
    )
    expected = ring._expected_chunks(prompts, 64)
    assert len(expected) == num_prompts
    for rank in range(PP_SIZE):
        assert log.launched[rank] == [sorted(c[0] for c in batch) for batch in expected], rank
        assert log.handled[rank] == list(range(num_prompts)), rank
        assert sorted(log.relayed[rank]) == log.relayed[rank], rank
        assert executors[rank].active_requests == []
        assert executors[rank].unhandled_batch_counter == 0
        assert all(mb is None for mb in executors[rank].micro_batches)
    # Ring closure: the first rank never runs more than the ring depth ahead.
    assert n == 8
    assert log.max_lead <= n
    assert len(launch_at) == num_prompts
    return launch_at, done_at


def _edge_held(launch_at, done_at, k: int) -> bool:
    """Whether the first rank launched micro-batch k after the second rank's
    device finished micro-batch k - 2."""
    return launch_at[k] >= done_at[(1, k - 2)]


# The first rank relays from iteration ring_depth - 1 on; its iteration k - 1 send
# gates launch k, so the edge exists for k >= ring_depth.
FIRST_GATED_LAUNCH = 8


def test_rendezvous_relay_gates_the_first_rank_on_the_second_stage(monkeypatch):
    """The default path: the relay is rendezvous, everything else eager."""
    launch_at, done_at = _run(monkeypatch, True, [0.001, 0.004, 0.001, 0.001], 0.0)
    gated = range(FIRST_GATED_LAUNCH, len(launch_at))
    assert all(_edge_held(launch_at, done_at, k) for k in gated), [
        k for k in gated if not _edge_held(launch_at, done_at, k)
    ]


def test_eager_relay_removes_the_edge_and_keeps_the_joins(monkeypatch):
    """With the relay eager, the first rank runs ahead of a slower second stage
    (bounded by the buffered transport and the ring closure, both checked in
    ``_run``), with the same chunks, relays and handling as before."""
    launch_at, done_at = _run(monkeypatch, False, [0.001, 0.004, 0.001, 0.001], 0.0)
    gated = range(FIRST_GATED_LAUNCH, len(launch_at))
    assert any(not _edge_held(launch_at, done_at, k) for k in gated)


def test_eager_relay_overlaps_first_rank_host_work_with_its_device(monkeypatch):
    """Equal stage costs c and host work h < c before every first-rank launch.
    Rendezvous puts h + 2c on every second micro-batch of the second stage
    (period (h + 2c) / 2); eager overlaps h with the first rank's own device
    (period about c)."""
    cost, host = 0.004, 0.0035
    steady = slice(FIRST_GATED_LAUNCH + 2, None)

    def period(done_at, num):
        last = [done_at[(PP_SIZE - 1, k)] for k in range(num)][steady]
        return (last[-1] - last[0]) / (len(last) - 1)

    launch_at, done_at = _run(monkeypatch, True, [cost] * PP_SIZE, host)
    rendezvous_period = period(done_at, len(launch_at))
    launch_at, done_at = _run(monkeypatch, False, [cost] * PP_SIZE, host)
    eager_period = period(done_at, len(launch_at))
    assert rendezvous_period >= (host + 2 * cost) / 2 * 0.95, rendezvous_period
    assert eager_period < 0.9 * rendezvous_period, (eager_period, rendezvous_period)
