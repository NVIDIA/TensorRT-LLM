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
"""Unit tests for the SMG adapter's load feed: the estimator, the record, the stream (no engine)."""

import asyncio
import math
from datetime import timedelta
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "smg_grpc_proto",
    reason='SMG gRPC adapter dependency not installed (pip install "tensorrt_llm[grpc-smg]")',
)

import grpc  # noqa: E402
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory  # noqa: E402

from tensorrt_llm.grpc.smg import load_feed  # noqa: E402
from tensorrt_llm.grpc.smg.bindings import common_pb2  # noqa: E402
from tensorrt_llm.grpc.smg.bindings import trtllm_service_pb2 as pb2  # noqa: E402
from tensorrt_llm.grpc.smg.load_tracker import (  # noqa: E402
    EngineStats,
    EngineStatsFold,
    KvFigures,
    LoadRecord,
    LoadTracker,
    RecordPacer,
    compose_record,
    load_changed,
    prefill_sample,
)
from tensorrt_llm.grpc.smg.request_manager import (  # noqa: E402
    GrpcRequestManager,
    create_sampling_params_from_proto,
)
from tensorrt_llm.grpc.smg.servicer import TrtllmServiceServicer  # noqa: E402

pytestmark = pytest.mark.cpu_only


class _Clock:
    """A monotonic clock the test advances."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _record(**overrides) -> LoadRecord:
    fields = dict(
        running_requests=2,
        waiting_requests=1,
        waiting_uncached_tokens=300,
        token_usage=0.25,
        gen_throughput=800.0,
        max_running_requests=64,
        age_ms=0,
        cache_hit_rate=0.5,
        num_used_tokens=4096,
        max_total_num_tokens=16384,
        prefill_pending_tokens=600,
        prefill_throughput=12000.0,
    )
    fields.update(overrides)
    return LoadRecord(**fields)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============================================================================
# The estimator
# ============================================================================


def test_prefill_capacity_unions_concurrency_keeps_idle_and_expires() -> None:
    clock = _Clock()
    tracker = LoadTracker(clock=clock)
    tracker.prefill_computed(1_000, 10.0, 12.0)
    assert tracker.estimate().prefill_throughput is None
    tracker.prefill_computed(2_000, 11.0, 13.0)
    # 3000 computed tokens over 3 active seconds, not 4 summed request seconds.
    assert tracker.estimate().prefill_throughput == 1_000.0
    tracker.generated(1)
    assert tracker.estimate().gen_throughput == 0.5
    clock.advance(30.0)
    assert tracker.estimate().prefill_throughput == 1_000.0
    clock.advance(31.0)
    assert tracker.estimate().prefill_throughput is None


def test_prefill_capacity_rejects_cache_hits_and_invalid_samples() -> None:
    tracker = LoadTracker(clock=_Clock())
    for tokens, start, end in [
        (0, 1.0, 2.0),
        (100, 2.0, 1.0),
        (100, 1.0, 1.0),
        (100, math.nan, 2.0),
        (100, 1.0, math.inf),
        (100, -1.0, 2.0),
    ]:
        tracker.prefill_computed(tokens, start, end)
    assert tracker.estimate().prefill_throughput is None
    tracker.prefill_computed(1_000, 1.0, 2.0)
    tracker.prefill_computed(1_000, 100.0, 101.0)
    # Idle engine time between the two requests is excluded.
    assert tracker.estimate().prefill_throughput == 1_000.0


def test_queued_token_work_is_the_youngest_waiting_prompts_discounted_by_the_hit_rate() -> None:
    tracker = LoadTracker(clock=_Clock())
    tracker.forwarded("a", 100)
    tracker.forwarded("b", 200)
    tracker.forwarded("c", 300)
    tracker.started("a", cached_tokens=50)
    estimate = tracker.estimate(num_waiting_reqs=1)
    assert (estimate.running, estimate.waiting) == (1, 2)
    assert estimate.cache_hit_rate == 0.5
    assert estimate.queued_token_work == 150
    assert estimate.prefill_pending_tokens == 500
    assert tracker.estimate().queued_token_work == 250
    assert tracker.estimate(num_waiting_reqs=0).queued_token_work == 0
    tracker.finished("a")
    tracker.started("unknown", cached_tokens=5)
    assert tracker.estimate().running == 0
    tracker.started("b", cached_tokens=400)
    assert tracker.estimate().cache_hit_rate == 250 / 300


def test_generation_throughput_covers_the_window_only() -> None:
    clock = _Clock()
    tracker = LoadTracker(clock=clock)
    tracker.generated(100)
    tracker.generated(0)
    assert tracker.estimate().gen_throughput == 50.0
    clock.advance(1.5)
    tracker.generated(20)
    assert tracker.estimate().gen_throughput == 60.0
    clock.advance(1.0)
    assert tracker.estimate().gen_throughput == 10.0


@pytest.mark.parametrize(
    ("prompt", "cached", "start", "end", "flags", "expected"),
    [
        (10, 4, 5.0, 6.0, {}, (6, 5.0, 6.0)),
        (10, 10, 5.0, 6.0, {}, None),
        (10, 0, 0.0, 6.0, {}, None),
        (10, 0, 5.0, None, {}, None),
        (10, 0, 5.0, 6.0, {"cancelled": True}, None),
        (10, 0, 5.0, 6.0, {"generation_only": True}, None),
    ],
)
def test_prefill_sample_rules(prompt, cached, start, end, flags, expected) -> None:
    assert prefill_sample(prompt, cached, start, end, **flags) == expected


# ============================================================================
# The engine's statistics and the record
# ============================================================================


def _iteration(rank: int, **overrides) -> dict:
    record = {
        "attentionDpRank": rank,
        "numActiveRequests": 3,
        "numQueuedRequests": 2,
        "maxNumActiveRequests": 32,
        "kvCacheStats": {"usedNumBlocks": 10, "maxNumBlocks": 100, "tokensPerBlock": 32},
        "inflightBatchingStats": {"numQueuedCtxTokens": 1_000},
    }
    record.update(overrides)
    return record


def test_engine_stats_read_the_iteration_record_and_fold_the_ranks() -> None:
    assert EngineStats.from_iteration({"iter": 1}, at=1.0) is None
    one = EngineStats.from_iteration(_iteration(0), at=1.0)
    assert (one.running, one.waiting, one.max_running, one.queued_prompt_tokens) == (
        3,
        2,
        32,
        1_000,
    )
    assert (one.used_blocks, one.max_blocks, one.tokens_per_block) == (10, 100, 32)
    bare = EngineStats.from_iteration({"numActiveRequests": 1}, at=2.0)
    assert (bare.rank, bare.waiting, bare.queued_prompt_tokens, bare.max_blocks) == (0, 0, None, 0)

    fold = EngineStatsFold()
    assert fold.current() is None and fold.kv() is None
    fold.observe(one)
    fold.observe(EngineStats.from_iteration(_iteration(1, numQueuedRequests=5), at=3.0))
    folded = fold.current()
    assert (folded.at, folded.running, folded.waiting, folded.max_running) == (1.0, 6, 7, 64)
    assert folded.queued_prompt_tokens == 2_000
    assert fold.kv() == KvFigures(used_tokens=640, max_tokens=6_400)
    assert fold.kv().usage == 0.1


def test_compose_record_takes_fresh_engine_counts_else_the_adapters_own() -> None:
    tracker = LoadTracker(clock=_Clock())
    engine = EngineStats.from_iteration(_iteration(0), at=10.0)
    kv = KvFigures(used_tokens=640, max_tokens=6_400)

    idle = compose_record(tracker.estimate(), engine, kv, now=10.5, max_running_fallback=8)
    assert (idle.running_requests, idle.waiting_requests, idle.waiting_uncached_tokens) == (0, 0, 0)
    assert (idle.token_usage, idle.num_used_tokens, idle.max_total_num_tokens) == (0.1, 640, 6_400)
    assert idle.max_running_requests == 32 and idle.prefill_throughput is None

    tracker.forwarded("a", 100)
    tracker.forwarded("b", 200)
    tracker.started("a", cached_tokens=50)
    fresh = compose_record(
        tracker.estimate(engine.waiting), engine, kv, now=10.5, max_running_fallback=8
    )
    assert (fresh.running_requests, fresh.waiting_requests) == (3, 2)
    assert fresh.waiting_uncached_tokens == 500  # the engine's queued prompt tokens, discounted
    assert fresh.age_ms == 500 and fresh.prefill_pending_tokens == 200

    stale = compose_record(tracker.estimate(), engine, None, now=13.0, max_running_fallback=8)
    assert (stale.running_requests, stale.waiting_requests, stale.max_running_requests) == (1, 1, 8)
    assert stale.waiting_uncached_tokens == 100 and stale.age_ms == 0
    assert (stale.token_usage, stale.num_used_tokens, stale.max_total_num_tokens) == (
        0.0,
        None,
        None,
    )


def test_load_changed_rules() -> None:
    base = _record()
    assert not load_changed(base, _record())
    assert load_changed(base, _record(running_requests=3))
    assert load_changed(base, _record(waiting_requests=0))
    assert load_changed(base, _record(waiting_uncached_tokens=None))
    assert load_changed(base, _record(max_running_requests=65))
    assert load_changed(base, _record(prefill_pending_tokens=601))
    assert load_changed(base, _record(token_usage=0.256))
    assert not load_changed(base, _record(token_usage=0.254))
    assert load_changed(base, _record(gen_throughput=851.0))
    assert not load_changed(base, _record(gen_throughput=830.0))
    assert load_changed(base, _record(prefill_throughput=None))
    assert load_changed(_record(prefill_throughput=None), base)
    assert load_changed(base, _record(prefill_throughput=12_060.0))
    assert not load_changed(base, _record(prefill_throughput=12_040.0))
    assert not load_changed(base, _record(cache_hit_rate=0.9, age_ms=5))


def test_record_pacer_sends_changes_at_once_and_heartbeats_with_backoff() -> None:
    pacer = RecordPacer(heartbeat_interval=1.0, heartbeat_backoff=5.0)
    record = _record()
    assert pacer.due(record, 0.0)
    assert not pacer.due(record, 0.5)
    assert pacer.due(record, 1.0)
    assert pacer.due(record, 2.0)
    assert not pacer.due(record, 3.0)
    assert pacer.due(record, 7.0)
    changed = _record(running_requests=5)
    assert pacer.due(changed, 7.1)
    assert not pacer.due(changed, 7.5)
    assert pacer.due(changed, 8.1)


# ============================================================================
# The wire
# ============================================================================


def _dynamic_messages():
    """Message classes of the proto shapes the installed package may predate.

    An ``EngineLoad`` that knows ``prefill_throughput`` (22), a ``SubscribeKvEventsRequest``
    that knows ``loads_only`` (3), and the ``GetLoads`` answer.
    """
    file = descriptor_pb2.FileDescriptorProto(
        name="test_load_feed_dynamic.proto", package="test.load_feed", syntax="proto3"
    )
    T = descriptor_pb2.FieldDescriptorProto

    def optional_double(message, name, number):
        oneof_index = len(message.oneof_decl)
        message.oneof_decl.add(name=f"_{name}")
        message.field.add(
            name=name,
            number=number,
            type=T.TYPE_DOUBLE,
            label=T.LABEL_OPTIONAL,
            proto3_optional=True,
            oneof_index=oneof_index,
        )

    load = file.message_type.add(name="EngineLoadNew")
    load.field.add(name="running_requests", number=1, type=T.TYPE_UINT32, label=T.LABEL_OPTIONAL)
    load.field.add(name="gen_throughput", number=5, type=T.TYPE_DOUBLE, label=T.LABEL_OPTIONAL)
    optional_double(load, "prefill_throughput", 22)

    request = file.message_type.add(name="SubscribeNew")
    request.field.add(
        name="start_sequence_number", number=1, type=T.TYPE_UINT64, label=T.LABEL_OPTIONAL
    )
    request.field.add(
        name="load_records_before_first_batch", number=2, type=T.TYPE_BOOL, label=T.LABEL_OPTIONAL
    )
    request.field.add(name="loads_only", number=3, type=T.TYPE_BOOL, label=T.LABEL_OPTIONAL)

    scheduler = file.message_type.add(name="SchedulerLoad")
    for number, (name, kind) in enumerate(
        [
            ("dp_rank", T.TYPE_INT32),
            ("num_running_reqs", T.TYPE_INT32),
            ("num_waiting_reqs", T.TYPE_INT32),
            ("num_total_reqs", T.TYPE_INT32),
            ("num_used_tokens", T.TYPE_INT32),
            ("max_total_num_tokens", T.TYPE_INT32),
            ("token_usage", T.TYPE_DOUBLE),
            ("gen_throughput", T.TYPE_DOUBLE),
            ("cache_hit_rate", T.TYPE_DOUBLE),
            ("utilization", T.TYPE_DOUBLE),
            ("max_running_requests", T.TYPE_INT32),
            ("num_waiting_uncached_tokens", T.TYPE_INT32),
        ],
        start=1,
    ):
        scheduler.field.add(name=name, number=number, type=kind, label=T.LABEL_OPTIONAL)
    optional_double(scheduler, "prefill_throughput", 13)

    response = file.message_type.add(name="GetLoadsResponse")
    response.field.add(name="timestamp", number=1, type=T.TYPE_STRING, label=T.LABEL_OPTIONAL)
    response.field.add(name="version", number=2, type=T.TYPE_STRING, label=T.LABEL_OPTIONAL)
    response.field.add(name="dp_rank_count", number=3, type=T.TYPE_INT32, label=T.LABEL_OPTIONAL)
    response.field.add(
        name="loads",
        number=4,
        type=T.TYPE_MESSAGE,
        label=T.LABEL_REPEATED,
        type_name=".test.load_feed.SchedulerLoad",
    )

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    classes = {
        name: message_factory.GetMessageClass(pool.FindMessageTypeByName(f"test.load_feed.{name}"))
        for name in ("EngineLoadNew", "SubscribeNew", "SchedulerLoad", "GetLoadsResponse")
    }
    return SimpleNamespace(**classes)


def test_engine_load_message_sets_the_optional_fields_and_carries_the_capacity() -> None:
    dynamic = _dynamic_messages()
    message = load_feed.engine_load_message(_record(), sample=3)
    assert (message.running_requests, message.waiting_requests, message.sample) == (2, 1, 3)
    assert message.load_only and message.waiting_uncached_tokens == 300
    assert message.HasField("cache_hit_rate") and message.HasField("prefill_pending_tokens")
    assert (message.num_used_tokens, message.max_total_num_tokens) == (4_096, 16_384)
    # A reader whose proto knows field 22 reads the capacity; one that does not keeps the bytes.
    newer = dynamic.EngineLoadNew.FromString(message.SerializeToString())
    assert newer.HasField("prefill_throughput") and newer.prefill_throughput == 12_000.0
    again = common_pb2.EngineLoad.FromString(message.SerializeToString())
    assert again.SerializeToString() == message.SerializeToString()

    absent = load_feed.engine_load_message(
        _record(
            waiting_uncached_tokens=None,
            cache_hit_rate=None,
            num_used_tokens=None,
            max_total_num_tokens=None,
            prefill_pending_tokens=None,
            prefill_throughput=None,
        ),
        sample=1,
    )
    for name in (
        "waiting_uncached_tokens",
        "cache_hit_rate",
        "num_used_tokens",
        "prefill_pending_tokens",
    ):
        assert not absent.HasField(name), name
    assert not dynamic.EngineLoadNew.FromString(absent.SerializeToString()).HasField(
        "prefill_throughput"
    )

    batch = load_feed.load_only_batch(_record(), dp_rank=0, sample=4, timestamp=12.5)
    assert (batch.sequence_number, batch.dp_rank, batch.timestamp) == (0, 0, 12.5)
    assert batch.load.load_only and batch.load.sample == 4 and not batch.events


def test_wants_loads_only_reads_the_field_whether_the_installed_proto_knows_it() -> None:
    dynamic = _dynamic_messages()
    assert not load_feed.wants_loads_only(common_pb2.SubscribeKvEventsRequest())
    assert not load_feed.wants_loads_only(
        common_pb2.SubscribeKvEventsRequest(load_records_before_first_batch=True)
    )
    newer = dynamic.SubscribeNew(load_records_before_first_batch=True, loads_only=True)
    assert load_feed.wants_loads_only(newer)
    assert not load_feed.wants_loads_only(dynamic.SubscribeNew(loads_only=False))
    # The gateway's request parsed by a package that predates the field keeps it unknown.
    older = common_pb2.SubscribeKvEventsRequest.FromString(newer.SerializeToString())
    assert "loads_only" not in older.DESCRIPTOR.fields_by_name
    assert load_feed.wants_loads_only(older)


def test_get_loads_response_lists_the_record_as_rank_zero() -> None:
    dynamic = _dynamic_messages()
    response = load_feed.get_loads_response([_record()], version="1.2.3", pb2=dynamic)
    assert response.version == "1.2.3" and response.dp_rank_count == 1 and response.timestamp
    (load,) = response.loads
    assert (load.dp_rank, load.num_running_reqs, load.num_waiting_reqs, load.num_total_reqs) == (
        0,
        2,
        1,
        3,
    )
    assert (load.num_used_tokens, load.max_total_num_tokens, load.max_running_requests) == (
        4_096,
        16_384,
        64,
    )
    assert (load.token_usage, load.utilization, load.gen_throughput, load.cache_hit_rate) == (
        0.25,
        0.25,
        800.0,
        0.5,
    )
    assert load.num_waiting_uncached_tokens == 300 and load.prefill_throughput == 12_000.0
    bare = load_feed.scheduler_load_message(
        _record(prefill_throughput=None, num_used_tokens=None, cache_hit_rate=None),
        dp_rank=1,
        pb2=dynamic,
    )
    assert (
        not bare.HasField("prefill_throughput") and bare.num_used_tokens == 0 and bare.dp_rank == 1
    )
    assert not load_feed.serves_get_loads()
    assert load_feed.serves_get_loads(SimpleNamespace(GetLoads=None))


def test_load_record_stream_sends_the_first_record_changes_and_heartbeats() -> None:
    clock = _Clock()
    records = iter(
        [_record(), _record(), _record(running_requests=3), _record(running_requests=3), None]
    )
    current = {"record": None}

    def source():
        current["record"] = next(records)
        return current["record"]

    async def sleep(seconds):
        clock.advance(seconds)

    stream = load_feed.LoadRecordStream(
        source,
        dp_rank=0,
        load_tick=0.6,
        heartbeat_interval=1.0,
        heartbeat_backoff=5.0,
        clock=clock,
        sleep=sleep,
    )

    async def collect():
        return [batch async for batch in stream.batches()]

    batches = _run(collect())
    # Tick 0: first record. Tick 0.6: unchanged, silent 0.6 < 1 s. Tick 1.2: changed. Tick 1.8: silent 0.6.
    # Then the source has no record: the stream ends.
    assert [(b.load.sample, b.load.running_requests) for b in batches] == [(1, 2), (2, 3)]
    assert all(b.sequence_number == 0 and b.dp_rank == 0 and b.load.load_only for b in batches)
    assert clock.now == pytest.approx(1000.0 + 4 * 0.6)

    ticks = {"n": 0}
    cancelled_stream = load_feed.LoadRecordStream(
        lambda: _record(), load_tick=0.1, clock=clock, sleep=sleep
    )

    def cancelled():
        ticks["n"] += 1
        return ticks["n"] > 3

    async def collect_until_cancelled():
        return [batch async for batch in cancelled_stream.batches(cancelled=cancelled)]

    assert [b.load.sample for b in _run(collect_until_cancelled())] == [1]


# ============================================================================
# The servicer and the request manager
# ============================================================================


class _Context:
    """Minimal mock for grpc.aio.ServicerContext."""

    def __init__(self, cancel_after: int = 0) -> None:
        self.code = None
        self.details = None
        self._polls = 0
        self._cancel_after = cancel_after

    def cancelled(self) -> bool:
        self._polls += 1
        return self._polls > self._cancel_after

    async def abort(self, code, details=""):
        self.code = code
        self.details = details
        raise grpc.aio.AbortError(code, details)


def test_servicer_refuses_event_subscriptions_and_streams_load_records(monkeypatch) -> None:
    manager = SimpleNamespace(load_record=lambda: _record())
    servicer = TrtllmServiceServicer(manager)
    monkeypatch.setattr(load_feed.LoadRecordStream, "__init__", _fast_stream_init)

    async def events():
        context = _Context()
        try:
            async for _ in servicer.SubscribeKvEvents(
                common_pb2.SubscribeKvEventsRequest(), context
            ):
                pass
        except grpc.aio.AbortError:
            pass
        return context.code, context.details

    assert _run(events()) == (grpc.StatusCode.UNIMPLEMENTED, load_feed.NO_KV_EVENTS)

    async def records():
        request = common_pb2.SubscribeKvEventsRequest.FromString(
            _dynamic_messages()
            .SubscribeNew(load_records_before_first_batch=True, loads_only=True)
            .SerializeToString()
        )
        return [
            batch async for batch in servicer.SubscribeKvEvents(request, _Context(cancel_after=2))
        ]

    batches = _run(records())
    assert [b.load.sample for b in batches] == [1]
    assert batches[0].load.running_requests == 2 and batches[0].load.load_only


_ORIGINAL_STREAM_INIT = load_feed.LoadRecordStream.__init__


def _fast_stream_init(self, source, **kwargs):
    """A stream whose ticks take no wall time (the servicer builds it with the defaults)."""
    clock = _Clock()

    async def sleep(seconds):
        clock.advance(seconds)

    kwargs.update(load_tick=0.1, clock=clock, sleep=sleep)
    _ORIGINAL_STREAM_INIT(self, source, **kwargs)


def test_servicer_get_loads_answers_with_the_installed_messages(monkeypatch) -> None:
    dynamic = _dynamic_messages()
    monkeypatch.setattr(load_feed, "trtllm_service_pb2", dynamic)
    servicer = TrtllmServiceServicer(
        SimpleNamespace(load_record=lambda: _record(running_requests=7))
    )
    response = _run(servicer.GetLoads(SimpleNamespace(), _Context()))
    assert response.dp_rank_count == 1 and response.loads[0].num_running_reqs == 7
    assert response.version


def _result(tokens, *, cached=2, finished=False, timing=None, finish_reason=None, error=None):
    metrics = None if timing is None else SimpleNamespace(timing_metrics=timing)
    return SimpleNamespace(
        cached_tokens=cached,
        finished=finished,
        error_msg=error,
        aborted=lambda: False,
        outputs=[
            SimpleNamespace(
                index=0,
                token_ids=list(tokens),
                request_perf_metrics=metrics,
                finish_reason=finish_reason,
            )
        ],
    )


def _timing(start: float, end: float):
    return SimpleNamespace(
        first_scheduled_time=timedelta(seconds=start), first_token_time=timedelta(seconds=end)
    )


class _FakeLlm:
    """An LLM whose generate_async replays scripted results, holding where a script says so."""

    def __init__(self, scripts, *, enable_iter_perf_stats=False, stats=()):
        self.args = SimpleNamespace(
            max_batch_size=16, enable_iter_perf_stats=enable_iter_perf_stats
        )
        self._scripts = list(scripts)
        self._stats = list(stats)
        self.released = asyncio.Event()

    def generate_async(self, inputs, sampling_params, **kwargs):
        script = self._scripts.pop(0)
        llm = self

        class Result:
            def __aiter__(self):
                return self._iterate()

            async def _iterate(self):
                for item in script:
                    if item == "hold":
                        await llm.released.wait()
                    else:
                        yield item

            def abort(self):
                pass

        return Result()

    def get_stats(self, timeout):
        if self._stats:
            return self._stats.pop(0)
        return []


def test_request_manager_feeds_the_record_from_the_requests_it_forwards() -> None:
    llm = _FakeLlm(
        [
            [_result([1], timing=_timing(10.0, 12.0)), "hold", _result([1, 2, 3], finished=True)],
            [_result([5], cached=1, finished=True, timing=_timing(11.0, 13.0))],
        ]
    )
    manager = GrpcRequestManager(llm)

    async def scenario():
        idle = manager.load_record()
        assert (idle.running_requests, idle.waiting_requests, idle.max_running_requests) == (
            0,
            0,
            16,
        )
        first = manager.generate("r1", [1, 2, 3, 4], sampling_params=None)
        await first.__anext__()
        held = manager.load_record()
        assert (held.running_requests, held.waiting_requests, held.prefill_throughput) == (
            1,
            0,
            None,
        )
        assert held.cache_hit_rate == 0.5 and held.gen_throughput == 0.5
        second = [r async for r in manager.generate("r2", [1, 2, 3, 4], sampling_params=None)]
        assert len(second) == 1
        # Two computed-token samples: (4-2)=2 over [10,12] and (4-1)=3 over [11,13]: 5 tokens / 3 s.
        assert manager.load_record().prefill_throughput == pytest.approx(5 / 3)
        llm.released.set()
        rest = [r async for r in first]
        assert len(rest) == 1
        done = manager.load_record()
        assert (done.running_requests, done.waiting_requests, done.waiting_uncached_tokens) == (
            0,
            0,
            0,
        )
        assert done.gen_throughput == pytest.approx((1 + 2 + 1) / 2)
        assert done.prefill_pending_tokens == 0

    _run(scenario())


def test_request_manager_reads_the_engine_statistics_while_requests_are_in_flight() -> None:
    llm = _FakeLlm(
        [[_result([1]), "hold", _result([1, 2], finished=True)]],
        enable_iter_perf_stats=True,
        stats=[[_iteration(0, numActiveRequests=1, numQueuedRequests=4)]],
    )
    manager = GrpcRequestManager(llm)

    async def scenario():
        manager.start_load_feed()
        manager.start_load_feed()
        stream = manager.generate("r1", [1, 2, 3, 4], sampling_params=None)
        await stream.__anext__()
        for _ in range(50):
            await asyncio.sleep(0.01)
            if manager.load_record().waiting_requests == 4:
                break
        record = manager.load_record()
        assert (record.running_requests, record.waiting_requests, record.max_running_requests) == (
            1,
            4,
            32,
        )
        assert record.waiting_uncached_tokens == 500 and record.token_usage == 0.1
        assert (record.num_used_tokens, record.max_total_num_tokens) == (320, 3_200)
        manager.stop_load_feed()
        llm.released.set()
        assert len([r async for r in stream]) == 1
        assert manager.load_record().running_requests == 0

    _run(scenario())
    assert GrpcRequestManager(_FakeLlm([]))._engine_emits_stats() is False


def test_sampling_params_carry_the_perf_metrics_request() -> None:
    config = pb2.SamplingConfig(beam_width=1, num_return_sequences=1)
    assert not create_sampling_params_from_proto(
        config, pb2.OutputConfig(), max_tokens=8
    ).return_perf_metrics
    params = create_sampling_params_from_proto(
        config, pb2.OutputConfig(return_perf_metrics=True), max_tokens=8
    )
    assert params.return_perf_metrics
