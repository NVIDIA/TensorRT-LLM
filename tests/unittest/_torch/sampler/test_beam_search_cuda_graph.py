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
"""State, ownership and lifecycle regressions for native beam sampling CUDA graphs."""

import copy
import dataclasses
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler import beam_search as beam
from tensorrt_llm._torch.pyexecutor.sampler import beam_search_graph as graphs
from tensorrt_llm._torch.pyexecutor.sampler.beam_search_graph import (
    BeamSearchGraphCaptureOwner as Owner,
)
from tensorrt_llm._torch.pyexecutor.sampler.beam_search_graph import (
    BeamSearchGraphProfile as Profile,
)
from tensorrt_llm._torch.pyexecutor.sampler.beam_search_graph import BeamSearchGraphRunner
from tensorrt_llm._torch.pyexecutor.sampler.beam_search_graph import BeamSearchGraphRunner as Runner
from tensorrt_llm._torch.pyexecutor.sampler.ops.flashinfer import radix_topk_op
from tensorrt_llm._torch.pyexecutor.sampler.sampler import TorchSampler

NATIVE = beam.beam_search_sampling_batch_cba


@pytest.fixture
def capture_runner() -> Iterator[BeamSearchGraphRunner]:
    runner = BeamSearchGraphRunner()
    try:
        with runner.allow_capture():
            yield runner
    finally:
        runner.clear()


def state(
    batch: int = 2, capacity: int = 4, window: int = 0, end_id: int = -1
) -> beam.BeamSearchMetadata:
    slots = 4
    store = beam.BeamSearchStore.create(
        cache_indirection_shape=(slots, capacity, 16),
        max_num_sequences=slots,
        max_beam_width=capacity,
    )
    fields = store.ensure_cba()
    for owner in (store, fields):
        for field in dataclasses.fields(owner):
            value = getattr(owner, field.name)
            if isinstance(value, torch.Tensor):
                value.zero_()
    store.beam_idx_arange.copy_(torch.arange(capacity, device="cuda", dtype=torch.int32))
    store.prompt_lens.fill_(2)
    fields.cba_normed_scores.fill_(-float("inf"))
    fields.cba_tokens.fill_(beam.BEAM_SEARCH_PAD_TOKEN)
    fields.cba_caps.fill_(capacity)
    cba = beam.CBAState(
        **{f.name: getattr(fields, f.name) for f in dataclasses.fields(fields)},
        end_ids=torch.full((slots,), end_id, device="cuda", dtype=torch.int32),
        prompt_lens=store.prompt_lens,
        original_tokens=store.original_tokens,
        batch_dones=store.batch_dones,
        max_seq_len=16,
        max_gen_len=8,
    )
    return beam.BeamSearchMetadata(
        cache_indirection=store.cache_indirection,
        cache_indirection_buffer=store.cache_indirection_buffer,
        cum_log_probs=store.cum_log_probs,
        new_log_probs=torch.zeros(slots, capacity, device="cuda"),
        seq_slots=torch.tensor([2, 0][:batch], device="cuda", dtype=torch.long),
        seq_lens=torch.full((batch,), 3, device="cuda", dtype=torch.int32),
        finished_beams=store.first_finish_reasons,
        pending_harvest=store.pending_harvest,
        predecessor_beams=store.predecessor_beams,
        beam_idx_arange=store.beam_idx_arange,
        stop_past_tokens=(
            torch.arange(window * slots * capacity, device="cuda", dtype=torch.int32).reshape(
                window, slots, capacity
            )
            if window
            else None
        ),
        cba=cba,
    )


def equal(actual: object, expected: object) -> None:
    if isinstance(actual, torch.Tensor):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(
            actual.contiguous().reshape(-1).view(torch.uint8),
            expected.contiguous().reshape(-1).view(torch.uint8),
        )
    elif dataclasses.is_dataclass(actual):
        for f in dataclasses.fields(actual):
            equal(getattr(actual, f.name), getattr(expected, f.name))
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal(a, b)
    else:
        assert actual == expected


def restore(dst: object, src: object) -> None:
    if isinstance(src, torch.Tensor):
        dst.copy_(src)
    elif dataclasses.is_dataclass(src):
        for f in dataclasses.fields(src):
            restore(getattr(dst, f.name), getattr(src, f.name))


def kwargs(width: int = 2, output: int = 4, stride: int | None = 4, probs: bool = True) -> dict:
    return dict(
        beam_width_in=width,
        beam_width_out=output,
        row_stride=stride,
        temperature=1.0,
        early_stopping=1,
        return_probs=probs,
    )


CASES = [
    (1, 4, 1, 4, 4, 32, False, None, None, None, 1, -1, 0),
    (2, 8, 2, 7, 8, 32, True, 1.0, 1.2, 0.0, 0, 3, 3),
    (2, 4, 4, 2, 4, 32, False, 0.7, -0.2, 0.4, 2, 3, 2),
    (2, 8, 2, 4, 8, 16384, True, 1.0, 0.0, 0.2, 1, -1, 0),
    (2, 8, 2, 4, 8, 32, True, 0.0, None, None, 1, -1, 2),
    (1, 4, 2, 4, 2, 32, True, None, 1.2, None, 2, 3, 0),
]


@pytest.mark.parametrize("case", CASES)
def test_state_and_outputs_match_with_padding_and_slot_reuse(case, capture_runner) -> None:
    (
        batch,
        cap,
        width,
        output,
        stride,
        vocab,
        probs,
        temp,
        penalty,
        diversity,
        early,
        end,
        window,
    ) = case
    torch.manual_seed(123)
    actual = state(batch, cap, window, end)
    reference, initial = copy.deepcopy(actual), copy.deepcopy(actual)
    options = {
        **kwargs(width, output, stride, probs),
        "temperature": temp,
        "length_penalty": penalty,
        "diversity_rate": diversity,
        "early_stopping": early,
    }
    retained = []
    for iteration in range(8):
        restore(actual, initial)
        restore(reference, initial)
        for value in (actual, reference):
            value.seq_slots.copy_(
                torch.tensor(([2, 0] if iteration % 2 else [1, 3])[:batch], device="cuda")
            )
            value.seq_lens.fill_(3 + iteration % 3)
            if width > 1 and iteration % 2:
                value.pending_harvest[value.seq_slots, 0] = True
        logits = torch.randn(batch * stride, vocab, device="cuda")
        logits.view(batch, stride, vocab)[:, width:].fill_(10000)
        expected = NATIVE(logits, beam_search_args=reference, **options)
        got = capture_runner.sample(logits, beam_search_args=actual, **options)
        equal(got, expected)
        equal(actual, reference)
        retained.append((got, tuple(x.clone() if x is not None else None for x in got)))
    assert capture_runner.capture_count == 1 and capture_runner.capture_failures == 0
    entry = next(iter(capture_runner._entries.values()))
    assert entry.logits.shape == (batch * width, vocab)
    for got, saved in retained:
        equal(got, saved)


def test_shared_pool_replays_in_arbitrary_order_without_clobbering_outputs(capture_runner) -> None:
    torch.manual_seed(22)
    actual, reference = state(), state()
    initial = copy.deepcopy(actual)
    retained = []
    for width in [1] * 4 + [2] * 4 + [4] * 4 + [2, 1, 4, 1, 2, 4, 2]:
        restore(actual, initial)
        restore(reference, initial)
        logits = torch.randn(8, 512, device="cuda")
        options = kwargs(width=width)
        got = capture_runner.sample(logits, beam_search_args=actual, **options)
        expected = NATIVE(logits, beam_search_args=reference, **options)
        equal(got, expected)
        equal(actual, reference)
        retained.append((got, tuple(x.clone() if x is not None else None for x in got)))
    assert capture_runner.capture_count == 3 and len(capture_runner._pools) == 1
    for got, saved in retained:
        equal(got, saved)


def test_sequential_state_progression(capture_runner) -> None:
    actual = state(window=3)
    actual.cba.max_gen_len = 12
    reference = copy.deepcopy(actual)
    options = kwargs(width=4)
    for iteration in range(8):
        actual.seq_lens.fill_(3 + iteration)
        reference.seq_lens.fill_(3 + iteration)
        logits = torch.randn(8, 128, device="cuda")
        equal(
            capture_runner.sample(logits, beam_search_args=actual, **options),
            NATIVE(logits, beam_search_args=reference, **options),
        )
        equal(actual, reference)
    assert capture_runner.capture_count == 1


@pytest.mark.parametrize(
    "field",
    [
        "cum_log_probs",
        "new_log_probs",
        "cache_indirection",
        "pending_harvest",
        "original_tokens",
        "end_ids",
    ],
)
def test_reallocated_state_gets_a_new_graph(field, capture_runner) -> None:
    actual = state()
    initial = copy.deepcopy(actual)
    for generation in range(2):
        if generation:
            owner = actual if hasattr(actual, field) else actual.cba
            setattr(owner, field, getattr(owner, field).clone())
        reference = copy.deepcopy(actual)
        for _ in range(5):
            restore(actual, initial)
            restore(reference, initial)
            logits = torch.randn(8, 32, device="cuda")
            equal(
                capture_runner.sample(logits, beam_search_args=actual, **kwargs()),
                NATIVE(logits, beam_search_args=reference, **kwargs()),
            )
            equal(actual, reference)
    assert capture_runner.capture_count == 2


def test_stop_buffer_resize_and_optional_buffer(capture_runner) -> None:
    actual = state()
    for window in [0, 2, 5, 0]:
        actual.stop_past_tokens = (
            torch.zeros(window, 4, 4, device="cuda", dtype=torch.int32) if window else None
        )
        initial = copy.deepcopy(actual)
        reference = copy.deepcopy(actual)
        for _ in range(5):
            restore(actual, initial)
            restore(reference, initial)
            logits = torch.randn(8, 32, device="cuda")
            equal(
                capture_runner.sample(logits, beam_search_args=actual, **kwargs()),
                NATIVE(logits, beam_search_args=reference, **kwargs()),
            )
            equal(actual, reference)
    assert capture_runner.capture_failures == 0


@pytest.mark.parametrize("vocab", [128, 16384])
@pytest.mark.parametrize("masked", [False, True])
def test_tied_logits_match_native(vocab, masked, capture_runner) -> None:
    actual = state()
    reference = copy.deepcopy(actual)
    initial = copy.deepcopy(actual)
    for iteration in range(6):
        restore(actual, initial)
        restore(reference, initial)
        logits = torch.zeros(8, vocab, device="cuda")
        if masked:
            logits[:, 8:] = -float("inf")
        got = capture_runner.sample(logits, beam_search_args=actual, **kwargs())
        expected = NATIVE(logits, beam_search_args=reference, **kwargs())
        equal(got, expected)
        equal(actual, reference)
    assert capture_runner.capture_count == 1


def test_noncontiguous_logits_and_implicit_row_stride(capture_runner) -> None:
    actual = state()
    reference = copy.deepcopy(actual)
    initial = copy.deepcopy(actual)
    for _ in range(6):
        restore(actual, initial)
        restore(reference, initial)
        logits = torch.randn(4, 128, device="cuda")[:, ::2]
        assert not logits.is_contiguous()
        options = kwargs(stride=None)
        equal(
            capture_runner.sample(logits, beam_search_args=actual, **options),
            NATIVE(logits, beam_search_args=reference, **options),
        )
        equal(actual, reference)
    assert capture_runner.capture_count == 1


def test_clear_drains_work_on_another_stream(capture_runner) -> None:
    actual = state()
    logits = torch.randn(8, 32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            capture_runner.sample(logits, beam_search_args=actual, **kwargs())
        torch.cuda._sleep(2_000_000)
        completed = torch.cuda.Event()
        completed.record()
    capture_runner.clear()
    assert completed.query()
    assert not capture_runner._entries and not capture_runner._pools


def prepare(cache, values, metadata, options):
    initial = copy.deepcopy(metadata)
    with cache.allow_capture():
        for _ in range(4):
            restore(metadata, initial)
            cache.sample(values, beam_search_args=metadata, **options)
    restore(metadata, initial)
    torch.cuda.synchronize()


def test_requests_never_capture_and_unseen_keys_remain_eager() -> None:
    cache = Runner()
    actual = state()
    reference = copy.deepcopy(actual)
    values = torch.randn(8, 32, device="cuda")
    for _ in range(8):
        equal(
            cache.sample(values, beam_search_args=actual, **kwargs()),
            NATIVE(values, beam_search_args=reference, **kwargs()),
        )
    assert cache.capture_count == 0 and not cache._entries
    prepare(cache, values, actual, kwargs())
    count = cache.capture_count
    reference = copy.deepcopy(actual)
    for _ in range(6):
        equal(
            cache.sample(values, beam_search_args=actual, **kwargs(width=3)),
            NATIVE(values, beam_search_args=reference, **kwargs(width=3)),
        )
        equal(actual, reference)
    assert cache.capture_count == count == 1 and len(cache._entries) == 1
    cache.clear()


def test_two_samplers_and_external_native_topk_overlap(monkeypatch) -> None:
    import flashinfer

    # Exercise the inter-CTA row-state buffer even if dispatch heuristics change.
    monkeypatch.setenv("FLASHINFER_TOPK_ALGO", "multi_cta")
    torch.manual_seed(9)
    rounds = 24
    caches = [Runner() for _ in range(2)]
    streams = [torch.cuda.Stream() for _ in range(3)]
    actual = [state() for _ in range(2)]
    references = [copy.deepcopy(value) for value in actual]
    values = [[torch.randn(8, 151936, device="cuda") for _ in range(rounds)] for _ in caches]
    extra = torch.randn(4, 151936, device="cuda")
    expected_extra = flashinfer.top_k(extra, 40, sorted=True, deterministic=True)
    expected = [[] for _ in caches]
    for i, cache in enumerate(caches):
        streams[i].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(streams[i]):
            prepare(cache, values[i][0], actual[i], kwargs())
        for logits in values[i]:
            expected[i].append(NATIVE(logits, beam_search_args=references[i], **kwargs()))
    torch.cuda.synchronize()
    barrier = threading.Barrier(3)

    def work(index):
        with torch.cuda.device(0), torch.cuda.stream(streams[index]):
            barrier.wait(timeout=20)
            outputs = []
            for step in range(rounds):
                if index == 2:
                    outputs.append(flashinfer.top_k(extra, 40, sorted=True, deterministic=True))
                else:
                    outputs.append(
                        caches[index].sample(
                            values[index][step], beam_search_args=actual[index], **kwargs()
                        )
                    )
            return outputs

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(work, i) for i in range(3)]
        outputs = [future.result(timeout=90) for future in futures]
    torch.cuda.synchronize()
    for i in range(2):
        equal(outputs[i], expected[i])
        equal(actual[i], references[i])
        assert caches[i].capture_count == 1 and caches[i]._replay_error is None
    for output in outputs[2]:
        equal(output, expected_extra)
    assert (
        caches[0]._workspaces[torch.device("cuda:0")].data_ptr()
        != caches[1]._workspaces[torch.device("cuda:0")].data_ptr()
    )
    for cache in caches:
        cache.clear()


def test_shared_static_input_slab_reuses_storage() -> None:
    cache = Runner()
    actual = state()
    values = torch.randn(8, 512, device="cuda")
    cache._input(8, 512, values.dtype, values.device)
    with cache.allow_capture():
        for width in [1, 2, 4]:
            for _ in range(4):
                cache.sample(values, beam_search_args=actual, **kwargs(width=width))
    assert cache.capture_count == 3
    entries = list(cache._entries.values())
    assert len({e.logits.untyped_storage().data_ptr() for e in entries}) == 1
    cache.clear()


def test_capture_stream_is_owned_and_matches_input_device(monkeypatch) -> None:
    cache = Runner()
    actual = state()
    values = torch.randn(8, 32, device="cuda")
    streams = []
    native = cache._call

    def observe(*args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            streams.append(torch.cuda.current_stream(values.device))
        return native(*args, **kwargs)

    monkeypatch.setattr(cache, "_call", observe)
    prepare(cache, values, actual, kwargs())
    assert len(streams) == 1
    assert streams[0] == cache._capture_streams[values.device]
    assert streams[0].device == values.device and streams[0] != torch.cuda.default_stream(
        values.device
    )
    cache.clear()
    assert not cache._capture_streams


@pytest.mark.parametrize("k", [40, 2049])
@pytest.mark.parametrize("ties", [False, True])
def test_workspace_keeps_native_radix_results(k, ties) -> None:
    import flashinfer

    values = torch.randn(4, 151936, device="cuda")
    if ties:
        values = (values * 4).round()
    workspace = torch.zeros(1024 * 1024, device="cuda", dtype=torch.uint8)
    equal(
        radix_topk_op(values, k, workspace),
        flashinfer.top_k(values, k, sorted=True, deterministic=True),
    )


def test_owner_registry_admits_only_sole_live_executor() -> None:
    first = Owner()
    try:
        with first.allow_capture() as admitted:
            assert admitted
        second = Owner()
        with first.allow_capture() as admitted:
            assert not admitted
        with second.allow_capture() as admitted:
            assert not admitted
        second.close()
        with first.allow_capture() as admitted:
            assert admitted
    finally:
        first.close()
    with first.allow_capture() as admitted:
        assert not admitted


def test_owner_registration_cannot_race_capture() -> None:
    first = Owner()
    started = threading.Event()
    finished = threading.Event()

    def register():
        started.set()
        owner = Owner()
        finished.set()
        return owner

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with first.allow_capture() as allowed:
                assert allowed
                future = pool.submit(register)
                assert started.wait(5) and not finished.wait(0.05)
            second = future.result(5)
            assert finished.is_set()
            second.close()
    finally:
        first.close()


def make_sampler(profiles: tuple[Profile, ...] = ()) -> TorchSampler:
    return TorchSampler(
        TorchSampler.Args(
            max_seq_len=16,
            max_draft_len=0,
            max_num_sequences=4,
            max_beam_width=4,
            max_total_draft_tokens=0,
            beam_search_graph_profiles=profiles,
        )
    )


def executor_warmup():
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    return PyExecutor._warmup_sampler_cuda_graphs


def executor(sampler: TorchSampler) -> NS:
    return NS(
        sampler=sampler,
        worker_started=False,
        is_warmup=False,
        is_encoder_decoder=False,
        dist=NS(world_size=1),
        draft_model_engine=None,
        kv_cache_transceiver=None,
        kv_connector_manager=None,
        _beam_graph_capture_owner=Owner(),
        execution_stream=torch.cuda.Stream(),
        model_engine=NS(model=NS(model_config=NS(pretrained_config=NS(vocab_size=32)))),
    )


def test_real_sampler_startup_restores_state_and_dispatches_handler(monkeypatch) -> None:
    sampler = make_sampler((Profile(beam_widths=(2, 4), batch_sizes=(1, 2)),))
    sampler.store.beam_search_store.ensure_cba()
    before = copy.deepcopy(sampler.store)
    finish = copy.deepcopy(sampler._finish_reasons_handler.store)
    ex = executor(sampler)
    try:
        executor_warmup()(ex)
        equal(sampler.store, before)
        equal(sampler._finish_reasons_handler.store, finish)
        runner = sampler._beam_search._graph_runner
        assert runner is not None and runner.capture_count == 4 and runner.capture_failures == 0
        assert not sampler.needs_cuda_graph_warmup
        entry = next(iter(runner._entries.values()))
        metadata = entry.metadata
        # Simulate admission of a valid context request after state restoration.
        for tensor in graphs._state_tensors(metadata):
            tensor.zero_()
        metadata.seq_slots.zero_()
        metadata.seq_lens.fill_(1)
        metadata.beam_idx_arange.copy_(torch.arange(4, device="cuda", dtype=torch.int32))
        metadata.cba.prompt_lens.fill_(1)
        metadata.cba.end_ids.fill_(-1)
        metadata.cba.cba_normed_scores.fill_(-float("inf"))
        metadata.cba.cba_tokens.fill_(-1)
        metadata.cba.cba_caps.fill_(4)
        step = beam.BeamSearchStep(
            1,
            2,
            1,
            torch.ones(1, device="cuda"),
            None,
            None,
            beam.BeamSearchEarlyStop.TRUE,
            computes_probs=False,
        )
        called = []
        original = runner.sample

        def record(*a, **kw):
            called.append(True)
            return original(*a, **kw)

        monkeypatch.setattr(runner, "sample", record)
        sampler._beam_search.sample(step, torch.randn(1, 32, device="cuda"), metadata=metadata)
        assert called and runner.replay_count == 5
        sampler.release_cuda_graphs()
        assert (
            sampler._beam_search._graph_runner is None
            and not runner._entries
            and not runner._workspaces
        )
    finally:
        ex._beam_graph_capture_owner.close()
        sampler.release_cuda_graphs()


def test_service_capture_config_reaches_native_startup() -> None:
    from tensorrt_llm._torch.pyexecutor._util import create_torch_sampler_args
    from tensorrt_llm.llmapi import BeamSearchCudaGraphConfig

    args = create_torch_sampler_args(
        max_seq_len=16,
        speculative_config=None,
        max_beam_width=4,
        disable_overlap_scheduler=False,
        enable_async_worker=False,
        enable_speculative_beam_history_d2h=False,
        max_num_sequences=4,
        beam_search_cuda_graph_config=BeamSearchCudaGraphConfig(
            max_batch_size=2, capture_configs=[dict(beam_width_array=[2, 4], max_capture_tokens=3)]
        ),
    )
    sampler = TorchSampler(args)
    ex = executor(sampler)
    try:
        executor_warmup()(ex)
        runner = sampler._beam_search._graph_runner
        assert runner is not None
        assert runner.capture_count == 6 and runner.capture_failures == 0
        assert not sampler.needs_cuda_graph_warmup
    finally:
        ex._beam_graph_capture_owner.close()
        sampler.release_cuda_graphs()


@pytest.mark.parametrize(
    "guard",
    [
        "disabled",
        "is_warmup",
        "worker_started",
        "is_encoder_decoder",
        "distributed",
        "draft",
        "transfer",
        "connector",
        "other",
    ],
)
def test_native_executor_guards_before_accessing_model(guard) -> None:
    ex = NS(
        sampler=NS(needs_cuda_graph_warmup=guard != "disabled"),
        is_warmup=guard == "is_warmup",
        worker_started=guard == "worker_started",
        is_encoder_decoder=guard == "is_encoder_decoder",
        dist=NS(world_size=2 if guard == "distributed" else 1),
        draft_model_engine=object() if guard == "draft" else None,
        kv_cache_transceiver=object() if guard == "transfer" else None,
        kv_connector_manager=object() if guard == "connector" else None,
        _beam_graph_capture_owner=Owner(),
    )
    other = Owner() if guard == "other" else None
    try:
        executor_warmup()(ex)
    finally:
        if other is not None:
            other.close()
        ex._beam_graph_capture_owner.close()


def test_async_sampler_declines_profiles() -> None:
    sampler = TorchSampler(
        TorchSampler.Args(
            max_seq_len=16,
            max_draft_len=0,
            max_num_sequences=4,
            max_beam_width=4,
            max_total_draft_tokens=0,
            enable_async_worker=True,
            beam_search_graph_profiles=(Profile(beam_widths=(2, 4), batch_sizes=(1,)),),
        )
    )
    assert not sampler.needs_cuda_graph_warmup


def test_incomplete_preparation_does_not_capture_during_requests() -> None:
    runner = Runner()
    actual = state()
    values = torch.randn(8, 32, device="cuda")
    with runner.allow_capture():
        runner.sample(values, beam_search_args=actual, **kwargs())
    for _ in range(8):
        runner.sample(values, beam_search_args=actual, **kwargs())
    assert runner.capture_count == 0
    runner.clear()


def test_one_owner_events_order_delayed_cross_stream_updates(monkeypatch) -> None:
    runner = Runner()
    actual = state()
    values = torch.ones(8, 32, device="cuda")

    def delayed(logits, args, step, **kw):
        torch.cuda._sleep(2000000)
        value = args.cum_log_probs
        value.copy_(value + 1)
        return value.clone(), None

    monkeypatch.setattr(runner, "_call", delayed)
    prepare(runner, values, actual, kwargs())
    actual.cum_log_probs.zero_()
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for index in range(24):
        with torch.cuda.stream(streams[index % 2]):
            outputs.append(runner.sample(values, beam_search_args=actual, **kwargs())[0])
    torch.cuda.synchronize()
    for index, value in enumerate(outputs):
        assert (value == index + 1).all().item()
    runner.clear()


@pytest.mark.parametrize("budget", [{"max_graphs": 0}, {"max_static_input_bytes": 0}])
def test_budget_fallback_preserves_state(budget) -> None:
    runner = Runner(**budget)
    actual = state()
    reference = copy.deepcopy(actual)
    with runner.allow_capture():
        for _ in range(6):
            values = torch.randn(8, 32, device="cuda")
            equal(
                runner.sample(values, beam_search_args=actual, **kwargs()),
                NATIVE(values, beam_search_args=reference, **kwargs()),
            )
            equal(actual, reference)
    assert runner.capture_count == 0
    runner.clear()


def test_failed_replay_poisons_only_owner(monkeypatch) -> None:
    runner = Runner()
    actual = state()
    values = torch.ones(8, 32, device="cuda")
    prepare(runner, values, actual, kwargs())
    entry = next(iter(runner._entries.values()))

    def fail():
        raise RuntimeError("launch failed")

    entry.graph = NS(replay=fail)
    with pytest.raises(RuntimeError, match="launch failed"):
        runner.sample(values, beam_search_args=actual, **kwargs())
    runner.clear()
    with pytest.raises(RuntimeError, match="invalid"):
        runner.sample(values, beam_search_args=actual, **kwargs())
    other = Runner()
    other.sample(values, beam_search_args=state(), **kwargs())
    other.clear()


def test_capture_failure_restores_prepare_state(monkeypatch) -> None:
    sampler = make_sampler((Profile(beam_widths=(2, 4), batch_sizes=(1,)),))
    sampler.store.beam_search_store.ensure_cba()
    before = copy.deepcopy(sampler.store)
    finish = copy.deepcopy(sampler._finish_reasons_handler.store)

    def fail(*args):
        raise RuntimeError("capture unsupported")

    monkeypatch.setattr(Runner, "_capture", fail)
    sampler.prepare_cuda_graphs(vocab_size=32)
    equal(sampler.store, before)
    equal(sampler._finish_reasons_handler.store, finish)
    assert sampler._beam_search._graph_runner.capture_failures == 2
    sampler.release_cuda_graphs()
