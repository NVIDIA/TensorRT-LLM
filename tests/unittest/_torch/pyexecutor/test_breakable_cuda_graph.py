# Adapted from SGLang's breakable CUDA graph tests.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tensorrt_llm._torch.pyexecutor.breakable_cuda_graph.breakable_cuda_graph as bcg
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor import breakable_cuda_graph_runner as runner_module
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    break_graph,
    eager_on_graph,
    is_in_breakable_cuda_graph,
)
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph.breakable_cuda_graph import _copy_output
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import (
    BreakableCUDAGraphRunner,
    BreakableCUDAGraphRunnerState,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer
from tensorrt_llm._torch.pyexecutor.engine.runners.interface import ScheduledInputs
from tensorrt_llm._torch.pyexecutor.model_engine import ModelEngine
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.utils import (
    get_per_request_prefill_cuda_graph_flag,
    make_weak_ref,
    set_per_request_prefill_cuda_graph_flag,
)
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("failure_point", ["begin", "end"])
def test_capture_failure_unwinds_owner_context(monkeypatch, failure_point):
    events = []

    class OwnerContext:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, _exc_value, _traceback):
            events.append(("exit", exc_type))

    monkeypatch.setattr(bcg, "nccl_window_graph_owner", lambda _pool: OwnerContext())
    capture = BreakableCUDAGraphCapture(BreakableCUDAGraph(), stream=torch.cuda.Stream())

    def fail():
        raise RuntimeError(f"capture {failure_point} failed")

    if failure_point == "begin":
        monkeypatch.setattr(capture, "_begin_new_segment", fail)
    else:
        monkeypatch.setattr(capture, "_begin_new_segment", lambda: None)
        monkeypatch.setattr(capture, "_end_current_segment", fail)

    with pytest.raises(RuntimeError, match=f"capture {failure_point} failed"):
        with capture:
            assert events == ["enter"]

    assert events == ["enter", ("exit", RuntimeError)]
    assert capture._exit_stack is None
    assert bcg._current_capture.get() is None
    assert bcg._current_stream.get() is None
    assert bcg._forked_streams.get() is None


def _capture(body):
    graph = BreakableCUDAGraph()
    with BreakableCUDAGraphCapture(graph, stream=torch.cuda.Stream()):
        body()
    return graph


def test_no_break_capture_and_repeated_replay():
    x = torch.zeros(4, device="cuda")
    output = torch.zeros_like(x)
    graph = _capture(lambda: output.copy_(x + 1))

    assert graph.num_segments == 1
    assert graph.num_breaks == 0
    for value in (5, 11):
        x.fill_(value)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.full_like(output, value + 1))


def test_single_and_multiple_breakpoints():
    @eager_on_graph
    def add_one(value):
        return value + 1

    @eager_on_graph
    def double(value):
        return value * 2

    x = torch.zeros(4, device="cuda")
    output = torch.zeros_like(x)

    def body():
        value = add_one(x + 1)
        value = double(value + 1)
        output.copy_(value)

    graph = _capture(body)
    assert graph.num_segments == 3
    assert graph.num_breaks == 2

    x.fill_(5)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 16))


def test_eager_output_storage_survives_allocator_churn():
    eager_output_ptrs = []
    eager_output_refs = []

    @eager_on_graph
    def add_one(value):
        output = value + 1
        eager_output_ptrs.append(output.data_ptr())
        eager_output_refs.append(weakref.ref(output))
        return output

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.zeros(1024, device="cuda")
        output = torch.zeros_like(x)
        graph = BreakableCUDAGraph()

        with BreakableCUDAGraphCapture(graph, stream=stream):
            output.copy_(add_one(x) * 2)

        captured_output_ptr = eager_output_ptrs[0]
        assert eager_output_refs[0]() is not None
        churn = [torch.full_like(x, value) for value in range(4)]
        assert all(tensor.data_ptr() != captured_output_ptr for tensor in churn)

        x.fill_(5)
        graph.replay()

    stream.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 12))
    for value, tensor in enumerate(churn):
        torch.testing.assert_close(tensor, torch.full_like(tensor, value))


def test_outside_capture():
    @eager_on_graph
    def outside(value):
        return value + 2

    value = torch.tensor([1.0, 2.0], device="cuda")
    torch.testing.assert_close(outside(value), value + 2)


def test_eager_on_graph_during_torch_compile():
    @eager_on_graph
    def add_one(value):
        return value + 1

    compiled_add_one = torch.compile(add_one, backend="eager", fullgraph=True)
    value = torch.ones(4, device="cuda")

    torch.testing.assert_close(compiled_add_one(value), value + 1)


def test_make_weak_ref_supports_value_types_and_rejects_objects():
    unsupported = object()
    with pytest.raises(TypeError, match="Invalid type"):
        make_weak_ref(unsupported)

    value = {"nested": (None, "value", [1, 2.0, True])}
    assert make_weak_ref(value) == value

    with pytest.raises(TypeError, match="Invalid type"):
        make_weak_ref({unsupported: "value"})


def test_break_graph_inserts_empty_breakpoint():
    x = torch.zeros(4, device="cuda")
    output = torch.zeros_like(x)

    def body():
        value = x + 1
        break_graph()
        output.copy_(value + 2)

    graph = _capture(body)
    assert graph.num_segments == 2
    assert graph.num_breaks == 1
    x.fill_(10)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 13))


def test_graphed_break_captures_only_with_a_pool_and_replays_per_key():
    calls = {"eager": 0, "body": 0}
    keys = {"value": "a"}
    scale = torch.ones(4, device="cuda")

    def eager_scale(value):
        calls["eager"] += 1
        value.mul_(scale)

    def body_scale(value):
        calls["body"] += 1
        value.mul_(scale)

    scale_in_place = eager_on_graph(
        eager_scale, graph_key=lambda value: keys["value"], graph_body=body_scale
    )
    x = torch.zeros(4, device="cuda")
    output = torch.zeros_like(x)

    def body():
        value = x + 1
        scale_in_place(value)
        output.copy_(value + 1)

    graph = _capture(body)
    assert calls == {"eager": 1, "body": 0}

    def replay(value, factor, pool=None):
        x.fill_(value)
        scale.fill_(factor)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay(break_graph_pool=pool)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.full_like(output, (value + 1) * factor + 1))

    # Without a pool a key without a graph runs eagerly.
    replay(1, 2)
    assert calls == {"eager": 2, "body": 0} and graph.num_break_graphs == 0
    # With one, the key's body runs once eagerly, then is captured.
    replay(2, 3, torch.cuda.graph_pool_handle())
    assert calls == {"eager": 2, "body": 2} and graph.num_break_graphs == 1
    # Its graph replays the body; a new key or None runs eagerly.
    replay(3, 4)
    assert calls == {"eager": 2, "body": 2}
    for key in ("b", None):
        keys["value"] = key
        replay(4, 5)
    assert calls == {"eager": 4, "body": 2} and graph.num_break_graphs == 1
    graph.reset()
    assert graph.num_break_graphs == 0


def test_output_writeback_for_tensor_dict_and_object():
    class Output:
        def __init__(self, tensor, label):
            self.tensor = tensor
            self.label = label

    tensor = torch.zeros(4, device="cuda")
    assert _copy_output(tensor, torch.full_like(tensor, 3)) is tensor
    torch.testing.assert_close(tensor, torch.full_like(tensor, 3))

    output_dict = {"value": torch.zeros(4, device="cuda")}
    assert _copy_output(output_dict, {"value": torch.ones(4, device="cuda")}) is output_dict
    torch.testing.assert_close(output_dict["value"], torch.ones(4, device="cuda"))

    output_object = Output(torch.zeros(4, device="cuda"), "old")
    assert (
        _copy_output(output_object, Output(torch.full((4,), 2.0, device="cuda"), "new"))
        is output_object
    )
    torch.testing.assert_close(output_object.tensor, torch.full_like(output_object.tensor, 2))
    assert output_object.label == "new"


def test_side_stream_is_joined_before_segment_end():
    x = torch.ones(4, device="cuda")
    output = torch.zeros_like(x)
    side_stream = torch.cuda.Stream()

    def body():
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            output.copy_((x + 1) * 2)

    graph = _capture(body)
    x.fill_(3)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 8))


def test_dropped_wait_stream_logs_callsite(caplog):
    side_stream = torch.cuda.Stream()

    def body():
        torch.cuda.current_stream().wait_stream(side_stream)

    with caplog.at_level(
        logging.WARNING,
        logger=("tensorrt_llm._torch.pyexecutor.breakable_cuda_graph.breakable_cuda_graph"),
    ):
        _capture(body)

    record = next(record for record in caplog.records if "Dropping a wait" in record.message)
    assert record.funcName == "body"


class _Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, value):
        self.forward_calls += 1
        return value + 1


class _LogitsProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0

    def forward(self, value):
        self.forward_calls += 1
        return value * 2


@pytest.mark.parametrize("through_model_engine", [False, True])
def test_runner_warmup_capture_execute_and_shared_output(through_model_engine):
    body = _Body().cuda()
    logits_processor = _LogitsProcessor().cuda()
    runner = BreakableCUDAGraphRunner(body)
    counters = {"outer": 0}
    inputs = {}

    def model_forward():
        counters["outer"] += 1
        if runner.is_capturing:
            return runner.capture_model_body(
                lambda: {"logits": logits_processor(body(inputs["value"]))}
            )
        return {"logits": logits_processor(body(inputs["value"]))}

    if through_model_engine:
        batch = ScheduledRequests()
        resources = object()

        def decoder_forward(forward_inputs, resource_manager, *, is_dummy):
            assert isinstance(forward_inputs, ScheduledInputs)
            assert forward_inputs.batch is batch
            assert resource_manager is resources
            assert is_dummy
            return model_forward()

        decoder_runner = object.__new__(DecoderRunner)
        decoder_runner._forward_decoder = decoder_forward
        decoder_runner._model = SimpleNamespace(extra_attrs={})
        engine = object.__new__(ModelEngine)
        engine._runner = decoder_runner
        engine._warmup_depth = 1

    def engine_forward():
        if through_model_engine:
            return engine.forward(batch, resources)
        return model_forward()

    inputs["value"] = torch.zeros((8, 4), device="cuda")
    runner.capture(8, engine_forward)
    first_shared_output = runner._shared_output
    inputs["value"] = torch.zeros((4, 4), device="cuda")
    runner.capture(4, engine_forward)

    assert runner.state == BreakableCUDAGraphRunnerState.IDLE
    assert counters == {"outer": 6}
    assert body.forward_calls == 6
    assert logits_processor.forward_calls == 6
    assert runner._shared_output is first_shared_output

    original_forward = body.forward
    inputs["value"].fill_(3)
    result = runner.execute(4, engine_forward)
    torch.cuda.synchronize()
    torch.testing.assert_close(result["logits"], torch.full((4, 4), 8.0, device="cuda"))
    assert counters == {"outer": 7}
    assert body.forward_calls == 6
    assert logits_processor.forward_calls == 7
    assert body.forward == original_forward


def test_runner_first_bucket_segments_share_one_memory_pool():
    class BreakableBody(nn.Module):
        @staticmethod
        @eager_on_graph
        def eager_add_one(value):
            return value + 1

        @staticmethod
        @eager_on_graph
        def eager_double(value):
            return value * 2

        def forward(self, value):
            value = self.eager_add_one(value + 1)
            return self.eager_double(value + 1)

    body = BreakableBody().cuda()
    runner = BreakableCUDAGraphRunner(body)
    inputs = {"value": torch.zeros((8, 4), device="cuda")}

    def engine_forward():
        if runner.is_capturing:
            return runner.capture_model_body(lambda: body(inputs["value"]))
        return body(inputs["value"])

    runner.capture(8, engine_forward)

    graph = runner._graphs[8]
    assert graph.num_segments == 3
    assert not runner.has_graphable_breaks(8)
    assert runner._memory_pool is not None
    assert all(segment.pool() == runner._memory_pool for segment in graph._segments)


def test_runner_captures_break_graphs_only_in_capture_break_graphs():
    calls = {"eager": 0}

    def eager_add_one(value):
        calls["eager"] += 1
        return value + 1

    class GraphedBreakBody(nn.Module):
        add_one = staticmethod(eager_on_graph(eager_add_one, graph_key=lambda value: 0))

        def forward(self, value):
            return self.add_one(value * 2) * 3

    body = GraphedBreakBody().cuda()
    runner = BreakableCUDAGraphRunner(body)
    inputs = {"value": torch.zeros((8, 4), device="cuda")}

    def engine_forward():
        if runner.is_capturing:
            return runner.capture_model_body(lambda: body(inputs["value"]))
        if runner.has_graph(8):
            return runner.execute(8, lambda: body(inputs["value"]))
        return body(inputs["value"])

    runner.capture(8, engine_forward)
    assert runner.has_graphable_breaks(8) and not runner.has_graphable_breaks(4)
    eager_calls = calls["eager"]
    runner.execute(8, lambda: body(inputs["value"]))
    assert calls["eager"] == eager_calls + 1 and runner.num_break_graphs == 0

    runner.capture_break_graphs(engine_forward)
    assert runner.num_break_graphs == 1 and runner.state == BreakableCUDAGraphRunnerState.IDLE
    eager_calls = calls["eager"]
    inputs["value"].fill_(2)
    result = runner.execute(8, lambda: body(inputs["value"]))
    torch.cuda.synchronize()
    torch.testing.assert_close(result, torch.full((8, 4), 15.0, device="cuda"))
    assert calls["eager"] == eager_calls

    runner.clear()
    assert runner.num_break_graphs == 0 and runner._break_graph_pool is None


def test_runner_capture_promotes_window_releases_before_warmup(monkeypatch):
    # Windows released on another stream must be reusable from the capture stream on every
    # rank before warmup requests any, else best-fit diverges across ranks and deadlocks.
    runner = BreakableCUDAGraphRunner(_Body().cuda())
    events = []
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner."
        "promote_synchronized_nccl_window_releases",
        lambda: events.append("promote"),
    )
    monkeypatch.setattr(runner, "warmup", lambda *_args: events.append("warmup"))

    def engine_forward():
        raise RuntimeError("stop after warmup")

    with pytest.raises(RuntimeError, match="stop after warmup"):
        runner.capture(4, engine_forward)

    assert events == ["promote", "warmup"]


def test_runner_clear_releases_memory_pool_after_graph_reset(monkeypatch):
    runner = BreakableCUDAGraphRunner(_Body().cuda())
    pool = (12345, 67890)
    events = []

    class Graph:
        @staticmethod
        def reset():
            events.append("reset")

    runner._graphs[4] = Graph()
    runner._memory_pool = pool
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner."
        "release_nccl_window_graph_owner",
        lambda released_pool: events.append(("release", released_pool)),
    )

    runner.clear()

    assert events == ["reset", ("release", pool)]
    assert runner._memory_pool is None


def test_runner_graph_miss_nested_execute_and_exception_recovery():
    body = _Body().cuda()
    runner = BreakableCUDAGraphRunner(body)
    with pytest.raises(KeyError, match="No BCG captured"):
        runner.execute(4, lambda: None)

    runner._graphs[4] = object()
    runner._outputs[4] = torch.zeros(1, device="cuda")
    original_forward = body.forward

    def nested():
        return runner.execute(4, lambda: None)

    with pytest.raises(RuntimeError, match="while runner is replay"):
        runner.execute(4, nested)
    assert runner.state == BreakableCUDAGraphRunnerState.IDLE
    assert body.forward == original_forward

    def fail():
        raise ValueError("expected")

    with pytest.raises(ValueError, match="expected"):
        runner.execute(4, fail)
    assert runner.state == BreakableCUDAGraphRunnerState.IDLE
    assert body.forward == original_forward


def test_runner_refreshes_pre_capture_state_through_a_model_wrapper():
    """Replay matches eager when the body reads state computed outside capture (Engram hashes)."""

    class StatefulBody(nn.Module):
        def __init__(self):
            super().__init__()
            self.hashes = torch.zeros((8, 4), device="cuda")

        def refresh(self, value):
            self.hashes[: value.shape[0]].copy_(value * 10)

        def forward(self, value):
            if not torch.cuda.is_current_stream_capturing():
                self.refresh(value)
            return value + self.hashes[: value.shape[0]]

    body = StatefulBody().cuda()

    class CausalLM(DecoderModelForCausalLM):
        def prepare_breakable_cuda_graph_replay(self, inputs):
            self.model.refresh(inputs["value"])

        def forward(self, value):
            return self.model(value=value)

    causal_lm = object.__new__(CausalLM)
    nn.Module.__init__(causal_lm)
    causal_lm.model = body

    class Wrapper(nn.Module):
        """Exposes the causal LM only as ``llm``, like a multimodal model."""

        def __init__(self):
            super().__init__()
            self.llm = causal_lm

        def forward(self, value):
            return self.llm(value)

    model = Wrapper()
    decoder_runner = object.__new__(DecoderRunner)
    decoder_runner._model = model
    decoder_runner._config = SimpleNamespace(
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE,
        prefill_cuda_graph_num_tokens=[4],
    )
    runner = decoder_runner._initialize_breakable_cuda_graph_runner()
    staged = torch.zeros((4, 4), device="cuda")

    def engine_forward():
        if runner.is_capturing:
            return runner.capture_model_body(lambda: model(staged))
        return model(staged)

    runner.capture(4, engine_forward)
    for value in (2.0, 7.0):
        staged.fill_(value)
        replayed = runner.execute(4, engine_forward).clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(replayed, model(staged))


def test_runner_warmup_exception_restores_idle_state():
    runner = BreakableCUDAGraphRunner(_Body().cuda())

    def fail():
        raise ValueError("expected")

    with pytest.raises(ValueError, match="expected"):
        runner.warmup(fail)
    assert runner.state == BreakableCUDAGraphRunnerState.IDLE


def test_runner_flag_covers_the_whole_forward():
    """Code outside the captured body sees the BCG flag in warmup, capture and replay alike."""
    body = _Body().cuda()
    runner = BreakableCUDAGraphRunner(body)
    inputs = {"value": torch.zeros((4, 4), device="cuda")}
    seen = []

    def engine_forward():
        seen.append(is_in_breakable_cuda_graph())
        if runner.is_capturing:
            return runner.capture_model_body(lambda: body(inputs["value"]))
        return body(inputs["value"])

    runner.capture(4, engine_forward)
    runner.execute(4, engine_forward)
    torch.cuda.synchronize()
    assert len(seen) == BreakableCUDAGraphRunner._WARMUP_STEPS + 2 and all(seen)
    assert not is_in_breakable_cuda_graph()


def test_runner_smaller_bucket_output_does_not_alias_shared_output():
    """A later, smaller capture must not place its body output inside the shared output buffer,
    or the captured copy into the shared buffer would overwrite its own source."""
    rows, cols = 4096, 4096  # 64 MiB float32 output, above the allocator's small-segment limit
    output_spans = {}

    class WorkspaceBody(nn.Module):
        def forward(self, value):
            workspace = torch.empty(5 * value.numel() // 2, dtype=value.dtype, device=value.device)
            del workspace
            output = value * 2 + 1
            output_spans[value.shape[0]] = (output.data_ptr(), output.nbytes)
            return output

    static_input = torch.randn(rows, cols, device="cuda")
    body = WorkspaceBody().cuda()
    runner = BreakableCUDAGraphRunner(body)
    active = {"rows": rows}

    def engine_forward():
        value = static_input[: active["rows"]]
        if runner.is_capturing:
            return runner.capture_model_body(lambda: body(value))
        return body(value)

    buckets = (rows, 3 * rows // 4)
    for num_tokens in buckets:
        active["rows"] = num_tokens
        runner.capture(num_tokens, engine_forward)
    assert runner.captured_num_tokens == sorted(buckets)

    shared_begin = runner._shared_output.data_ptr()
    shared_end = shared_begin + runner._shared_output.nbytes
    output_begin, output_bytes = output_spans[buckets[1]]
    assert output_begin == shared_begin or not (
        output_begin < shared_end and shared_begin < output_begin + output_bytes
    )

    for num_tokens in reversed(buckets):
        active["rows"] = num_tokens
        result = runner.execute(num_tokens, engine_forward)
        torch.cuda.synchronize()
        torch.testing.assert_close(result, static_input[:num_tokens] * 2 + 1, rtol=0, atol=0)


def _make_padding_preparer(breakable_runner):
    preparer = object.__new__(DecoderInputPreparer)
    preparer._config = SimpleNamespace(
        enable_attention_dp=False,
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE,
        prefill_cuda_graph_num_tokens=[4, 8, 16],
        use_mrope=False,
    )
    preparer.mapping = None
    preparer.dist = None
    preparer._buffers = SimpleNamespace(
        input_ids_cuda=torch.ones(32, dtype=torch.int, device="cuda"),
        position_ids_cuda=torch.ones(32, dtype=torch.int, device="cuda"),
    )
    preparer.breakable_cuda_graph_runner = breakable_runner
    return preparer


@pytest.mark.parametrize(
    "state, captured, num_tokens, expected_tokens",
    [
        # Idle: pad only to a captured bucket; 8 was not captured.
        ("idle", [4, 16], 5, 16),
        ("idle", [4, 16], 3, 4),
        # Idle without any captured bucket: no padding, eager forward.
        ("idle", [], 5, 5),
        # Capture and warmup pad to the configured bucket being recorded.
        ("capturing", [], 5, 8),
        ("warming_up", [16], 5, 8),
        # Without a breakable runner: the configured buckets.
        (None, None, 5, 8),
    ],
)
def test_prefill_padding_targets_replayable_buckets(state, captured, num_tokens, expected_tokens):
    runner = state and SimpleNamespace(
        is_capturing=state == "capturing",
        is_warming_up=state == "warming_up",
        captured_num_tokens=captured,
    )
    preparer = _make_padding_preparer(runner)
    attn_metadata = SimpleNamespace(padded_num_tokens=None)
    try:
        padded = preparer.pad_tokens(attn_metadata, num_tokens, num_contexts=1)
        assert padded == expected_tokens
        assert get_per_request_prefill_cuda_graph_flag() == (expected_tokens != num_tokens)
        expected_padded = expected_tokens if expected_tokens != num_tokens else None
        assert attn_metadata.padded_num_tokens == expected_padded
        assert torch.all(preparer._buffers.input_ids_cuda[num_tokens:expected_tokens] == 0)
    finally:
        set_per_request_prefill_cuda_graph_flag(False)


class _PositionBody(nn.Module):
    """Decoder body stand-in whose output reads the (mrope) position ids."""

    def forward(self, input_ids, attn_metadata, position_ids, inputs_embeds=None, **kwargs):
        return position_ids.reshape(3, -1).t().float() * 2.0


class _EchoWorker:
    """Spec worker stand-in returning the target hidden states."""

    reads_lm_head_dtype_logits = False

    def takes_vocab_sharded_target_logits(self, *args):
        return False

    def target_logits(self, hidden_states, *args):
        return hidden_states

    def __call__(self, **kwargs):
        return kwargs["hidden_states"].clone()


def test_padded_context_replay_with_spec_worker_matches_eager():
    """A padded context batch's positions are a strided view of the engine buffer. A hoisted
    ``.contiguous()`` copy (the pre-fix forward) is a fresh tensor the replayed body never reads: it
    reads the freed capture-time copy (capture-stream pool), so the replay returns stale positions."""
    body = _PositionBody().cuda()
    model = object.__new__(SpecDecOneEngineForCausalLM)
    nn.Module.__init__(model)
    model.model = body
    model.spec_worker = _EchoWorker()
    model.spec_decode_fast_paths = True
    model.layer_idx = -1
    model.lm_head = model.logits_processor = model.draft_model = None
    runner = BreakableCUDAGraphRunner(body)
    buffer = torch.zeros((3, 1, 32), dtype=torch.int32, device="cuda")
    padded = buffer[..., :16]
    assert not padded.is_contiguous()
    attn_metadata = SimpleNamespace(num_contexts=1, padded_num_tokens=16, num_tokens=12)
    spec_metadata = SimpleNamespace(
        gather_ids=torch.arange(12, device="cuda"), is_layer_capture=lambda layer_idx: False
    )

    def forward():
        return model.forward(attn_metadata, position_ids=padded, spec_metadata=spec_metadata)

    def engine_forward():
        if runner.is_capturing:
            return runner.capture_model_body(forward)
        return forward()

    buffer.copy_(torch.arange(96, dtype=torch.int32, device="cuda").view(3, 1, 32))
    runner.capture(16, engine_forward)
    buffer.copy_(torch.arange(1000, 1096, dtype=torch.int32, device="cuda").view(3, 1, 32))
    expected = forward()
    replayed = runner.execute(16, forward)
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed, expected)


class _PreparedBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0
        self.fail_capture = lambda: False

    def forward(self, value, config, optional=None, scale=1, owner=None, **kwargs):
        self.forward_calls += 1
        if self.fail_capture():
            raise ValueError("prepared capture failed")
        output = value + config["offset"][0]
        for feature in config["features"]:
            output = output + feature
        if optional is not None:
            output = output + optional
        if owner is not None:
            output = output + owner.offset
        if kwargs.get("host_scale") is not None:
            output = output * kwargs["host_scale"].item()
        return (output + kwargs.get("extra", 0)) * scale


def _prepared_runner(agree_replay=None):
    body = _PreparedBody().cuda()
    runner = BreakableCUDAGraphRunner(
        body, prepare_inputs=lambda inputs: inputs, agree_replay=agree_replay
    )
    inputs = {
        "value": torch.zeros((4, 4), device="cuda"),
        "config": {
            "features": [torch.ones((4, 4), device="cuda")],
            "offset": (torch.full((4, 4), 2.0, device="cuda"),),
        },
        "owner": None,
        "extra": torch.full((4, 4), 3.0, device="cuda"),
        "host_scale": torch.tensor(1.0),
    }

    def engine_forward():
        if runner.is_capturing:
            return runner.capture_model_body(lambda: body(**inputs))
        return body(**inputs)

    return body, runner, inputs, engine_forward


def test_runner_prepared_inputs_refresh_owned_nested_storage(monkeypatch):
    body, runner, inputs, engine_forward = _prepared_runner()
    assert runner.stages_inputs
    source_ref = weakref.ref(inputs["value"])
    source_ptr = inputs["value"].data_ptr()
    runner.capture(4, engine_forward)
    captured = runner._inputs[4]
    captured_ptr = captured["value"].data_ptr()
    assert captured_ptr != source_ptr
    assert captured["config"]["features"][0] is not inputs["config"]["features"][0]
    assert captured["config"]["offset"][0] is not inputs["config"]["offset"][0]

    replay_calls = []
    original_replay = runner.replay

    def replay(num_tokens):
        replay_calls.append(num_tokens)
        return original_replay(num_tokens)

    monkeypatch.setattr(runner, "replay", replay)
    for value in (5, 11):
        inputs["value"] = torch.full((4, 4), float(value), device="cuda")
        inputs["config"] = {
            "features": [torch.full((4, 4), float(value + 1), device="cuda")],
            "offset": (torch.full((4, 4), float(value + 2), device="cuda"),),
        }
        inputs["extra"] = torch.full((4, 4), float(value + 3), device="cuda")
        assert source_ref() is None
        churn = [torch.full_like(inputs["value"], index) for index in range(8)]
        assert all(tensor.data_ptr() != captured_ptr for tensor in churn)
        expected = body(**inputs)
        eager_calls = body.forward_calls

        actual = runner.execute(4, engine_forward)
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert body.forward_calls == eager_calls
        assert captured["value"].data_ptr() == captured_ptr
        for index, tensor in enumerate(churn):
            torch.testing.assert_close(tensor, torch.full_like(tensor, index))
    assert replay_calls == [4, 4]


def test_runner_prepared_inputs_share_the_largest_bucket_storage(monkeypatch):
    """A smaller bucket stages its token-major device inputs in the leading rows of the
    largest bucket's; alternate replays still read only the current step's rows."""
    body = _PreparedBody().cuda()
    runner = BreakableCUDAGraphRunner(body, prepare_inputs=lambda inputs: inputs)
    inputs = {}

    def set_inputs(num_tokens, value):
        rows = torch.arange(num_tokens, device="cuda", dtype=torch.float32)[:, None]

        def full(offset):
            return torch.full((num_tokens, 4), float(value + offset), device="cuda") + rows

        inputs.update(value=full(0), config={"features": [full(1)], "offset": (full(2),)})
        # A host tensor and a tensor whose other dimensions change with the bucket.
        inputs.update(extra=full(3), host_scale=torch.tensor(1.0))
        inputs.update(positions=torch.zeros((3, 1, num_tokens), device="cuda"))

    def engine_forward():
        if runner.is_capturing:
            return runner.capture_model_body(lambda: body(**inputs))
        return body(**inputs)

    for num_tokens in (8, 4):
        set_inputs(num_tokens, 0)
        runner.capture(num_tokens, engine_forward)
    largest, smaller = runner._inputs[8], runner._inputs[4]
    for name in ("value", "extra"):
        assert smaller[name].data_ptr() == largest[name].data_ptr()
        assert smaller[name].shape == (4, 4)
    assert (
        smaller["config"]["features"][0].data_ptr() == largest["config"]["features"][0].data_ptr()
    )
    assert smaller["config"]["offset"][0].data_ptr() == largest["config"]["offset"][0].data_ptr()
    for name in ("host_scale", "positions"):
        assert smaller[name].untyped_storage().data_ptr() != (
            largest[name].untyped_storage().data_ptr()
        )
    calls_after_capture = body.forward_calls
    replay_calls = []
    original_replay = runner.replay

    def replay(num_tokens):
        replay_calls.append(num_tokens)
        return original_replay(num_tokens)

    monkeypatch.setattr(runner, "replay", replay)
    steps = [(8, 5), (4, 11), (8, 17), (4, 23)]
    for num_tokens, value in steps:
        set_inputs(num_tokens, value)
        actual = runner.execute(num_tokens, engine_forward)
        torch.cuda.synchronize()

        # value + offset + feature + extra, each with its row index added.
        rows = torch.arange(num_tokens, device="cuda", dtype=torch.float32)[:, None]
        expected = torch.full((num_tokens, 4), 4.0 * value + 6, device="cuda") + 4 * rows
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert body.forward_calls == calls_after_capture
    assert replay_calls == [num_tokens for num_tokens, _ in steps]


def test_stage_inputs_share_only_views_with_the_layout_of_a_clone():
    """A smaller bucket views the largest bucket's input only when both have the same
    rank and are contiguous; a scalar or a transposed largest input is cloned instead."""
    largest = runner_module._stage_inputs(
        {
            "rows": torch.zeros((8, 4), device="cuda"),
            "scalar": torch.zeros((), device="cuda"),
            "transposed": torch.zeros((4, 8), device="cuda").t(),
        },
        None,
    )
    inputs = {
        "rows": torch.ones((4, 4), device="cuda"),
        "scalar": torch.ones((4,), device="cuda"),
        "transposed": torch.ones((4, 4), device="cuda"),
    }
    staged = runner_module._stage_inputs(inputs, largest)

    assert staged["rows"].data_ptr() == largest["rows"].data_ptr()
    for name in ("scalar", "transposed"):
        assert (
            staged[name].untyped_storage().data_ptr() != largest[name].untyped_storage().data_ptr()
        )
        assert staged[name].stride() == inputs[name].stride()
    for name, value in inputs.items():
        torch.testing.assert_close(staged[name], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "change, mismatch",
    [
        ("shape", "inputs['config']['features'][0]"),
        ("dtype", "inputs['value']"),
        ("optional", "inputs['optional']"),
        ("list_length", "inputs['config']['features']"),
        ("list_type", "inputs['config']['features']"),
        ("scalar", "inputs['scale']"),
        ("owner", "inputs['owner']"),
    ],
)
def test_runner_prepared_inputs_incompatible_values_fall_back_eager(monkeypatch, change, mismatch):
    body, runner, inputs, engine_forward = _prepared_runner()
    runner.capture(4, engine_forward)
    logged = []
    monkeypatch.setattr(
        runner_module.logger, "info_once", lambda message, key: logged.append((message, key))
    )
    original_forward = body.forward
    if change == "shape":
        inputs["config"]["features"][0] = torch.full((1, 4), 5.0, device="cuda")
    elif change == "dtype":
        inputs["value"] = inputs["value"].double()
    elif change == "optional":
        inputs["optional"] = torch.full_like(inputs["value"], 5)
    elif change == "list_length":
        inputs["config"]["features"].append(torch.full_like(inputs["value"], 5))
    elif change == "list_type":
        inputs["config"]["features"] = tuple(inputs["config"]["features"])
    elif change == "scalar":
        inputs["scale"] = 2
    else:
        inputs["owner"] = SimpleNamespace(offset=7.0)
    expected = body(**inputs)
    eager_calls = body.forward_calls

    def unexpected_replay(_num_tokens):
        pytest.fail("incompatible prepared inputs must not replay")

    monkeypatch.setattr(runner, "replay", unexpected_replay)
    for step in (1, 2):
        actual = runner.execute(4, engine_forward)
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert body.forward_calls == eager_calls + step
        assert runner.num_eager_fallbacks == step
    assert body.forward == original_forward
    assert runner.state == BreakableCUDAGraphRunnerState.IDLE
    # One info-level key per bucket (the logger prints a key once), naming the input.
    assert [key for _, key in logged] == ["breakable_prefill_eager_fallback_4"] * 2
    assert f"{mismatch} differs from the capture" in logged[0][0]


@pytest.mark.parametrize("late_input", ["extra", "optional"])
@pytest.mark.parametrize("matches", [(True, True), (True, False), (False, True)])
def test_runner_prepared_inputs_attention_dp_ranks_agree_before_copying(
    monkeypatch, matches, late_input
):
    """Two attention DP ranks share their staged-input checks: both replay or both run eagerly.

    The mismatching rank changes a late input, ``extra``'s shape or ``optional``, which
    was None at capture, so a rank that copied while it checked would already have
    written ``value`` into the graph's inputs.
    """
    ranks = []
    for rank, rank_matches in enumerate(matches):
        local_answers = []

        def agree_replay(replayable, rank=rank, local_answers=local_answers):
            local_answers.append(replayable)
            # Stand-in for the all-gather: the other rank's answer is its scripted check.
            return all(
                replayable if peer == rank else peer_matches
                for peer, peer_matches in enumerate(matches)
            )

        body, runner, inputs, engine_forward = _prepared_runner(agree_replay)
        runner.capture(4, engine_forward)
        ranks.append((body, runner, inputs, engine_forward, local_answers, rank_matches))

    for body, runner, inputs, engine_forward, local_answers, rank_matches in ranks:
        captured = runner._inputs[4]
        inputs["value"] = torch.full((4, 4), 5.0, device="cuda")
        if not rank_matches and late_input == "extra":
            inputs["extra"] = torch.full((1, 4), 3.0, device="cuda")
        elif not rank_matches:
            inputs["optional"] = torch.ones((4, 4), device="cuda")
        expected = body(**inputs)
        eager_calls = body.forward_calls
        replay_calls = []
        original_replay = runner.replay

        def replay(num_tokens, original_replay=original_replay, replay_calls=replay_calls):
            replay_calls.append(num_tokens)
            return original_replay(num_tokens)

        monkeypatch.setattr(runner, "replay", replay)
        logged = []
        monkeypatch.setattr(
            runner_module.logger, "info_once", lambda message, key: logged.append(message)
        )
        actual = runner.execute(4, engine_forward)
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert local_answers == [rank_matches]
        if all(matches):
            assert replay_calls == [4]
            assert body.forward_calls == eager_calls
            assert runner.num_eager_fallbacks == 0 and logged == []
            torch.testing.assert_close(captured["value"], inputs["value"], rtol=0, atol=0)
        else:
            assert replay_calls == []
            assert body.forward_calls == eager_calls + 1
            assert runner.num_eager_fallbacks == 1
            reason = (
                "another attention DP rank's inputs differ"
                if rank_matches
                else f"inputs['{late_input}'] differs"
            )
            assert len(logged) == 1 and reason in logged[0]
            # Nothing was copied into the graph's inputs before the ranks agreed.
            torch.testing.assert_close(
                captured["value"], torch.zeros_like(captured["value"]), rtol=0, atol=0
            )
        assert runner.state == BreakableCUDAGraphRunnerState.IDLE


@pytest.mark.parametrize("stages_inputs", [False, True])
@pytest.mark.parametrize("enable_attention_dp, tp_size", [(False, 2), (True, 1), (True, 2)])
@pytest.mark.parametrize("peer_replayable", [True, False])
def test_decoder_runner_agrees_staged_replay_only_under_attention_dp(
    stages_inputs, enable_attention_dp, tp_size, peer_replayable
):
    """Under attention DP with more than one TP rank, a runner that stages inputs shares
    each rank's check with the same TP all-gather as the prefill graph vote. Otherwise
    there is no added all-gather, as in the decode graph agreement."""
    import numpy as np
    from _torch.helpers import make_breakable_prefill_runner, make_stand_in_multimodal_model

    from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4ForCausalLM
    from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4ForConditionalGeneration

    owner = make_stand_in_multimodal_model(
        Gemma4ForConditionalGeneration, Gemma4ForCausalLM, _PreparedBody()
    )
    sent = []

    def tp_allgather_int64(values):
        # Rank 0 is this rank; rank 1 answers peer_replayable.
        sent.append(list(values))
        return np.asarray([list(values), [int(peer_replayable)]], dtype=np.int64)

    runner = make_breakable_prefill_runner(
        # The text decoder alone keeps the unstaged text body.
        owner if stages_inputs else owner.llm,
        4,
        enable_attention_dp=enable_attention_dp,
        tp_size=tp_size,
        dist=SimpleNamespace(tp_allgather_int64=tp_allgather_int64),
    )
    assert runner.stages_inputs is stages_inputs
    if not (stages_inputs and enable_attention_dp and tp_size > 1):
        assert runner._agree_replay is None
        return
    assert runner._agree_replay(True) is peer_replayable
    assert runner._agree_replay(False) is False
    assert sent == [[1], [0]]


@pytest.mark.parametrize("source_kind", ["mutable_owner", "cpu_tensor"])
def test_runner_prepared_inputs_mutable_host_values_fall_back_eager(monkeypatch, source_kind):
    body, runner, inputs, engine_forward = _prepared_runner()
    if source_kind == "mutable_owner":
        inputs["owner"] = SimpleNamespace(offset=0.5)
    else:
        inputs["host_scale"] = torch.tensor(1.0)
    runner.capture(4, engine_forward)
    if source_kind == "mutable_owner":
        inputs["owner"].offset = 7.0
    else:
        inputs["host_scale"].fill_(7)
    expected = body(**inputs)
    eager_calls = body.forward_calls

    def unexpected_replay(_num_tokens):
        pytest.fail("changed mutable host values must not replay captured Python decisions")

    monkeypatch.setattr(runner, "replay", unexpected_replay)
    actual = runner.execute(4, engine_forward)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert body.forward_calls == eager_calls + 1


def test_runner_prepared_warmup_normalizes_positional_inputs_under_the_flag():
    body = _PreparedBody().cuda()
    normalization_calls = []

    def prepare_inputs(inputs):
        normalization_calls.append(
            (inputs["optional"], inputs["scale"], is_in_breakable_cuda_graph())
        )
        inputs["value"] = inputs["value"] + 2
        return inputs

    runner = BreakableCUDAGraphRunner(body, prepare_inputs=prepare_inputs)
    original_forward = body.forward
    value = torch.zeros((4, 4), device="cuda")
    config = {"offset": (torch.zeros_like(value),), "features": []}
    outputs = []

    runner.warmup(lambda: outputs.append(body(value, config)), steps=2)

    assert normalization_calls == [(None, 1, True), (None, 1, True)]
    assert body.forward_calls == 2
    assert body.forward == original_forward
    assert runner.state == BreakableCUDAGraphRunnerState.IDLE
    for output in outputs:
        torch.testing.assert_close(output, torch.full_like(value, 2), rtol=0, atol=0)


def test_runner_prepared_input_storage_is_released_on_clear_and_capture_failure():
    body, runner, _inputs, engine_forward = _prepared_runner()
    runner.capture(4, engine_forward)
    staged_ref = weakref.ref(runner._inputs[4]["value"])
    runner.clear()
    assert runner._inputs == {}
    assert staged_ref() is None

    original_forward = body.forward
    body.fail_capture = lambda: runner.is_capturing
    with pytest.raises(ValueError, match="prepared capture failed"):
        runner.capture(4, engine_forward)

    assert runner._inputs == {}
    assert not runner.has_graph(4)
    assert runner._memory_pool is None
    assert body.forward == original_forward
    assert runner.state == BreakableCUDAGraphRunnerState.IDLE


def test_runner_prepared_inputs_follow_the_whole_forward_flag():
    """Staged runners keep main's flag scope: outer code sees the flag in all three phases."""
    body = _Body().cuda()
    runner = BreakableCUDAGraphRunner(body, prepare_inputs=lambda inputs: inputs)
    inputs = {"value": torch.zeros((4, 4), device="cuda")}
    seen = []

    def engine_forward():
        seen.append(is_in_breakable_cuda_graph())
        if runner.is_capturing:
            return runner.capture_model_body(lambda: body(**inputs))
        return body(**inputs)

    runner.capture(4, engine_forward)
    inputs["value"] = torch.full((4, 4), 5.0, device="cuda")
    replayed = runner.execute(4, engine_forward)
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed, torch.full((4, 4), 6.0, device="cuda"), rtol=0, atol=0)
    assert len(seen) == BreakableCUDAGraphRunner._WARMUP_STEPS + 2 and all(seen)
    assert not is_in_breakable_cuda_graph()


def test_runner_prepared_body_replays_after_an_encoder_cuda_graph():
    """An encoder graph replayed in the outer forward feeds the staged body replay."""
    encoder_input = torch.zeros((4, 4), device="cuda")
    encoder_graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        encoder_output = encoder_input * 3  # warm up outside capture
        with torch.cuda.graph(encoder_graph, stream=stream):
            encoder_output = encoder_input * 3
    torch.cuda.current_stream().wait_stream(stream)
    body = _Body().cuda()
    runner = BreakableCUDAGraphRunner(body, prepare_inputs=lambda inputs: inputs)

    def outer():
        encoder_graph.replay()
        # Fusion makes a fresh tensor every step, as fuse_input_embeds does.
        return body(value=encoder_output + 0)

    def engine_forward():
        return runner.capture_model_body(outer) if runner.is_capturing else outer()

    runner.capture(4, engine_forward)
    for value in (2.0, 7.0):
        encoder_input.fill_(value)
        replayed = runner.execute(4, engine_forward)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            replayed, torch.full_like(replayed, 3 * value + 1), rtol=0, atol=0
        )
    encoder_graph.reset()
    runner.clear()


@torch.inference_mode()
def test_gemma4_opted_in_decoder_captures_and_refreshes_ple_ids():
    from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4ForCausalLM
    from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4ForConditionalGeneration

    class Body(nn.Module):
        hidden_size_per_layer_input = 1

        def __init__(self):
            super().__init__()
            self.forward_calls = 0

        def forward(self, input_ids=None, inputs_embeds=None, ple_input_ids=None):
            self.forward_calls += 1
            return inputs_embeds + ple_input_ids[:, None]

    from _torch.helpers import (
        check_stand_in_body_replay,
        make_breakable_prefill_runner,
        make_stand_in_multimodal_model,
    )

    body = Body()
    owner = make_stand_in_multimodal_model(Gemma4ForConditionalGeneration, Gemma4ForCausalLM, body)
    runner = make_breakable_prefill_runner(owner, 4)
    assert runner.stages_inputs

    def multimodal_step(inputs):
        inputs["input_ids"] = torch.arange(4, device="cuda")
        inputs["inputs_embeds"] = torch.full((4, 4), 3.0, device="cuda")
        return inputs["inputs_embeds"] + inputs["input_ids"][:, None]

    inputs = {
        "input_ids": torch.zeros(4, dtype=torch.long, device="cuda"),
        "inputs_embeds": torch.zeros((4, 4), device="cuda"),
    }
    check_stand_in_body_replay(runner, body, inputs, [multimodal_step], num_tokens=4)
