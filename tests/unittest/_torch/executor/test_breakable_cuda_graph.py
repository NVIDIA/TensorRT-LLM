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
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
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
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import DecoderCudaGraphs
from tensorrt_llm._torch.pyexecutor.engine.runners.interface import ScheduledInputs
from tensorrt_llm._torch.pyexecutor.model_engine import ModelEngine
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.utils import make_weak_ref
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
    assert runner._memory_pool is not None
    assert all(segment.pool() == runner._memory_pool for segment in graph._segments)


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


def test_runner_warmup_exception_restores_idle_state():
    runner = BreakableCUDAGraphRunner(_Body().cuda())

    def fail():
        raise ValueError("expected")

    with pytest.raises(ValueError, match="expected"):
        runner.warmup(fail)
    assert runner.state == BreakableCUDAGraphRunnerState.IDLE


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


def _prepared_runner():
    body = _PreparedBody().cuda()
    runner = BreakableCUDAGraphRunner(body, prepare_inputs=lambda inputs: inputs)
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


@pytest.mark.parametrize(
    "change", ["shape", "dtype", "optional", "list_length", "list_type", "scalar", "owner"]
)
def test_runner_prepared_inputs_incompatible_values_fall_back_eager(monkeypatch, change):
    body, runner, inputs, engine_forward = _prepared_runner()
    runner.capture(4, engine_forward)
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
    actual = runner.execute(4, engine_forward)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert body.forward_calls == eager_calls + 1
    assert body.forward == original_forward
    assert runner.state == BreakableCUDAGraphRunnerState.IDLE


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


def test_runner_warmup_normalizes_positional_inputs_and_defaults():
    body = _PreparedBody().cuda()
    normalization_calls = []

    def prepare_inputs(inputs):
        normalization_calls.append((inputs["optional"], inputs["scale"]))
        inputs["value"] = inputs["value"] + 2
        return inputs

    runner = BreakableCUDAGraphRunner(body, prepare_inputs=prepare_inputs)
    original_forward = body.forward
    value = torch.zeros((4, 4), device="cuda")
    config = {"offset": (torch.zeros_like(value),), "features": []}
    outputs = []

    runner.warmup(lambda: outputs.append(body(value, config)), steps=2)

    assert normalization_calls == [(None, 1), (None, 1)]
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


class _MultimodalOwner(MultimodalModelMixin):
    def __init__(self, decoder, disable_mm_encoder=False):
        self._decoder = decoder
        self.model_config = SimpleNamespace(disable_mm_encoder=disable_mm_encoder)
        self.mm_encoder = None if disable_mm_encoder else object()

    @property
    def language_model(self):
        return self._decoder


class _SupportedMultimodalDecoder(DecoderModelForCausalLM):
    supports_multimodal_breakable_cuda_graph = True


def _decoder_stub(supported=True):
    decoder = object.__new__(_SupportedMultimodalDecoder if supported else DecoderModelForCausalLM)
    nn.Module.__init__(decoder)
    decoder.model = _Body()
    return decoder


@pytest.mark.parametrize("owner_kind", ["property", "text", "handoff", "llm_only"])
def test_decoder_runner_resolves_breakable_graph_body(owner_kind):
    decoder = _decoder_stub(supported=owner_kind != "text")
    owner = (
        decoder
        if owner_kind == "text"
        else _MultimodalOwner(decoder, disable_mm_encoder=owner_kind in ("handoff", "llm_only"))
    )
    if owner_kind == "llm_only":
        owner.llm = decoder
        engine = object.__new__(ModelEngine)
        engine.model = owner
        engine.llm_args = SimpleNamespace(
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE,
            disable_mm_encoder=True,
        )
        engine.mapping = SimpleNamespace(pp_size=2)
        engine._validate_breakable_cuda_graph_compatibility()
    decoder_runner = object.__new__(DecoderRunner)
    decoder_runner._model = owner
    decoder_runner._mapping = SimpleNamespace(pp_size=2 if owner_kind == "llm_only" else 1)
    decoder_runner._config = SimpleNamespace(
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE
    )

    runner = decoder_runner._initialize_breakable_cuda_graph_runner()

    assert runner.layer_model is decoder.model
    if owner_kind in ("property", "handoff"):
        assert not hasattr(owner, "llm")
        assert runner._prepare_inputs == owner.prepare_breakable_cuda_graph_inputs
    else:
        assert runner._prepare_inputs is None
        assert runner._input_signature is None
    assert runner._inputs == {}


@pytest.mark.parametrize(
    ("decoder_body", "pp_size", "error"),
    [(False, 1, "DecoderModelForCausalLM language_model"), (True, 2, "pipeline parallelism")],
)
def test_multimodal_breakable_graph_guard_explains_unsupported_case(decoder_body, pp_size, error):
    engine = object.__new__(ModelEngine)
    engine.llm_args = SimpleNamespace(
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE,
        disable_mm_encoder=False,
    )
    engine.model = _MultimodalOwner(_decoder_stub() if decoder_body else object())
    engine.mapping = SimpleNamespace(pp_size=pp_size)

    with pytest.raises(ValueError, match=error):
        engine._validate_breakable_cuda_graph_compatibility()


def _multimodal_prefill(owner, run_model=None):
    decoder_runner = object.__new__(DecoderRunner)
    decoder_runner._model = owner
    decoder_runner._mapping = SimpleNamespace(pp_size=1)
    decoder_runner._config = SimpleNamespace(
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE,
        disable_mm_encoder=False,
    )
    graphs = object.__new__(DecoderCudaGraphs)
    graphs.breakable_cuda_graph_runner = decoder_runner._initialize_breakable_cuda_graph_runner()

    def forward(inputs):
        return graphs._run(
            None,
            inputs,
            run_model or (lambda current: owner.language_model.model(**current)),
            lambda _current: None,
            attn_metadata=inputs.get("attn_metadata"),
            draft_kv_cache_manager=None,
            moe_load_balancer=None,
            enable_spec_decode=False,
            gather_context_logits=False,
            position_id_offsets=None,
        )

    return graphs.breakable_cuda_graph_runner, forward


@pytest.mark.parametrize("disable_mm_encoder", [False, True])
def test_glm_multimodal_changed_host_schedule_stays_eager(disable_mm_encoder):
    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
    from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import Glm5NextMamba2Metadata
    from tensorrt_llm._torch.models.modeling_glm5_next import Glm5NextForCausalLM, Glm5NextModel

    class ObserveContext(nn.Module):
        def forward(self, *, hidden_states, runtime_ctx, **_kwargs):
            scalar = 100 * runtime_ctx.num_contexts + runtime_ctx.num_ctx_tokens
            return hidden_states + scalar + runtime_ctx.cached_lens[0]

    metadata = object.__new__(AttentionMetadata)
    metadata.kv_cache_manager = object()
    metadata._seq_lens = torch.tensor([4], dtype=torch.int32)
    metadata._num_contexts = 1
    metadata._num_ctx_tokens = metadata._num_tokens = 4
    metadata.kv_lens_cuda = torch.tensor([4, 0], dtype=torch.int32, device="cuda")
    mamba = object.__new__(Glm5NextMamba2Metadata)
    mamba.glm_block_tables = torch.zeros((2, 1), dtype=torch.long, device="cuda")
    mamba.glm_ctx_cu_seqlens = [0, 4]
    mamba.glm_cached_lens_host = [0]
    metadata.mamba_metadata = mamba
    body = object.__new__(Glm5NextModel)
    nn.Module.__init__(body)
    body.schedule = SimpleNamespace(num_layers=1)
    body.layers = nn.ModuleList([ObserveContext()])
    body._fused_hc_ok = lambda: False
    body.expand_streams = body.collapse_streams = lambda value: value
    decoder = object.__new__(Glm5NextForCausalLM)
    nn.Module.__init__(decoder)
    decoder.model = body
    owner = _MultimodalOwner(decoder, disable_mm_encoder=disable_mm_encoder)
    owner.llm = decoder
    runner, forward = _multimodal_prefill(
        owner,
        lambda current: body(
            attn_metadata=current["attn_metadata"], inputs_embeds=current["inputs_embeds"]
        ),
    )
    inputs = {
        "input_ids": torch.zeros(4, dtype=torch.long, device="cuda"),
        "attn_metadata": metadata,
        "inputs_embeds": torch.ones((4, 4), device="cuda"),
    }

    assert runner is None
    torch.testing.assert_close(forward(inputs), torch.full_like(inputs["inputs_embeds"], 105))
    mamba.glm_cached_lens_host = [7]
    metadata.kv_lens_cuda.copy_(torch.tensor([11, 0], dtype=torch.int32, device="cuda"))
    torch.testing.assert_close(forward(inputs), torch.full_like(inputs["inputs_embeds"], 112))
    metadata._seq_lens = torch.tensor([3, 1], dtype=torch.int32)
    metadata._num_ctx_tokens = 3
    metadata.kv_lens_cuda.copy_(torch.tensor([10, 1], dtype=torch.int32, device="cuda"))
    mamba.glm_ctx_cu_seqlens = [0, 3]
    mamba.glm_cached_lens_host = [7, 0]

    torch.testing.assert_close(forward(inputs), torch.full_like(inputs["inputs_embeds"], 111))
    assert inputs["attn_metadata"] is metadata
    assert inputs["input_ids"].numel() == 4


@pytest.mark.parametrize("disable_mm_encoder", [False, True])
def test_deepseek_multimodal_changed_ids_and_image_mask_stay_eager(disable_mm_encoder):
    from _torch.modules.test_engram_replay import provider

    from tensorrt_llm._torch.models.modeling_deepseekv41 import DeepseekV41ForCausalLM

    hash_provider = provider()
    reference_provider = provider()

    class EngramBody(nn.Module):
        def forward(self, input_ids, image_mask):
            return hash_provider.compute_hashes(input_ids, token_mask=~image_mask)[0].float()

    decoder = object.__new__(DeepseekV41ForCausalLM)
    nn.Module.__init__(decoder)
    decoder.model = EngramBody()
    owner = _MultimodalOwner(decoder, disable_mm_encoder=disable_mm_encoder)
    owner.llm = decoder
    runner, forward = _multimodal_prefill(owner)
    inputs = {
        "input_ids": torch.zeros(4, dtype=torch.long, device="cuda"),
        "image_mask": torch.zeros(4, dtype=torch.bool, device="cuda"),
    }

    assert runner is None
    original = forward(inputs).clone()
    inputs["input_ids"].copy_(torch.tensor([1, 2, 4, 5], device="cuda"))
    inputs["image_mask"].copy_(torch.tensor([False, True, False, False], device="cuda"))
    expected = reference_provider.compute_hashes(
        inputs["input_ids"], token_mask=~inputs["image_mask"]
    )[0].float()
    actual = forward(inputs).clone()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, original)
    assert torch.count_nonzero(actual[1]).item() == 0
    inputs["image_mask"].zero_()
    unmasked = forward(inputs)
    expected = reference_provider.compute_hashes(inputs["input_ids"])[0].float()
    torch.testing.assert_close(unmasked, expected, rtol=0, atol=0)
    assert not torch.equal(unmasked, actual)


@pytest.mark.parametrize("disable_mm_encoder", [False, True])
def test_multimodal_unverified_decoder_requires_explicit_support(disable_mm_encoder):
    decoder = object.__new__(DecoderModelForCausalLM)
    nn.Module.__init__(decoder)
    decoder.model = _Body()
    engine = object.__new__(ModelEngine)
    engine.model = _MultimodalOwner(decoder, disable_mm_encoder=disable_mm_encoder)
    engine.model.llm = decoder
    engine.mapping = SimpleNamespace(pp_size=1)
    engine.llm_args = SimpleNamespace(
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.DISABLED,
        disable_mm_encoder=disable_mm_encoder,
    )
    engine._validate_breakable_cuda_graph_compatibility()
    runner, _forward = _multimodal_prefill(engine.model)
    assert runner is None
    engine.llm_args.prefill_cuda_graph_backend = PrefillCudaGraphBackend.BREAKABLE

    with pytest.raises(ValueError, match="support"):
        engine._validate_breakable_cuda_graph_compatibility()


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

    decoder = object.__new__(Gemma4ForCausalLM)
    nn.Module.__init__(decoder)
    body = decoder.model = Body()
    owner = object.__new__(Gemma4ForConditionalGeneration)
    nn.Module.__init__(owner)
    owner.llm = decoder
    owner.model_config = SimpleNamespace(disable_mm_encoder=False)
    runner, _forward = _multimodal_prefill(owner)
    assert runner is not None
    inputs = {
        "input_ids": torch.zeros(4, dtype=torch.long, device="cuda"),
        "inputs_embeds": torch.zeros((4, 4), device="cuda"),
    }

    def outer():
        return body(**inputs)

    def engine():
        return runner.capture_model_body(outer) if runner.is_capturing else outer()

    try:
        runner.capture(4, engine)
        assert runner.has_graph(4)
        calls_after_capture = body.forward_calls
        inputs["input_ids"] = torch.arange(4, device="cuda")
        inputs["inputs_embeds"] = torch.full((4, 4), 3.0, device="cuda")
        actual = runner.execute(4, engine)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            actual, inputs["inputs_embeds"] + inputs["input_ids"][:, None], rtol=0, atol=0
        )
        assert body.forward_calls == calls_after_capture
    finally:
        runner.clear()


@pytest.mark.parametrize("input_kind", ["inputs_embeds", "embedding_handle"])
def test_llm_only_pipeline_breakable_graph_rejects_embedding_handoff_before_forward(input_kind):
    decoder_runner = object.__new__(DecoderRunner)
    decoder = _decoder_stub()
    decoder_runner._model = _MultimodalOwner(decoder, disable_mm_encoder=True)
    decoder_runner._model.llm = decoder
    decoder_runner._mapping = SimpleNamespace(pp_size=2)
    decoder_runner._config = SimpleNamespace(
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE
    )

    def unexpected_forward(**_kwargs):
        pytest.fail("unsupported embedding handoff must fail before model forward")

    decoder_runner._model_caller = unexpected_forward
    embedding = torch.ones((1, 4))
    inputs = {"inputs_embeds": embedding}
    if input_kind == "embedding_handle":
        inputs = {
            "multimodal_params": [
                SimpleNamespace(multimodal_data={"multimodal_embedding": embedding})
            ]
        }

    with pytest.raises(ValueError, match="single pipeline stage"):
        decoder_runner._model_forward(is_dummy=False, **inputs)


@pytest.mark.parametrize("multimodal_owner", [False, True])
@pytest.mark.parametrize("with_multimodal_params", [False, True])
def test_llm_only_pipeline_breakable_graph_preserves_text_forward(
    multimodal_owner, with_multimodal_params
):
    decoder = _decoder_stub()
    decoder_runner = object.__new__(DecoderRunner)
    decoder_runner._model = (
        _MultimodalOwner(decoder, disable_mm_encoder=True) if multimodal_owner else decoder
    )
    if multimodal_owner:
        decoder_runner._model.llm = decoder
    decoder_runner._mapping = SimpleNamespace(pp_size=2)
    decoder_runner._config = SimpleNamespace(
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE
    )
    decoder_runner._eager_workspace_reclaimer = None
    decoder_runner._moe_workspace_reclaimer = None
    metadata = object()
    text_params = (
        [SimpleNamespace(multimodal_data={"mrope_config": {}})] if with_multimodal_params else None
    )
    result = object()
    received = []

    def forward(**kwargs):
        received.append(kwargs)
        return result

    decoder_runner._model_caller = forward

    assert (
        decoder_runner._model_forward(
            is_dummy=False, attn_metadata=metadata, multimodal_params=text_params
        )
        is result
    )
    assert received == [{"attn_metadata": metadata, "multimodal_params": text_params}]


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("fail_body", [False, True])
def test_runner_eager_marker_covers_only_prepared_body(prepared, fail_body):
    flags = []

    class Body(nn.Module):
        def forward(self, value):
            flags.append(("body", is_in_breakable_cuda_graph()))
            if fail_body:
                raise ValueError("eager body failed")
            return value + 1

    def unexpected_normalization(_inputs):
        pytest.fail("run_eager must not normalize a warmup body twice")

    body = Body()
    runner = BreakableCUDAGraphRunner(
        body, prepare_inputs=unexpected_normalization if prepared else None
    )
    original_forward = body.forward

    def outer_forward():
        flags.append(("outer", is_in_breakable_cuda_graph()))
        return body(torch.zeros(4, device="cuda"))

    if fail_body:
        with pytest.raises(ValueError, match="eager body failed"):
            runner.run_eager(outer_forward)
    else:
        torch.testing.assert_close(
            runner.run_eager(outer_forward), torch.ones(4, device="cuda"), rtol=0, atol=0
        )

    assert flags == [("outer", False), ("body", prepared)]
    assert body.forward == original_forward
    assert runner.state == BreakableCUDAGraphRunnerState.IDLE
    assert not is_in_breakable_cuda_graph()


def test_runner_prepared_warmup_normalizes_once_inside_eager_body():
    body = _Body()
    normalization_flags = []

    def prepare_inputs(inputs):
        normalization_flags.append(is_in_breakable_cuda_graph())
        return {"value": inputs["value"] + 2}

    runner = BreakableCUDAGraphRunner(body, prepare_inputs=prepare_inputs)
    original_forward = body.forward
    value = torch.zeros(4, device="cuda")
    outputs = []

    runner.warmup(lambda: outputs.append(runner.run_eager(lambda: body(value))), steps=2)

    assert normalization_flags == [True, True]
    assert body.forward_calls == 2
    assert body.forward == original_forward
    assert runner.state == BreakableCUDAGraphRunnerState.IDLE
    for output in outputs:
        torch.testing.assert_close(output, torch.full_like(value, 3), rtol=0, atol=0)


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("fail_replay", [False, True])
def test_runner_execute_marker_keeps_multimodal_outer_eager(monkeypatch, prepared, fail_replay):
    body = _Body()
    runner = BreakableCUDAGraphRunner(
        body, prepare_inputs=(lambda inputs: inputs) if prepared else None
    )
    value = torch.zeros((4, 4), device="cuda")
    flags = []

    def outer_forward():
        flags.append(("outer", is_in_breakable_cuda_graph()))
        return body(value)

    def engine_forward():
        if runner.is_capturing:
            return runner.capture_model_body(outer_forward)
        return outer_forward()

    runner.capture(4, engine_forward)
    original_forward = body.forward
    original_replay = runner.replay
    flags.clear()

    def replay(num_tokens):
        flags.append(("replay", is_in_breakable_cuda_graph()))
        if fail_replay:
            raise ValueError("body replay failed")
        return original_replay(num_tokens)

    monkeypatch.setattr(runner, "replay", replay)
    if fail_replay:
        with pytest.raises(ValueError, match="body replay failed"):
            runner.execute(4, engine_forward)
    else:
        torch.testing.assert_close(
            runner.execute(4, engine_forward), torch.ones_like(value), rtol=0, atol=0
        )

    assert flags == [("outer", not prepared), ("replay", True)]
    assert body.forward == original_forward
    assert runner.state == BreakableCUDAGraphRunnerState.IDLE
    assert not is_in_breakable_cuda_graph()
