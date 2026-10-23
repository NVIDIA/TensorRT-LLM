# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Workflow metadata and current inference-boundary contracts (CPU only)."""

import ast
import inspect
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from profiler_test_utils import decode_profiler_metadata

from tensorrt_llm import _profiler as profiler
from tensorrt_llm._torch.pyexecutor import cuda_graph_runner, py_executor
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import runner as decoder_runner

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("context_tokens", "generation_requests", "expected"),
    [(16, 0, "trtllm::prefill"), (0, 4, "trtllm::decode"), (16, 4, "trtllm::mixed")],
)
def test_phase_tracks_scheduled_work(
    context_tokens: int, generation_requests: int, expected: str
) -> None:
    assert py_executor._forward_workflow_region(context_tokens, generation_requests) == expected


@pytest.mark.skipif(
    not hasattr(torch.profiler, "ExecutionTraceObserver"), reason="requires ExecutionTraceObserver"
)
def test_direct_tensor_metadata_and_decorator_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}))
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    scalar = torch.tensor(7, dtype=torch.int64)
    result = object()
    path = tmp_path / "execution.json"
    observer = torch.profiler.ExecutionTraceObserver().register_callback(str(path))

    @profiler.record_function_with_metadata("workflow::decorated")
    def wrapped(value: torch.Tensor, *, offset: torch.Tensor, nested: object) -> object:
        assert value is tensor and offset is scalar
        assert nested is container
        return result

    container = {"ignored": [tensor]}
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], execution_trace_observer=observer
        ) as trace:
            assert wrapped(tensor, offset=scalar, nested=container) is result
            with profiler.record_function_with_metadata("workflow::context", tensor, container):
                torch.ones(1)
    finally:
        observer.unregister_callback()
    nodes = {node["name"]: node for node in json.loads(path.read_text())["nodes"]}
    region, metadata = decode_profiler_metadata(nodes["workflow::decorated"]["inputs"]["values"][0])
    assert region == "workflow"
    assert metadata == {
        "workflow.tensor_count": 2,
        "workflow.tensor.0.shape": [4, 3],
        "workflow.tensor.0.dtype": "torch.float32",
        "workflow.tensor.0.stride": [1, 4],
        "workflow.tensor.0.device": "cpu",
        "workflow.tensor.1.shape": [],
        "workflow.tensor.1.dtype": "torch.int64",
        "workflow.tensor.1.stride": [],
        "workflow.tensor.1.device": "cpu",
    }
    assert (
        decode_profiler_metadata(nodes["workflow::context"]["inputs"]["values"][0])[1][
            "workflow.tensor_count"
        ]
        == 1
    )
    chrome_path = tmp_path / "chrome.json"
    trace.export_chrome_trace(str(chrome_path))
    names = {event.get("name") for event in json.loads(chrome_path.read_text())["traceEvents"]}
    assert {"workflow::decorated", "workflow::context"} <= names
    assert inspect.signature(wrapped).parameters.keys() == {"value", "offset", "nested"}


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_metadata_factory_is_lazy(
    monkeypatch: pytest.MonkeyPatch, active: bool, selected: bool
) -> None:
    monkeypatch.setattr(
        profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}) if selected else frozenset()
    )
    calls = 0

    def fail_metadata(tensor: torch.Tensor) -> dict[str, profiler.MetadataValue]:
        nonlocal calls
        calls += 1
        raise RuntimeError("observational metadata failure")

    monkeypatch.setattr(profiler, "_tensor_metadata", fail_metadata)
    tensor = torch.ones(1)

    @profiler.record_function_with_metadata("workflow::lazy")
    def wrapped(value: torch.Tensor) -> torch.Tensor:
        return value

    if active:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
            assert wrapped(tensor) is tensor
    else:
        assert wrapped(tensor) is tensor
    assert calls == int(active and selected)


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("error", [RuntimeError("operation"), KeyboardInterrupt("operation")])
def test_wrapped_exception_identity(
    monkeypatch: pytest.MonkeyPatch, selected: bool, error: BaseException
) -> None:
    monkeypatch.setattr(
        profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}) if selected else frozenset()
    )

    @profiler.record_function_with_metadata("workflow::exception")
    def wrapped() -> None:
        raise error

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
        with pytest.raises(type(error)) as caught:
            wrapped()
        assert caught.value is error
        with pytest.raises(type(error)) as caught:
            with profiler.record_function_with_metadata("workflow::exception_context"):
                raise error
        assert caught.value is error


def test_none_selection_suppresses_real_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset())
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        with profiler.record_function_with_metadata("workflow::disabled", torch.ones(1)):
            torch.ones(1)
    assert "workflow::disabled" not in {event.name for event in trace.events()}


@pytest.mark.parametrize(
    ("context_tokens", "generation_requests", "phase", "fails"),
    [
        (16, 0, "trtllm::prefill", False),
        (0, 4, "trtllm::decode", False),
        (16, 4, "trtllm::mixed", False),
        (16, 4, "trtllm::mixed", True),
    ],
)
def test_executor_forward_preserves_replay_and_diagnostic_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    context_tokens: int,
    generation_requests: int,
    phase: str,
    fails: bool,
) -> None:
    """Execute the current forward owner with only CUDA/model operations replaced."""
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}))
    monkeypatch.setattr(py_executor.ExpertStatistic, "set_iter", Mock())
    events: list[str] = []
    result = object()
    failure = RuntimeError("model failed")
    default_stream = SimpleNamespace(wait_stream=lambda stream: events.append("default_wait"))
    execution_stream = SimpleNamespace(wait_stream=lambda stream: events.append("execution_wait"))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: default_stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    route_capture = SimpleNamespace(
        set_iter=lambda iteration: events.append("route_iter"),
        prepare=lambda *args, **kwargs: events.append("route_prepare"),
        finish_forward=lambda: events.append("route_finish"),
        abort_forward=lambda: events.append("route_abort"),
    )
    batch = SimpleNamespace(
        context_requests=[SimpleNamespace(context_chunk_size=context_tokens)]
        if context_tokens
        else [],
        generation_requests=[SimpleNamespace() for _ in range(generation_requests)],
        num_context_requests=int(context_tokens > 0),
        num_generation_requests=generation_requests,
    )

    def forward(*args: object, **kwargs: object) -> object:
        assert args == (batch, resources, None)
        assert kwargs == {"cache_indirection_buffer": cache_indirection}
        events.append("model")
        if fails:
            raise failure
        return result

    resources = object()
    cache_indirection = object()
    executor = SimpleNamespace(
        iter_counter=7,
        model_engine=SimpleNamespace(route_capture=route_capture, is_warmup=False, forward=forward),
        enable_kv_cache_reuse=False,
        _route_cache_capacity=None,
        _iter_adp_dummy_ctx_tokens=0,
        _iter_adp_dummy_gen_tokens=0,
        execution_stream=execution_stream,
        resource_manager=resources,
        sampler=SimpleNamespace(get_cache_indirection=lambda: cache_indirection),
        _compute_adp_dummy_tokens=lambda requests: (3, 2),
        _attach_encoder_output_to_execution_stream=lambda requests: events.append("attach"),
        _mark_cross_kv_projection_consumed=lambda requests: events.append("consume"),
        _kv_connector_wait_for_save=lambda: events.append("save"),
        _maybe_record_hang_diagnostic_phase=lambda name, *args, **kwargs: events.append(name),
        _handle_errors=lambda message: events.append(message),
    )
    observer = torch.profiler.ExecutionTraceObserver().register_callback(
        str(tmp_path / "forward.json")
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], execution_trace_observer=observer
    ) as trace:
        actual = py_executor.PyExecutor._forward_step(executor, batch)
    observer.unregister_callback()
    nodes = json.loads((tmp_path / "forward.json").read_text())["nodes"]
    for name in ("trtllm::forward", phase):
        node = next(node for node in nodes if node["name"] == name)
        fields = decode_profiler_metadata(node["inputs"]["values"][0])[1]
        assert fields["workflow.executor.iteration"] == 7
        assert fields["workflow.executor.is_warmup"] is False
        assert (
            fields["workflow.batch.live_requests"] == int(context_tokens > 0) + generation_requests
        )
        assert fields["workflow.batch.scheduled_context_tokens"] == context_tokens
    names = {event.name for event in trace.events()}
    assert {"trtllm::forward", phase} <= names
    phase_event = next(event for event in trace.events() if event.name == phase)
    assert phase_event.cpu_parent is not None
    assert phase_event.cpu_parent.name == "trtllm::forward"
    assert executor._iter_adp_dummy_ctx_tokens == 3
    assert executor._iter_adp_dummy_gen_tokens == 2
    prefix = ["forward_call", "route_iter", "route_prepare", "execution_wait", "attach", "model"]
    if fails:
        assert actual is None
        assert events == [*prefix, "forward_error", "route_abort", "model failed"]
    else:
        assert actual is result
        assert events == [
            *prefix,
            "forward_returned",
            "consume",
            "default_wait",
            "save",
            "route_finish",
            "forward_complete",
        ]


@pytest.mark.parametrize("active,selected", [(False, True), (True, False), (True, True)])
def test_explicit_metadata_is_lazy_and_observational(
    monkeypatch: pytest.MonkeyPatch, active: bool, selected: bool, tmp_path: Path
) -> None:
    """Missing factory gating or swallowing a body exception breaks this contract."""
    monkeypatch.setattr(
        profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}) if selected else frozenset()
    )
    calls = []
    value = torch.ones(2)
    failure = RuntimeError("body")

    def metadata(tensor: torch.Tensor, *, fails: bool = False) -> dict[str, profiler.MetadataValue]:
        calls.append(tensor)
        if fails:
            raise ValueError("metadata")
        return {"workflow.owner.count": 3}

    @profiler.record_function_with_metadata("workflow::explicit", metadata_factory=metadata)
    def wrapped(tensor: torch.Tensor, *, fails: bool = False) -> torch.Tensor:
        if fails:
            raise failure
        return tensor

    observer = torch.profiler.ExecutionTraceObserver().register_callback(
        str(tmp_path / "explicit.json")
    )
    with (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], execution_trace_observer=observer
        )
        if active
        else nullcontext()
    ):
        assert wrapped(value) is value
        with pytest.raises(RuntimeError) as caught:
            wrapped(value, fails=True)
        assert caught.value is failure
    observer.unregister_callback()
    assert len(calls) == 2 * int(active and selected)
    assert inspect.signature(wrapped).parameters.keys() == {"tensor", "fails"}
    assert wrapped.__name__ == "wrapped"
    if active and selected:
        node = next(
            node
            for node in json.loads((tmp_path / "explicit.json").read_text())["nodes"]
            if node["name"] == "workflow::explicit"
        )
        fields = decode_profiler_metadata(node["inputs"]["values"][0])[1]
        assert fields["workflow.owner.count"] == 3
        assert fields["workflow.tensor.0.shape"] == [2]


def test_batch_dummy_counts_mapping_and_microbatch_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Counting padded lists as live occupancy or dropping PP slot must fail."""
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}))
    mapping = SimpleNamespace(
        rank=3,
        world_size=8,
        tp_rank=1,
        tp_size=2,
        pp_rank=1,
        pp_size=2,
        cp_rank=0,
        cp_size=2,
        moe_ep_rank=1,
        moe_ep_size=2,
    )
    owner = SimpleNamespace(
        iter_counter=19,
        dist=SimpleNamespace(mapping=mapping),
        model_engine=SimpleNamespace(is_warmup=True),
        enable_attention_dp=True,
    )
    batch = SimpleNamespace(
        context_requests=[
            SimpleNamespace(context_chunk_size=5),
            SimpleNamespace(context_chunk_size=2, is_attention_dp_dummy=True),
            SimpleNamespace(context_chunk_size=3, is_dummy_request=True),
        ],
        generation_requests=[
            SimpleNamespace(),
            SimpleNamespace(is_cuda_graph_dummy=True, is_dummy_request=True),
        ],
    )
    observer = torch.profiler.ExecutionTraceObserver().register_callback(
        str(tmp_path / "batch.json")
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], execution_trace_observer=observer
    ):
        with profiler.record_function_with_metadata(
            "batch",
            metadata_factory=lambda: py_executor._forward_workflow_metadata(
                owner, batch, microbatch_id=2
            ),
        ):
            torch.ones(1)
    observer.unregister_callback()
    node = next(
        node
        for node in json.loads((tmp_path / "batch.json").read_text())["nodes"]
        if node["name"] == "batch"
    )
    fields = decode_profiler_metadata(node["inputs"]["values"][0])[1]
    assert fields["workflow.executor.microbatch_slot"] == 2
    assert fields["workflow.attention_dp_enabled"] is True
    assert fields["workflow.rank"] == 3 and fields["workflow.moe_ep_rank"] == 1
    assert fields["workflow.batch.scheduled_context_requests"] == 3
    assert fields["workflow.batch.scheduled_generation_requests"] == 2
    assert fields["workflow.batch.live_requests"] == 2
    assert fields["workflow.batch.live_context_requests"] == 1
    assert fields["workflow.batch.live_generation_requests"] == 1
    assert fields["workflow.batch.attention_dp_dummy_requests"] == 1
    assert fields["workflow.batch.cuda_graph_dummy_requests"] == 1
    assert fields["workflow.batch.generic_dummy_requests"] == 2
    assert fields["workflow.batch.scheduled_context_tokens"] == 10
    assert fields["workflow.batch.live_context_tokens"] == 5


@pytest.mark.parametrize("warmup_only", [False, True])
def test_decoder_capture_and_replay_named_capacity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, warmup_only: bool
) -> None:
    """Actual graph owners must not present padded token rows as live requests."""
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}))
    monkeypatch.setattr(
        cuda_graph_runner, "with_multi_stream", lambda value: nullcontext(), raising=False
    )
    monkeypatch.setattr(
        cuda_graph_runner, "piecewise_cuda_graph", lambda value: nullcontext(), raising=False
    )
    monkeypatch.setattr(
        cuda_graph_runner, "make_weak_ref", lambda value: lambda: value, raising=False
    )
    graph = SimpleNamespace(replay=lambda: None, pool=lambda: None, reset=lambda: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(
        cuda_graph_runner, "nccl_window_graph_capture", lambda *args, **kwargs: nullcontext()
    )
    key = cuda_graph_runner.KeyType(batch_size=4, draft_len=2)
    attn = object()
    inputs = {
        "input_ids": torch.arange(12),
        "position_ids": torch.arange(12).reshape(1, 12),
        "attn_metadata": attn,
    }
    owner = object.__new__(cuda_graph_runner.CUDAGraphRunner)
    vars(owner).update(
        config=SimpleNamespace(use_mrope=False),
        max_beam_width=1,
        padding_enabled=True,
        is_warmup_only=warmup_only,
        WARMUP_STEPS=1,
        shared_static_tensors={
            "input_ids": torch.zeros(12, dtype=torch.int64),
            "position_ids": torch.zeros(1, 12, dtype=torch.int64),
        },
        graph_metadata={},
        graphs={},
        graph_outputs={},
        memory_pool=object(),
        _pre_replay_hooks=[],
    )
    observer = torch.profiler.ExecutionTraceObserver().register_callback(
        str(tmp_path / "graph.json")
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], execution_trace_observer=observer
    ) as trace:
        cuda_graph_runner.CUDAGraphRunner.capture(
            owner, key, lambda inputs: inputs["input_ids"] + 1, inputs
        )
        if not warmup_only:
            for live_count in (1, 3):
                batch = SimpleNamespace(
                    context_requests=[],
                    generation_requests=[SimpleNamespace() for _ in range(live_count)],
                )
                executor = SimpleNamespace(
                    iter_counter=live_count, model_engine=SimpleNamespace(is_warmup=False)
                )
                with profiler.record_function_with_metadata(
                    "parent",
                    metadata_factory=lambda: py_executor._forward_workflow_metadata(
                        executor, batch
                    ),
                ):
                    cuda_graph_runner.CUDAGraphRunner.replay(
                        owner, key, inputs, position_id_offsets=None
                    )
    observer.unregister_callback()
    nodes = json.loads((tmp_path / "graph.json").read_text())["nodes"]
    graph_nodes = [node for node in nodes if node["name"].startswith("trtllm::cuda_graph_")]
    assert len(graph_nodes) == (1 if warmup_only else 3)
    for node in graph_nodes:
        fields = decode_profiler_metadata(node["inputs"]["values"][0])[1]
        assert fields["workflow.graph.kind"] == "decoder"
        assert fields["workflow.graph.key.batch_size"] == 4
        assert fields["workflow.graph.key.draft_len"] == 2
        assert fields["workflow.graph.key.short_seq_len_mode"] is False
        assert "workflow.graph.key.is_first_draft" not in fields
        assert fields["workflow.graph.key.sample_type"] == "FULL"
        assert fields["workflow.graph.batch_capacity"] == 4
        assert fields["workflow.graph.token_capacity"] == 12
        assert fields["workflow.graph.input_token_rows"] == 12
        assert fields["workflow.graph.padding_enabled"] is True
        assert "workflow.batch.live_requests" not in fields
        assert "workflow.graph.padding_requests" not in fields
    first = decode_profiler_metadata(graph_nodes[0]["inputs"]["values"][0])[1]
    assert first["workflow.graph.operation"] == ("warmup_only" if warmup_only else "capture")
    if not warmup_only:
        parent_counts = [
            decode_profiler_metadata(node["inputs"]["values"][0])[1]["workflow.batch.live_requests"]
            for node in nodes
            if node["name"] == "parent"
        ]
        assert parent_counts == [1, 3]
    trace.export_chrome_trace(str(tmp_path / "chrome.json"))
    assert "trtllm::cuda_graph_capture" in {
        event.get("name")
        for event in json.loads((tmp_path / "chrome.json").read_text())["traceEvents"]
    }


@pytest.mark.parametrize("operation", ["capture", "replay"])
def test_encoder_graph_preserves_source_and_key_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, operation: str
) -> None:
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}))
    result = object()
    owner = SimpleNamespace(
        feature_mode=True,
        padding_enabled=True,
        is_warmup_only=False,
        WARMUP_STEPS=1,
        _replay_features=lambda key, inputs: result,
        _prepare_feature_capture=lambda key, inputs: (inputs, None),
        _get_capture_stream=lambda: None,
        _contains_nested_tensor=lambda output: False,
        graph_metadata={},
        graphs={},
        graph_outputs={},
        memory_pool=object(),
    )
    monkeypatch.setattr(
        cuda_graph_runner, "with_multi_stream", lambda value: nullcontext(), raising=False
    )
    monkeypatch.setattr(
        cuda_graph_runner, "piecewise_cuda_graph", lambda value: nullcontext(), raising=False
    )
    monkeypatch.setattr(cuda_graph_runner, "make_weak_ref", lambda value: value, raising=False)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: SimpleNamespace(pool=lambda: None))
    monkeypatch.setattr(
        cuda_graph_runner, "nccl_window_graph_capture", lambda *args, **kwargs: nullcontext()
    )
    inputs = {
        "input_features": torch.ones(2, 8),
        "_encoder_source_seq_lens": [3, 3],
        "attn_metadata": object(),
    }
    observer = torch.profiler.ExecutionTraceObserver().register_callback(
        str(tmp_path / "encoder.json")
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], execution_trace_observer=observer
    ):
        if operation == "replay":
            actual = cuda_graph_runner.EncoderCUDAGraphRunner.replay(owner, (4, 12, 3), inputs)
        else:
            actual = cuda_graph_runner.EncoderCUDAGraphRunner.capture(
                owner, (4, 12, 3), lambda inputs: result, inputs
            )
        assert actual is result
    observer.unregister_callback()
    node = next(
        node
        for node in json.loads((tmp_path / "encoder.json").read_text())["nodes"]
        if node["name"] == f"trtllm::cuda_graph_{operation}"
    )
    fields = decode_profiler_metadata(node["inputs"]["values"][0])[1]
    assert fields["workflow.graph.kind"] == "encoder"
    assert fields["workflow.graph.operation"] == operation
    assert fields["workflow.graph.key.padded_batch_size"] == 4
    assert fields["workflow.graph.key.padded_total_tokens"] == 12
    assert fields["workflow.graph.key.max_seq_len_bucket"] == 3
    assert fields["workflow.graph.input_feature_shape"] == [2, 8]
    assert fields["workflow.graph.source_requests"] == 2
    assert fields["workflow.graph.source_tokens"] == 6
    assert fields["workflow.graph.padding_tokens"] == 6
    assert "workflow.graph.key.draft_len" not in fields


@pytest.mark.parametrize("is_dummy", [False, True])
def test_eager_model_physical_rows_are_not_generation_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, is_dummy: bool
) -> None:
    """Speculative rows must retain their physical dimension, not request count."""
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}))
    tensor = torch.ones(9, dtype=torch.int64)
    owner = SimpleNamespace(
        _model_caller=lambda **kwargs: kwargs["input_ids"],
        _eager_workspace_reclaimer=None,
        _moe_workspace_reclaimer=None,
    )
    observer = torch.profiler.ExecutionTraceObserver().register_callback(
        str(tmp_path / "model.json")
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], execution_trace_observer=observer
    ):
        assert (
            decoder_runner.DecoderRunner.model_forward(
                owner, is_dummy=is_dummy, input_ids=tensor, attn_metadata=object()
            )
            is tensor
        )
    observer.unregister_callback()
    node = next(
        node
        for node in json.loads((tmp_path / "model.json").read_text())["nodes"]
        if node["name"] == "trtllm::model_forward"
    )
    fields = decode_profiler_metadata(node["inputs"]["values"][0])[1]
    assert fields["workflow.model.input_token_rows"] == 9
    assert fields["workflow.executor.is_warmup"] is is_dummy
    assert "workflow.batch.live_generation_requests" not in fields


@pytest.mark.parametrize("route", ["_forward_step_inter_pp", "_forward_step"])
def test_pp_loop_call_propagates_actual_slot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, route: str
) -> None:
    """Execute the PP loop's real call expression and intermediate forward owner."""
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}))
    batch = SimpleNamespace(
        context_requests=[], generation_requests=[SimpleNamespace()], context_requests_last_chunk=[]
    )
    owner = SimpleNamespace(
        iter_counter=6,
        model_engine=SimpleNamespace(is_warmup=False, runtime_draft_len=0),
        perf_manager=SimpleNamespace(record_perf_events=lambda *args: nullcontext()),
        _update_request_states=lambda batch: None,
        sampler=SimpleNamespace(SampleState=lambda **kwargs: None),
    )

    @profiler.record_function_with_metadata(
        "pp-forward", metadata_factory=py_executor._forward_workflow_metadata
    )
    def forward(
        executor: object, scheduled_requests: object, *, microbatch_id: int | None = None
    ) -> None:
        pass

    owner._forward_step = lambda *args, **kwargs: forward(owner, *args, **kwargs)
    owner._forward_step_inter_pp = (
        lambda *args, **kwargs: py_executor.PyExecutor._forward_step_inter_pp(
            owner, *args, **kwargs
        )
    )
    monkeypatch.setattr(torch.cuda, "Event", lambda: SimpleNamespace(record=lambda: None))
    monkeypatch.setattr(py_executor, "SamplerEvent", lambda **kwargs: None, raising=False)
    tree = ast.parse(Path(py_executor.__file__).read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PyExecutor"
    )
    loop = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_executor_loop_pp"
    )
    call = next(
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == route
    )
    expression = compile(ast.fix_missing_locations(ast.Expression(body=call)), "pp-call", "eval")
    observer = torch.profiler.ExecutionTraceObserver().register_callback(str(tmp_path / "pp.json"))
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], execution_trace_observer=observer
    ):
        eval(
            expression,
            {
                "self": owner,
                "scheduled_batch": batch,
                "microbatch_id": 3,
                "gpu_forward_start": None,
                "gpu_forward_end": None,
            },
        )
    observer.unregister_callback()
    node = next(
        node
        for node in json.loads((tmp_path / "pp.json").read_text())["nodes"]
        if node["name"] == "pp-forward"
    )
    fields = decode_profiler_metadata(node["inputs"]["values"][0])[1]
    assert fields["workflow.executor.microbatch_slot"] == 3
    assert fields["workflow.executor.iteration"] == 6


@pytest.mark.parametrize("pp_size,expected", [(1, 0), (2, None), (None, None)])
def test_microbatch_slot_unknown_is_not_a_numeric_sentinel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pp_size: int | None, expected: int | None
) -> None:
    monkeypatch.setattr(profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}))
    owner = SimpleNamespace(iter_counter=1, model_engine=SimpleNamespace())
    if pp_size is not None:
        owner.dist = SimpleNamespace(mapping=SimpleNamespace(pp_size=pp_size))
    batch = SimpleNamespace(context_requests=[], generation_requests=[])
    observer = torch.profiler.ExecutionTraceObserver().register_callback(
        str(tmp_path / "unknown.json")
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], execution_trace_observer=observer
    ):
        with profiler.record_function_with_metadata(
            "unknown", metadata_factory=lambda: py_executor._forward_workflow_metadata(owner, batch)
        ):
            pass
    observer.unregister_callback()
    node = next(
        node
        for node in json.loads((tmp_path / "unknown.json").read_text())["nodes"]
        if node["name"] == "unknown"
    )
    fields = decode_profiler_metadata(node["inputs"]["values"][0])[1]
    if expected is None:
        assert "workflow.executor.microbatch_slot" not in fields
    else:
        assert fields["workflow.executor.microbatch_slot"] == expected
    assert "workflow.attention_dp_enabled" not in fields
    assert "workflow.executor.is_warmup" not in fields


@pytest.mark.parametrize("selected", [False, True])
def test_graph_metadata_failure_keeps_replay_body_exception(
    monkeypatch: pytest.MonkeyPatch, selected: bool
) -> None:
    monkeypatch.setattr(
        profiler, "_CUSTOM_REGIONS", frozenset({"workflow"}) if selected else frozenset()
    )
    calls = []

    def fail_metadata(*args: object) -> dict[str, profiler.MetadataValue]:
        calls.append(None)
        raise ValueError("observational")

    monkeypatch.setattr(cuda_graph_runner, "_decoder_graph_workflow_metadata", fail_metadata)
    error = RuntimeError("graph replay body")

    def replay() -> None:
        raise error

    key = cuda_graph_runner.KeyType(batch_size=1, draft_len=0)
    attn = object()
    owner = object.__new__(cuda_graph_runner.CUDAGraphRunner)
    vars(owner).update(
        config=SimpleNamespace(use_mrope=False),
        max_beam_width=1,
        graph_metadata={key: {"attn_metadata": attn, "spec_metadata": None}},
        shared_static_tensors={"input_ids": torch.zeros(1), "position_ids": torch.zeros(1, 1)},
        graphs={key: SimpleNamespace(replay=replay, reset=lambda: None)},
        graph_outputs={},
        memory_pool=None,
        _pre_replay_hooks=[],
    )
    owner._get_num_tokens_for_key = lambda key: 1
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
        with pytest.raises(RuntimeError) as caught:
            cuda_graph_runner.CUDAGraphRunner.replay(
                owner,
                key,
                {
                    "input_ids": torch.ones(1),
                    "position_ids": torch.ones(1, 1),
                    "attn_metadata": attn,
                },
            )
        assert caught.value is error
    assert len(calls) == int(selected)


@pytest.mark.parametrize(
    ("filename", "owner", "boundaries"),
    [
        (
            "py_executor.py",
            "PyExecutor",
            {
                "_prepare_and_schedule_batch": "trtllm::prepare_batch",
                "_schedule": "trtllm::schedule",
                "_forward_step": "trtllm::forward",
                "_sample_async": "trtllm::sampling",
                "_update_requests": "trtllm::update_requests",
                "_handle_responses": "trtllm::handle_responses",
            },
        ),
        (
            "engine/runners/decoder/inputs.py",
            "DecoderInputPreparer",
            {"prepare_inputs": "trtllm::prepare_inputs"},
        ),
        (
            "engine/runners/decoder/runner.py",
            "DecoderRunner",
            {"model_forward": "trtllm::model_forward"},
        ),
        (
            "cuda_graph_runner.py",
            "CUDAGraphRunner",
            {"capture": "trtllm::cuda_graph_capture", "replay": "trtllm::cuda_graph_replay"},
        ),
        (
            "cuda_graph_runner.py",
            "EncoderCUDAGraphRunner",
            {"capture": "trtllm::cuda_graph_capture", "replay": "trtllm::cuda_graph_replay"},
        ),
    ],
)
def test_annotations_stay_at_current_owner_boundaries(
    filename: str, owner: str, boundaries: dict[str, str]
) -> None:
    """Missing a boundary or wrapping the entire executor loop breaks trace granularity."""
    tree = ast.parse((Path(py_executor.__file__).parent / filename).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    for method, region in boundaries.items():
        names = [
            decorator.args[0].value
            for decorator in methods[method].decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "record_function_with_metadata"
        ]
        assert names == [region]
    for name, method in methods.items():
        if "executor_loop" in name:
            assert not any(
                isinstance(node, ast.Name) and node.id == "record_function_with_metadata"
                for decorator in method.decorator_list
                for node in ast.walk(decorator)
            )
