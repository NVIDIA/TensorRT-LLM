# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import inspect
from dataclasses import dataclass, field

import pytest
import torch
from torch._subclasses import FakeTensorMode

from tensorrt_llm._torch import nccl_window_graph, nccl_window_tensor_scope

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def mock_synchronize_releases(monkeypatch):
    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "synchronize_nccl_window_buffer_releases",
        lambda: None,
    )


def test_pool_owner_is_value_based():
    assert nccl_window_graph._shared_pool_owner(
        (12345, 67890)
    ) == nccl_window_graph._shared_pool_owner(tuple([12345, 67890]))


def test_pool_owner_rejects_invalid_handles():
    with pytest.raises(TypeError, match="CUDA graph pool handle"):
        nccl_window_graph._shared_pool_owner(None)
    with pytest.raises(TypeError, match="Invalid CUDA graph pool handle"):
        nccl_window_graph._shared_pool_owner((1, "not-an-int"))


def test_release_removes_owner_after_backend_release(monkeypatch):
    pool = (23456, 78901)
    owner = nccl_window_graph._shared_pool_owner(pool)
    released = []
    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "release_nccl_window_graph_owner",
        released.append,
    )

    nccl_window_graph.release_nccl_window_graph_owner(pool)

    replacement_owner = nccl_window_graph._shared_pool_owner(pool)
    assert released == [owner]
    assert replacement_owner > owner


def test_release_retains_owner_when_backend_release_fails(monkeypatch):
    pool = (34567, 89012)
    owner = nccl_window_graph._shared_pool_owner(pool)

    def fail_release(_owner):
        raise RuntimeError("backend release failed")

    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "release_nccl_window_graph_owner",
        fail_release,
    )

    with pytest.raises(RuntimeError, match="backend release failed"):
        nccl_window_graph.release_nccl_window_graph_owner(pool)

    assert nccl_window_graph._shared_pool_owner(pool) == owner


def test_graph_capture_sets_and_restores_owner(monkeypatch):
    pool = (45678, 90123)
    graph = object()
    owner = nccl_window_graph._shared_pool_owner(pool)
    events = []

    def set_owner(value):
        events.append(("owner", value))

    @contextlib.contextmanager
    def capture(captured_graph, *, pool, **kwargs):
        events.append(("capture", captured_graph, pool, kwargs))
        try:
            yield
        finally:
            events.append(("capture_exit",))

    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "set_nccl_window_graph_owner",
        set_owner,
    )
    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "synchronize_nccl_window_buffer_releases",
        lambda: events.append(("synchronize_releases",)),
    )
    monkeypatch.setattr(nccl_window_graph.torch.cuda, "graph", capture)

    with nccl_window_graph.nccl_window_graph_capture(
        graph, pool, capture_error_mode="thread_local"
    ):
        events.append(("body",))

    assert events == [
        ("owner", owner),
        ("synchronize_releases",),
        (
            "capture",
            graph,
            pool,
            {"capture_error_mode": "thread_local"},
        ),
        ("body",),
        ("capture_exit",),
        ("owner", nccl_window_graph._EAGER_OWNER),
    ]


def test_graph_owner_sets_and_restores_owner(monkeypatch):
    pool = (51234, 6789)
    owner = nccl_window_graph._shared_pool_owner(pool)
    owners = []

    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "set_nccl_window_graph_owner",
        owners.append,
    )

    with nccl_window_graph.nccl_window_graph_owner(pool):
        assert owners == [owner]

    assert owners == [owner, nccl_window_graph._EAGER_OWNER]


def test_graph_capture_abandons_new_owner_when_capture_entry_fails(monkeypatch):
    pool = (56789, 1234)
    graph = object()
    events = []

    def set_owner(value):
        events.append(("owner", value))

    def capture(captured_graph, *, pool, **kwargs):
        events.append(("capture", captured_graph, pool, kwargs))
        raise RuntimeError("capture failed")

    def release_owner(value):
        events.append(("release", value))

    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "set_nccl_window_graph_owner",
        set_owner,
    )
    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "release_nccl_window_graph_owner",
        release_owner,
    )
    monkeypatch.setattr(nccl_window_graph.torch.cuda, "graph", capture)

    with pytest.raises(RuntimeError, match="capture failed"):
        with nccl_window_graph.nccl_window_graph_capture(graph, pool):
            pytest.fail("capture body should not run")

    owner = events[0][1]
    assert events == [
        ("owner", owner),
        ("capture", graph, pool, {}),
        ("owner", nccl_window_graph._EAGER_OWNER),
    ]
    assert nccl_window_graph._pool_key(pool) not in nccl_window_graph._pool_owners


@pytest.mark.parametrize("failure_stage", ["body", "exit"])
def test_graph_capture_abandons_new_owner_when_capture_does_not_complete(
    monkeypatch, failure_stage
):
    pool = (57901, 2345)
    owners = []
    released = []

    @contextlib.contextmanager
    def capture(*args, **kwargs):
        del args, kwargs
        yield
        if failure_stage == "exit":
            raise RuntimeError("exit failed")

    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "set_nccl_window_graph_owner",
        owners.append,
    )
    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "release_nccl_window_graph_owner",
        released.append,
    )
    monkeypatch.setattr(nccl_window_graph.torch.cuda, "graph", capture)

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        with nccl_window_graph.nccl_window_graph_capture(object(), pool):
            if failure_stage == "body":
                raise RuntimeError("body failed")

    owner = owners[0]
    assert owners == [owner, nccl_window_graph._EAGER_OWNER]
    assert released == []
    assert nccl_window_graph._pool_key(pool) not in nccl_window_graph._pool_owners


def test_graph_capture_failure_preserves_existing_owner(monkeypatch):
    pool = (59012, 3456)
    owner = nccl_window_graph._shared_pool_owner(pool)
    released = []

    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "set_nccl_window_graph_owner",
        lambda value: None,
    )
    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "release_nccl_window_graph_owner",
        released.append,
    )
    monkeypatch.setattr(
        nccl_window_graph.torch.cuda,
        "graph",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )

    with pytest.raises(RuntimeError, match="body failed"):
        with nccl_window_graph.nccl_window_graph_capture(object(), pool):
            raise RuntimeError("body failed")

    assert released == []
    assert nccl_window_graph._shared_pool_owner(pool) == owner


def test_nested_graph_capture_restores_previous_owner(monkeypatch):
    outer_pool = (67890, 2345)
    inner_pool = (78901, 3456)
    outer_owner = nccl_window_graph._shared_pool_owner(outer_pool)
    inner_owner = nccl_window_graph._shared_pool_owner(inner_pool)
    owners = []

    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "set_nccl_window_graph_owner",
        owners.append,
    )
    monkeypatch.setattr(
        nccl_window_graph.torch.cuda,
        "graph",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )

    with nccl_window_graph.nccl_window_graph_capture(object(), outer_pool):
        with nccl_window_graph.nccl_window_graph_capture(object(), inner_pool):
            pass

    assert owners == [
        outer_owner,
        inner_owner,
        outer_owner,
        nccl_window_graph._EAGER_OWNER,
    ]


@pytest.fixture
def tensor_scope_events(monkeypatch):
    events = []

    monkeypatch.setattr(
        nccl_window_tensor_scope,
        "_cuda_tensors",
        lambda value, tensors: tensors.extend(value),
    )
    monkeypatch.setattr(
        nccl_window_tensor_scope.torch.ops.trtllm,
        "begin_nccl_window_tensor_scope",
        lambda tensors: events.append(("begin", tensors)),
    )
    monkeypatch.setattr(
        nccl_window_tensor_scope.torch.ops.trtllm,
        "end_nccl_window_tensor_scope",
        lambda tensors, failed: events.append(("end", tensors, failed)),
    )
    return events


def test_tensor_scope_transfers_outputs_and_releases_other_leases(tensor_scope_events):
    inputs = [object(), object()]
    outputs = [object()]

    with nccl_window_tensor_scope.nccl_window_tensor_scope(inputs) as scope:
        scope.escape(outputs)

    assert tensor_scope_events == [("begin", inputs), ("end", outputs, False)]


def test_tensor_scope_quarantines_adopted_leases_on_failure(tensor_scope_events):
    inputs = [object()]

    with pytest.raises(RuntimeError, match="scope failed"):
        with nccl_window_tensor_scope.nccl_window_tensor_scope(inputs):
            raise RuntimeError("scope failed")

    assert tensor_scope_events == [("begin", inputs), ("end", inputs, True)]


def test_tensor_scope_collects_cuda_tensors_from_dataclass():
    @dataclass
    class StructuredOutput:
        hidden_states: torch.Tensor
        residual: torch.Tensor | None = None
        cross: object | None = None

    with FakeTensorMode():
        hidden_states = torch.empty(1, device="cuda")
        residual = torch.empty(1, device="cuda")

    output = StructuredOutput(hidden_states, residual)
    output.cross = output
    tensors = []
    nccl_window_tensor_scope._cuda_tensors(output, tensors)

    assert tensors == [hidden_states, residual]


def test_tensor_scope_skips_uninitialized_dataclass_fields():
    @dataclass
    class PartiallyInitialized:
        hidden_states: torch.Tensor
        deferred: torch.Tensor = field(init=False)

    with FakeTensorMode():
        hidden_states = torch.empty(1, device="cuda")

    tensors = []
    nccl_window_tensor_scope._cuda_tensors(PartiallyInitialized(hidden_states), tensors)

    assert tensors == [hidden_states]


def test_eager_decoder_layer_hooks_scope_each_invocation(tensor_scope_events):
    from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer

    class TestLayer(DecoderLayer):
        def forward(self, hidden_states, **kwargs):
            return hidden_states

    layer = TestLayer()
    handles = nccl_window_tensor_scope.install_eager_nccl_window_tensor_scopes(layer)
    output = [object()]
    try:
        assert layer(output) is output
    finally:
        for handle in handles:
            handle.remove()

    assert [event[0] for event in tensor_scope_events] == ["begin", "end"]
    assert tensor_scope_events[-1][-1] is False


def test_eager_decoder_layer_hook_propagates_forward_failure(tensor_scope_events):
    from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer

    class TestLayer(DecoderLayer):
        def forward(self, hidden_states, **kwargs):
            raise RuntimeError("layer failed")

    layer = TestLayer()
    handles = nccl_window_tensor_scope.install_eager_nccl_window_tensor_scopes(layer)
    try:
        with pytest.raises(RuntimeError, match="layer failed"):
            layer([object()])
    finally:
        for handle in handles:
            handle.remove()

    assert tensor_scope_events[-1][-1] is True


def test_each_decoder_layer_scope_survives_torch_compile():
    from tensorrt_llm._torch.compilation.nccl_window import insert_nccl_window_tensor_scopes
    from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer

    class TestLayer(DecoderLayer):
        def forward(self, hidden_states, **kwargs):
            return hidden_states + 1

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Exceed Dynamo's default specialization limit. A Python forward
            # wrapper used to specialize once per layer and fail this case.
            self.layers = torch.nn.ModuleList(TestLayer() for _ in range(20))

        def forward(self, hidden_states):
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            return hidden_states

    graphs = []

    def capture_graph(gm, _example_inputs):
        insert_nccl_window_tensor_scopes(gm)
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    old_limit = torch._dynamo.config.cache_size_limit
    try:
        torch._dynamo.config.cache_size_limit = 16
        compiled = torch.compile(TestModel(), backend=capture_graph, fullgraph=True)
        result = compiled(torch.zeros(1))
    finally:
        torch._dynamo.config.cache_size_limit = old_limit
        torch._dynamo.reset()

    assert torch.equal(result, torch.full((1,), 20.0))
    targets = [node.target for node in graphs[0].graph.nodes]
    begin = torch.ops.trtllm.begin_nccl_window_tensor_scope.default
    end = torch.ops.trtllm.end_nccl_window_tensor_scope.default
    assert targets.count(begin) == 20
    assert targets.count(end) == 20

    active_scopes = 0
    for target in targets:
        active_scopes += target == begin
        active_scopes -= target == end
        assert active_scopes in (0, 1)
    assert active_scopes == 0


def test_decoder_layer_scope_preserves_concrete_forward_signature():
    from tensorrt_llm._torch.modules.decoder_layer import DecoderLayer

    class TestLayer(DecoderLayer):
        def forward(self, hidden_states, residual, block_residual=None, **kwargs):
            return hidden_states

    assert list(inspect.signature(TestLayer.forward).parameters) == [
        "self",
        "hidden_states",
        "residual",
        "block_residual",
        "kwargs",
    ]


def test_tensor_scope_compilation_metadata_pins_boundaries_to_primary_stream(monkeypatch):
    from tensorrt_llm._torch.compilation.multi_stream import auto_multi_stream
    from tensorrt_llm._torch.compilation.utils import inplace_info

    begin = torch.ops.trtllm.begin_nccl_window_tensor_scope.default
    end = torch.ops.trtllm.end_nccl_window_tensor_scope.default
    assert inplace_info()[begin] == {1: "inputs"}
    assert inplace_info()[end] == {1: "outputs"}

    graph = torch.fx.Graph()
    hidden_states = graph.placeholder("hidden_states")
    begin_node = graph.call_function(begin, kwargs={"inputs": [hidden_states]})
    output = graph.call_function(torch.ops.aten.add.Tensor, args=(hidden_states, 1))
    end_node = graph.call_function(end, kwargs={"outputs": [output], "failed": False})
    graph.output(output)

    monkeypatch.setattr(auto_multi_stream, "estimate_time", lambda _node: 1)
    dag = auto_multi_stream.MultiStreamDAG(torch.fx.GraphModule(torch.nn.Module(), graph))
    dag.assign_streams(2)

    assert dag.nodes[begin_node].stream.id == 0
    assert dag.nodes[end_node].stream.id == 0
