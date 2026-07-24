# Adapted from SGLang's breakable CUDA graph tests.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import weakref

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    break_graph,
    eager_on_graph,
)
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph.breakable_cuda_graph import _copy_output
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import (
    BreakableCUDAGraphRunner,
    BreakableCUDAGraphRunnerState,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


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
    @eager_on_graph(True)
    def add_one(value):
        return value + 1

    @eager_on_graph(True)
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


def test_disabled_and_outside_capture():
    @eager_on_graph(False)
    def disabled(value):
        return value + 1

    @eager_on_graph(True)
    def outside(value):
        return value + 2

    value = torch.tensor([1.0, 2.0], device="cuda")
    torch.testing.assert_close(disabled(value), value + 1)
    torch.testing.assert_close(outside(value), value + 2)


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


def test_runner_warmup_capture_execute_and_shared_output():
    body = _Body().cuda()
    logits_processor = _LogitsProcessor().cuda()
    runner = BreakableCUDAGraphRunner(body)
    counters = {"outer": 0}
    inputs = {}

    def engine_forward():
        counters["outer"] += 1
        if runner.is_capturing:
            return runner.capture_model_body(
                lambda: {"logits": logits_processor(body(inputs["value"]))}
            )
        return {"logits": logits_processor(body(inputs["value"]))}

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
