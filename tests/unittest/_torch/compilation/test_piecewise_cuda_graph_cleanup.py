# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
from typing import Callable
from unittest.mock import patch

import pytest
import torch
from torch.fx import symbolic_trace

from tensorrt_llm._torch.compilation.piecewise_optimizer import PiecewiseRunner
from tensorrt_llm._torch.compilation.utils import capture_piecewise_cuda_graph
from tensorrt_llm._torch.utils import piecewise_cuda_graph


class _ResetFailureGraph:
    def reset(self) -> None:
        raise RuntimeError("reset failed")


class _RecordingGraph:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def _make_runner(default_callable: Callable[[], object] | None = None) -> PiecewiseRunner:
    return PiecewiseRunner(
        graph=symbolic_trace(torch.nn.Identity()),
        name="test",
        compile_time_num_tokens=1,
        runtime_num_tokens_idx=None,
        capture_num_tokens=[1, 2],
        graph_pool_handle=None,
        default_callable=default_callable or (lambda: None),
        enable_inductor=False,
        is_first_runner=False,
        is_last_runner=False,
    )


def test_clear_continues_after_reset_failure() -> None:
    runner = _make_runner()
    remaining_graph = _RecordingGraph()
    runner.entries[1].cuda_graph = _ResetFailureGraph()
    runner.entries[2].cuda_graph = remaining_graph

    runner.clear_cuda_graphs()

    assert remaining_graph.reset_count == 1
    for entry in runner.entries.values():
        assert entry.cuda_graph is None
        assert entry.warmup_count == 0
        assert entry.input_addresses is None
        assert entry.output_addresses is None
        assert entry.output is None


def test_capture_failure_resets_graph_before_entry_commit() -> None:
    def fail_capture() -> None:
        raise ValueError("capture failed")

    runner = _make_runner(fail_capture)
    runner.entries[1].warmup_count = 3
    graph = _RecordingGraph()
    capture_stream = object()
    restored_stream = object()
    extra_attrs = {}

    with (
        piecewise_cuda_graph(True),
        capture_piecewise_cuda_graph(True),
        patch("torch.cuda.CUDAGraph", return_value=graph),
        patch("torch.cuda.graph", return_value=nullcontext()),
        patch("torch.cuda.current_stream", side_effect=[capture_stream, restored_stream]),
        patch(
            "tensorrt_llm._torch.compilation.piecewise_optimizer.get_model_extra_attrs",
            return_value=extra_attrs,
        ),
        pytest.raises(ValueError, match="capture failed"),
    ):
        runner()

    assert graph.reset_count == 1
    assert runner.entries[1].cuda_graph is None
    assert extra_attrs["global_stream"] is restored_stream
