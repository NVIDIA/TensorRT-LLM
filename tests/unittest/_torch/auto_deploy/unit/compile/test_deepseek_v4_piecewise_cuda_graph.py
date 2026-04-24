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
"""DeepSeek V4 CUDA graph runtime config and piecewise split tests."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.fx import Graph, GraphModule

from tensorrt_llm._torch.auto_deploy.compile.backends import torch_cudagraph
from tensorrt_llm._torch.auto_deploy.compile.piecewise_runner import (
    ADPiecewiseRunner,
    DynamicOpWrapper,
)
from tensorrt_llm._torch.auto_deploy.compile.piecewise_utils import (
    is_dynamic_cached_op,
    is_metadata_prep,
    needs_out_buffer,
    split_graph_at_dynamic_ops,
    submod_has_cuda_ops,
)
from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs

_REPO_ROOT = Path(__file__).resolve().parents[6]
_REGISTRY_CONFIGS_DIR = _REPO_ROOT / "examples" / "auto_deploy" / "model_registry" / "configs"
_DEEPSEEK_V4_CONFIG = _REGISTRY_CONFIGS_DIR / "deepseek_v4_flash.yaml"
_DUMMY_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_CONFIGURED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]


class _FakeOpOverload:
    def __init__(self, op_name: str) -> None:
        self._op_name = op_name

    def name(self) -> str:
        return self._op_name


def torch_deepseek_v4_sparse_attention(
    x: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    y = x + 1
    if out is not None:
        out.copy_(y)
        return torch.empty(0, dtype=x.dtype, device=x.device)
    return y


def deepseek_v4_prepare_cache_metadata(x: torch.Tensor) -> torch.Tensor:
    return x


def _make_call_function_node(op_name: str) -> SimpleNamespace:
    return SimpleNamespace(op="call_function", target=_FakeOpOverload(op_name))


def _make_toy_graph(dynamic_fn: Any = torch_deepseek_v4_sparse_attention) -> GraphModule:
    root = nn.Module()
    root.static_0 = nn.Linear(4, 4, bias=False)
    root.static_1 = nn.Linear(4, 4, bias=False)

    graph = Graph()
    x = graph.placeholder("x")
    first_static = graph.call_module("static_0", args=(x,))
    dynamic = graph.call_function(dynamic_fn, args=(first_static,))
    second_static = graph.call_module("static_1", args=(dynamic,))
    graph.output(second_static)
    graph.lint()
    return GraphModule(root, graph)


@pytest.mark.parametrize(
    "op_name",
    [
        "auto_deploy::torch_deepseek_v4_sparse_attention",
        "auto_deploy::triton_deepseek_v4_sparse_attention_with_cache",
        "auto_deploy::torch_deepseek_v4_moe",
        "auto_deploy::triton_mxfp4_moe",
        "auto_deploy::triton_mxfp4_moe_ep",
        "auto_deploy::deepseek_v4_prepare_cache_metadata",
        "auto_deploy::torch_deepseek_v4_prepare_cache_metadata",
        "auto_deploy::triton_deepseek_v4_prepare_cache_metadata",
    ],
)
def test_deepseek_v4_reserved_ops_are_dynamic(op_name: str) -> None:
    assert is_dynamic_cached_op(_make_call_function_node(op_name))


def test_deepseek_v4_sparse_attention_splits_static_regions_around_dynamic_op() -> None:
    split_info = split_graph_at_dynamic_ops(_make_toy_graph())

    assert split_info.dynamic_submod_indices == [1]
    assert split_info.static_submod_indices == [0, 2]
    assert submod_has_cuda_ops(split_info.split_gm.submod_0)
    assert submod_has_cuda_ops(split_info.split_gm.submod_2)
    assert needs_out_buffer(split_info.split_gm.submod_1)


def test_piecewise_prepare_wraps_static_regions_around_deepseek_v4_dynamic_op(
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: ("fake_pool",))

    piecewise = torch_cudagraph.PiecewiseCapturedGraph(
        _make_toy_graph(),
        piecewise_num_tokens=[4],
        capture_lm_head=True,
    )

    piecewise.prepare()

    assert isinstance(piecewise.split_gm.submod_0, ADPiecewiseRunner)
    assert isinstance(piecewise.split_gm.submod_1, DynamicOpWrapper)
    assert isinstance(piecewise.split_gm.submod_2, ADPiecewiseRunner)
    assert piecewise._wrapped_dynamic_indices == {1}


def test_deepseek_v4_cache_metadata_prep_is_dynamic_without_out_buffer() -> None:
    split_info = split_graph_at_dynamic_ops(_make_toy_graph(deepseek_v4_prepare_cache_metadata))
    metadata_submod = split_info.split_gm.submod_1

    assert is_metadata_prep(metadata_submod)
    assert not needs_out_buffer(metadata_submod)


def test_deepseek_v4_config_parses_cuda_graph_batch_sizes() -> None:
    args = LlmArgs(model=_DUMMY_MODEL, yaml_extra=[str(_DEEPSEEK_V4_CONFIG)])

    assert args.enable_chunked_prefill
    assert args.cuda_graph_config.batch_sizes == _CONFIGURED_BATCH_SIZES
    assert args.cuda_graph_config.max_batch_size == max(_CONFIGURED_BATCH_SIZES)
    assert args.transforms["compile_model"]["cuda_graph_batch_sizes"] == _CONFIGURED_BATCH_SIZES
    assert args.transforms["compile_model"]["piecewise_enabled"]
    assert not args.transforms["multi_stream_moe"]["enabled"]


def test_torch_cudagraph_compile_keeps_monolithic_when_piecewise_disabled(monkeypatch) -> None:
    def fake_capture_graph(
        self: torch_cudagraph.CapturedGraph,
        get_args_kwargs: Any,
        batch_sizes: list[int],
    ) -> None:
        self._out_spec = None

    def fail_piecewise_construction(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("PiecewiseCapturedGraph should not be constructed")

    monkeypatch.setattr(torch_cudagraph.CapturedGraph, "capture_graph", fake_capture_graph)
    monkeypatch.setattr(torch_cudagraph, "PiecewiseCapturedGraph", fail_piecewise_construction)

    model = nn.Identity()
    compiler = torch_cudagraph.TorchCudagraphCompiler(
        model=model,
        get_args_kwargs_for_compile=lambda batch_size: ((torch.ones(batch_size, 4),), {}),
        cuda_graph_batch_sizes=[1, 2],
        piecewise_enabled=False,
    )

    compiled = compiler.compile()

    assert isinstance(compiled, torch_cudagraph.CapturedGraph)
    assert not isinstance(compiled, torch_cudagraph.DualModeCapturedGraph)
    assert compiled.model is model


class _FakeMonolithic(nn.Module):
    _output_dynamic_dim = 0

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Identity()
        self.num_calls = 0

    def forward(self, *args: Any, **kwargs: Any) -> str:
        self.num_calls += 1
        return "decode"


class _FakePiecewise(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.original_model = nn.Identity()
        self.piecewise_num_tokens = _CONFIGURED_BATCH_SIZES
        self.num_calls = 0

    def forward(self, *args: Any, **kwargs: Any) -> str:
        self.num_calls += 1
        return "piecewise"


def test_decode_only_runtime_uses_monolithic_path_with_configured_piecewise_buckets() -> None:
    monolithic = _FakeMonolithic()
    piecewise = _FakePiecewise()
    runtime = torch_cudagraph.DualModeCapturedGraph(monolithic, piecewise)

    result = runtime(
        input_ids=torch.ones(3, 1, dtype=torch.int64),
        batch_info_host=torch.tensor([0, 0, 0, 0, 3, 3], dtype=torch.int64),
    )

    assert result == "decode"
    assert monolithic.num_calls == 1
    assert piecewise.num_calls == 0
    assert runtime._find_nearest_bucket(33) == 64
    assert runtime._find_nearest_bucket(65) is None
