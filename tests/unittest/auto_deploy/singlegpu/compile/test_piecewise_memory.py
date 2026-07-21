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
"""Memory regression tests for piecewise CUDA graph capture."""

import gc
import weakref
from contextlib import contextmanager
from typing import Callable

import pytest
import torch
import torch.nn as nn
from torch.fx import Graph, GraphModule

from tensorrt_llm._torch.auto_deploy.compile import piecewise_runner as piecewise_runner_mod
from tensorrt_llm._torch.auto_deploy.compile.backends.torch_cudagraph import PiecewiseCapturedGraph
from tensorrt_llm._torch.auto_deploy.compile.piecewise_runner import (
    ADPiecewiseRunner,
    DynamicOpWrapper,
    MetadataWrapper,
)

_MIB = 1 << 20


class _FakeDynamicOp:
    """Callable FX target that mimics a registered AutoDeploy custom op."""

    def __init__(self, qualified_name: str, impl: Callable):
        self._name = qualified_name
        self._impl = impl
        short_name = qualified_name.split("::")[-1].replace(".", "_")
        self.__name__ = short_name
        self.__qualname__ = short_name
        self.__module__ = __name__

    def name(self):
        return self._name

    def __call__(self, *args, **kwargs):
        return self._impl(*args, **kwargs)


def _out_buffer_dynamic_op(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    result = x + 1
    if out is not None:
        out.copy_(result)
        return torch.empty((0,), dtype=x.dtype, device=x.device)
    return result


def _metadata_dynamic_op(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    metadata = torch.arange(x.shape[0], dtype=torch.int32, device=x.device)
    return metadata, metadata.view(1, -1)


_OUT_BUFFER_TARGET = _FakeDynamicOp(
    "auto_deploy::trtllm_attention_mha_with_cache",
    _out_buffer_dynamic_op,
)
_METADATA_TARGET = _FakeDynamicOp(
    "auto_deploy::flashinfer_attention_prepare_metadata",
    _metadata_dynamic_op,
)


def _large_output_from_tensor(x: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return x.new_empty((x.shape[0], vocab_size))


def _large_output_from_metadata(
    metadata: tuple[torch.Tensor, torch.Tensor],
    x: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    return x.new_empty((metadata[0].shape[0], vocab_size))


def _cleanup_cuda() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _cuda_used_bytes() -> int:
    _cleanup_cuda()
    free_mem, total_mem = torch.cuda.mem_get_info()
    return total_mem - free_mem


def _prime_cuda_graph(device: torch.device) -> None:
    """Pay one-time CUDA graph allocator overhead before measuring this test."""
    static = torch.zeros(1, device=device)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        static.add_(1)
    torch.cuda.synchronize()

    del graph, static
    _cleanup_cuda()


def _module_contains_target(module: nn.Module, target_name: str) -> bool:
    graph = getattr(module, "graph", None)
    if graph is None:
        return False
    for node in graph.nodes:
        target = node.target
        name = target.name() if hasattr(target, "name") else str(target)
        if target_name in name:
            return True
    return False


def _find_dynamic_submodule(pcg: PiecewiseCapturedGraph, target_name: str) -> tuple[int, nn.Module]:
    assert pcg.split_info is not None
    assert pcg.split_gm is not None
    for idx in pcg.split_info.dynamic_submod_indices:
        module = getattr(pcg.split_gm, f"submod_{idx}")
        inner = (
            module.submodule if isinstance(module, (DynamicOpWrapper, MetadataWrapper)) else module
        )
        if _module_contains_target(inner, target_name):
            return idx, module
    raise AssertionError(f"Dynamic submodule containing {target_name} was not found")


def _assert_trailing_static_partition_is_eager(pcg: PiecewiseCapturedGraph) -> None:
    assert pcg.split_info is not None
    assert pcg.split_gm is not None
    last_dynamic_idx = max(pcg.split_info.dynamic_submod_indices)
    trailing_static_indices = [
        idx for idx in pcg.split_info.static_submod_indices if idx > last_dynamic_idx
    ]
    assert len(trailing_static_indices) == 1

    trailing = getattr(pcg.split_gm, f"submod_{trailing_static_indices[0]}")
    assert not isinstance(trailing, ADPiecewiseRunner)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _assert_static_runners_captured_with_graph_pool(
    pcg: PiecewiseCapturedGraph,
    bucket: int,
) -> None:
    assert pcg._static_runners

    pools = [runner.graph_pool for runner in pcg._static_runners.values()]
    assert all(pool is not None for pool in pools)
    assert len({pool for pool in pools}) == 1

    for runner in pcg._static_runners.values():
        entry = runner.entries.get(bucket)
        assert entry is not None
        assert entry.cuda_graph is not None


def _assert_no_dynamic_out_buffers(pcg: PiecewiseCapturedGraph) -> None:
    for runner in pcg._static_runners.values():
        for entry in runner.entries.values():
            assert entry.dynamic_out_bufs == {}


def _build_piecewise_memory_model(
    policy: str,
    hidden_size: int,
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> GraphModule:
    root = nn.Module()
    root.add_module(
        "proj",
        nn.Linear(hidden_size, hidden_size, bias=False).to(device=device, dtype=dtype),
    )

    graph = Graph()
    x = graph.placeholder("x")
    y = graph.call_module("proj", args=(x,))

    if policy == "out_buffer":
        dynamic = graph.create_node("call_function", _OUT_BUFFER_TARGET, args=(y,), name="attn")
        out = graph.call_function(_large_output_from_tensor, args=(dynamic, vocab_size))
    elif policy == "metadata_wrapper":
        metadata = graph.create_node("call_function", _METADATA_TARGET, args=(y,), name="metadata")
        out = graph.call_function(_large_output_from_metadata, args=(metadata, y, vocab_size))
    else:
        raise AssertionError(f"Unexpected policy: {policy}")

    graph.output(out)
    gm = GraphModule(root, graph)
    gm.eval()
    return gm


@pytest.fixture(autouse=True)
def _reset_piecewise_state():
    ADPiecewiseRunner.set_current_num_tokens(None)
    ADPiecewiseRunner.set_current_phase("replay")
    yield
    ADPiecewiseRunner.set_current_num_tokens(None)
    ADPiecewiseRunner.set_current_phase("replay")
    if torch.cuda.is_available():
        _cleanup_cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_piecewise_metadata_wrapper_buffers_are_bounded():
    """MetadataWrapper must not retain storage proportional to logits output."""
    device = torch.device("cuda")
    dtype = torch.float16
    hidden_size = 16
    bucket = 1024
    # Produces a 256 MiB logits-like output: large enough to stand above
    # normal CUDA graph-pool overhead, but still cheap because it is empty().
    vocab_size = 131072
    logits_output_bytes = bucket * vocab_size * torch.tensor([], dtype=dtype).element_size()

    _cleanup_cuda()
    free_mem, _ = torch.cuda.mem_get_info()
    if free_mem < 4 * logits_output_bytes:
        pytest.skip(
            "Not enough free GPU memory for the piecewise memory regression guard: "
            f"free={free_mem // _MIB} MiB"
        )

    model = _build_piecewise_memory_model(
        policy="metadata_wrapper",
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        device=device,
        dtype=dtype,
    )
    pcg = PiecewiseCapturedGraph(model, piecewise_num_tokens=[bucket])
    pcg.prepare()

    _, dynamic_submodule = _find_dynamic_submodule(pcg, "flashinfer_attention_prepare_metadata")
    assert isinstance(dynamic_submodule, MetadataWrapper)
    _assert_trailing_static_partition_is_eager(pcg)

    x = torch.randn(bucket, hidden_size, dtype=dtype, device=device)

    def get_args_kwargs(num_tokens: int):
        return (x[:num_tokens],), {}

    _prime_cuda_graph(device)
    before = _cuda_used_bytes()

    with torch.inference_mode():
        pcg.warmup_and_capture(get_args_kwargs, warmup_iters=2)

    after = _cuda_used_bytes()
    retained_bytes = max(0, after - before)
    assert retained_bytes < logits_output_bytes, (
        "Incremental piecewise CUDA graph capture retained a logits-sized allocation. "
        f"retained={retained_bytes // _MIB} MiB, "
        f"logits_output={logits_output_bytes // _MIB} MiB"
    )

    _assert_static_runners_captured_with_graph_pool(pcg, bucket)
    _assert_no_dynamic_out_buffers(pcg)

    stable_outputs = dynamic_submodule._stable_outputs[bucket]
    metadata_bytes = sum(_tensor_nbytes(t) for t in stable_outputs if isinstance(t, torch.Tensor))
    expected_metadata_bytes = 2 * bucket * torch.tensor([], dtype=torch.int32).element_size()
    assert metadata_bytes <= expected_metadata_bytes
    assert metadata_bytes < logits_output_bytes

    with torch.inference_mode():
        replay_out = pcg(x, num_tokens=bucket)
    assert replay_out.shape == (bucket, vocab_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_piecewise_out_buffer_uses_shared_graph_pool(monkeypatch):
    """OUT_BUFFER dynamic outputs are allocated by a captured runner graph pool."""
    device = torch.device("cuda")
    dtype = torch.float16
    hidden_size = 16
    bucket = 128
    vocab_size = 1024
    capture_pools = []
    original_cuda_graph = torch.cuda.graph

    @contextmanager
    def recording_cuda_graph(*args, **kwargs):
        pool = kwargs.get("pool")
        if len(args) > 1:
            pool = args[1]
        capture_pools.append(pool)
        with original_cuda_graph(*args, **kwargs):
            yield

    monkeypatch.setattr(torch.cuda, "graph", recording_cuda_graph)

    model = _build_piecewise_memory_model(
        policy="out_buffer",
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        device=device,
        dtype=dtype,
    )
    pcg = PiecewiseCapturedGraph(model, piecewise_num_tokens=[bucket])
    pcg.prepare()

    dynamic_idx, dynamic_submodule = _find_dynamic_submodule(pcg, "trtllm_attention_mha_with_cache")
    assert isinstance(dynamic_submodule, DynamicOpWrapper)
    assert dynamic_submodule.dynamic_submod_id == dynamic_idx
    _assert_trailing_static_partition_is_eager(pcg)

    x = torch.randn(bucket, hidden_size, dtype=dtype, device=device)

    def get_args_kwargs(num_tokens: int):
        return (x[:num_tokens],), {}

    with torch.inference_mode():
        pcg.warmup_and_capture(get_args_kwargs, warmup_iters=2)

    _assert_static_runners_captured_with_graph_pool(pcg, bucket)

    pools = {runner.graph_pool for runner in pcg._static_runners.values()}
    assert dynamic_submodule.preceding_runner.graph_pool in pools

    out_buf = dynamic_submodule.preceding_runner.get_dynamic_out_buf(bucket, dynamic_idx)
    assert out_buf is not None
    assert out_buf.shape == (bucket, hidden_size)
    assert out_buf.dtype == dtype

    assert capture_pools
    assert set(capture_pools) <= pools
    assert None not in capture_pools


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_piecewise_out_buffer_releases_strong_ref_after_capture(monkeypatch):
    """After finalize_capture(), OUT_BUFFER storage is held only by graph-pool lifetime."""
    device = torch.device("cuda")
    dtype = torch.float16
    hidden_size = 16
    bucket = 128
    vocab_size = 1024
    tensor_refs_by_ptr = {}

    def fake_make_weak_ref(value):
        if isinstance(value, torch.Tensor):
            ptr = value.data_ptr()
            tensor_refs_by_ptr[ptr] = weakref.ref(value)
            return ("weak_tensor", ptr, tuple(value.shape), value.dtype)
        return value

    monkeypatch.setattr(piecewise_runner_mod, "make_weak_ref", fake_make_weak_ref)

    model = _build_piecewise_memory_model(
        policy="out_buffer",
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        device=device,
        dtype=dtype,
    )
    pcg = PiecewiseCapturedGraph(model, piecewise_num_tokens=[bucket])
    pcg.prepare()

    dynamic_idx, dynamic_submodule = _find_dynamic_submodule(pcg, "trtllm_attention_mha_with_cache")
    assert isinstance(dynamic_submodule, DynamicOpWrapper)

    x = torch.randn(bucket, hidden_size, dtype=dtype, device=device)

    def get_args_kwargs(num_tokens: int):
        return (x[:num_tokens],), {}

    with torch.inference_mode():
        pcg.warmup_and_capture(get_args_kwargs, warmup_iters=2)

    entry = dynamic_submodule.preceding_runner.entries[bucket]
    weak_marker = entry.dynamic_out_bufs[dynamic_idx]
    assert weak_marker[0] == "weak_tensor"
    assert weak_marker[2] == (bucket, hidden_size)
    assert weak_marker[3] == dtype

    out_buf_ref = tensor_refs_by_ptr[weak_marker[1]]
    gc.collect()
    assert out_buf_ref() is None
