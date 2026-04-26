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
    DEEPSEEK_V4_ATTENTION_CUDA_GRAPH_CONTRACT,
    DEEPSEEK_V4_CUDA_GRAPH_CACHE_RESOURCE_ARG_NAMES,
    DEEPSEEK_V4_CUDA_GRAPH_METADATA_ARG_NAMES,
    DEEPSEEK_V4_CUDA_GRAPH_WORKSPACE_ARG_NAME,
    DEEPSEEK_V4_CUDA_GRAPH_WORKSPACE_SIZE_ARG_NAME,
    DEEPSEEK_V4_ROUTE_METADATA_CUDA_GRAPH_CONTRACT,
    get_deepseek_v4_cuda_graph_kernel_contract,
    is_dynamic_cached_op,
    is_metadata_prep,
    needs_out_buffer,
    split_graph_at_dynamic_ops,
    submod_has_cuda_ops,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo, SequenceInfo
from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs
from tensorrt_llm._torch.utils import make_weak_ref

_REPO_ROOT = Path(__file__).resolve().parents[6]
_REGISTRY_CONFIGS_DIR = _REPO_ROOT / "examples" / "auto_deploy" / "model_registry" / "configs"
_DEEPSEEK_V4_CONFIG = _REGISTRY_CONFIGS_DIR / "deepseek_v4_flash.yaml"
_DUMMY_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_CONFIGURED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
_DSV4_LOCAL_HEADS = 8
_DSV4_HEAD_DIM = 512
_DSV4_ROPE_DIM = 64
_DSV4_WINDOW_SIZE = 128
_DSV4_SOFTMAX_SCALE = 0.04419417382415922
_DSV4_RMS_NORM_EPS = 1.0e-6

_SUPPORTED_TRITON_REPLAY_OUT_PTRS: list[int] = []


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


def triton_deepseek_v4_sparse_attention_v2_with_cache(
    x: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    if out is None:
        raise AssertionError("supported DSV4 Triton replay must receive caller-owned out=")
    _SUPPORTED_TRITON_REPLAY_OUT_PTRS.append(out.data_ptr())
    out.copy_(x + 2)
    return out.new_empty(0)


def deepseek_v4_prepare_cache_metadata(x: torch.Tensor) -> torch.Tensor:
    return x


def deepseek_v4_route_metadata_prep(x: torch.Tensor) -> torch.Tensor:
    return x


def torch_deepseek_v4_sparse_attention_v2_with_cache(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    freqs_cis_table: torch.Tensor,
    position_ids: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    softmax_scale: float,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: int,
    rms_norm_eps: float,
    rope_dim: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    del (
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        freqs_cis_table,
        position_ids,
        batch_info_host,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        softmax_scale,
        window_size,
        compress_ratio,
        max_compressed_len,
        rms_norm_eps,
        rope_dim,
    )
    y = q + 1
    if out is not None:
        out.copy_(y)
        return torch.empty(0, dtype=q.dtype, device=q.device)
    return y


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


def _make_dynamic_only_graph(dynamic_fn: Any = torch_deepseek_v4_sparse_attention) -> GraphModule:
    graph = Graph()
    x = graph.placeholder("x")
    dynamic = graph.call_function(dynamic_fn, args=(x,))
    graph.output(dynamic)
    graph.lint()
    return GraphModule(nn.Module(), graph)


def _make_cached_attention_graph() -> GraphModule:
    graph = Graph()
    placeholders = {
        name: graph.placeholder(name)
        for name in (
            "q",
            "kv",
            "attn_sink",
            "topk_idxs",
            "compressor_kv",
            "compressor_gate",
            "compressor_ape",
            "compressor_norm_weight",
            "freqs_cis_table",
            "position_ids",
            *DEEPSEEK_V4_CUDA_GRAPH_METADATA_ARG_NAMES,
            *DEEPSEEK_V4_CUDA_GRAPH_CACHE_RESOURCE_ARG_NAMES,
        )
    }
    q = graph.call_function(torch.relu, args=(placeholders["q"],))
    dynamic = graph.call_function(
        torch_deepseek_v4_sparse_attention_v2_with_cache,
        args=(
            q,
            placeholders["kv"],
            placeholders["attn_sink"],
            placeholders["topk_idxs"],
            placeholders["compressor_kv"],
            placeholders["compressor_gate"],
            placeholders["compressor_ape"],
            placeholders["compressor_norm_weight"],
            placeholders["freqs_cis_table"],
            placeholders["position_ids"],
            placeholders["batch_info_host"],
            placeholders["seq_len_host"],
            placeholders["input_pos_host"],
            placeholders["cu_seqlen_host"],
            placeholders["cache_loc_host"],
            placeholders["cu_num_pages_host"],
            placeholders["swa_cache"],
            placeholders["mhc_cache"],
            placeholders["compressor_kv_cache"],
            placeholders["compressor_gate_cache"],
            1.0,
            128,
            4,
            2048,
            1.0e-6,
            64,
        ),
    )
    graph.output(dynamic)
    graph.lint()
    return GraphModule(nn.Module(), graph)


class _FakeReplayRunner:
    def __init__(self, out_buf: torch.Tensor) -> None:
        self.out_buf = out_buf
        self.requests: list[tuple[int, int]] = []

    def get_dynamic_out_buf(self, num_tokens: int, dynamic_submod_id: int) -> torch.Tensor:
        self.requests.append((num_tokens, dynamic_submod_id))
        return self.out_buf


def _batch_info(num_decode: int) -> torch.Tensor:
    batch_info = BatchInfo()
    batch_info.update([0, 0, 0, 0, num_decode, num_decode])
    batch_info.update_tokens_gather_info(num_decode, False)
    return batch_info.serialize()


def _freqs_cis_table(max_position: int = 16) -> torch.Tensor:
    freqs = 1.0 / (
        10000.0 ** (torch.arange(0, _DSV4_ROPE_DIM, 2, dtype=torch.float32) / _DSV4_ROPE_DIM)
    )
    phases = torch.arange(max_position, dtype=torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
    return torch.polar(torch.ones_like(phases), phases)


def _make_cuda_ratio0_decode_args(
    batch_size: int = 2,
    active_sequences: int = 1,
    prefix_len: int = 1,
) -> list[Any]:
    device = torch.device("cuda")
    torch.manual_seed(4096 + batch_size + active_sequences + prefix_len)
    q = torch.randn(
        batch_size,
        1,
        _DSV4_LOCAL_HEADS,
        _DSV4_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    kv = torch.randn(batch_size, 1, _DSV4_HEAD_DIM, dtype=torch.bfloat16, device=device)
    attn_sink = torch.linspace(-0.5, 0.5, _DSV4_LOCAL_HEADS, dtype=torch.float32, device=device)
    topk_idxs = torch.full(
        (batch_size, 1, _DSV4_WINDOW_SIZE),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk_idxs[:active_sequences, 0, 0] = 0
    topk_idxs[:active_sequences, 0, 1] = prefix_len

    cache_loc_host = torch.arange(active_sequences, dtype=torch.int32, device=device)
    cu_num_pages_host = torch.arange(active_sequences + 1, dtype=torch.int32, device=device)
    swa_cache = torch.zeros(
        active_sequences,
        1,
        _DSV4_WINDOW_SIZE,
        1,
        _DSV4_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    mhc_cache = torch.zeros_like(swa_cache)
    compressor_cache = torch.empty(
        active_sequences,
        1,
        _DSV4_WINDOW_SIZE,
        1,
        0,
        dtype=torch.bfloat16,
        device=device,
    )
    if active_sequences:
        swa_cache[:, 0, 0, 0].normal_()

    return [
        q,
        kv,
        attn_sink,
        topk_idxs,
        torch.empty(batch_size, 1, 0, dtype=torch.bfloat16, device=device),
        torch.empty(batch_size, 1, 0, dtype=torch.bfloat16, device=device),
        torch.empty(0, 0, dtype=torch.bfloat16, device=device),
        torch.empty(0, dtype=torch.float32, device=device),
        _freqs_cis_table().to(device),
        torch.zeros(batch_size, 1, dtype=torch.int32, device=device),
        _batch_info(active_sequences).to(device),
        torch.full((active_sequences,), prefix_len + 1, dtype=torch.int32, device=device),
        torch.full((active_sequences,), prefix_len, dtype=torch.int32, device=device),
        torch.arange(active_sequences + 1, dtype=torch.int32, device=device),
        cache_loc_host,
        cu_num_pages_host,
        swa_cache,
        mhc_cache,
        compressor_cache,
        compressor_cache.clone(),
        _DSV4_SOFTMAX_SCALE,
        _DSV4_WINDOW_SIZE,
        0,
        None,
        _DSV4_RMS_NORM_EPS,
        _DSV4_ROPE_DIM,
    ]


@pytest.mark.parametrize(
    "op_name",
    [
        "auto_deploy::torch_deepseek_v4_sparse_attention",
        "auto_deploy::torch_deepseek_v4_sparse_attention_v2",
        "auto_deploy::torch_deepseek_v4_sparse_attention_v2_with_cache",
        "auto_deploy::triton_deepseek_v4_sparse_attention_with_cache",
        "auto_deploy::triton_deepseek_v4_sparse_attention_v2_with_cache",
        "auto_deploy::torch_deepseek_v4_moe",
        "auto_deploy::triton_mxfp4_moe",
        "auto_deploy::triton_mxfp4_moe_ep",
        "auto_deploy::deepseek_v4_prepare_cache_metadata",
        "auto_deploy::torch_deepseek_v4_prepare_cache_metadata",
        "auto_deploy::triton_deepseek_v4_prepare_cache_metadata",
        "auto_deploy::triton_deepseek_v4_prepare_indexer_metadata",
        "auto_deploy::triton_deepseek_v4_prepare_route_metadata",
        "auto_deploy::triton_deepseek_v4_route_metadata_prep",
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


def test_deepseek_v4_route_metadata_prep_is_dynamic_without_out_buffer() -> None:
    split_info = split_graph_at_dynamic_ops(_make_toy_graph(deepseek_v4_route_metadata_prep))
    metadata_submod = split_info.split_gm.submod_1

    assert is_metadata_prep(metadata_submod)
    assert not needs_out_buffer(metadata_submod)


def test_deepseek_v4_cuda_graph_contract_documents_buffers_and_workspaces() -> None:
    attention_contract = get_deepseek_v4_cuda_graph_kernel_contract(
        "auto_deploy::triton_deepseek_v4_sparse_attention_v2_with_cache"
    )
    route_contract = get_deepseek_v4_cuda_graph_kernel_contract(
        "auto_deploy::triton_deepseek_v4_route_metadata_prep"
    )

    assert attention_contract is DEEPSEEK_V4_ATTENTION_CUDA_GRAPH_CONTRACT
    assert attention_contract.uses_out_buffer
    assert attention_contract.workspace_arg_name == DEEPSEEK_V4_CUDA_GRAPH_WORKSPACE_ARG_NAME
    assert (
        attention_contract.workspace_size_arg_name == DEEPSEEK_V4_CUDA_GRAPH_WORKSPACE_SIZE_ARG_NAME
    )
    assert attention_contract.metadata_arg_names == DEEPSEEK_V4_CUDA_GRAPH_METADATA_ARG_NAMES
    assert attention_contract.cache_resource_arg_names == (
        DEEPSEEK_V4_CUDA_GRAPH_CACHE_RESOURCE_ARG_NAMES
    )

    assert route_contract is DEEPSEEK_V4_ROUTE_METADATA_CUDA_GRAPH_CONTRACT
    assert not route_contract.uses_out_buffer
    assert route_contract.workspace_arg_name == DEEPSEEK_V4_CUDA_GRAPH_WORKSPACE_ARG_NAME
    assert route_contract.workspace_size_arg_name == DEEPSEEK_V4_CUDA_GRAPH_WORKSPACE_SIZE_ARG_NAME


def test_deepseek_v4_out_buffer_convention_reuses_caller_buffer() -> None:
    dynamic_submod = _make_dynamic_only_graph()
    torch_cudagraph._inject_out_param(dynamic_submod)
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    out = torch.empty_like(x)

    result = dynamic_submod(x, out=out)

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, x + 1)


def test_deepseek_v4_triton_dynamic_replay_reuses_preallocated_out_buffer() -> None:
    _SUPPORTED_TRITON_REPLAY_OUT_PTRS.clear()
    dynamic_submod = _make_dynamic_only_graph(triton_deepseek_v4_sparse_attention_v2_with_cache)
    torch_cudagraph._inject_out_param(dynamic_submod)
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    out_buf = torch.empty_like(x)
    runner = _FakeReplayRunner(out_buf)
    wrapper = DynamicOpWrapper(dynamic_submod, preceding_runner=runner, dynamic_submod_id=7)

    ADPiecewiseRunner.set_current_phase("replay")
    ADPiecewiseRunner.set_current_num_tokens(4)
    try:
        result = wrapper(x)
    finally:
        ADPiecewiseRunner.set_current_num_tokens(None)

    assert runner.requests == [(4, 7)]
    assert _SUPPORTED_TRITON_REPLAY_OUT_PTRS == [out_buf.data_ptr()]
    assert result.data_ptr() == out_buf.data_ptr()
    torch.testing.assert_close(out_buf, x + 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_triton_attention_replay_uses_device_kernel_not_torch_reference(
    monkeypatch,
) -> None:
    pytest.importorskip("triton")

    from tensorrt_llm._torch.auto_deploy.custom_ops.attention import deepseek_v4_attention

    triton_sparse_attention = pytest.importorskip(
        "tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_deepseek_v4_sparse_attention"
    )
    deepseek_v4_ratio0_swa_triton_skip_reason = (
        triton_sparse_attention.deepseek_v4_ratio0_swa_triton_skip_reason
    )

    monkeypatch.delenv("TRTLLM_AD_DSV4_SPARSE_ATTENTION_FORCE_TORCH", raising=False)
    args = _make_cuda_ratio0_decode_args()
    out = torch.full_like(args[0], float("nan"))
    active_sequences = int(args[10][4].item())

    reason = deepseek_v4_ratio0_swa_triton_skip_reason(
        args[0],
        args[1],
        args[2],
        args[3],
        args[11],
        args[12],
        args[13],
        args[14],
        args[15],
        args[16],
        _DSV4_WINDOW_SIZE,
        0,
        out,
        active_sequences,
    )
    if reason is not None:
        pytest.skip(f"DSV4 ratio-0 Triton replay path is not locally supported: {reason}")

    def fail_reference_fallback(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("supported DSV4 Triton replay fell back to the Torch reference")

    monkeypatch.setattr(
        deepseek_v4_attention,
        "torch_deepseek_v4_sparse_attention_v2_with_cache",
        fail_reference_fallback,
    )

    result = torch.ops.auto_deploy.triton_deepseek_v4_sparse_attention_v2_with_cache.default(
        *args,
        out=out,
    )

    assert result.numel() == 0
    assert result.device == out.device
    assert not torch.isnan(out[:active_sequences]).any()
    torch.testing.assert_close(
        out[active_sequences:],
        torch.zeros_like(out[active_sequences:]),
    )


def test_deepseek_v4_cached_attention_keeps_cache_resources_as_graph_inputs() -> None:
    split_info = split_graph_at_dynamic_ops(_make_cached_attention_graph())
    dynamic_submod = getattr(split_info.split_gm, f"submod_{split_info.dynamic_submod_indices[0]}")
    placeholder_names = {
        node.target for node in dynamic_submod.graph.nodes if node.op == "placeholder"
    }
    get_attr_names = {node.target for node in dynamic_submod.graph.nodes if node.op == "get_attr"}

    assert set(DEEPSEEK_V4_CUDA_GRAPH_CACHE_RESOURCE_ARG_NAMES).issubset(placeholder_names)
    assert not set(DEEPSEEK_V4_CUDA_GRAPH_CACHE_RESOURCE_ARG_NAMES) & get_attr_names
    assert needs_out_buffer(dynamic_submod)


def test_deepseek_v4_config_parses_cuda_graph_batch_sizes() -> None:
    args = LlmArgs(model=_DUMMY_MODEL, yaml_extra=[str(_DEEPSEEK_V4_CONFIG)])

    assert args.enable_chunked_prefill
    assert args.cuda_graph_config.batch_sizes == _CONFIGURED_BATCH_SIZES
    assert args.cuda_graph_config.max_batch_size == max(_CONFIGURED_BATCH_SIZES)
    assert args.transforms["compile_model"]["cuda_graph_batch_sizes"] == _CONFIGURED_BATCH_SIZES
    assert not args.transforms["compile_model"]["piecewise_enabled"]
    assert not args.transforms["multi_stream_moe"]["enabled"]


def test_piecewise_setup_uses_rectangular_prefill_for_unflattened_sequence_info() -> None:
    seq_info = SequenceInfo(
        max_seq_len=256,
        max_batch_size=64,
        max_num_tokens=512,
        tokens_per_block=32,
    )

    torch_cudagraph._setup_piecewise_mixed_batch(seq_info, 512)
    named_args = seq_info.named_args

    assert named_args["input_ids"].shape == (2, 256)
    assert named_args["position_ids"].shape == (2, 256)
    assert seq_info.batch_info.get_num_sequences() == (2, 0, 0)
    assert seq_info.batch_info.get_num_tokens() == (512, 0, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_make_weak_ref_accepts_complex_cuda_rope_buffers() -> None:
    freqs_cis = torch.randn(16, 8, device="cuda", dtype=torch.complex64)

    weak_ref = make_weak_ref(freqs_cis)

    assert weak_ref.dtype == torch.complex64
    assert weak_ref.shape == freqs_cis.shape
    assert weak_ref.data_ptr() == freqs_cis.data_ptr()


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


class _FakePaddedPiecewise(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.original_model = nn.Identity()
        self.piecewise_num_tokens = [4]

    def forward(self, *args: Any, num_tokens: int | None = None, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        assert num_tokens is not None
        return torch.arange(num_tokens * 2, dtype=torch.float32).reshape(num_tokens, 2)


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


def test_piecewise_prefill_ignores_padded_bucket_output_slots() -> None:
    runtime = torch_cudagraph.DualModeCapturedGraph(_FakeMonolithic(), _FakePaddedPiecewise())

    result = runtime(
        input_ids=torch.ones(3, 1, dtype=torch.int64),
        batch_info_host=torch.tensor([1, 3, 0, 0, 0, 0], dtype=torch.int64),
    )

    assert result.shape == (3, 2)
    torch.testing.assert_close(
        result,
        torch.tensor(
            [
                [0.0, 1.0],
                [2.0, 3.0],
                [4.0, 5.0],
            ]
        ),
    )
