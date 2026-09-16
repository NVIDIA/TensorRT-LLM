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

"""Dispatch contract of ``FallbackFmha.is_supported``.

The native op silently drops requests it cannot serve, so the fallback has to
refuse them itself rather than lower them and read back an untouched output.
"""

from __future__ import annotations

import dataclasses
import inspect
from contextlib import nullcontext
from threading import Thread
from types import SimpleNamespace

import pytest
import torch
from fmha_test_utils import FakeAttention, make_fake_metadata, make_fmha_forward_args
from torch._dynamo.testing import CompileCounter

from tensorrt_llm._torch.attention.backends.fmha import fallback as fallback_backend
from tensorrt_llm._torch.attention.backends.fmha.combined import CombinedFmha
from tensorrt_llm._torch.attention.backends.fmha.fallback import (
    FallbackFmha,
    _AttentionOpCache,
    _legacy_forward_args,
    _set_context_workspace_shape,
)
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaParams, StaticAttentionConfig
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm.functional import AttentionMaskType


@pytest.mark.parametrize(
    ("is_cross", "update_kv_cache", "expected"),
    (
        (False, False, False),
        (False, True, True),
        (True, False, True),
    ),
)
def test_fallback_support_matches_thop_kv_update_contract(is_cross, update_kv_cache, expected):
    """Do not dispatch requests that the native attention op rejects."""
    fmha = object.__new__(FallbackFmha)
    metadata = SimpleNamespace(is_cross=is_cross)
    forward_args = make_fmha_forward_args(update_kv_cache=update_kv_cache)

    assert fmha.is_supported(None, None, None, metadata, forward_args) is expected


def test_fallback_rejects_raw_fp8_input():
    """Do not dispatch raw FP8 QKV to the native attention op."""
    fmha = object.__new__(FallbackFmha)
    metadata = SimpleNamespace(is_cross=False)
    forward_args = make_fmha_forward_args(update_kv_cache=True)
    q = torch.empty((1, 128), dtype=torch.float8_e4m3fn)

    assert not fmha.is_supported(q, None, None, metadata, forward_args)


def test_context_workspace_shape_uses_active_batch_extents():
    params = FmhaParams(max_num_requests=2048, max_context_length=8192)

    _set_context_workspace_shape(
        params,
        num_contexts=1,
        num_ctx_tokens=1,
        host_context_lengths=torch.tensor([1], dtype=torch.int32),
        host_past_key_value_lengths=torch.tensor([0], dtype=torch.int32),
        host_total_kv_lens=torch.tensor([1], dtype=torch.int32),
        max_context_q_len_override=None,
    )

    assert params.num_seqs == 1
    assert params.num_requests == 1
    assert params.num_tokens == 1
    assert params.input_seq_length == 1
    assert params.max_past_kv_length == 0
    assert params.total_kv_len == 1


def test_context_workspace_shape_clears_context_for_generation_only():
    params = FmhaParams(
        num_seqs=2048,
        num_requests=2048,
        num_tokens=8192,
        input_seq_length=8192,
        max_past_kv_length=8192,
    )

    _set_context_workspace_shape(
        params,
        num_contexts=0,
        num_ctx_tokens=0,
        host_context_lengths=torch.empty(0, dtype=torch.int32),
        host_past_key_value_lengths=torch.empty(0, dtype=torch.int32),
        host_total_kv_lens=torch.empty(0, dtype=torch.int32),
        max_context_q_len_override=None,
    )

    assert params.num_seqs == 0
    assert params.num_requests == 0
    assert params.num_tokens == 0
    assert params.input_seq_length == 0
    assert params.max_past_kv_length == 0
    assert params.total_kv_len == 0


def test_prepare_workspace_keeps_scheduler_and_multi_ctas_counters_separate(monkeypatch):
    scheduler_counter = torch.empty(1, dtype=torch.uint32)
    multi_ctas_kv_counter = torch.empty(64, dtype=torch.uint8)
    allocations = []

    def allocate_counter(counter, device, num_heads, max_num_sequences):
        allocations.append((counter, device, num_heads, max_num_sequences))
        return multi_ctas_kv_counter

    class AttentionOp:
        def get_attention_workspace_size(self, *args):
            return 0

    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.fmha.fallback.get_multi_ctas_kv_counter",
        allocate_counter,
    )
    attn = FakeAttention()
    attn.num_heads = 8
    fmha = FallbackFmha(attn)
    fmha.attention_op = lambda params: AttentionOp()
    q = torch.empty((1, 32))
    workspace = torch.empty(0, dtype=torch.uint8)
    forward_args = make_fmha_forward_args(
        output=torch.empty_like(q),
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=128,
        fmha_scheduler_counter=scheduler_counter,
    )
    metadata = make_fake_metadata(
        num_generations=1,
        num_contexts=0,
        num_ctx_tokens=0,
        max_num_sequences=4,
        max_num_requests=2,
        host_total_kv_lens=torch.tensor([0, 1], dtype=torch.int32),
        effective_beam_width=1,
    )

    fmha.prepare_workspace(q, None, None, metadata, forward_args, workspace)

    assert allocations == [(None, q.device, 8, 4)]
    # Execution gets fresh parameters, not the private sizing parameters.
    params = fmha._build_params(q, None, None, metadata, forward_args, workspace)
    assert params.multi_ctas_kv_counter is None
    fmha._to_thop_params(params)
    assert params.fwd.fmha_scheduler_counter is scheduler_counter
    assert params.multi_ctas_kv_counter is multi_ctas_kv_counter


@pytest.fixture
def native_ops(monkeypatch):
    created = []

    class AttentionOp:
        def __init__(self, config):
            self.config = config
            self.calls = []
            created.append(self)

        def get_attention_workspace_size(self, params, *args):
            return 0

        def run_context(self, params):
            self.calls.append(params)

        def run_generation(self, params):
            self.calls.append(params)

    monkeypatch.setattr(StaticAttentionConfig, "to_thop_config", lambda self: self)
    monkeypatch.setattr(fallback_backend.thop, "AttentionOp", AttentionOp)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    fallback_backend._compat_attention_ops.clear()
    yield created
    fallback_backend._compat_attention_ops.clear()


def test_attention_op_cache_separates_static_configurations(native_ops):
    attn = FakeAttention()
    fallback = FallbackFmha(attn)
    params = FmhaParams(
        fwd=make_fmha_forward_args(),
        qkv_or_q=torch.empty((1, 12), dtype=torch.bfloat16),
        output=torch.empty((1, 4), dtype=torch.bfloat16),
        num_heads=1,
        num_kv_heads=1,
        head_size=4,
    )
    causal_op = fallback.attention_op(params)
    assert fallback.attention_op(params) is causal_op
    # Dynamic buffers and shapes are not part of the runner's static configuration.
    dynamic = dataclasses.replace(
        params, qkv_or_q=torch.empty((2, 12), dtype=torch.bfloat16), num_tokens=2
    )
    assert fallback.attention_op(dynamic) is causal_op
    full = dataclasses.replace(
        params, fwd=make_fmha_forward_args(attention_mask=PredefinedAttentionMask.FULL)
    )
    assert fallback.attention_op(full) is not causal_op
    assert FallbackFmha(attn).attention_op(params) is not causal_op
    assert len(native_ops) == 3
    fallback.release()
    assert fallback.attention_op(params) is not causal_op


def test_attention_op_cache_isolates_devices_and_threads(native_ops, monkeypatch):
    cache = _AttentionOpCache()
    config = StaticAttentionConfig(num_heads=8)
    first = cache.get(config, torch.device("cuda:0"))
    assert cache.get(config, torch.device("cuda:0")) is first
    assert cache.get(config, torch.device("cuda:1")) is not first
    other_thread = Thread()
    monkeypatch.setattr(fallback_backend, "current_thread", lambda: other_thread)
    assert cache.get(config, torch.device("cuda:0")) is not first
    assert len(native_ops) == 3


def test_compat_attention_caches_static_args_but_not_inputs(native_ops):
    # Supply neutral values for the legacy signature's unrelated optional features.
    kwargs = {
        name: {bool: False, int: 0, float: 1.0}.get(parameter.annotation)
        for name, parameter in inspect.signature(FallbackFmha.attention).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    kwargs.update(
        output=torch.empty((1, 4), dtype=torch.bfloat16),
        workspace=torch.empty(0, dtype=torch.uint8),
        sequence_length=torch.tensor([1], dtype=torch.int32),
        host_past_key_value_lengths=torch.tensor([0], dtype=torch.int32),
        host_total_kv_lens=torch.tensor([1, 0], dtype=torch.int32),
        context_lengths=torch.tensor([1], dtype=torch.int32),
        host_context_lengths=torch.tensor([1], dtype=torch.int32),
        kv_cache_block_offsets=torch.zeros((1, 1, 2, 4), dtype=torch.int32),
        host_kv_cache_pool_pointers=torch.zeros((1, 2), dtype=torch.int64),
        host_kv_cache_pool_mapping=torch.zeros((1, 2), dtype=torch.int32),
        num_heads=1,
        num_kv_heads=1,
        head_size=4,
        tokens_per_block=32,
        max_num_requests=1,
        max_context_length=128,
        max_seq_len=128,
        attention_window_size=128,
        mask_type=AttentionMaskType.causal,
        num_contexts=1,
        num_ctx_tokens=1,
        beam_width=1,
        predicted_tokens_per_seq=1,
        q=torch.zeros((1, 12), dtype=torch.bfloat16),
    )
    FallbackFmha.attention(**kwargs)
    first = native_ops[0]
    kwargs["q"] = torch.ones((1, 12), dtype=torch.bfloat16)
    kwargs["host_past_key_value_lengths"] = torch.tensor([4], dtype=torch.int32)
    FallbackFmha.attention(**kwargs)
    assert len(native_ops) == 1
    assert len(first.calls) == 2
    assert first.calls[0] is not first.calls[1]
    assert first.calls[1].qkv_or_q.data_ptr() == kwargs["q"].data_ptr()
    assert first.calls[1].max_past_kv_length == 4
    assert first.calls[1].local_layer_idx == 0
    kwargs["mask_type"] = AttentionMaskType.padding
    FallbackFmha.attention(**kwargs)
    assert len(native_ops) == 2
    assert native_ops[-1].config.mask_type == AttentionMaskType.padding
    kwargs["output"] = torch.empty((1, 4), dtype=torch.float8_e4m3fn)
    FallbackFmha.attention(**kwargs)
    assert len(native_ops) == 3
    assert native_ops[-1].config.is_fp8_out

    kwargs.update(
        output=torch.empty((1, 4), dtype=torch.bfloat16),
        mask_type=AttentionMaskType.causal,
        attention_input_type=AttentionInputType.generation_only,
        num_ctx_tokens=0,
        # Decode-only Q, but metadata still has one prefill before decode.
        host_context_lengths=torch.tensor([16, 1], dtype=torch.int32),
        host_past_key_value_lengths=torch.tensor([0, 4], dtype=torch.int32),
        host_total_kv_lens=torch.tensor([16, 5], dtype=torch.int32),
        context_lengths=torch.tensor([16, 1], dtype=torch.int32),
        sequence_length=torch.tensor([16, 5], dtype=torch.int32),
    )
    FallbackFmha.attention(**kwargs)
    assert len(native_ops) == 3
    decode = first.calls[-1]
    assert decode.seq_offset == 1
    assert decode.num_seqs == 1
    assert decode.input_seq_length == 1
    assert decode.sequence_length.tolist() == [5]


def test_legacy_forward_args_preserves_nested_native_inputs():
    tensor = torch.empty(1)
    args = _legacy_forward_args(
        {
            "mask_type": AttentionMaskType.padding,
            "kv_scale_orig_quant": tensor,
            "out_scale": tensor,
            "latent_cache": tensor,
            "q_pe": tensor,
            "sage_attn_qk_int8": True,
            "sparse_attn_indices": tensor,
            "skip_softmax_threshold_scale_factor_prefill": 0.5,
            "chunked_prefill_buffer_batch_size": None,
        }
    )
    assert args.mask_type == AttentionMaskType.padding
    assert args.kv_scale_orig_quant is tensor
    assert args.out_scale is tensor
    assert args.latent_cache is tensor
    assert args.q_pe is tensor
    assert args.sage_attn_qk_int8
    assert args.sparse_runtime_params.sparse_attn_indices is tensor
    assert args.sparse_runtime_params.threshold_scale_factor_prefill == 0.5
    assert args.chunked_prefill_buffer_batch_size == 1


@pytest.mark.parametrize(
    ("method_name", "combined"),
    [
        ("prepare_workspace", False),
        ("run_context", False),
        ("run_mla_context", False),
        ("run_generation", False),
        ("run_mla_generation", False),
        ("prepare_workspace", True),
        ("run_context", True),
        ("run_generation", True),
    ],
)
def test_fallback_native_phases_stay_eager_under_compile(
    monkeypatch: pytest.MonkeyPatch, method_name: str, combined: bool
) -> None:
    torch._dynamo.reset()
    calls = []

    class AttentionOp:
        def get_attention_workspace_size(self, params: FmhaParams, *args: int) -> int:
            assert not torch.compiler.is_compiling()
            calls.append("prepare_workspace")
            return 16

        def run_context(self, params: FmhaParams) -> None:
            assert not torch.compiler.is_compiling()
            calls.append("run_context")
            params.output.copy_(params.qkv_or_q + 2)

        def run_generation(self, params: FmhaParams) -> None:
            assert not torch.compiler.is_compiling()
            calls.append("run_generation")
            params.output.copy_(params.qkv_or_q + 3)

        def run_mla_generation(self, params: FmhaParams) -> None:
            assert not torch.compiler.is_compiling()
            calls.append("run_mla_generation")
            params.output.copy_(params.qkv_or_q + 4)

    def to_thop_params(self: FallbackFmha, params: FmhaParams) -> FmhaParams:
        assert not torch.compiler.is_compiling()
        return params

    monkeypatch.setattr(FallbackFmha, "_to_thop_params", to_thop_params)
    op = AttentionOp()
    attn = FakeAttention()
    fallback = FallbackFmha(attn)
    fallback.attention_op = lambda params: op
    fmha = fallback
    if combined:
        fmha = CombinedFmha(attn)
        fmha.set_fmha_impls(fallback, fallback)
    method = getattr(fmha, method_name)
    workspace = torch.empty(0, dtype=torch.uint8)
    output = torch.empty((1, 4))
    params = FmhaParams(
        fwd=make_fmha_forward_args(output=output, attention_window_size=1),
        output=output,
        workspace=workspace,
        host_context_lengths=torch.tensor([1], dtype=torch.int32),
        host_past_key_value_lengths=torch.tensor([0], dtype=torch.int32),
        cyclic_attention_window_size=1,
        max_attention_window_size=1,
    )
    metadata = make_fake_metadata(
        num_contexts=1,
        num_ctx_tokens=1,
        host_total_kv_lens=torch.tensor([1, 0]),
        prompt_lens_cpu_runtime=params.host_context_lengths,
        kv_lens_runtime=params.host_past_key_value_lengths,
    )

    def invoke(x: torch.Tensor) -> torch.Tensor:
        params.qkv_or_q = x + 1
        if method_name == "prepare_workspace":
            method(params.qkv_or_q, None, None, metadata, params.fwd, workspace)
            return params.qkv_or_q + workspace.numel()
        method(params)
        return params.output + 1

    counter = CompileCounter()
    compiled = torch.compile(invoke, backend=counter, fullgraph=False)
    expected_method = "run_context" if method_name == "run_mla_context" else method_name
    offset = {
        "prepare_workspace": 17,
        "run_context": 4,
        "run_generation": 5,
        "run_mla_generation": 6,
    }[expected_method]
    for value in (0.0, 1.0):
        x = torch.full((1, 4), value)
        torch.testing.assert_close(compiled(x), x + offset)
    assert calls == [expected_method] * (
        4 if combined and method_name == "prepare_workspace" else 2
    )
    assert counter.frame_count >= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for native FMHA")
@pytest.mark.parametrize("backend", ["eager", "inductor", "cuda_graph"])
def test_fallback_compiled_context_matches_reference(backend: str, monkeypatch) -> None:
    """Exercise workspace sizing and native context execution without an outer custom op."""
    torch._dynamo.reset()
    torch.manual_seed(0)
    seq_len, num_heads, head_dim = 64, 4, 128
    attn = TrtllmAttention(
        layer_idx=0,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
    )
    fmha = FallbackFmha(attn)
    created = []
    native_op = fallback_backend.thop.AttentionOp

    def create_op(config):
        op = native_op(config)
        created.append(op)
        return op

    monkeypatch.setattr(fallback_backend.thop, "AttentionOp", create_op)
    if backend == "cuda_graph":
        monkeypatch.setattr(fallback_backend, "_compat_attention_ops", _AttentionOpCache())
        monkeypatch.setattr(fmha, "attention_op", fallback_backend._compat_attention_op)
    lengths = torch.tensor([seq_len], dtype=torch.int32)
    metadata = make_fake_metadata(
        num_contexts=1,
        num_ctx_tokens=seq_len,
        max_context_length=seq_len,
        max_seq_len=seq_len,
        effective_workspace=torch.empty(0, dtype=torch.uint8, device="cuda"),
        kv_lens_cuda_runtime=lengths.cuda(),
        kv_lens_runtime=torch.zeros_like(lengths),
        prompt_lens_cuda_runtime=lengths.cuda(),
        prompt_lens_cpu_runtime=lengths,
        host_total_kv_lens=torch.tensor([seq_len, 0], dtype=torch.int32),
    )

    def invoke(qkv: torch.Tensor) -> torch.Tensor:
        output = torch.empty((seq_len, num_heads * head_dim), dtype=qkv.dtype, device=qkv.device)
        fmha.forward(
            qkv,
            None,
            None,
            metadata,
            make_fmha_forward_args(
                output=output,
                attention_mask=PredefinedAttentionMask.FULL,
                attention_window_size=seq_len,
            ),
        )
        return output

    graphs = []
    if backend == "cuda_graph":

        def compiled(qkv):
            graph = torch.cuda.CUDAGraph()
            # invoke() was warmed on the default stream; capture switches streams.
            with torch.cuda.graph(graph):
                output = invoke(qkv)
            graph.replay()
            graphs.append(graph)
            return output
    else:
        compiled = torch.compile(invoke, backend=backend, fullgraph=False)
    for _ in range(2):
        q, k, v = [
            torch.randn((seq_len, num_heads, head_dim), dtype=torch.bfloat16, device="cuda")
            for _ in range(3)
        ]
        qkv = torch.cat([x.flatten(1) for x in (q, k, v)], dim=-1)
        reference = (
            torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1).float(), k.transpose(0, 1).float(), v.transpose(0, 1).float()
            )
            .transpose(0, 1)
            .reshape(seq_len, -1)
        )
        expected = invoke(qkv)
        actual = compiled(qkv)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual.float(), reference, rtol=2e-2, atol=2e-2)
    assert len(created) == 1
