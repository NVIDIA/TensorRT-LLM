# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side control flow of dense FP4 MLA chunked prefill (cached context)."""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.attention.backends.trtllm as trtllm_backend
import tensorrt_llm._torch.attention.mla as mla_module
from tensorrt_llm._torch.attention.backends.fmha.fp4_mla import Fp4MlaFmha
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.pyexecutor.py_executor_creator import _validate_fp4_mla_chunk_size

pytestmark = pytest.mark.cpu_only

_NUM_HEADS = 2
_QK_NOPE = 4
_QK_ROPE = 2
_V_HEAD = 4
_KV_LORA_RANK = 8


def _fp4_attention() -> TrtllmAttention:
    attention = object.__new__(TrtllmAttention)
    attention.is_mla_enable = True
    attention.has_fp4_kv_cache = True
    attention.sparse_params = None
    attention.layer_idx = 3
    attention.mla_params = SimpleNamespace(
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE,
        qk_nope_head_dim=_QK_NOPE,
    )
    return attention


def _context_metadata(*, chunked_prefill: bool) -> TrtllmAttentionMetadata:
    metadata = object.__new__(TrtllmAttentionMetadata)
    metadata.kv_cache_manager = object()
    metadata.enable_context_mla_with_cached_kv = True
    metadata.num_ctx_cached_tokens = 256
    metadata.runtime_features = SimpleNamespace(chunked_prefill=chunked_prefill)
    return metadata


@pytest.mark.parametrize("chunked_prefill", [False, True])
def test_fp4_cached_context_dispatch(monkeypatch: pytest.MonkeyPatch, chunked_prefill: bool):
    class ReachedFp4ChunkedPrefill(Exception):
        pass

    def fp4_chunked_prefill(*args, **kwargs):
        raise ReachedFp4ChunkedPrefill

    def unexpected(*args, **kwargs):
        raise AssertionError("FP4 MLA must not reach the cached-KV C++ paths.")

    mla = SimpleNamespace(
        mha=_fp4_attention(),
        cached_warmup_forward_context_with_cached_kv=lambda *args: None,
        num_heads_tp=_NUM_HEADS,
        qk_nope_head_dim=_QK_NOPE,
        qk_rope_head_dim=_QK_ROPE,
        kv_lora_rank=_KV_LORA_RANK,
        v_head_dim=_V_HEAD,
        forward_context_with_chunked_prefill_fp4=fp4_chunked_prefill,
        forward_context_with_chunked_prefill=unexpected,
        forward_context_with_cached_kv=unexpected,
        forward_context_default=unexpected,
    )
    q = torch.empty(1, _NUM_HEADS * (_QK_NOPE + _QK_ROPE), dtype=torch.bfloat16)
    args = (q, None, None, None, _context_metadata(chunked_prefill=chunked_prefill), None)
    if chunked_prefill:
        with pytest.raises(ReachedFp4ChunkedPrefill):
            MLA.forward_context(mla, *args, latent_cache=torch.empty(0))
    else:
        with pytest.raises(NotImplementedError, match="enable_chunked_prefill"):
            MLA.forward_context(mla, *args, latent_cache=torch.empty(0))


def test_fp4_chunk_gather_dispatches_to_fp4_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def gather(metadata, layer_idx, **kwargs):
        captured.update(kwargs, layer_idx=layer_idx)
        return "kv", "k_pe"

    monkeypatch.setattr(trtllm_backend, "load_fp4_mla_chunked_kv_cache", gather)
    metadata = SimpleNamespace(kv_cache_manager=object())
    result = _fp4_attention().load_chunked_kv_cache_for_mla(
        metadata,
        num_ctx_cached_tokens=5,
        cu_chunked_seq_len="cu",
        chunked_global_offset="offset",
        chunked_max_seq_len=3,
        out_dtype=torch.bfloat16,
    )
    assert result == ("kv", "k_pe")
    assert captured == dict(
        layer_idx=3,
        num_ctx_cached_tokens=5,
        cu_chunked_seq_len="cu",
        chunked_global_offset="offset",
        chunked_max_seq_len=3,
        out_dtype=torch.bfloat16,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE,
    )


def test_fp4_context_partition_uses_fp4_mla_fmha(monkeypatch: pytest.MonkeyPatch) -> None:
    def partition(self, q, k, v, metadata, forward_args, *, kv_lens_cuda, kv_lens_cpu):
        return ("partition", q, kv_lens_cuda, kv_lens_cpu)

    monkeypatch.setattr(Fp4MlaFmha, "forward_context_partition", partition)
    attention = _fp4_attention()
    attention._fmha_manager = SimpleNamespace(fmha_libs=[object(), object.__new__(Fp4MlaFmha)])
    assert attention.forward_fp4_mla_context_partition(
        "q", "k", "v", None, None, kv_lens_cuda="lens", kv_lens_cpu="host_lens"
    ) == ("partition", "q", "lens", "host_lens")

    attention._fmha_manager = SimpleNamespace(fmha_libs=[object()])
    with pytest.raises(RuntimeError, match="Fp4MlaFmha"):
        attention.forward_fp4_mla_context_partition(
            "q", "k", "v", None, None, kv_lens_cuda="lens", kv_lens_cpu="host_lens"
        )


def _bind_cached_kv_context_hooks(mla: SimpleNamespace) -> SimpleNamespace:
    """Give a stub module the base MLA cached-KV context hooks."""
    mla._expand_kv_to_k_v = lambda *args: MLA._expand_kv_to_k_v(mla, *args)
    mla._cached_kv_context_inputs = lambda q: MLA._cached_kv_context_inputs(mla, q)
    return mla


def test_fp4_chunked_prefill_merges_cached_partitions_then_current_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mla_module, "maybe_compiled_cat", torch.cat)
    num_ctx_tokens = 3
    calls = []
    rotated_k_pe = 7.0

    def append(q, latent_cache, metadata):
        calls.append(("append",))
        # The fused FP4 scatter rotates the current K in place.
        latent_cache[:, _KV_LORA_RANK:] = rotated_k_pe

    def load(*, metadata, num_ctx_cached_tokens, chunked_max_seq_len, out_dtype, **kwargs):
        calls.append(("load", num_ctx_cached_tokens, chunked_max_seq_len))
        return (
            torch.zeros(num_ctx_cached_tokens, _KV_LORA_RANK),
            torch.zeros(num_ctx_cached_tokens, _QK_ROPE),
        )

    def partition(q, k, v, metadata, forward_args, *, kv_lens_cuda, kv_lens_cpu):
        assert forward_args.attention_input_type == AttentionInputType.context_only
        assert forward_args.latent_cache is None
        assert forward_args.softmax_stats_tensor is mla.temp_softmax_stats_tensor
        assert k.shape[0] == v.shape[0] == int(kv_lens_cpu.sum())
        # The base module leaves Q / K / V conversion to the attention op.
        assert forward_args.quant_q_buffer is None
        k_pe = k.view(-1, _NUM_HEADS, _QK_NOPE + _QK_ROPE)[..., _QK_NOPE:]
        calls.append(("partition", forward_args.attention_mask, kv_lens_cpu.tolist(), k_pe))

    def merge(attn_output, temp_output, stats, temp_stats, merge_op, metadata):
        assert stats is mla.softmax_stats_tensor and temp_stats is mla.temp_softmax_stats_tensor
        calls.append(("merge", merge_op.tolist()))

    mla = SimpleNamespace(
        mha=SimpleNamespace(
            append_fp4_mla_context_cache=append,
            load_chunked_kv_cache_for_mla=load,
            forward_fp4_mla_context_partition=partition,
            merge_attention_for_mla=merge,
        ),
        num_heads_tp=_NUM_HEADS,
        qk_nope_head_dim=_QK_NOPE,
        qk_rope_head_dim=_QK_ROPE,
        qk_head_dim=_QK_NOPE + _QK_ROPE,
        kv_lora_rank=_KV_LORA_RANK,
        v_head_dim=_V_HEAD,
        out_scale=None,
        kv_b_proj=lambda x: torch.zeros(x.shape[0], _NUM_HEADS * (_QK_NOPE + _V_HEAD)),
    )
    _bind_cached_kv_context_hooks(mla)
    # Two context requests with 4 + 2 cached tokens, gathered in two loops.
    chunked_seq_len = torch.tensor([[3, 2], [1, 0]], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_ctx_tokens=num_ctx_tokens,
        num_contexts=2,
        chunked_loop_num=2,
        cu_chunked_seq_len=torch.tensor([[0, 3, 5], [0, 1, 1]]),
        host_cu_chunked_seq_len=torch.tensor([[0, 3, 5], [0, 1, 1]]),
        chunked_global_offset=torch.tensor([[0, 0], [3, 2], [4, 2]]),
        max_chunk_len_per_loop=[3, 1],
        chunked_seq_len=chunked_seq_len,
        host_chunked_seq_len=chunked_seq_len,
        merge_op_tensor=torch.tensor([[2, 2], [1, 0], [1, 1]]),
        prompt_lens_cuda_runtime=torch.tensor([2, 1], dtype=torch.int32),
        prompt_lens_cpu_runtime=torch.tensor([2, 1], dtype=torch.int32),
        runtime_features=SimpleNamespace(chunked_prefill_buffer_batch_size=1),
    )
    q = torch.zeros(num_ctx_tokens, _NUM_HEADS * (_QK_NOPE + _QK_ROPE))
    output = torch.empty(num_ctx_tokens, _NUM_HEADS * _V_HEAD)
    latent_cache = torch.zeros(num_ctx_tokens, _KV_LORA_RANK + _QK_ROPE)

    result = MLA.forward_context_with_chunked_prefill_fp4(
        mla,
        q,
        torch.zeros(num_ctx_tokens, _KV_LORA_RANK),
        latent_cache,
        metadata,
        output,
    )

    assert result is output
    assert [call[0] for call in calls] == [
        "append",
        "load",
        "partition",
        "merge",
        "load",
        "partition",
        "merge",
        "partition",
        "merge",
    ]
    assert calls[1][1:] == (5, 3) and calls[4][1:] == (1, 1)
    masks = [call[1] for call in calls if call[0] == "partition"]
    assert masks == [
        PredefinedAttentionMask.FULL,
        PredefinedAttentionMask.FULL,
        PredefinedAttentionMask.CAUSAL,
    ]
    partitions = [call for call in calls if call[0] == "partition"]
    assert [call[2] for call in partitions] == [[3, 2], [1, 0], [2, 1]]
    # The current chunk attends with its rotated K from latent_cache.
    assert bool((partitions[-1][3] == rotated_k_pe).all())
    assert [call[1] for call in calls if call[0] == "merge"] == [[2, 2], [1, 0], [1, 1]]


def test_fp4_chunked_prefill_hands_every_partition_the_cached_kv_context_inputs() -> None:
    """``_cached_kv_context_inputs`` is called once, after the FP4 append rotated Q, and its
    Q buffer and KV expansion serve every cached partition and the current chunk."""
    num_ctx_tokens = 3
    events = []
    q_fp8 = torch.ones(1)

    def append(q, latent_cache, metadata):
        events.append("append")
        q.fill_(5.0)

    def load(*, metadata, num_ctx_cached_tokens, **kwargs):
        return (
            torch.zeros(num_ctx_cached_tokens, _KV_LORA_RANK),
            torch.zeros(num_ctx_cached_tokens, _QK_ROPE),
        )

    def inputs(q):
        # RoPE has been applied before the Q buffer is built.
        assert bool((q == 5.0).all())
        events.append("inputs")
        return q_fp8, expand

    def expand(compressed_kv, k_pe):
        events.append(("expand", compressed_kv.shape[0], k_pe.shape[0]))
        num_tokens = compressed_kv.shape[0]
        return (
            torch.zeros(num_tokens, _NUM_HEADS * (_QK_NOPE + _QK_ROPE)),
            torch.zeros(num_tokens, _NUM_HEADS * _V_HEAD),
        )

    def partition(q, k, v, metadata, forward_args, *, kv_lens_cuda, kv_lens_cpu):
        assert forward_args.quant_q_buffer is q_fp8
        events.append(("partition", int(kv_lens_cpu.sum())))

    mla = SimpleNamespace(
        mha=SimpleNamespace(
            append_fp4_mla_context_cache=append,
            load_chunked_kv_cache_for_mla=load,
            forward_fp4_mla_context_partition=partition,
            merge_attention_for_mla=lambda *args: None,
        ),
        _cached_kv_context_inputs=inputs,
        num_heads_tp=_NUM_HEADS,
        qk_nope_head_dim=_QK_NOPE,
        qk_rope_head_dim=_QK_ROPE,
        qk_head_dim=_QK_NOPE + _QK_ROPE,
        kv_lora_rank=_KV_LORA_RANK,
        v_head_dim=_V_HEAD,
        out_scale=None,
    )
    chunked_seq_len = torch.tensor([[3, 2]], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_ctx_tokens=num_ctx_tokens,
        num_contexts=2,
        chunked_loop_num=1,
        cu_chunked_seq_len=torch.tensor([[0, 3, 5]]),
        host_cu_chunked_seq_len=torch.tensor([[0, 3, 5]]),
        chunked_global_offset=torch.tensor([[0, 0], [3, 2]]),
        max_chunk_len_per_loop=[3],
        chunked_seq_len=chunked_seq_len,
        host_chunked_seq_len=chunked_seq_len,
        merge_op_tensor=torch.tensor([[2, 2], [1, 1]]),
        prompt_lens_cuda_runtime=torch.tensor([2, 1], dtype=torch.int32),
        prompt_lens_cpu_runtime=torch.tensor([2, 1], dtype=torch.int32),
        runtime_features=SimpleNamespace(chunked_prefill_buffer_batch_size=1),
    )
    MLA.forward_context_with_chunked_prefill_fp4(
        mla,
        torch.zeros(num_ctx_tokens, _NUM_HEADS * (_QK_NOPE + _QK_ROPE)),
        torch.zeros(num_ctx_tokens, _KV_LORA_RANK),
        torch.zeros(num_ctx_tokens, _KV_LORA_RANK + _QK_ROPE),
        metadata,
        torch.empty(num_ctx_tokens, _NUM_HEADS * _V_HEAD),
    )
    assert events == [
        "append",
        "inputs",
        ("expand", 5, 5),
        ("partition", 5),
        ("expand", 3, 3),
        ("partition", 3),
    ]


def test_fp4_mla_chunked_prefill_requires_tile_aligned_chunks() -> None:
    _validate_fp4_mla_chunk_size(
        uses_fp4_mla=True, enable_chunked_prefill=True, max_num_tokens=8192
    )
    _validate_fp4_mla_chunk_size(
        uses_fp4_mla=True, enable_chunked_prefill=False, max_num_tokens=8191
    )
    _validate_fp4_mla_chunk_size(
        uses_fp4_mla=False, enable_chunked_prefill=True, max_num_tokens=8191
    )
    with pytest.raises(ValueError, match="multiple of 16"):
        _validate_fp4_mla_chunk_size(
            uses_fp4_mla=True, enable_chunked_prefill=True, max_num_tokens=8191
        )
