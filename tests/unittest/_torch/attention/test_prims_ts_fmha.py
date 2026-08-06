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

import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from packaging.version import Version

import tensorrt_llm._torch.attention_backend.fmha.prims_ts as prims_ts_module
from tensorrt_llm._torch.attention_backend.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention_backend.fmha.phased import FmhaParams
from tensorrt_llm._torch.attention_backend.fmha.prims_ts import PrimsTSFmha
from tensorrt_llm._torch.attention_backend.fmha.registry import get_enabled_fmha_lib_classes
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm.bindings import DataType


class _TensorSpec:
    """Minimal tensor-like object for the pure support predicate."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        device: str = "cuda",
        contiguous: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device(device)
        self.ndim = len(shape)
        self._contiguous = contiguous

    def is_contiguous(self) -> bool:
        return self._contiguous

    def numel(self) -> int:
        return math.prod(self.shape)


class _Attention:
    def __init__(
        self,
        *,
        head_dim: int = 128,
        is_mla: bool = False,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = (1 if is_mla else 2) if num_kv_heads is None else num_kv_heads
        self.head_dim = head_dim
        self.is_mla_enable = is_mla
        self.kv_lora_rank = 512 if is_mla else None
        self.qk_rope_head_dim = 64 if is_mla else None
        self.qk_nope_head_dim = 128 if is_mla else None
        self.v_head_dim = 128 if is_mla else None
        self.predicted_tokens_per_seq = 1
        self.sparse_params = None
        self.position_embedding_type = 0
        self.quant_mode = 0
        self.q_scaling = 1.0
        self.attention_chunk_size = 0
        self.rope_dim = head_dim
        self.local_layer_idx = 0
        self.rope_params = SimpleNamespace(
            dim=head_dim,
            theta=10000.0,
            scale_type=0,
            scale=1.0,
            max_positions=4096,
        )
        self.rotary_inv_freq = None
        self.rotary_cos_sin = None


def _support_result(
    *,
    attention_input_type: AttentionInputType,
    head_dim: int = 128,
    num_heads: int = 8,
    num_kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    output_dtype: torch.dtype | None = None,
    kv_dtype: DataType | None = None,
    tokens_per_block: int = 32,
    is_mla: bool = False,
    is_fused_qkv: bool = True,
    has_separate_kv: bool = False,
    has_paged_cache: bool = True,
    is_cross: bool = False,
    beam_width: int = 1,
    use_spec_decoding: bool = False,
    is_spec_dec_tree: bool = False,
    has_attention_sinks: bool = False,
    has_relative_attention_bias: bool = False,
    has_sparse_attention: bool = False,
    position_embedding_type: int = 0,
    kv_lora_rank: int | None = None,
    qk_rope_head_dim: int | None = None,
    has_output: bool = True,
    attention_window_size: int = 128,
    attention_chunk_size: int = 0,
    max_seq_len: int = 128,
    kv_layout: str = "HND",
    num_kv_cache_pools: int = 1,
    use_kv_cache_v2: bool = False,
    enable_swa_scratch_reuse: bool = False,
) -> tuple[bool, str]:
    attn = _Attention(
        head_dim=head_dim,
        is_mla=is_mla,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    attn.position_embedding_type = position_embedding_type
    attn.attention_chunk_size = attention_chunk_size
    if has_sparse_attention:
        attn.sparse_params = SimpleNamespace(algorithm="mqa_gqa")
    if kv_lora_rank is not None:
        attn.kv_lora_rank = kv_lora_rank
    if qk_rope_head_dim is not None:
        attn.qk_rope_head_dim = qk_rope_head_dim

    output_dtype = dtype if output_dtype is None else output_dtype
    if kv_dtype is None:
        kv_dtype = DataType.BF16 if dtype == torch.bfloat16 else DataType.HALF
    q_width = attn.num_heads * head_dim
    if is_fused_qkv and not is_mla:
        q_width += 2 * attn.num_kv_heads * head_dim
    q = _TensorSpec((4, q_width), dtype)
    output_width = attn.num_heads * (512 if is_mla else head_dim)
    forward_args = AttentionForwardArgs(
        output=_TensorSpec((4, output_width), output_dtype) if has_output else None,
        attention_input_type=attention_input_type,
        attention_mask=PredefinedAttentionMask.CAUSAL,
        attention_window_size=attention_window_size,
        attention_sinks=torch.empty(1) if has_attention_sinks else None,
        relative_attention_bias=torch.empty(1) if has_relative_attention_bias else None,
        is_fused_qkv=is_fused_qkv,
    )
    if attention_input_type == AttentionInputType.context_only:
        num_contexts, num_generations, num_ctx_tokens = 1, 0, 4
        kv_lens = [4]
    elif attention_input_type == AttentionInputType.generation_only:
        num_contexts, num_generations, num_ctx_tokens = 0, 4, 0
        kv_lens = [128, 96, 64, 32]
    else:
        num_contexts, num_generations, num_ctx_tokens = 1, 1, 3
        kv_lens = [3, 128]
    metadata = SimpleNamespace(
        helix_position_offsets=None,
        num_sparse_topk=0,
        use_spec_decoding=use_spec_decoding,
        is_spec_dec_tree=is_spec_dec_tree,
        is_spec_decoding_enabled=use_spec_decoding,
        kv_cache_block_offsets=torch.empty(1) if has_paged_cache else None,
        host_kv_cache_pool_pointers=torch.empty(1),
        host_kv_cache_pool_mapping=torch.zeros((1, 2), dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(
            dtype=kv_dtype,
            impl=(
                SimpleNamespace(get_page_index_upper_bound=lambda *args: 128)
                if use_kv_cache_v2
                else SimpleNamespace()
            ),
            enable_swa_scratch_reuse=enable_swa_scratch_reuse,
            num_local_layers=1,
            num_pools=num_kv_cache_pools,
        ),
        is_cross=is_cross,
        beam_width=beam_width,
        tokens_per_block=tokens_per_block,
        kv_layout=kv_layout,
        num_contexts=num_contexts,
        num_generations=num_generations,
        num_ctx_tokens=num_ctx_tokens,
        kv_lens_runtime=torch.tensor(kv_lens, dtype=torch.int32),
        max_seq_len=max_seq_len,
    )
    fmha = PrimsTSFmha(attn)
    fmha._get_kv_page_offset = Mock(return_value=1)
    k = _TensorSpec((4, attn.num_kv_heads * head_dim), dtype) if has_separate_kv else None
    v = _TensorSpec((4, attn.num_kv_heads * head_dim), dtype) if has_separate_kv else None
    return fmha._is_supported_with_reason(q, k, v, attn, metadata, forward_args)


@pytest.mark.parametrize(
    "case",
    [
        {
            "attention_input_type": AttentionInputType.context_only,
            "head_dim": 128,
        },
        {
            "attention_input_type": AttentionInputType.mixed,
            "head_dim": 256,
            "dtype": torch.float16,
        },
        {
            "attention_input_type": AttentionInputType.generation_only,
            "head_dim": 64,
            "dtype": torch.float16,
        },
        {
            "attention_input_type": AttentionInputType.generation_only,
            "head_dim": 576,
            "is_mla": True,
            "num_heads": 128,
        },
        {
            "attention_input_type": AttentionInputType.context_only,
            "num_kv_cache_pools": 2,
            "use_kv_cache_v2": True,
        },
    ],
    ids=["context", "mixed", "generation", "mla-generation", "v2-multi-pool"],
)
def test_supported_matrix(case: dict) -> None:
    supported, reason = _support_result(**case)

    assert supported, reason


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        (
            {"attention_input_type": AttentionInputType.context_only, "head_dim": 64},
            "context head dimension",
        ),
        (
            {"attention_input_type": AttentionInputType.generation_only, "head_dim": 96},
            "decode head dimension",
        ),
        (
            {"attention_input_type": AttentionInputType.context_only, "tokens_per_block": 8},
            "page size",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "has_paged_cache": False,
            },
            "paged KV-cache block offsets",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "is_fused_qkv": False,
                "has_separate_kv": True,
            },
            "only fused QKV",
        ),
        (
            {"attention_input_type": AttentionInputType.context_only, "is_cross": True},
            "cross attention",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "has_sparse_attention": True,
            },
            "sparse attention",
        ),
        (
            {
                "attention_input_type": AttentionInputType.generation_only,
                "use_spec_decoding": True,
            },
            "speculative decoding",
        ),
        (
            {
                "attention_input_type": AttentionInputType.generation_only,
                "is_spec_dec_tree": True,
                "use_spec_decoding": True,
            },
            "speculative decoding",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "has_attention_sinks": True,
            },
            "attention sinks",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "has_relative_attention_bias": True,
            },
            "relative attention bias",
        ),
        (
            {"attention_input_type": AttentionInputType.generation_only, "beam_width": 2},
            "beam search",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "position_embedding_type": 4,
            },
            "position embedding type",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "kv_dtype": DataType.HALF,
            },
            "query and KV-cache dtypes",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "output_dtype": torch.float16,
            },
            "output dtype must match",
        ),
        (
            {"attention_input_type": AttentionInputType.context_only, "has_output": False},
            "output tensor",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "num_heads": 7,
                "num_kv_heads": 2,
            },
            "divisible",
        ),
        (
            {
                "attention_input_type": AttentionInputType.generation_only,
                "num_heads": 64,
                "num_kv_heads": 1,
            },
            "GQA ratio",
        ),
        (
            {
                "attention_input_type": AttentionInputType.generation_only,
                "dtype": torch.float8_e4m3fn,
                "kv_dtype": DataType.FP8,
            },
            "query dtype",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "head_dim": 576,
                "is_mla": True,
            },
            "generation-only",
        ),
        (
            {
                "attention_input_type": AttentionInputType.generation_only,
                "head_dim": 576,
                "is_mla": True,
                "kv_lora_rank": 256,
            },
            "kv_lora_rank=512",
        ),
        (
            {
                "attention_input_type": AttentionInputType.generation_only,
                "head_dim": 576,
                "is_mla": True,
                "qk_rope_head_dim": 32,
            },
            "qk_rope_head_dim=64",
        ),
        (
            {
                "attention_input_type": AttentionInputType.generation_only,
                "head_dim": 576,
                "is_mla": True,
                "num_heads": 129,
            },
            "at most 128 local query heads",
        ),
        (
            {
                "attention_input_type": AttentionInputType.generation_only,
                "head_dim": 640,
                "is_mla": True,
            },
            "latent plus RoPE dimensions",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "attention_window_size": 64,
            },
            "cyclic TRT-LLM page tables",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "attention_window_size": 1,
            },
            "cyclic TRT-LLM page tables",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "attention_chunk_size": 64,
            },
            "chunked context attention",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "kv_layout": "NHD",
            },
            "HND KV-cache layout",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "num_kv_cache_pools": 2,
            },
            "V1 with multiple memory pools",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "use_kv_cache_v2": True,
                "enable_swa_scratch_reuse": True,
            },
            "V2 SWA scratch reuse",
        ),
    ],
    ids=[
        "context-head-dim",
        "generation-head-dim",
        "page-size",
        "no-paged-cache",
        "separate-qkv",
        "cross",
        "sparse",
        "spec-decode",
        "tree-mask",
        "sinks",
        "relative-bias",
        "beam-search",
        "alibi",
        "kv-dtype-mismatch",
        "output-dtype-mismatch",
        "missing-output",
        "heads-not-divisible",
        "generation-head-ratio",
        "fp8",
        "mla-context",
        "mla-kv-rank",
        "mla-rope-dim",
        "mla-too-many-heads",
        "mla-head-dim",
        "sliding-window",
        "one-token-window",
        "chunked-context",
        "nhd-cache",
        "v1-multi-pool",
        "v2-swa-scratch-reuse",
    ],
)
def test_unsupported_matrix_falls_through(case: dict, expected_reason: str) -> None:
    supported, reason = _support_result(**case)

    assert not supported
    assert expected_reason in reason


@pytest.mark.parametrize(
    ("sm", "cutlass_version", "compiler_version", "expected"),
    [
        (100, "4.7.0", "13.3", True),
        (103, "4.7.0", "13.3", True),
        (100, "4.7.0", "13.4", True),
        (100, "4.7.0", "13.2", False),
        (120, "4.7.0", "13.3", False),
        (100, "4.6.2", "13.3", False),
    ],
)
def test_static_availability_gate(
    monkeypatch: pytest.MonkeyPatch,
    sm: int,
    cutlass_version: str,
    compiler_version: str,
    expected: bool,
) -> None:
    target_version = Mock(
        side_effect=lambda *, min_version: Version(compiler_version) >= Version(min_version)
    )
    cutlass = SimpleNamespace(target_version=target_version)

    def import_cutlass_module(module_name: str) -> object:
        if module_name == "cutlass":
            return cutlass
        assert module_name == "cutlass.experimental.task_scheduling"
        return object()

    monkeypatch.setattr(prims_ts_module, "get_sm_version", lambda: sm)
    monkeypatch.setattr(prims_ts_module, "version", lambda _: cutlass_version)
    monkeypatch.setattr(prims_ts_module, "import_module", import_cutlass_module)
    monkeypatch.setattr(PrimsTSFmha, "_missing_fused_nanobind_ops", staticmethod(lambda: []))

    assert PrimsTSFmha.is_available(_Attention()) is expected
    if sm in (100, 103) and Version(cutlass_version) >= Version("4.7.0"):
        target_version.assert_called_once_with(min_version="13.3")
    else:
        target_version.assert_not_called()


def test_static_availability_gate_fails_closed_when_compiler_query_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cutlass = SimpleNamespace(target_version=Mock(side_effect=RuntimeError("query failed")))
    monkeypatch.setattr(prims_ts_module, "get_sm_version", lambda: 100)
    monkeypatch.setattr(prims_ts_module, "version", lambda _: "4.7.0")
    monkeypatch.setattr(
        prims_ts_module,
        "import_module",
        lambda module_name: cutlass if module_name == "cutlass" else object(),
    )
    monkeypatch.setattr(PrimsTSFmha, "_missing_fused_nanobind_ops", staticmethod(lambda: []))

    assert not PrimsTSFmha.is_available(_Attention())
    cutlass.target_version.assert_called_once_with(min_version="13.3")


def test_unsupported_cutlass_compiler_excludes_prims_ts_from_fmha_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_version = Mock(
        side_effect=lambda *, min_version: Version("13.2") >= Version(min_version)
    )
    cutlass = SimpleNamespace(target_version=target_version)
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts,fallback")
    monkeypatch.setattr(prims_ts_module, "get_sm_version", lambda: 100)
    monkeypatch.setattr(prims_ts_module, "version", lambda _: "4.7.0")
    monkeypatch.setattr(
        prims_ts_module,
        "import_module",
        lambda module_name: cutlass if module_name == "cutlass" else object(),
    )
    monkeypatch.setattr(PrimsTSFmha, "_missing_fused_nanobind_ops", staticmethod(lambda: []))

    available_classes = [
        fmha_cls
        for fmha_cls in get_enabled_fmha_lib_classes()
        if fmha_cls.is_available(_Attention())
    ]

    assert available_classes == [FallbackFmha]
    target_version.assert_called_once_with(min_version="13.3")


def test_v2_total_page_bound_is_not_expanded() -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    get_page_index_upper_bound = Mock(return_value=4096)
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            impl=SimpleNamespace(get_page_index_upper_bound=get_page_index_upper_bound),
            blocks_in_primary_pool=4096,
            num_local_layers=24,
        )
    )

    assert fmha._get_total_num_blocks(metadata) == 4096
    assert get_page_index_upper_bound.call_args.args[0] == 0


@pytest.mark.parametrize("is_mla", [False, True], ids=["standard", "mla"])
def test_v1_total_page_bound_excludes_slots_before_selected_layer(is_mla: bool) -> None:
    attn = _Attention(is_mla=is_mla)
    attn.local_layer_idx = 3
    fmha = PrimsTSFmha(attn)
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            impl=SimpleNamespace(),
            blocks_in_primary_pool=64,
            num_local_layers=4,
        ),
        host_kv_cache_pool_mapping=torch.tensor(
            [[0, 0], [0, 1], [0, 2], [0, 3]], dtype=torch.int32
        ),
    )

    kv_factor = 1 if is_mla else 2
    assert fmha._get_total_num_blocks(metadata) == 64 * 4 * kv_factor - 3 * kv_factor


def test_kv_page_offset_uses_v2_manager_displacement() -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(kv_offset=torch.tensor([0, 128])),
        host_kv_cache_pool_mapping=torch.tensor([[1, 0]], dtype=torch.int32),
    )

    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == 128


def test_kv_page_offset_is_inferred_from_v1_host_tables() -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    host_offsets = torch.tensor(
        [[[[0, 1, 2], [64, 65, 66]], [[3, 4, 5], [67, 68, 69]]]],
        dtype=torch.int32,
    )
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            kv_offset=None,
            host_kv_cache_block_offsets=host_offsets,
        ),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
    )

    assert fmha._get_kv_page_offset(fmha.attn, metadata, 1) == 64


def test_exact_context_csr_compacts_live_pages() -> None:
    block_tables = torch.tensor(
        [
            [[10, 11, 12, 13], [110, 111, 112, 113]],
            [[20, 21, 22, 23], [120, 121, 122, 123]],
        ],
        dtype=torch.int32,
    )

    indptr, indices, last_page_lengths = PrimsTSFmha._make_exact_context_csr(
        block_tables,
        torch.tensor([33, 64], dtype=torch.int32),
        32,
    )

    torch.testing.assert_close(indptr, torch.tensor([0, 2, 4], dtype=torch.int32))
    torch.testing.assert_close(indices, torch.tensor([10, 11, 20, 21], dtype=torch.int32))
    torch.testing.assert_close(last_page_lengths, torch.tensor([1, 32], dtype=torch.int32))


def test_fixed_stride_csr_reuses_stable_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    block_tables = torch.tensor(
        [
            [[10, 11, 12], [110, 111, 112]],
            [[20, 21, 22], [120, 121, 122]],
        ],
        dtype=torch.int32,
    )

    indptr, indices = fmha._make_fixed_stride_csr(block_tables, 2)
    first_storage = indices.data_ptr()
    torch.testing.assert_close(indptr, torch.tensor([0, 3, 6], dtype=torch.int32))
    torch.testing.assert_close(
        indices,
        torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int32),
    )

    block_tables[:, 0].add_(100)
    _, updated_indices = fmha._make_fixed_stride_csr(block_tables, 2)

    assert updated_indices.data_ptr() == first_storage
    torch.testing.assert_close(
        updated_indices,
        torch.tensor([110, 111, 112, 120, 121, 122], dtype=torch.int32),
    )


def test_mla_aligned_sequence_lengths_use_source_storage() -> None:
    fmha = PrimsTSFmha(_Attention(head_dim=576, is_mla=True))
    sequence_lengths = torch.tensor([33, 64], dtype=torch.int32)
    assert sequence_lengths.data_ptr() % 16 == 0

    actual = fmha._get_mla_sequence_lengths(sequence_lengths, 2)

    assert actual.data_ptr() == sequence_lengths.data_ptr()


def test_context_launcher_receives_v1_cache_views_and_exact_csr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 120
    q_processed = torch.empty((3, attn.num_heads, attn.head_dim), dtype=torch.bfloat16)
    kv_pool = torch.empty((12, attn.num_kv_heads, 32, attn.head_dim), dtype=torch.bfloat16)
    block_tables = torch.tensor(
        [
            [[0, 1, 2, 3], [6, 7, 8, 9]],
            [[2, 3, 4, 5], [8, 9, 10, 11]],
        ],
        dtype=torch.int32,
    )
    cu_q_seqlens = torch.tensor([0, 1, 3], dtype=torch.int32)
    context_preprocess = Mock(
        return_value=(
            q_processed,
            kv_pool,
            block_tables,
            None,
            1.0,
            1.0,
            None,
            cu_q_seqlens,
            None,
            2,
            64,
            -1,
        )
    )
    context_postprocess = Mock()
    launcher = Mock()
    monkeypatch.setattr(
        prims_ts_module.thop,
        "trtllm_gen_context_preprocess",
        context_preprocess,
    )
    monkeypatch.setattr(
        prims_ts_module.thop,
        "trtllm_gen_context_postprocess",
        context_postprocess,
    )
    monkeypatch.setattr(prims_ts_module, "_run_prims_context", launcher)

    host_block_offsets = torch.tensor(
        [
            [
                [[0, 1, 2, 3], [6, 7, 8, 9]],
                [[2, 3, 4, 5], [8, 9, 10, 11]],
                [[4, 5, 0, 0], [10, 11, 0, 0]],
            ]
        ],
        dtype=torch.int32,
    )
    metadata = SimpleNamespace(
        kv_cache_block_offsets=torch.empty((3, 2, 4), dtype=torch.int32),
        host_kv_cache_pool_pointers=torch.tensor([1234], dtype=torch.int64),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(
            kv_offset=None,
            host_kv_cache_block_offsets=host_block_offsets,
        ),
        kv_lens_runtime=torch.tensor([7, 33, 64], dtype=torch.int32),
    )
    output = torch.empty((3, attn.num_heads, attn.head_dim), dtype=torch.bfloat16)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.context_only,
        attention_window_size=64,
        is_fused_qkv=True,
    )
    params = FmhaParams(
        attn=attn,
        meta=metadata,
        fwd=forward_args,
        workspace=torch.empty(32, dtype=torch.uint8),
        qkv_input=torch.empty(
            (3, (attn.num_heads + 2 * attn.num_kv_heads) * attn.head_dim),
            dtype=torch.bfloat16,
        ),
        context_buf=output,
        sequence_lengths=torch.tensor([33, 64], dtype=torch.int32),
        context_lengths=torch.tensor([1, 2], dtype=torch.int32),
        input_seq_length=2,
        max_past_kv_length=64,
        max_attention_window_size=64,
        cyclic_attention_window_size=64,
        num_tokens=3,
        seq_offset=1,
        tokens_per_block=32,
        kv_factor=2,
        total_num_blocks=24,
        batch_size=2,
    )

    fmha.run_context(params)

    launcher.assert_called_once()
    args = launcher.call_args.args
    kwargs = launcher.call_args.kwargs
    assert args[0] is q_processed
    assert args[3] is cu_q_seqlens
    k_cache, v_cache = args[1], args[2]
    assert k_cache.shape == v_cache.shape == (6, attn.num_kv_heads, 32, attn.head_dim)
    assert v_cache.storage_offset() - k_cache.storage_offset() == 6 * math.prod(kv_pool.shape[1:])
    torch.testing.assert_close(args[4], torch.tensor([0, 2, 4], dtype=torch.int32))
    torch.testing.assert_close(args[5], torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    torch.testing.assert_close(args[6], torch.tensor([1, 32], dtype=torch.int32))
    assert kwargs["page_size"] == 32
    assert kwargs["kv_layout"] == "HND"
    assert kwargs["mask_type"] == "causal"
    assert kwargs["window_left"] == -1
    assert kwargs["sm_scale"] == pytest.approx(1.0 / math.sqrt(attn.head_dim))
    assert kwargs["output_scale"] == 1.0
    assert kwargs["out_dtype"] == torch.bfloat16
    assert kwargs["out"] is output
    context_preprocess.assert_called_once()
    context_postprocess.assert_called_once()


def test_generation_launcher_receives_v2_bound_and_stable_native_csr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 120
    prims_workspace = torch.full((64,), 7, dtype=torch.uint8)
    fmha._prims_workspace = prims_workspace
    q_processed = torch.empty((2, attn.num_heads, attn.head_dim), dtype=torch.bfloat16)
    kv_pool = torch.empty((12, attn.num_kv_heads, 32, attn.head_dim), dtype=torch.bfloat16)
    block_tables = torch.tensor(
        [
            [[0, 1, 2, 3], [6, 7, 8, 9]],
            [[2, 3, 4, 5], [8, 9, 10, 11]],
        ],
        dtype=torch.int32,
    )
    generation_preprocess = Mock(
        return_value=(
            q_processed,
            kv_pool,
            block_tables,
            None,
            1.0,
            1.0,
            None,
            None,
            1,
            64,
            15,
            False,
        )
    )
    launcher = Mock()
    monkeypatch.setattr(
        prims_ts_module.thop,
        "trtllm_gen_generation_preprocess",
        generation_preprocess,
    )
    monkeypatch.setattr(prims_ts_module, "_run_prims_decode", launcher)

    get_page_index_upper_bound = Mock(return_value=12)
    metadata = SimpleNamespace(
        beam_width=1,
        kv_cache_block_offsets=torch.empty((2, 2, 4), dtype=torch.int32),
        host_kv_cache_pool_pointers=torch.tensor([1234], dtype=torch.int64),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(
            impl=SimpleNamespace(get_page_index_upper_bound=get_page_index_upper_bound),
            kv_offset=torch.tensor([6], dtype=torch.int32),
        ),
    )
    total_num_blocks = fmha._get_total_num_blocks(metadata)
    output = torch.empty((2, attn.num_heads, attn.head_dim), dtype=torch.bfloat16)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=64,
        is_fused_qkv=True,
    )
    sequence_lengths = torch.tensor([0, 33, 64], dtype=torch.int32)[1:]
    assert sequence_lengths.data_ptr() % 16 != 0
    params = FmhaParams(
        attn=attn,
        meta=metadata,
        fwd=forward_args,
        workspace=torch.empty(32, dtype=torch.uint8),
        qkv_input=torch.empty(
            (2, (attn.num_heads + 2 * attn.num_kv_heads) * attn.head_dim),
            dtype=torch.bfloat16,
        ),
        context_buf=output,
        sequence_lengths=sequence_lengths,
        input_seq_length=1,
        max_past_kv_length=64,
        max_attention_window_size=64,
        cyclic_attention_window_size=64,
        num_tokens=2,
        seq_offset=1,
        tokens_per_block=32,
        kv_factor=2,
        total_num_blocks=total_num_blocks,
        num_requests=2,
    )

    fmha.run_generation(params)

    launcher.assert_called_once()
    args = launcher.call_args.args
    kwargs = launcher.call_args.kwargs
    assert args[0].shape == (2, attn.num_heads, attn.head_dim)
    assert args[0].data_ptr() == q_processed.data_ptr()
    k_cache, v_cache = args[1]
    assert k_cache.shape == v_cache.shape == (6, attn.num_kv_heads, 32, attn.head_dim)
    assert v_cache.storage_offset() - k_cache.storage_offset() == 6 * math.prod(kv_pool.shape[1:])
    assert args[2] is prims_workspace
    torch.testing.assert_close(args[3], torch.tensor([0, 4, 8], dtype=torch.int32))
    torch.testing.assert_close(args[4], torch.tensor([0, 1, 2, 3, 2, 3, 4, 5], dtype=torch.int32))
    assert args[4].data_ptr() == fmha._page_indices_buffer.data_ptr()
    torch.testing.assert_close(args[5], params.sequence_lengths)
    assert args[5].data_ptr() == params.sequence_lengths.data_ptr()
    assert args[5].data_ptr() % 4 == 0
    assert args[6] == 128
    assert kwargs["seq_len_q"] == 1
    assert kwargs["bmm1_scale"] == pytest.approx(1.0 / math.sqrt(attn.head_dim))
    assert kwargs["bmm2_scale"] == 1.0
    assert kwargs["out"].shape == (2, attn.num_heads, attn.head_dim)
    assert kwargs["out"].data_ptr() == output.data_ptr()
    assert kwargs["out_dtype"] == torch.bfloat16
    assert kwargs["mask_type"] == "causal"
    assert kwargs["window_left"] == 15
    assert kwargs["kv_layout"] == "HND"
    assert torch.count_nonzero(prims_workspace) == 0
    preprocess_args = generation_preprocess.call_args.args
    assert preprocess_args[15] == params.seq_offset
    assert preprocess_args[39] == total_num_blocks
    get_page_index_upper_bound.assert_called_once()


def test_mla_launcher_receives_v2_bound_and_dense_page_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = _Attention(head_dim=576, is_mla=True, num_heads=4)
    fmha = PrimsTSFmha(attn)
    prims_workspace = torch.empty(96, dtype=torch.uint8)
    fmha._prims_workspace = prims_workspace
    kv_cache = torch.empty((20, 1, 32, 576), dtype=torch.bfloat16)
    block_tables = torch.tensor(
        [
            [[0, 1, 2], [10, 11, 12]],
            [[3, 4, 5], [13, 14, 15]],
        ],
        dtype=torch.int32,
    )
    build_metadata = Mock(return_value=(kv_cache, block_tables, None))
    launcher = Mock()
    monkeypatch.setattr(
        prims_ts_module.thop,
        "build_trtllm_gen_kv_cache_metadata",
        build_metadata,
    )
    monkeypatch.setattr(prims_ts_module, "_run_prims_mla_decode", launcher)

    get_page_index_upper_bound = Mock(return_value=20)
    metadata = SimpleNamespace(
        beam_width=1,
        kv_cache_block_offsets=torch.empty((2, 2, 3), dtype=torch.int32),
        host_kv_cache_pool_pointers=torch.tensor([1234], dtype=torch.int64),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(
            impl=SimpleNamespace(get_page_index_upper_bound=get_page_index_upper_bound)
        ),
    )
    total_num_blocks = fmha._get_total_num_blocks(metadata)
    output = torch.empty((2, attn.num_heads, 512), dtype=torch.bfloat16)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=64,
        is_fused_qkv=True,
    )
    sequence_lengths = torch.tensor([0, 33, 64], dtype=torch.int32)[1:]
    assert sequence_lengths.data_ptr() % 16 != 0
    params = FmhaParams(
        attn=attn,
        meta=metadata,
        fwd=forward_args,
        workspace=torch.empty(0, dtype=torch.uint8),
        qkv_input=torch.empty((2, attn.num_heads * 576), dtype=torch.bfloat16),
        context_buf=output,
        sequence_lengths=sequence_lengths,
        input_seq_length=1,
        max_past_kv_length=64,
        max_attention_window_size=64,
        cyclic_attention_window_size=64,
        num_tokens=2,
        seq_offset=2,
        tokens_per_block=32,
        kv_factor=1,
        total_num_blocks=total_num_blocks,
        num_requests=2,
    )

    fmha.run_mla_generation(params)

    launcher.assert_called_once()
    args = launcher.call_args.args
    kwargs = launcher.call_args.kwargs
    assert args[0].shape == (2, 1, attn.num_heads, 576)
    assert args[0].data_ptr() == params.qkv_input.data_ptr()
    assert args[1] is kv_cache
    assert args[2] is prims_workspace
    assert args[3:5] == (512, 64)
    torch.testing.assert_close(args[5], torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32))
    assert args[5].data_ptr() == fmha._page_indices_buffer.data_ptr()
    torch.testing.assert_close(args[6], params.sequence_lengths)
    assert args[6].data_ptr() == fmha._sequence_lengths_buffer.data_ptr()
    assert args[6].data_ptr() % 16 == 0
    assert args[7] == 96
    assert kwargs["max_seq_len_q"] == 1
    assert kwargs["out"].shape == (2, 1, attn.num_heads, 512)
    assert kwargs["out"].data_ptr() == output.data_ptr()
    assert kwargs["bmm1_scale"] == pytest.approx(1.0 / math.sqrt(128 + 64))
    assert kwargs["bmm2_scale"] == 1.0
    assert kwargs["mask_type"] == "causal"
    assert kwargs["out_dtype"] == torch.bfloat16
    builder_args = build_metadata.call_args.args
    assert builder_args[8] == total_num_blocks
    assert builder_args[10] == params.seq_offset
    assert builder_args[11] == 2
    assert builder_args[12] == torch.bfloat16
    get_page_index_upper_bound.assert_called_once()


def test_workspace_growth_retains_graph_visible_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        prims_ts_module.thop,
        "get_trtllm_gen_generation_workspace_layout",
        lambda *args, **kwargs: {"total_size": 0},
    )
    required_bytes = iter((32, 64))
    monkeypatch.setattr(
        prims_ts_module,
        "_get_prims_decode_workspace_size",
        lambda *args, **kwargs: next(required_bytes),
    )
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 1
    q = torch.empty((2, 12 * 128), dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        kv_cache_block_offsets=torch.empty((2, 2, 4), dtype=torch.int32),
        max_num_requests=2,
        num_contexts=0,
        num_generations=2,
        num_ctx_tokens=0,
        tokens_per_block=32,
        kv_lens_runtime=torch.tensor([64, 96], dtype=torch.int32),
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((2, 8 * 128), dtype=torch.bfloat16),
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=128,
    )
    workspace = torch.empty(0, dtype=torch.uint8)

    fmha.prepare_workspace(q, None, None, metadata, forward_args, workspace)
    first_workspace = fmha._get_prims_workspace()
    fmha.prepare_workspace(q, None, None, metadata, forward_args, workspace)

    assert fmha._get_prims_workspace().numel() == 64
    assert len(fmha._retained_prims_workspaces) == 1
    assert fmha._retained_prims_workspaces[0] is first_workspace


def test_workspace_cannot_grow_during_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 1
    fmha._prims_workspace = torch.zeros(16, dtype=torch.uint8)
    fmha._metadata_row_capacity = 2
    fmha._metadata_column_capacity = 4
    fmha._page_indices_buffer = torch.empty((2, 4), dtype=torch.int32)
    fmha._fixed_indptr_buffer = torch.empty(3, dtype=torch.int32)
    fmha._sequence_lengths_buffer = torch.empty(2, dtype=torch.int32)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        prims_ts_module.thop,
        "get_trtllm_gen_generation_workspace_layout",
        lambda *args, **kwargs: {"total_size": 0},
    )
    monkeypatch.setattr(
        prims_ts_module,
        "_get_prims_decode_workspace_size",
        lambda *args, **kwargs: 32,
    )
    metadata = SimpleNamespace(
        kv_cache_block_offsets=torch.empty((2, 2, 4), dtype=torch.int32),
        max_num_requests=2,
        num_contexts=0,
        num_generations=2,
        num_ctx_tokens=0,
        tokens_per_block=32,
        kv_lens_runtime=torch.tensor([64, 96], dtype=torch.int32),
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((2, 8 * 128), dtype=torch.bfloat16),
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=128,
    )

    with pytest.raises(
        RuntimeError,
        match="PrimTS workspace must be allocated before CUDA graph capture",
    ):
        fmha.prepare_workspace(
            torch.empty((2, 12 * 128), dtype=torch.bfloat16),
            None,
            None,
            metadata,
            forward_args,
            torch.empty(0, dtype=torch.uint8),
        )


def test_metadata_buffers_cannot_grow_during_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fmha = PrimsTSFmha(_Attention())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(
        RuntimeError,
        match="PrimTS metadata buffers must be allocated before CUDA graph capture",
    ):
        fmha._ensure_metadata_buffers(torch.device("cpu"), 2, 4)


def test_phased_forward_routes_mixed_batch_to_context_and_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    context_calls = []
    generation_calls = []
    run_context = Mock(side_effect=lambda params: context_calls.append(replace(params)))
    run_generation = Mock(side_effect=lambda params: generation_calls.append(replace(params)))
    monkeypatch.setattr(fmha, "prepare_workspace", Mock())
    monkeypatch.setattr(fmha, "run_context", run_context)
    monkeypatch.setattr(fmha, "run_generation", run_generation)

    q = torch.empty((5, 128), dtype=torch.bfloat16)
    output = torch.empty((5, attn.num_heads * attn.head_dim), dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        kv_cache_block_offsets=torch.empty(1),
        effective_workspace=torch.empty(0, dtype=torch.int8),
        num_contexts=1,
        num_ctx_tokens=3,
        num_generations=2,
        cache_indirection=None,
        beam_width=1,
        tokens_per_block=32,
        kv_lens_cuda_runtime=torch.tensor([3, 65, 97], dtype=torch.int32),
        kv_lens_runtime=torch.tensor([3, 65, 97], dtype=torch.int32),
        prompt_lens_cuda_runtime=torch.tensor([3, 1, 1], dtype=torch.int32),
        prompt_lens_cpu_runtime=torch.tensor([3, 1, 1], dtype=torch.int32),
        is_spec_decoding_enabled=False,
        is_cross=False,
        kv_cache_manager=None,
    )
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.mixed,
        attention_window_size=128,
    )

    fmha.forward(q, None, None, metadata, forward_args)

    run_context.assert_called_once()
    context_params = context_calls[0]
    assert context_params.num_tokens == 3
    assert context_params.seq_offset == 0
    assert context_params.batch_size == 1
    assert context_params.attention_input is not None
    assert context_params.attention_input.shape[0] == 3
    assert context_params.context_buf is not None
    assert context_params.context_buf.shape == (3, attn.num_heads, attn.head_dim)

    run_generation.assert_called_once()
    generation_params = generation_calls[0]
    assert generation_params.num_tokens == 2
    assert generation_params.seq_offset == 1
    assert generation_params.num_requests == 2
    assert generation_params.attention_input is not None
    assert generation_params.attention_input.shape[0] == 2
    assert generation_params.context_buf is not None
    assert generation_params.context_buf.shape == (2, attn.num_heads, attn.head_dim)
