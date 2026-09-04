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
import tensorrt_llm._torch.attention_backend.fmha.utils as fmha_utils
import tensorrt_llm._torch.attention_backend.prims_ts as prims_ts_package
import tensorrt_llm._torch.attention_backend.prims_ts.context as prims_context_module
import tensorrt_llm._torch.attention_backend.prims_ts.decode as prims_decode_module
import tensorrt_llm._torch.attention_backend.prims_ts.mla_decode as prims_mla_module
from tensorrt_llm._torch.attention_backend.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention_backend.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention_backend.fmha.phased import FmhaParams
from tensorrt_llm._torch.attention_backend.fmha.prims_ts import PrimsTSFmha
from tensorrt_llm._torch.attention_backend.fmha.registry import get_enabled_fmha_lib_classes
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
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


def _make_v1_manager(**attributes: object) -> KVCacheManager:
    manager = object.__new__(KVCacheManager)
    for name, value in attributes.items():
        setattr(manager, name, value)
    return manager


def _make_v2_manager(**attributes: object) -> KVCacheManagerV2:
    manager = object.__new__(KVCacheManagerV2)
    for name, value in attributes.items():
        setattr(manager, name, value)
    return manager


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
    is_cuda_graph: bool = False,
    has_attention_sinks: bool = False,
    has_relative_attention_bias: bool = False,
    has_sparse_attention: bool = False,
    has_sparse_runtime_metadata: bool = False,
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
    phase: FmhaPhase | None = None,
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
    if has_sparse_runtime_metadata:
        forward_args.sparse_runtime_params.sparse_kv_indices = torch.empty(1)
    if attention_input_type == AttentionInputType.context_only:
        num_contexts, num_generations, num_ctx_tokens = 1, 0, 4
        kv_lens = [4]
    elif attention_input_type == AttentionInputType.generation_only:
        num_contexts, num_generations, num_ctx_tokens = 0, 4, 0
        kv_lens = [128, 96, 64, 32]
    else:
        num_contexts, num_generations, num_ctx_tokens = 1, 1, 3
        kv_lens = [3, 128]
    if use_kv_cache_v2:
        kv_cache_manager = _make_v2_manager(
            dtype=kv_dtype,
            impl=SimpleNamespace(get_page_index_upper_bound=lambda *args: 128),
            enable_swa_scratch_reuse=enable_swa_scratch_reuse,
            num_local_layers=1,
            num_pools=num_kv_cache_pools,
            kv_offset=torch.full((num_kv_cache_pools,), 128, dtype=torch.int32),
        )
    else:
        kv_cache_manager = _make_v1_manager(
            dtype=kv_dtype,
            impl=SimpleNamespace(),
            num_local_layers=1,
            num_pools=num_kv_cache_pools,
            host_kv_cache_block_offsets=torch.tensor(
                [[[[0], [128]]]],
                dtype=torch.int32,
            ),
        )
    metadata = SimpleNamespace(
        helix_position_offsets=None,
        num_sparse_topk=0,
        use_spec_decoding=use_spec_decoding,
        is_cuda_graph=is_cuda_graph,
        is_spec_dec_tree=is_spec_dec_tree,
        is_spec_dec_dynamic_tree=False,
        is_spec_decoding_enabled=use_spec_decoding,
        kv_cache_block_offsets=torch.empty(1) if has_paged_cache else None,
        host_kv_cache_pool_pointers=torch.empty(1),
        host_kv_cache_pool_mapping=torch.zeros((1, 2), dtype=torch.int32),
        kv_cache_manager=kv_cache_manager,
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
    k = _TensorSpec((4, attn.num_kv_heads * head_dim), dtype) if has_separate_kv else None
    v = _TensorSpec((4, attn.num_kv_heads * head_dim), dtype) if has_separate_kv else None
    return fmha._is_supported_with_reason(
        q,
        k,
        v,
        attn,
        metadata,
        forward_args,
        phase=phase,
    )


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
        {
            "attention_input_type": AttentionInputType.context_only,
            "num_heads": 64,
            "num_kv_heads": 1,
        },
    ],
    ids=[
        "context",
        "mixed",
        "generation",
        "mla-generation",
        "v2-multi-pool",
        "context-gqa-ratio-over-32",
    ],
)
def test_supported_matrix(case: dict) -> None:
    supported, reason = _support_result(**case)

    assert supported, reason


@pytest.mark.parametrize("phase", [FmhaPhase.CONTEXT, FmhaPhase.GENERATION])
def test_phase_support_check_preserves_whole_request_semantics(phase: FmhaPhase) -> None:
    supported, reason = _support_result(
        attention_input_type=AttentionInputType.mixed,
        head_dim=64,
        phase=phase,
    )

    assert not supported
    assert "context head dimension" in reason


def test_is_supported_accepts_and_forwards_phase_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    support_check = Mock(return_value=(True, ""))
    monkeypatch.setattr(fmha, "_is_supported_with_reason", support_check)
    q = Mock(spec=torch.Tensor)
    metadata = SimpleNamespace()
    forward_args = AttentionForwardArgs()

    assert fmha.is_supported(
        q,
        None,
        None,
        metadata,
        forward_args,
        phase=FmhaPhase.GENERATION,
    )
    support_check.assert_called_once_with(
        q,
        None,
        None,
        attn,
        metadata,
        forward_args,
        phase=FmhaPhase.GENERATION,
    )


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
                "is_cuda_graph": True,
            },
            "context planning is not CUDA-graph capturable",
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
                "attention_input_type": AttentionInputType.context_only,
                "has_sparse_runtime_metadata": True,
            },
            "sparse attention metadata",
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
        "context-cuda-graph",
        "sparse",
        "sparse-runtime-metadata",
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
        kv_cache_manager=_make_v2_manager(
            impl=SimpleNamespace(get_page_index_upper_bound=get_page_index_upper_bound)
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
        kv_cache_manager=_make_v1_manager(
            impl=SimpleNamespace(
                get_primary_pool_data=lambda _: torch.empty(64, dtype=torch.uint8)
            ),
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
        kv_cache_manager=_make_v2_manager(
            impl=SimpleNamespace(get_page_index_upper_bound=lambda *args: 128),
            kv_offset=torch.tensor([0, 128]),
        ),
        host_kv_cache_pool_mapping=torch.tensor([[1, 0]], dtype=torch.int32),
    )

    assert (
        fmha_utils.get_kv_page_offset(
            fmha.attn,
            metadata,
            0,
            cache=fmha._kv_page_offset_cache,
        )
        == 128
    )


def test_kv_page_offset_is_inferred_from_v1_host_tables() -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    host_offsets = torch.tensor(
        [[[[0, 1, 2], [64, 65, 66]], [[3, 4, 5], [67, 68, 69]]]],
        dtype=torch.int32,
    )
    metadata = SimpleNamespace(
        kv_cache_manager=_make_v1_manager(
            impl=SimpleNamespace(),
            host_kv_cache_block_offsets=host_offsets,
        ),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
    )

    assert (
        fmha_utils.get_kv_page_offset(
            fmha.attn,
            metadata,
            1,
            cache=fmha._kv_page_offset_cache,
        )
        == 64
    )


def test_fixed_block_tables_are_zero_copy_plane_zero_view() -> None:
    block_tables = torch.tensor(
        [
            [[10, 11, 12], [110, 111, 112]],
            [[20, 21, 22], [120, 121, 122]],
            [[30, 31, 32], [130, 131, 132]],
        ],
        dtype=torch.int32,
    )
    attn = _Attention()
    fmha = PrimsTSFmha(attn)

    actual = fmha._get_fixed_block_tables(block_tables, 2)

    assert actual.shape == (2, 3)
    assert actual.stride() == (6, 1)
    assert actual.data_ptr() == block_tables.data_ptr()
    torch.testing.assert_close(
        actual,
        torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32),
    )

    block_tables[1, 0, 2] = 99
    block_tables[:, 1].add_(1000)

    torch.testing.assert_close(
        actual,
        torch.tensor([[10, 11, 12], [20, 21, 99]], dtype=torch.int32),
    )


def test_sequence_lengths_are_live_source_view() -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    sequence_lengths = torch.tensor([0, 33, 64, 96], dtype=torch.int32)[1:]

    actual = fmha._get_sequence_lengths(sequence_lengths, 2)

    assert actual.shape == (2,)
    assert actual.stride() == (1,)
    assert actual.data_ptr() == sequence_lengths.data_ptr()
    sequence_lengths[1] = 65
    torch.testing.assert_close(actual, torch.tensor([33, 65], dtype=torch.int32))


def test_context_wrapper_plans_once_and_reads_live_fixed_metadata(
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
    cu_kv_seqlens = torch.tensor([0, 7, 18], dtype=torch.int32)
    fmha_workspace = torch.empty(0, dtype=torch.uint8)
    context_preprocess = Mock(
        return_value=(
            q_processed,
            kv_pool,
            block_tables,
            None,
            1.0,
            1.0,
            fmha_workspace,
            cu_q_seqlens,
            cu_kv_seqlens,
            2,
            64,
            -1,
        )
    )
    context_postprocess = Mock()
    wrapper = Mock()
    wrapper_factory = Mock(return_value=wrapper)
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
    monkeypatch.setattr(
        prims_context_module,
        "BatchPrefillPagedTSWrapper",
        wrapper_factory,
    )

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
        kv_cache_manager=_make_v1_manager(
            impl=SimpleNamespace(),
            host_kv_cache_block_offsets=host_block_offsets,
        ),
        kv_lens_runtime=torch.tensor([7, 33, 64], dtype=torch.int32),
        max_context_length=8,
        max_seq_len=128,
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

    wrapper_factory.assert_called_once_with(kv_layout="HND")
    wrapper.plan.assert_called_once()
    plan_args = wrapper.plan.call_args.args
    plan_kwargs = wrapper.plan.call_args.kwargs
    assert plan_args == ()
    assert plan_kwargs == {
        "device": q_processed.device,
        "batch_size": 2,
        "max_seq_len_q": 8,
        "max_kv_len": 128,
        "num_qo_heads": attn.num_heads,
        "num_kv_heads": attn.num_kv_heads,
        "head_dim": attn.head_dim,
        "q_dtype": torch.bfloat16,
        "kv_dtype": torch.bfloat16,
        "out_dtype": torch.bfloat16,
        "page_size": 32,
        "mask_type": "causal",
        "window_left": -1,
        "sm_scale": pytest.approx(1.0 / math.sqrt(attn.head_dim)),
        "output_scale": 1.0,
    }
    wrapper.run.assert_called_once()
    run_args = wrapper.run.call_args.args
    run_kwargs = wrapper.run.call_args.kwargs
    assert run_args[0] is q_processed
    k_cache, v_cache = run_args[1], run_args[2]
    assert k_cache.shape == v_cache.shape == (6, attn.num_kv_heads, 32, attn.head_dim)
    assert v_cache.storage_offset() - k_cache.storage_offset() == 6 * math.prod(kv_pool.shape[1:])
    fixed_block_tables = run_kwargs["block_tables"]
    seq_lens_kv = run_kwargs["seq_lens_kv"]
    first_metadata_ptrs = (
        run_args[3].data_ptr(),
        fixed_block_tables.data_ptr(),
        seq_lens_kv.data_ptr(),
    )
    assert run_kwargs["out"] is output
    assert run_kwargs["validate"] is False
    assert run_args[3] is cu_q_seqlens
    torch.testing.assert_close(seq_lens_kv, torch.tensor([33, 64], dtype=torch.int32))
    assert seq_lens_kv.data_ptr() == params.sequence_lengths.data_ptr()
    assert fixed_block_tables.shape == (2, 4)
    assert fixed_block_tables.stride() == (8, 1)
    assert fixed_block_tables.data_ptr() == block_tables.data_ptr()
    torch.testing.assert_close(
        fixed_block_tables,
        torch.tensor([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=torch.int32),
    )
    context_preprocess.assert_called_once()
    context_postprocess.assert_called_once()
    assert context_preprocess.call_args.kwargs["skip_fmha_workspace"] is True
    assert context_postprocess.call_args.kwargs["skip_fmha_workspace"] is True

    block_tables[:, 0].add_(1)
    block_tables[:, 1].add_(100)
    params.sequence_lengths.copy_(torch.tensor([64, 33], dtype=torch.int32))
    cu_kv_seqlens.copy_(torch.tensor([0, 4, 9], dtype=torch.int32))
    fmha.run_context(params)

    wrapper_factory.assert_called_once_with(kv_layout="HND")
    wrapper.plan.assert_called_once()
    assert wrapper.run.call_count == 2
    second_run_args = wrapper.run.call_args.args
    second_run_kwargs = wrapper.run.call_args.kwargs
    assert (
        second_run_args[3].data_ptr(),
        second_run_kwargs["block_tables"].data_ptr(),
        second_run_kwargs["seq_lens_kv"].data_ptr(),
    ) == first_metadata_ptrs
    torch.testing.assert_close(
        second_run_kwargs["seq_lens_kv"],
        torch.tensor([64, 33], dtype=torch.int32),
    )
    torch.testing.assert_close(
        second_run_kwargs["block_tables"],
        torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=torch.int32),
    )


@pytest.mark.parametrize(
    (
        "use_split_kv",
        "use_separate_reduction_kernel",
        "use_cluster_smem_reduction",
        "requires_control_reset",
    ),
    (
        pytest.param(False, False, False, False, id="direct"),
        pytest.param(True, False, False, True, id="fused-global-reduction"),
        pytest.param(True, True, False, False, id="separate-reduction"),
        pytest.param(True, False, True, False, id="cluster-smem-reduction"),
    ),
)
def test_generation_wrapper_plans_once_and_reads_live_fixed_metadata(
    monkeypatch: pytest.MonkeyPatch,
    use_split_kv: bool,
    use_separate_reduction_kernel: bool,
    use_cluster_smem_reduction: bool,
    requires_control_reset: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 120
    fmha._decode_workspace_offset_bytes = 0
    fmha._decode_workspace_required_bytes = 64
    fmha_workspace = torch.empty(0, dtype=torch.uint8)
    q_processed = torch.empty((2, attn.num_heads, attn.head_dim), dtype=torch.bfloat16)
    kv_pool = torch.empty((12, attn.num_kv_heads, 32, attn.head_dim), dtype=torch.bfloat16)
    block_tables = torch.tensor(
        [
            [[0, 1, 2, 3], [6, 7, 8, 9]],
            [[2, 3, 4, 5], [8, 9, 10, 11]],
        ],
        dtype=torch.int32,
    )
    workspace = torch.full((64,), 7, dtype=torch.uint8)
    split_kv_counter = workspace[32:40].view(torch.int32)
    generation_preprocess = Mock(
        return_value=(
            q_processed,
            kv_pool,
            block_tables,
            None,
            1.0,
            1.0,
            fmha_workspace,
            None,
            1,
            64,
            15,
            False,
        )
    )
    wrapper = Mock()
    wrapper._plan_state = SimpleNamespace(
        policy=(
            ("use_split_kv", use_split_kv),
            ("use_separate_reduction_kernel", use_separate_reduction_kernel),
            ("use_cluster_smem_reduction", use_cluster_smem_reduction),
        ),
        workspace=SimpleNamespace(split_kv_counter=split_kv_counter),
    )
    wrapper_factory = Mock(return_value=wrapper)
    monkeypatch.setattr(
        prims_ts_module.thop,
        "trtllm_gen_generation_preprocess",
        generation_preprocess,
    )
    monkeypatch.setattr(
        prims_ts_package,
        "get_prims_ts_batch_decode_workspace_size",
        Mock(return_value=64),
    )
    monkeypatch.setattr(
        prims_decode_module,
        "BatchDecodePagedTSWrapper",
        wrapper_factory,
    )

    get_page_index_upper_bound = Mock(return_value=12)
    metadata = SimpleNamespace(
        beam_width=1,
        kv_cache_block_offsets=torch.empty((2, 2, 4), dtype=torch.int32),
        host_kv_cache_pool_pointers=torch.tensor([1234], dtype=torch.int64),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
        kv_cache_manager=_make_v2_manager(
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
        workspace=workspace,
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
        batch_size=2,
        num_requests=2,
    )

    fmha.run_generation(params)

    wrapper_factory.assert_called_once_with(kv_layout="HND")
    wrapper.plan.assert_called_once()
    plan_args = wrapper.plan.call_args.args
    plan_kwargs = wrapper.plan.call_args.kwargs
    assert plan_args == (
        params.workspace.device,
        2,
        attn.num_heads,
        attn.num_kv_heads,
        attn.head_dim,
        32,
        128,
    )
    decode_workspace = plan_kwargs["workspace_buffer"]
    assert decode_workspace.data_ptr() == params.workspace.data_ptr()
    assert decode_workspace.numel() == 64
    assert {key: value for key, value in plan_kwargs.items() if key != "workspace_buffer"} == {
        "max_seq_len_q": 1,
        "packed_query": False,
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
        "o_data_type": torch.bfloat16,
        "mask_type": "causal",
        "window_left": 15,
    }
    assert fmha._decode_wrappers[2] is wrapper
    wrapper.run.assert_called_once()
    run_args = wrapper.run.call_args.args
    run_kwargs = wrapper.run.call_args.kwargs
    assert run_args[0].shape == (2, attn.num_heads, attn.head_dim)
    assert run_args[0].data_ptr() == q_processed.data_ptr()
    k_cache, v_cache = run_args[1]
    assert k_cache.shape == v_cache.shape == (6, attn.num_kv_heads, 32, attn.head_dim)
    assert v_cache.storage_offset() - k_cache.storage_offset() == 6 * math.prod(kv_pool.shape[1:])
    assert run_args[2].data_ptr() == params.sequence_lengths.data_ptr()
    fixed_block_tables = run_kwargs["block_tables"]
    assert fixed_block_tables.shape == (2, 4)
    assert fixed_block_tables.stride() == (8, 1)
    assert fixed_block_tables.data_ptr() == block_tables.data_ptr()
    torch.testing.assert_close(
        fixed_block_tables,
        torch.tensor([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=torch.int32),
    )
    assert run_kwargs["bmm1_scale"] == pytest.approx(1.0 / math.sqrt(attn.head_dim))
    assert run_kwargs["bmm2_scale"] == 1.0
    assert run_kwargs["out"].shape == (2, attn.num_heads, attn.head_dim)
    assert run_kwargs["out"].data_ptr() == output.data_ptr()
    assert run_kwargs["validate"] is False
    if requires_control_reset:
        assert torch.count_nonzero(split_kv_counter) == 0
    else:
        torch.testing.assert_close(
            split_kv_counter,
            torch.full_like(split_kv_counter, 0x07070707),
        )
    torch.testing.assert_close(params.workspace[:32], torch.full((32,), 7, dtype=torch.uint8))
    torch.testing.assert_close(params.workspace[40:], torch.full((24,), 7, dtype=torch.uint8))
    preprocess_args = generation_preprocess.call_args.args
    assert preprocess_args[15] == params.seq_offset
    assert preprocess_args[39] == total_num_blocks
    assert generation_preprocess.call_args.kwargs["skip_fmha_workspace"] is True
    get_page_index_upper_bound.assert_called_once()

    block_tables[:, 0].add_(20)
    block_tables[:, 1].add_(200)
    sequence_lengths.add_(1)
    params.workspace.fill_(9)
    fmha.run_generation(params)

    wrapper_factory.assert_called_once_with(kv_layout="HND")
    wrapper.plan.assert_called_once()
    assert wrapper.run.call_count == 2
    second_run_kwargs = wrapper.run.call_args.kwargs
    assert second_run_kwargs["block_tables"].data_ptr() == block_tables.data_ptr()
    assert second_run_kwargs["block_tables"].stride() == (8, 1)
    torch.testing.assert_close(
        second_run_kwargs["block_tables"],
        torch.tensor([[20, 21, 22, 23], [22, 23, 24, 25]], dtype=torch.int32),
    )
    torch.testing.assert_close(run_args[2], torch.tensor([34, 65], dtype=torch.int32))
    if requires_control_reset:
        assert torch.count_nonzero(split_kv_counter) == 0
    else:
        torch.testing.assert_close(
            split_kv_counter,
            torch.full_like(split_kv_counter, 0x09090909),
        )
    torch.testing.assert_close(params.workspace[:32], torch.full((32,), 9, dtype=torch.uint8))
    torch.testing.assert_close(params.workspace[40:], torch.full((24,), 9, dtype=torch.uint8))


def test_decode_layer_adapters_bind_the_same_shared_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrappers = [Mock(), Mock()]
    wrapper_factory = Mock(side_effect=wrappers)
    monkeypatch.setattr(
        prims_decode_module,
        "BatchDecodePagedTSWrapper",
        wrapper_factory,
    )
    attentions = [_Attention(), _Attention()]
    layers = [PrimsTSFmha(attn) for attn in attentions]
    shared_workspace = torch.empty(64, dtype=torch.uint8)

    def get_wrapper(layer: PrimsTSFmha) -> object:
        return layer._get_or_plan_decode_wrapper(
            shared_workspace,
            batch_size=2,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=32,
            seq_len_q=1,
            max_kv_len=64,
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            mask_type="causal",
            window_left=-1,
        )

    first_results = [get_wrapper(layer) for layer in layers]
    second_results = [get_wrapper(layer) for layer in layers]

    assert first_results == second_results == wrappers
    assert wrapper_factory.call_count == 2
    for wrapper in wrappers:
        wrapper.plan.assert_called_once()
        assert wrapper.plan.call_args.kwargs["workspace_buffer"] is shared_workspace


def test_context_wrapper_cache_plans_each_batch_once_and_reuses_a_b_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrappers = [Mock(), Mock()]
    wrapper_factory = Mock(side_effect=wrappers)
    monkeypatch.setattr(
        prims_context_module,
        "BatchPrefillPagedTSWrapper",
        wrapper_factory,
    )
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    k_cache = torch.empty((8, 2, 32, 128), dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)

    def get_wrapper(batch_size: int) -> object:
        q = torch.empty((batch_size, 8, 128), dtype=torch.bfloat16)
        return fmha._get_or_plan_context_wrapper(
            q,
            k_cache,
            v_cache,
            batch_size=batch_size,
            max_seq_len_q=128,
            max_kv_len=256,
            page_size=32,
            mask_type="causal",
            window_left=-1,
            sm_scale=1.0 / math.sqrt(128),
            output_dtype=torch.bfloat16,
        )

    first_a = get_wrapper(1)
    profile_b = get_wrapper(2)
    second_a = get_wrapper(1)

    assert first_a is second_a is wrappers[0]
    assert profile_b is wrappers[1]
    assert wrapper_factory.call_count == 2
    for wrapper in wrappers:
        wrapper.plan.assert_called_once()
    assert set(fmha._context_wrappers) == {1, 2}


def test_decode_wrapper_cache_plans_each_batch_once_and_reuses_a_b_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrappers = [Mock(), Mock()]
    wrapper_factory = Mock(side_effect=wrappers)
    monkeypatch.setattr(
        prims_decode_module,
        "BatchDecodePagedTSWrapper",
        wrapper_factory,
    )
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    workspace = torch.empty(64, dtype=torch.uint8)

    def get_wrapper(batch_size: int) -> object:
        return fmha._get_or_plan_decode_wrapper(
            workspace,
            batch_size=batch_size,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=32,
            seq_len_q=1,
            max_kv_len=256,
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            mask_type="causal",
            window_left=-1,
        )

    first_a = get_wrapper(1)
    profile_b = get_wrapper(2)
    second_a = get_wrapper(1)

    assert first_a is second_a is wrappers[0]
    assert profile_b is wrappers[1]
    assert wrapper_factory.call_count == 2
    for wrapper in wrappers:
        wrapper.plan.assert_called_once()
    assert set(fmha._decode_wrappers) == {1, 2}


def test_workspace_allocation_change_invalidates_only_workspace_bound_wrappers() -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    first_workspace = torch.empty(32, dtype=torch.uint8)
    fmha._update_workspace_allocation(first_workspace)
    fmha._context_wrappers[1] = Mock()
    fmha._decode_wrappers[1] = Mock()
    fmha._mla_decode_wrappers[1] = Mock()

    second_workspace = torch.empty(64, dtype=torch.uint8)
    fmha._update_workspace_allocation(second_workspace)

    assert set(fmha._context_wrappers) == {1}
    assert fmha._decode_wrappers == {}
    assert fmha._mla_decode_wrappers == {}


def _get_test_mla_wrapper(
    fmha: PrimsTSFmha,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    mask_type: str = "causal",
) -> object:
    return fmha._get_or_plan_mla_decode_wrapper(
        workspace_buffer,
        batch_size=int(block_tables.shape[0]),
        num_heads=4,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        page_size=32,
        max_seq_len_q=1,
        max_kv_len=96,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        mask_type=mask_type,
    )


def test_mla_eager_wrapper_plans_once_and_reads_live_fixed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = _Attention(head_dim=576, is_mla=True, num_heads=4)
    fmha = PrimsTSFmha(attn)
    kv_cache = torch.empty((20, 1, 32, 576), dtype=torch.bfloat16)
    block_tables = torch.tensor(
        [
            [[0, 1, 2], [10, 11, 12]],
            [[3, 4, 5], [13, 14, 15]],
        ],
        dtype=torch.int32,
    )
    build_metadata = Mock(return_value=(kv_cache, block_tables, None))
    wrapper = Mock()
    wrapper_factory = Mock(return_value=wrapper)
    monkeypatch.setattr(
        prims_ts_module.thop,
        "build_trtllm_gen_kv_cache_metadata",
        build_metadata,
    )
    monkeypatch.setattr(
        prims_ts_package,
        "get_prims_ts_batch_decode_mla_workspace_size",
        Mock(return_value=64),
    )
    monkeypatch.setattr(
        prims_mla_module,
        "BatchMLADecodePagedTSWrapper",
        wrapper_factory,
    )

    metadata = SimpleNamespace(
        is_cuda_graph=False,
        beam_width=1,
        kv_cache_block_offsets=torch.empty((2, 2, 3), dtype=torch.int32),
        host_kv_cache_pool_pointers=torch.tensor([1234], dtype=torch.int64),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
    )
    output = torch.empty((2, attn.num_heads, 512), dtype=torch.bfloat16)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=64,
        is_fused_qkv=True,
    )
    sequence_lengths = torch.tensor([33, 64], dtype=torch.int32)
    assert sequence_lengths.data_ptr() % 16 == 0
    params = FmhaParams(
        attn=attn,
        meta=metadata,
        fwd=forward_args,
        workspace=torch.empty(64, dtype=torch.uint8),
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
        total_num_blocks=20,
        batch_size=2,
        num_requests=2,
    )

    fmha.run_mla_generation(params)

    wrapper_factory.assert_called_once_with()
    wrapper.plan.assert_called_once()
    plan_args = wrapper.plan.call_args.args
    plan_kwargs = wrapper.plan.call_args.kwargs
    assert plan_args == (
        params.workspace.device,
        2,
        attn.num_heads,
        512,
        64,
        32,
        96,
    )
    workspace_buffer = plan_kwargs["workspace_buffer"]
    assert workspace_buffer.data_ptr() == params.workspace.data_ptr()
    assert workspace_buffer.numel() == 64
    assert {key: value for key, value in plan_kwargs.items() if key != "workspace_buffer"} == {
        "max_seq_len_q": 1,
        "packed_query": False,
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
        "o_data_type": torch.bfloat16,
        "mask_type": "causal",
    }
    wrapper.run.assert_called_once()
    run_args = wrapper.run.call_args.args
    run_kwargs = wrapper.run.call_args.kwargs
    assert run_args[0].shape == (2, 1, attn.num_heads, 576)
    assert run_args[0].data_ptr() == params.qkv_input.data_ptr()
    assert run_args[1] is kv_cache
    assert run_kwargs["block_tables"].shape == (2, 3)
    assert run_kwargs["block_tables"].stride() == (6, 1)
    assert run_kwargs["block_tables"].data_ptr() == block_tables.data_ptr()
    torch.testing.assert_close(
        run_kwargs["block_tables"],
        torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32),
    )
    assert run_kwargs["seq_lens"].data_ptr() == sequence_lengths.data_ptr()
    torch.testing.assert_close(run_kwargs["seq_lens"], sequence_lengths)
    assert run_kwargs["out"].shape == (2, 1, attn.num_heads, 512)
    assert run_kwargs["out"].data_ptr() == output.data_ptr()
    assert run_kwargs["bmm1_scale"] == pytest.approx(1.0 / math.sqrt(128 + 64))
    assert run_kwargs["bmm2_scale"] == 1.0
    assert run_kwargs["validate"] is False

    block_tables[:, 0].add_(20)
    block_tables[:, 1].add_(200)
    sequence_lengths.add_(1)
    fmha.run_mla_generation(params)

    wrapper_factory.assert_called_once_with()
    wrapper.plan.assert_called_once()
    assert wrapper.run.call_count == 2
    assert run_kwargs["block_tables"].data_ptr() == block_tables.data_ptr()
    assert run_kwargs["block_tables"].stride() == (6, 1)
    torch.testing.assert_close(
        run_kwargs["block_tables"],
        torch.tensor([[20, 21, 22], [23, 24, 25]], dtype=torch.int32),
    )
    torch.testing.assert_close(run_kwargs["seq_lens"], torch.tensor([34, 65], dtype=torch.int32))
    assert fmha._mla_decode_wrappers[2] is wrapper


def test_mla_wrapper_cache_plans_each_batch_once_and_reuses_a_b_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = _Attention(head_dim=576, is_mla=True, num_heads=4)
    fmha = PrimsTSFmha(attn)
    wrappers = [Mock(), Mock()]
    wrapper_factory = Mock(side_effect=wrappers)
    monkeypatch.setattr(
        prims_mla_module,
        "BatchMLADecodePagedTSWrapper",
        wrapper_factory,
    )
    block_tables = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    seq_lens = torch.tensor([33, 64], dtype=torch.int32)
    workspace = torch.empty(64, dtype=torch.uint8)

    first = _get_test_mla_wrapper(fmha, block_tables, seq_lens, workspace)
    cached_first = _get_test_mla_wrapper(fmha, block_tables, seq_lens, workspace)
    block_pointer_hit = _get_test_mla_wrapper(
        fmha,
        block_tables.clone(),
        seq_lens,
        workspace,
    )
    seq_pointer_hit = _get_test_mla_wrapper(
        fmha,
        block_tables,
        seq_lens.clone(),
        workspace,
    )
    profile_b = _get_test_mla_wrapper(
        fmha,
        block_tables[:1],
        seq_lens[:1],
        workspace,
    )
    profile_a_again = _get_test_mla_wrapper(
        fmha,
        block_tables,
        seq_lens,
        workspace,
    )

    assert all(
        result is wrappers[0]
        for result in (
            first,
            cached_first,
            block_pointer_hit,
            seq_pointer_hit,
            profile_a_again,
        )
    )
    assert profile_b is wrappers[1]
    assert wrapper_factory.call_count == 2
    for wrapper in wrappers:
        wrapper.plan.assert_called_once()
        assert wrapper.plan.call_args.kwargs["workspace_buffer"] is workspace
    assert fmha._mla_decode_wrappers[2] is wrappers[0]
    assert fmha._mla_decode_wrappers[1] is wrappers[1]


def test_mla_wrapper_capture_uses_cached_plan_and_rejects_plan_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capturing = False
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: capturing,
    )
    wrapper = Mock()
    wrapper_factory = Mock(return_value=wrapper)
    monkeypatch.setattr(
        prims_mla_module,
        "BatchMLADecodePagedTSWrapper",
        wrapper_factory,
    )
    attn = _Attention(head_dim=576, is_mla=True, num_heads=4)
    fmha = PrimsTSFmha(attn)
    block_tables = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    seq_lens = torch.tensor([33], dtype=torch.int32)
    workspace = torch.empty(64, dtype=torch.uint8)
    planned = _get_test_mla_wrapper(fmha, block_tables, seq_lens, workspace)
    capturing = True

    cached = _get_test_mla_wrapper(fmha, block_tables, seq_lens, workspace)
    with pytest.raises(RuntimeError, match="must be planned before CUDA graph capture"):
        _get_test_mla_wrapper(
            fmha,
            torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32),
            torch.tensor([33, 64], dtype=torch.int32),
            workspace,
        )

    assert cached is planned is wrapper
    wrapper_factory.assert_called_once_with()
    wrapper.plan.assert_called_once()


@pytest.mark.parametrize("is_cuda_graph", [False, True], ids=["eager", "cuda-graph"])
def test_mla_wrapper_receives_v2_bound_and_shared_workspace(
    monkeypatch: pytest.MonkeyPatch,
    is_cuda_graph: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = _Attention(head_dim=576, is_mla=True, num_heads=4)
    fmha = PrimsTSFmha(attn)
    kv_cache = torch.empty((20, 1, 32, 576), dtype=torch.bfloat16)
    block_tables = torch.tensor(
        [
            [[0, 1, 2], [10, 11, 12]],
            [[3, 4, 5], [13, 14, 15]],
        ],
        dtype=torch.int32,
    )
    build_metadata = Mock(return_value=(kv_cache, block_tables, None))
    wrapper = Mock()
    wrapper_factory = Mock(return_value=wrapper)
    monkeypatch.setattr(
        prims_ts_module.thop,
        "build_trtllm_gen_kv_cache_metadata",
        build_metadata,
    )
    monkeypatch.setattr(
        prims_ts_package,
        "get_prims_ts_batch_decode_mla_workspace_size",
        Mock(return_value=96),
    )
    monkeypatch.setattr(
        prims_mla_module,
        "BatchMLADecodePagedTSWrapper",
        wrapper_factory,
    )

    get_page_index_upper_bound = Mock(return_value=20)
    metadata = SimpleNamespace(
        is_cuda_graph=is_cuda_graph,
        beam_width=1,
        kv_cache_block_offsets=torch.empty((2, 2, 3), dtype=torch.int32),
        host_kv_cache_pool_pointers=torch.tensor([1234], dtype=torch.int64),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
        kv_cache_manager=_make_v2_manager(
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
        workspace=torch.empty(96, dtype=torch.uint8),
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
        batch_size=2,
        num_requests=2,
    )

    fmha.run_mla_generation(params)

    wrapper_factory.assert_called_once_with()
    wrapper.plan.assert_called_once()
    plan_args = wrapper.plan.call_args.args
    plan_kwargs = wrapper.plan.call_args.kwargs
    assert plan_args == (
        params.workspace.device,
        2,
        attn.num_heads,
        512,
        64,
        32,
        96,
    )
    assert plan_kwargs["packed_query"] is False
    assert plan_kwargs["workspace_buffer"].data_ptr() == params.workspace.data_ptr()
    assert plan_kwargs["workspace_buffer"].numel() == 96
    wrapper.run.assert_called_once()
    run_args = wrapper.run.call_args.args
    run_kwargs = wrapper.run.call_args.kwargs
    assert run_args[0].shape == (2, 1, attn.num_heads, 576)
    assert run_args[0].data_ptr() == params.qkv_input.data_ptr()
    assert run_args[1] is kv_cache
    torch.testing.assert_close(
        run_kwargs["block_tables"],
        torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32),
    )
    assert run_kwargs["block_tables"].shape == (2, 3)
    assert run_kwargs["block_tables"].stride() == (6, 1)
    assert run_kwargs["block_tables"].data_ptr() == block_tables.data_ptr()
    torch.testing.assert_close(run_kwargs["seq_lens"], params.sequence_lengths)
    assert run_kwargs["seq_lens"].data_ptr() == params.sequence_lengths.data_ptr()
    assert run_kwargs["out"].shape == (2, 1, attn.num_heads, 512)
    assert run_kwargs["out"].data_ptr() == output.data_ptr()
    assert run_kwargs["bmm1_scale"] == pytest.approx(1.0 / math.sqrt(128 + 64))
    assert run_kwargs["bmm2_scale"] == 1.0
    assert run_kwargs["validate"] is False
    builder_args = build_metadata.call_args.args
    assert builder_args[8] == total_num_blocks
    assert builder_args[10] == params.seq_offset
    assert builder_args[11] == 2
    assert builder_args[12] == torch.bfloat16
    get_page_index_upper_bound.assert_called_once()


@pytest.mark.parametrize("is_cuda_graph", [False, True], ids=["eager", "cuda-graph"])
def test_mla_prepare_workspace_sizes_caller_owned_workspace(
    monkeypatch: pytest.MonkeyPatch,
    is_cuda_graph: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    workspace_size = Mock(return_value=48)
    monkeypatch.setattr(
        prims_ts_package,
        "get_prims_ts_batch_decode_mla_workspace_size",
        workspace_size,
    )
    attn = _Attention(head_dim=576, is_mla=True, num_heads=4)
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 1
    q = torch.empty((2, attn.num_heads * 576), dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        is_cuda_graph=is_cuda_graph,
        kv_cache_block_offsets=torch.empty((2, 2, 3), dtype=torch.int32),
        max_num_requests=2,
        num_contexts=0,
        num_generations=2,
        num_ctx_tokens=0,
        tokens_per_block=32,
        kv_lens_runtime=torch.tensor([33, 64], dtype=torch.int32),
    )
    output = torch.empty((2, attn.num_heads * 512), dtype=torch.bfloat16)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=96,
    )

    workspace = torch.empty(0, dtype=torch.uint8)
    fmha.prepare_workspace(
        q,
        None,
        None,
        metadata,
        forward_args,
        workspace,
    )

    workspace_size.assert_called_once()
    assert workspace.numel() == 48


def test_mla_prepare_workspace_preserves_cached_wrappers_with_stable_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    stream = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=stream))
    workspace_size = Mock(return_value=48)
    monkeypatch.setattr(
        prims_ts_package,
        "get_prims_ts_batch_decode_mla_workspace_size",
        workspace_size,
    )
    attn = _Attention(head_dim=576, is_mla=True, num_heads=4)
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 1
    workspace = torch.empty(48, dtype=torch.uint8)
    fmha._update_workspace_allocation(workspace)
    wrapper = Mock()
    fmha._mla_decode_wrappers[2] = wrapper
    q = torch.empty((2, attn.num_heads * 576), dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        is_cuda_graph=True,
        kv_cache_block_offsets=torch.empty((2, 2, 3), dtype=torch.int32),
        max_num_requests=2,
        num_contexts=0,
        num_generations=2,
        num_ctx_tokens=0,
        tokens_per_block=32,
        kv_lens_runtime=torch.tensor([33, 64], dtype=torch.int32),
    )
    output = torch.empty((2, attn.num_heads * 512), dtype=torch.bfloat16)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=96,
    )
    fmha.prepare_workspace(q, None, None, metadata, forward_args, workspace)

    stream.synchronize.assert_not_called()
    assert fmha._mla_decode_wrappers[2] is wrapper
    assert workspace.numel() == 48
    workspace_size.assert_called_once()


def test_mla_caller_workspace_grows_across_plan_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    required_bytes = iter((32, 64))
    monkeypatch.setattr(
        prims_ts_package,
        "get_prims_ts_batch_decode_mla_workspace_size",
        lambda *args, **kwargs: next(required_bytes),
    )
    attn = _Attention(head_dim=576, is_mla=True, num_heads=4)
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 1
    q = torch.empty((2, attn.num_heads * 576), dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        is_cuda_graph=False,
        kv_cache_block_offsets=torch.empty((2, 2, 3), dtype=torch.int32),
        max_num_requests=2,
        num_contexts=0,
        num_generations=2,
        num_ctx_tokens=0,
        tokens_per_block=32,
        kv_lens_runtime=torch.tensor([33, 64], dtype=torch.int32),
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((2, attn.num_heads * 512), dtype=torch.bfloat16),
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=96,
    )
    workspace = torch.empty(0, dtype=torch.uint8)

    fmha.prepare_workspace(q, None, None, metadata, forward_args, workspace)
    assert workspace.numel() == 32

    dense_forward_args = replace(
        forward_args,
        attention_mask=PredefinedAttentionMask.FULL,
    )
    fmha.prepare_workspace(q, None, None, metadata, dense_forward_args, workspace)

    assert workspace.numel() == 64


def test_decode_prepare_workspace_reserves_tail_after_compact_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        prims_ts_module.thop,
        "get_trtllm_gen_generation_workspace_layout",
        lambda *args, **kwargs: {"total_size": 64},
    )
    monkeypatch.setattr(
        prims_ts_package,
        "get_prims_ts_batch_decode_workspace_size",
        lambda *args, **kwargs: 48,
    )
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 1
    metadata = SimpleNamespace(
        kv_cache_block_offsets=torch.empty((2, 2, 4), dtype=torch.int32),
        max_num_requests=2,
        num_contexts=0,
        num_generations=2,
        num_ctx_tokens=0,
        tokens_per_block=32,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((2, 8 * 128), dtype=torch.bfloat16),
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=128,
    )
    workspace = torch.empty(0, dtype=torch.uint8)

    fmha.prepare_workspace(
        torch.empty((2, 12 * 128), dtype=torch.bfloat16),
        None,
        None,
        metadata,
        forward_args,
        workspace,
    )

    assert fmha._decode_workspace_offset_bytes == 64
    assert fmha._decode_workspace_required_bytes == 48
    assert workspace.numel() == 112
    decode_workspace = fmha._get_decode_workspace(workspace)
    assert decode_workspace.data_ptr() == workspace.data_ptr() + 64
    assert decode_workspace.numel() == 48


def test_decode_workspace_tail_is_stable_across_mixed_context_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    context_layout = Mock(
        side_effect=(
            {"total_size": 64},
            {"total_size": 320},
        )
    )
    monkeypatch.setattr(
        prims_ts_module.thop,
        "get_trtllm_gen_context_workspace_layout",
        context_layout,
    )
    monkeypatch.setattr(
        prims_ts_module.thop,
        "get_trtllm_gen_generation_workspace_layout",
        lambda *args, **kwargs: {"total_size": 64},
    )
    monkeypatch.setattr(
        prims_ts_package,
        "get_prims_ts_batch_decode_workspace_size",
        lambda *args, **kwargs: 48,
    )
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 1
    metadata = SimpleNamespace(
        kv_cache_block_offsets=torch.empty((4, 2, 4), dtype=torch.int32),
        max_num_requests=4,
        num_contexts=2,
        num_generations=2,
        num_ctx_tokens=4,
        tokens_per_block=32,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((6, 8 * 128), dtype=torch.bfloat16),
        attention_input_type=AttentionInputType.mixed,
        attention_window_size=128,
    )
    q = torch.empty((6, 12 * 128), dtype=torch.bfloat16)
    workspace = torch.empty(1024, dtype=torch.uint8)

    fmha.prepare_workspace(q, None, None, metadata, forward_args, workspace)
    first_workspace = fmha._get_decode_workspace(workspace)
    cached_wrapper = Mock()
    fmha._decode_wrappers[2] = cached_wrapper

    fmha.prepare_workspace(q, None, None, metadata, forward_args, workspace)
    second_workspace = fmha._get_decode_workspace(workspace)

    assert context_layout.call_count == 2
    assert fmha._decode_wrappers[2] is cached_wrapper
    assert fmha._decode_workspace_offset_bytes == 960
    assert first_workspace.data_ptr() == second_workspace.data_ptr()
    assert first_workspace.numel() == second_workspace.numel() == 48


def test_workspace_cannot_grow_during_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 1
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        prims_ts_module.thop,
        "get_trtllm_gen_generation_workspace_layout",
        lambda *args, **kwargs: {"total_size": 32},
    )
    monkeypatch.setattr(
        prims_ts_package,
        "get_prims_ts_batch_decode_workspace_size",
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
        match="PrimTS caller workspace must be sized before CUDA graph capture",
    ):
        fmha.prepare_workspace(
            torch.empty((2, 12 * 128), dtype=torch.bfloat16),
            None,
            None,
            metadata,
            forward_args,
            torch.empty(16, dtype=torch.uint8),
        )


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
    assert context_params.num_requests == 1
    assert context_params.attention_input is not None
    assert context_params.attention_input.shape[0] == 3
    assert context_params.context_buf is not None
    assert context_params.context_buf.shape == (3, attn.num_heads, attn.head_dim)

    run_generation.assert_called_once()
    generation_params = generation_calls[0]
    assert generation_params.num_tokens == 2
    assert generation_params.seq_offset == 1
    assert generation_params.batch_size == 2
    assert generation_params.num_requests == 2
    assert generation_params.attention_input is not None
    assert generation_params.attention_input.shape[0] == 2
    assert generation_params.context_buf is not None
    assert generation_params.context_buf.shape == (2, attn.num_heads, attn.head_dim)
