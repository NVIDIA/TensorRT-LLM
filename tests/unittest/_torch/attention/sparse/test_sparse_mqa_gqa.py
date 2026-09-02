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

"""Executable examples and regression tests for sparse MQA/GQA compute.

The token-sparse tests replace the model-specific sparse selector with static
token-index lists, then exercise the same backend path used by
``TrtllmAttention``:

1. Build request-local ``int32`` token indices.
2. Translate them to paged KV-cache pool indices.
3. Return them from the sparse prediction hooks.
4. Call ``TrtllmAttention.forward`` and compare with a PyTorch reference.

The block-sparse tests pass static block-index lists to the MSA FMHA wrapper
and compare its paged MQA/GQA output with an independent PyTorch reference.
Algorithm-independent sparse framework tests remain in
``test_sparse_attention.py``.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pytest
import torch
from utils.util import getSMVersion

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.fmha.msa_sparse_gqa import run_msa_sparse_gqa
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.attention_backend.sparse.dsa.kernels import (
    triton_convert_req_index_to_global_index,
)
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import msa_package_available
from tensorrt_llm._torch.attention_backend.sparse.params import SparseParams
from tensorrt_llm._torch.attention_backend.trtllm import (
    TrtllmAttention,
    TrtllmAttentionMetadata,
    generate_spec_decoding_packed_mask,
    generate_spec_decoding_position_offsets,
)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

ATOL = 2e-2
RTOL = 2e-2
FP8_ATOL = 4e-1
FP8_RTOL = 4e-2
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
SUPPORTED_KV_CACHE_DTYPES = (*SUPPORTED_DTYPES, torch.float8_e4m3fn)
SUPPORTED_HEAD_DIMS = (64, 80, 128, 256)
SUPPORTED_FP8_HEAD_DIMS = (64, 128, 256)
SUPPORTED_TEST_PAGE_SIZES = (8, 16, 32, 64, 128, 256, 512)
TOKEN_SPARSE_PAGE_TEST_KV_LEN = 64
MAX_Q_HEADS_PER_KV_HEAD = 32
SUPPORTED_SPARSE_MQA_GQA_SMS = (100, 103)

pytestmark = pytest.mark.skipif(
    getSMVersion() not in SUPPORTED_SPARSE_MQA_GQA_SMS,
    reason="Sparse MQA/GQA requires an SM100 or SM103 GPU",
)


@pytest.fixture(autouse=True)
def _force_trtllm_gen_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep token-sparse tests on the internal TRTLLM-Gen fallback path."""
    monkeypatch.setenv("TLLM_FMHA_LIBS", "fallback")


def _fp8_qdq(tensor: torch.Tensor) -> torch.Tensor:
    """Apply unit-scale E4M3 quantize-dequantize to a reference tensor."""
    return tensor.to(torch.float8_e4m3fn).to(tensor.dtype)


# Kernel contract and static selector adapter.


@dataclass(kw_only=True, frozen=True)
class SparseMqaGqaScenario:
    """Kernel geometry shared by context and generation scenarios.

    The validation mirrors the supported static token-sparse kernel contract,
    so every scenario is also a compact declaration of a supported shape.
    """

    dtype: torch.dtype = torch.bfloat16
    kvcache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 1
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    page_size: int = 32
    num_pages: int = 16
    batch_size: int = 1
    num_sparse_topk: int = 64

    def __post_init__(self) -> None:
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError("Model QKV dtype must be BF16 or FP16")
        if self.kvcache_dtype not in SUPPORTED_KV_CACHE_DTYPES:
            raise ValueError("KV-cache dtype must be BF16, FP16, or E4M3 FP8")
        if self.kvcache_dtype != torch.float8_e4m3fn and self.kvcache_dtype != self.dtype:
            raise ValueError("A non-FP8 KV-cache dtype must match the model QKV dtype")
        if self.head_dim not in SUPPORTED_HEAD_DIMS:
            raise ValueError(f"head_dim must be one of {SUPPORTED_HEAD_DIMS}")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.q_heads_per_kv_head > MAX_Q_HEADS_PER_KV_HEAD:
            raise ValueError(f"at most {MAX_Q_HEADS_PER_KV_HEAD} query heads may share one KV head")
        if self.page_size < 8 or self.page_size & (self.page_size - 1):
            raise ValueError("page_size must be a power of two and at least 8")
        if self.num_sparse_topk <= 0 or self.num_sparse_topk % 4 != 0:
            raise ValueError("num_sparse_topk must be a positive multiple of 4")

    @property
    def q_heads_per_kv_head(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def kv_pool_num_pages(self) -> int:
        return self.batch_size * self.num_pages


@dataclass(kw_only=True, frozen=True)
class ContextScenario(SparseMqaGqaScenario):
    """Packed context requests used for cache compaction or sparse compute."""

    seq_lens: Tuple[int, ...] = (128,)

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.seq_lens) != self.batch_size:
            raise ValueError(
                f"seq_lens length {len(self.seq_lens)} must match batch_size {self.batch_size}"
            )

    @property
    def max_seq_len(self) -> int:
        return max(self.seq_lens)

    @property
    def nnz_q(self) -> int:
        return sum(self.seq_lens)


@dataclass(kw_only=True, frozen=True)
class GenerationScenario(SparseMqaGqaScenario):
    """One decode token per request with an existing paged KV history."""

    past_kv_lens: Tuple[int, ...] = (256,)
    query_len: int = 1
    fp8_output: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.past_kv_lens) != self.batch_size:
            raise ValueError(
                f"past_kv_lens length {len(self.past_kv_lens)} must match batch_size {self.batch_size}"
            )
        if self.query_len < 1:
            raise ValueError("query_len must be positive")
        if self.fp8_output and self.kvcache_dtype != torch.float8_e4m3fn:
            raise ValueError("FP8 output testing requires an FP8 KV cache")

    @property
    def num_generations(self) -> int:
        return self.batch_size

    @property
    def nnz_q(self) -> int:
        return self.batch_size * self.query_len

    @property
    def max_query_len(self) -> int:
        return self.query_len

    @property
    def has_draft_tokens(self) -> bool:
        return self.max_query_len > 1


@dataclass(kw_only=True, frozen=True)
class BlockSparseGqaScenario:
    """Packed block-sparse MQA/GQA inputs for the MSA FMHA backend."""

    q_lens: Tuple[int, ...] = (1,)
    kv_lens: Tuple[int, ...] = (2176,)
    num_q_heads: int = 16
    num_kv_heads: int = 1
    head_dim: int = 128
    page_size: int = 128
    topk: int = 16
    dtype: torch.dtype = torch.bfloat16
    qo_offsets: Optional[Tuple[int, ...]] = None
    active_blocks: Optional[int] = None
    shuffle_pages: bool = False
    per_token_blocks: bool = False

    def __post_init__(self) -> None:
        if len(self.q_lens) != len(self.kv_lens):
            raise ValueError("q_lens and kv_lens must describe the same batch")
        if self.qo_offsets is not None and len(self.qo_offsets) != len(self.q_lens):
            raise ValueError("qo_offsets must describe the same batch as q_lens")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        if self.q_heads_per_kv_head not in (2, 4, 8, 16):
            raise ValueError("block-sparse MQA/GQA supports 2, 4, 8, or 16 Q heads per KV head")
        if self.head_dim != 128 or self.page_size != 128:
            raise ValueError("MSA block-sparse MQA/GQA requires head_dim=page_size=128")
        if self.topk not in (4, 8, 16, 32):
            raise ValueError("MSA block-sparse MQA/GQA supports Top-K 4, 8, 16, or 32")
        if self.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError("MSA block-sparse MQA/GQA supports BF16 or E4M3 FP8 Q/K/V")
        active_blocks = self.topk if self.active_blocks is None else self.active_blocks
        if not 0 < active_blocks <= self.topk:
            raise ValueError("active_blocks must be in [1, topk]")
        for q_len, kv_len, qo_offset in zip(
            self.q_lens,
            self.kv_lens,
            self.causal_offsets,
            strict=True,
        ):
            if q_len <= 0 or q_len > kv_len:
                raise ValueError("each q_len must be positive and no larger than kv_len")
            if qo_offset < 0 or qo_offset + q_len > kv_len:
                raise ValueError("each qo_offset must place every query inside its KV sequence")
            if kv_len % self.page_size:
                raise ValueError("each kv_len must be a multiple of page_size")
            if kv_len // self.page_size <= active_blocks:
                raise ValueError("each request needs more KV pages than selected active blocks")

    @property
    def q_heads_per_kv_head(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    @property
    def total_q(self) -> int:
        return sum(self.q_lens)

    @property
    def selected_blocks(self) -> int:
        return self.topk if self.active_blocks is None else self.active_blocks

    @property
    def causal_offsets(self) -> Tuple[int, ...]:
        if self.qo_offsets is not None:
            return self.qo_offsets
        return tuple(
            kv_len - q_len for q_len, kv_len in zip(self.q_lens, self.kv_lens, strict=True)
        )


class _SparseMqaGqaParams(SparseParams):
    """Token-granular parameters that select the internal MQA/GQA path."""

    algorithm: str = "mqa_gqa"

    @property
    def indices_block_size(self) -> int:
        return 1


@dataclass
class _SparseMqaGqaMetadata(TrtllmAttentionMetadata):
    """Attention metadata extended with the static sparse Top-K."""

    num_sparse_topk: int = 64


class _StaticSparseMqaGqaAttention(TrtllmAttention):
    """Backend adapter that replaces a model selector with static indices."""

    def __init__(
        self,
        *args,
        sparse_kv_indices: Optional[torch.Tensor] = None,
        sparse_kv_offsets: Optional[torch.Tensor] = None,
        sparse_attn_indices: Optional[torch.Tensor] = None,
        sparse_attn_offsets: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        kwargs["sparse_params"] = _SparseMqaGqaParams()
        kwargs["pos_embd_params"] = None
        super().__init__(*args, **kwargs)

        self._sparse_kv_indices = sparse_kv_indices
        self._sparse_kv_offsets = sparse_kv_offsets
        self._sparse_attn_indices = sparse_attn_indices
        self._sparse_attn_offsets = sparse_attn_offsets

    def sparse_kv_predict(self, q, k, metadata, forward_args: AttentionForwardArgs):
        return self._sparse_kv_indices, self._sparse_kv_offsets

    def sparse_attn_predict(self, q, k, metadata, forward_args: AttentionForwardArgs):
        return self._sparse_attn_indices, self._sparse_attn_offsets


@dataclass(kw_only=True)
class _ContextInputs:
    """Packed context tensors plus their paged KV-cache metadata."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    kv_cache_manager: KVCacheManager
    request_ids: List[int]
    metadata: _SparseMqaGqaMetadata

    @property
    def fused_qkv(self) -> torch.Tensor:
        return torch.cat([self.q, self.k, self.v], dim=1)


@dataclass(kw_only=True)
class _GenerationInputs:
    """Decode tensors, local token selections, and populated paged KV cache."""

    q: torch.Tensor
    k_new: torch.Tensor
    v_new: torch.Tensor
    local_sparse_attn_indices: torch.Tensor
    kv_cache_manager: KVCacheManager
    request_ids: List[int]
    metadata: _SparseMqaGqaMetadata

    @property
    def fused_qkv(self) -> torch.Tensor:
        return torch.cat([self.q, self.k_new, self.v_new], dim=1)


# Sparse KV-cache feature tests.


_SPARSE_KV_CASES = [
    pytest.param(
        ContextScenario(batch_size=2, seq_lens=(48, 64), num_pages=8),
        id="batch2_var_seq",
    ),
    pytest.param(
        ContextScenario(batch_size=4, seq_lens=(96, 112, 128, 144), num_pages=16),
        id="batch4_var_seq",
    ),
    pytest.param(
        ContextScenario(batch_size=1, seq_lens=(256,), num_pages=8),
        id="batch1_seq256",
    ),
    pytest.param(
        ContextScenario(batch_size=3, seq_lens=(64, 96, 128), num_pages=12),
        id="batch3_var_seq",
    ),
    pytest.param(
        ContextScenario(
            batch_size=1,
            seq_lens=(128,),
            num_pages=8,
            num_heads=8,
            num_kv_heads=8,
        ),
        id="mha_8q8kv",
    ),
]


@pytest.mark.parametrize("scenario", _SPARSE_KV_CASES)
def test_prefill_sparse_kv_compaction(scenario: ContextScenario) -> None:
    """Sparse KV selection compacts the cache without changing dense prefill output.

    This test calls ``attention.forward``, but supplies only
    ``sparse_kv_indices``. Without ``sparse_attn_indices``, attention compute is
    dense; the sparse feature under test is the selected K/V write into the
    paged cache.
    """
    inputs = _create_context_inputs(scenario)
    local_sparse_kv_indices, sparse_kv_offsets = _make_context_kv_indices(
        scenario.seq_lens,
        scenario.num_kv_heads,
        scenario.num_sparse_topk,
        inputs.q.device,
    )
    attention = _StaticSparseMqaGqaAttention(
        layer_idx=0,
        num_heads=scenario.num_heads,
        head_dim=scenario.head_dim,
        num_kv_heads=scenario.num_kv_heads,
        quant_config=_quant_config(scenario),
        sparse_kv_indices=local_sparse_kv_indices,
        sparse_kv_offsets=sparse_kv_offsets,
    )

    try:
        reference_output = _reference_dense_context_attention(
            inputs.q, inputs.k, inputs.v, scenario
        )
        expected_kvs = _build_expected_compacted_kv(
            inputs.k,
            inputs.v,
            local_sparse_kv_indices,
            sparse_kv_offsets,
            scenario,
        )

        output = attention.forward(inputs.fused_qkv, None, None, inputs.metadata)
        assert output.shape == reference_output.shape
        torch.testing.assert_close(output, reference_output, atol=ATOL, rtol=RTOL)

        compacted_kv_lens = tuple(
            int((sparse_kv_offsets[i + 1] - sparse_kv_offsets[i]).item())
            for i in range(scenario.batch_size)
        )
        actual_kvs = _read_paged_kv_cache(
            inputs.kv_cache_manager,
            inputs.request_ids,
            compacted_kv_lens,
            scenario,
            scenario.dtype,
        )
        for batch_idx, ((actual_k, actual_v), (expected_k, expected_v)) in enumerate(
            zip(actual_kvs, expected_kvs, strict=True)
        ):
            torch.testing.assert_close(
                actual_k,
                expected_k,
                atol=ATOL,
                rtol=RTOL,
                msg=f"K cache mismatch for batch {batch_idx} after sparse compaction",
            )
            torch.testing.assert_close(
                actual_v,
                expected_v,
                atol=ATOL,
                rtol=RTOL,
                msg=f"V cache mismatch for batch {batch_idx} after sparse compaction",
            )
    finally:
        inputs.kv_cache_manager.shutdown()


# Sparse MQA/GQA computation tests.
#
# Sparse MQA/GQA support matrix:
#
# Values after "tested" are regression coverage, not narrower support constraints.
#
#   Parameter               Token-sparse                   Block-sparse
#   Sparse block size       1 token; tested                128 tokens; tested
#   GPU architecture        SM100 and SM103;               SM100 and SM103;
#                           tested on SM100                 tested on SM100
#   Compute phase           Packed prefill, single-token,  Packed prefill, single-token,
#                           and linear draft decode;        linear multi-query compute,
#                           tested q_len 1 and 4            and mixed batches; tested; the
#                                                          integrated MiniMax-M3 decode uses 1
#   Attention type          MQA/GQA; Q heads divisible    MQA/GQA; Q heads divisible
#                           by KV heads                    by KV heads
#   Q heads per KV head     <= 32; tested 2, 3, 4, 8,      2, 4, 8, or 16; tested all
#                           16, 24, 31, and 32              integrated MiniMax-M3 uses 16
#   Q/KV head counts        No additional discrete limit;  No additional discrete kernel limit;
#                           tested Q={6,8,16,32,48,62,64},  tested Q={4,8,16,32}, KV={1,2}
#                           KV={1,2,4,8}
#   Attention input dtype   BF16 or FP16; tested both      BF16 or E4M3 FP8; tested both
#   Q/K/V input layout      Fused QKV; tested              Q [T,Hq,D], paged K/V
#                                                          [P,Hkv,128,D]; tested
#   Kernel output           BF16/FP16 for all head dims;   BF16; tested
#                           E4M3 FP8 for 64/128/256;
#                           tested all dtype/dim pairs
#   KV-cache dtype          BF16/FP16 for all head dims;   BF16 or E4M3 FP8; tested both
#                           E4M3 FP8 for 64/128/256;
#                           tested all dtype/dim pairs
#   Q/K/V head dimension   64, 80, 128, or 256;           128; tested
#                           tested all
#   KV-cache layout         Paged; power-of-two page size  Paged HND; page size 128;
#                           >= 8; tested 8 through 512      shuffled physical pages and
#                                                          strided outer page storage tested
#   Sparse indices          int32 physical token indices   int32 request-local block indices
#                           per KV head/query; tested       per KV head/query; per-token lists,
#                                                          -1 padding, and physical remap tested
#   Sparse Top-K            Positive multiple of 4;        Prefill kernel accepts 4, 8, 16,
#                           tested 4, 32, 64, and 128       or 32 and tests cover all; the
#                                                          integrated MiniMax-M3 path uses 16
#   Attention semantics     Causal; tested                 Causal with per-request Q offsets;
#                                                          bottom-right and custom offsets tested


# Token-granular sparse computation (block_size=1).


_PREFILL_COMPUTE_CASES = [
    pytest.param(
        ContextScenario(
            batch_size=2,
            seq_lens=(128, 64),
            num_pages=8,
            num_kv_heads=1,
            num_heads=8,
        ),
        id="mqa_8q1kv",
    ),
    pytest.param(
        ContextScenario(
            batch_size=2,
            seq_lens=(128, 64),
            num_pages=8,
            num_kv_heads=2,
            num_heads=8,
        ),
        id="gqa_8q2kv",
    ),
    pytest.param(
        ContextScenario(
            batch_size=3,
            seq_lens=(64, 96, 128),
            num_pages=12,
            num_kv_heads=4,
            num_heads=16,
        ),
        id="gqa_16q4kv_batch3",
    ),
    pytest.param(
        ContextScenario(
            batch_size=2,
            seq_lens=(64, 128),
            num_pages=8,
            num_kv_heads=4,
            num_heads=32,
        ),
        id="gqa_32q4kv",
    ),
    pytest.param(
        ContextScenario(
            batch_size=2,
            seq_lens=(64, 128),
            num_pages=8,
            num_kv_heads=1,
            num_heads=8,
            num_sparse_topk=4,
        ),
        id="topk4_very_sparse",
    ),
    pytest.param(
        ContextScenario(
            batch_size=2,
            seq_lens=(64, 128),
            num_pages=8,
            num_kv_heads=2,
            num_heads=8,
            num_sparse_topk=128,
        ),
        id="topk128_near_dense",
    ),
]


_GENERATION_CORRECTNESS_CASES = [
    pytest.param(
        GenerationScenario(batch_size=2, past_kv_lens=(96, 128), num_pages=16),
        id="batch2_var_kv",
    ),
    pytest.param(
        GenerationScenario(
            batch_size=4,
            past_kv_lens=(192, 224, 256, 288),
            num_pages=32,
        ),
        id="batch4_var_kv",
    ),
    pytest.param(
        GenerationScenario(batch_size=1, past_kv_lens=(64,), num_pages=8),
        id="batch1_kv64",
    ),
    pytest.param(
        GenerationScenario(
            batch_size=3,
            past_kv_lens=(128, 160, 192),
            num_pages=24,
        ),
        id="batch3_var_kv",
    ),
    pytest.param(
        GenerationScenario(
            num_heads=8,
            num_kv_heads=1,
            batch_size=2,
            past_kv_lens=(96, 128),
            num_pages=16,
        ),
        id="mqa_8q1kv",
    ),
    pytest.param(
        GenerationScenario(
            num_heads=16,
            num_kv_heads=4,
            batch_size=2,
            past_kv_lens=(128, 256),
            num_pages=16,
        ),
        id="gqa_16q4kv",
    ),
    pytest.param(
        GenerationScenario(
            num_heads=8,
            num_kv_heads=4,
            batch_size=2,
            past_kv_lens=(128, 256),
            num_pages=16,
        ),
        id="gqa_8q4kv",
    ),
    pytest.param(
        GenerationScenario(
            num_heads=32,
            num_kv_heads=1,
            batch_size=1,
            past_kv_lens=(128,),
            num_pages=8,
        ),
        id="mqa_group32_boundary",
    ),
    pytest.param(
        GenerationScenario(
            num_heads=64,
            num_kv_heads=2,
            batch_size=1,
            past_kv_lens=(128,),
            num_pages=8,
        ),
        id="gqa_group32_boundary",
    ),
    pytest.param(
        GenerationScenario(
            num_heads=6,
            num_kv_heads=2,
            batch_size=1,
            past_kv_lens=(64,),
            num_pages=8,
        ),
        id="gqa_group3_non_power_of_two",
    ),
    pytest.param(
        GenerationScenario(
            num_heads=32,
            num_kv_heads=2,
            batch_size=1,
            past_kv_lens=(64,),
            num_pages=8,
        ),
        id="gqa_group16",
    ),
    pytest.param(
        GenerationScenario(
            num_heads=48,
            num_kv_heads=2,
            batch_size=1,
            past_kv_lens=(64,),
            num_pages=8,
        ),
        id="gqa_group24",
    ),
    pytest.param(
        GenerationScenario(
            num_heads=62,
            num_kv_heads=2,
            batch_size=1,
            past_kv_lens=(64,),
            num_pages=8,
        ),
        id="gqa_group31",
    ),
    pytest.param(
        GenerationScenario(
            num_heads=8,
            num_kv_heads=4,
            batch_size=1,
            past_kv_lens=(64,),
            query_len=4,
            num_pages=8,
            num_sparse_topk=32,
        ),
        id="gqa_2to1_with_3_draft_tokens",
    ),
    pytest.param(
        GenerationScenario(
            batch_size=1,
            past_kv_lens=(128,),
            num_pages=8,
            num_sparse_topk=4,
        ),
        id="topk4_min",
    ),
    pytest.param(
        GenerationScenario(
            batch_size=2,
            past_kv_lens=(32, 256),
            num_pages=16,
            num_sparse_topk=128,
        ),
        id="topk128_exceeds_some",
    ),
    pytest.param(
        GenerationScenario(
            batch_size=8,
            past_kv_lens=(64, 96, 128, 160, 192, 224, 256, 288),
            num_pages=64,
        ),
        id="batch8_varied",
    ),
    pytest.param(
        GenerationScenario(
            page_size=64,
            batch_size=2,
            past_kv_lens=(64, 192),
            num_pages=8,
        ),
        id="page_size_64",
    ),
]


_GENERATION_SUPPORT_CASES = (
    [
        pytest.param(
            GenerationScenario(
                dtype=dtype,
                kvcache_dtype=dtype,
                num_heads=8,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                batch_size=1,
                past_kv_lens=(64,),
                num_pages=4,
                num_sparse_topk=32,
            ),
            id=(
                f"support_{str(dtype).removeprefix('torch.')}_h{head_dim}_"
                f"{'mqa' if num_kv_heads == 1 else 'gqa_2to1'}"
            ),
        )
        for dtype in SUPPORTED_DTYPES
        for head_dim in SUPPORTED_HEAD_DIMS
        for num_kv_heads in (1, 4)
        # These two option combinations are already covered by correctness cases.
        if not (dtype == torch.bfloat16 and head_dim == 128)
    ]
    + [
        pytest.param(
            GenerationScenario(
                dtype=torch.bfloat16,
                kvcache_dtype=torch.float8_e4m3fn,
                num_heads=8,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                batch_size=1,
                past_kv_lens=(64,),
                num_pages=8,
                num_sparse_topk=32,
                fp8_output=fp8_output,
            ),
            id=(
                f"support_fp8_kv_h{head_dim}_"
                f"{'mqa' if num_kv_heads == 1 else 'gqa_2to1'}_"
                f"{'fp8' if fp8_output else 'bf16'}_output"
            ),
        )
        for head_dim in SUPPORTED_FP8_HEAD_DIMS
        for num_kv_heads in (1, 4)
        for fp8_output in (False, True)
    ]
    + [
        pytest.param(
            GenerationScenario(
                page_size=page_size,
                batch_size=1,
                past_kv_lens=(TOKEN_SPARSE_PAGE_TEST_KV_LEN,),
                num_pages=max(
                    4,
                    (TOKEN_SPARSE_PAGE_TEST_KV_LEN + page_size) // page_size,
                ),
                num_sparse_topk=32,
            ),
            id=f"support_page_size_{page_size}",
        )
        for page_size in SUPPORTED_TEST_PAGE_SIZES
        if page_size not in (32, 64)
    ]
)


@pytest.mark.parametrize("scenario", _PREFILL_COMPUTE_CASES)
def test_prefill_sparse_mqa_gqa(scenario: ContextScenario) -> None:
    """Prefill sparse indices drive sparse compute and compacted KV writes."""
    inputs = _create_context_inputs(scenario)
    try:
        available_kv_lens = tuple(
            token_idx + 1 for seq_len in scenario.seq_lens for token_idx in range(seq_len)
        )
        local_attn_indices = _make_sparse_attention_indices(
            available_kv_lens,
            scenario.num_kv_heads,
            scenario.num_sparse_topk,
            inputs.q.device,
        )
        cache_pool_indices = _local_to_cache_pool_indices(
            local_attn_indices, inputs.metadata, layer_idx=0
        )
        local_sparse_kv_indices, sparse_kv_offsets = _make_context_kv_indices(
            scenario.seq_lens,
            scenario.num_kv_heads,
            scenario.num_sparse_topk,
            inputs.q.device,
        )

        attention = _StaticSparseMqaGqaAttention(
            layer_idx=0,
            num_heads=scenario.num_heads,
            head_dim=scenario.head_dim,
            num_kv_heads=scenario.num_kv_heads,
            quant_config=_quant_config(scenario),
            sparse_kv_indices=local_sparse_kv_indices,
            sparse_kv_offsets=sparse_kv_offsets,
            sparse_attn_indices=cache_pool_indices,
        )
        reference_output = _reference_sparse_context_attention(
            inputs.q, inputs.k, inputs.v, local_attn_indices, scenario
        )

        output = attention.forward(inputs.fused_qkv, None, None, inputs.metadata)
        expected_shape = (scenario.nnz_q, scenario.num_heads * scenario.head_dim)
        assert output.shape == expected_shape
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, reference_output, atol=ATOL, rtol=RTOL)
    finally:
        inputs.kv_cache_manager.shutdown()


@pytest.mark.parametrize(
    "scenario",
    _GENERATION_CORRECTNESS_CASES + _GENERATION_SUPPORT_CASES,
)
def test_generation_sparse_mqa_gqa(scenario: GenerationScenario) -> None:
    """Decode static token selections match the paged-cache PyTorch reference."""
    inputs = _create_generation_inputs(scenario)
    try:
        cache_pool_indices = _local_to_cache_pool_indices(
            inputs.local_sparse_attn_indices, inputs.metadata, layer_idx=0
        )
        attention = _StaticSparseMqaGqaAttention(
            layer_idx=0,
            num_heads=scenario.num_heads,
            head_dim=scenario.head_dim,
            num_kv_heads=scenario.num_kv_heads,
            quant_config=_quant_config(scenario),
            sparse_attn_indices=cache_pool_indices,
        )

        kv_caches = _read_paged_kv_cache(
            inputs.kv_cache_manager,
            inputs.request_ids,
            scenario.past_kv_lens,
            scenario,
            scenario.dtype,
        )
        reference_q = inputs.q
        reference_k_new = inputs.k_new
        reference_v_new = inputs.v_new
        if scenario.kvcache_dtype == torch.float8_e4m3fn:
            reference_q = _fp8_qdq(reference_q)
            reference_k_new = _fp8_qdq(reference_k_new)
            reference_v_new = _fp8_qdq(reference_v_new)
        reference_output = _reference_sparse_generation_attention(
            reference_q,
            kv_caches,
            reference_k_new,
            reference_v_new,
            inputs.local_sparse_attn_indices,
            scenario,
        )

        forward_args = None
        if scenario.fp8_output:
            forward_args = AttentionForwardArgs(
                out_scale=torch.ones(1, dtype=torch.float32, device=inputs.q.device)
            )
        output = attention.forward(
            inputs.fused_qkv,
            None,
            None,
            inputs.metadata,
            forward_args=forward_args,
        )
        expected_shape = (scenario.nnz_q, scenario.num_heads * scenario.head_dim)
        assert output.shape == expected_shape
        uses_fp8 = scenario.kvcache_dtype == torch.float8_e4m3fn or scenario.fp8_output
        output_for_comparison = output.float() if uses_fp8 else output
        if scenario.fp8_output:
            reference_output = reference_output.to(torch.float8_e4m3fn)
        reference_for_comparison = reference_output.float() if uses_fp8 else reference_output
        assert torch.isfinite(output_for_comparison).all()
        if scenario.fp8_output:
            assert output.dtype == torch.float8_e4m3fn
        torch.testing.assert_close(
            output_for_comparison,
            reference_for_comparison,
            atol=FP8_ATOL if uses_fp8 else ATOL,
            rtol=FP8_RTOL if uses_fp8 else RTOL,
        )
    finally:
        inputs.kv_cache_manager.shutdown()


# Block-granular sparse computation (block_size=128).


_BLOCK_SPARSE_GQA_CASES = [
    pytest.param(
        BlockSparseGqaScenario(
            q_lens=(1, 1, 1, 1),
            kv_lens=(2176, 2304, 2432, 2560),
        ),
        id="msa_mqa_single_token_varlen_batch4",
    ),
    pytest.param(
        BlockSparseGqaScenario(
            q_lens=(4, 4),
            kv_lens=(2176, 2304),
            num_q_heads=32,
            num_kv_heads=2,
            per_token_blocks=True,
        ),
        id="msa_gqa_linear_draft_tokens",
    ),
    *[
        pytest.param(
            BlockSparseGqaScenario(
                q_lens=(1,),
                kv_lens=(2176,),
                num_q_heads=2 * q_heads_per_kv_head,
                num_kv_heads=2,
            ),
            id=f"msa_gqa_single_token_{q_heads_per_kv_head}q_per_kv",
        )
        for q_heads_per_kv_head in (2, 4, 8)
    ],
    pytest.param(
        BlockSparseGqaScenario(
            q_lens=(1, 4, 33),
            kv_lens=(1152, 1280, 1408),
            num_q_heads=8,
            num_kv_heads=2,
            active_blocks=3,
            shuffle_pages=True,
            per_token_blocks=True,
        ),
        id="msa_gqa_mixed_varlen_shuffled_pages_padded_indices",
    ),
    pytest.param(
        BlockSparseGqaScenario(
            q_lens=(33,),
            kv_lens=(640,),
            num_q_heads=4,
            num_kv_heads=2,
            topk=4,
        ),
        id="msa_gqa_2q_per_kv_topk4",
    ),
    pytest.param(
        BlockSparseGqaScenario(
            q_lens=(33,),
            kv_lens=(1152,),
            num_q_heads=8,
            num_kv_heads=2,
            topk=8,
        ),
        id="msa_gqa_4q_per_kv_topk8",
    ),
    pytest.param(
        BlockSparseGqaScenario(
            q_lens=(33,),
            kv_lens=(2176,),
            num_q_heads=16,
            num_kv_heads=2,
            topk=16,
        ),
        id="msa_gqa_8q_per_kv_topk16",
    ),
    pytest.param(
        BlockSparseGqaScenario(
            q_lens=(33,),
            kv_lens=(4224,),
            num_q_heads=32,
            num_kv_heads=2,
            topk=32,
        ),
        id="msa_gqa_16q_per_kv_topk32",
    ),
    pytest.param(
        BlockSparseGqaScenario(
            q_lens=(33,),
            kv_lens=(2176,),
            num_q_heads=16,
            num_kv_heads=1,
            dtype=torch.float8_e4m3fn,
            per_token_blocks=True,
        ),
        id="msa_mqa_fp8_qkv_bf16_output",
    ),
    pytest.param(
        BlockSparseGqaScenario(
            q_lens=(33, 40),
            kv_lens=(2176, 2304),
            qo_offsets=(512, 1024),
            num_q_heads=16,
            num_kv_heads=2,
        ),
        id="msa_gqa_custom_per_request_q_offsets",
    ),
    pytest.param(
        BlockSparseGqaScenario(
            q_lens=(1, 1),
            kv_lens=(2176, 2304),
            num_q_heads=32,
            num_kv_heads=2,
            dtype=torch.float8_e4m3fn,
        ),
        id="msa_gqa_fp8_single_token",
    ),
]


@pytest.mark.parametrize("scenario", _BLOCK_SPARSE_GQA_CASES)
def test_block_sparse_mqa_gqa(scenario: BlockSparseGqaScenario) -> None:
    """MSA-selected KV blocks match a direct PyTorch block-sparse reference."""
    if not msa_package_available():
        pytest.skip("fmha_sm100 (MSA) is not importable")

    inputs = _create_block_sparse_gqa_inputs(scenario)
    output = torch.empty(
        scenario.total_q,
        scenario.num_q_heads,
        scenario.head_dim,
        dtype=torch.bfloat16,
        device=inputs["q"].device,
    )
    run_msa_sparse_gqa(
        inputs["q"],
        inputs["k_paged"],
        inputs["v_paged"],
        inputs["kv_block_indexes"],
        kv_indices=inputs["kv_indices"],
        sm_scale=scenario.head_dim**-0.5,
        qo_lens_cpu=torch.tensor(scenario.q_lens, dtype=torch.int32),
        kv_lens_cpu=torch.tensor(scenario.kv_lens, dtype=torch.int32),
        qo_offset_cpu=torch.tensor(scenario.causal_offsets, dtype=torch.int32),
        causal=True,
        head_dim=scenario.head_dim,
        out=output,
        use_fp8=scenario.dtype == torch.float8_e4m3fn,
    )
    torch.cuda.synchronize()

    reference = _reference_block_sparse_gqa(inputs, scenario)
    output_float = output.float()
    reference_float = reference.float()
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output_float).all()
    cosine_similarity = torch.nn.functional.cosine_similarity(
        output_float.flatten(), reference_float.flatten(), dim=0
    )
    threshold = 0.999 if scenario.dtype == torch.float8_e4m3fn else 0.9999
    assert cosine_similarity > threshold


# Sparse index and paged KV-cache helpers.


def _quant_config(s: SparseMqaGqaScenario) -> Optional[QuantConfig]:
    if isinstance(s, GenerationScenario) and s.fp8_output:
        return QuantConfig(
            quant_algo=QuantAlgo.FP8,
            kv_cache_quant_algo=QuantAlgo.FP8,
        )
    if s.kvcache_dtype == torch.float8_e4m3fn:
        return QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    return None


def _create_kv_cache_manager(
    s: SparseMqaGqaScenario, kv_cache: Optional[torch.Tensor] = None
) -> KVCacheManager:
    """Create kv cache manager for testing."""
    kv_cache_config = KvCacheConfig(max_tokens=s.kv_pool_num_pages * s.page_size)
    mapping = Mapping(world_size=1, tp_size=1, rank=0)

    manager = KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=s.num_layers,
        num_kv_heads=s.num_kv_heads,
        head_dim=s.head_dim,
        tokens_per_block=s.page_size,
        max_seq_len=s.kv_pool_num_pages * s.page_size,
        max_batch_size=s.batch_size,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(s.kvcache_dtype)),
    )

    if kv_cache is not None:
        for i in range(s.num_layers):
            manager.get_buffers(i, kv_layout="HND").copy_(kv_cache[i])

    return manager


def _make_context_kv_indices(
    seq_lens: Tuple[int, ...],
    num_kv_heads: int,
    num_sparse_topk: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate sparse kv indices for context phase.

    For each request, pick min(num_sparse_topk, seq_len) indices from [0, seq_len).
    Returns (indices [num_kv_heads, total_sparse], offsets [num_requests + 1]).
    """
    all_indices = []
    offsets = [0]

    for seq_len in seq_lens:
        pick = min(num_sparse_topk, seq_len)
        batch_indices = []
        for _ in range(num_kv_heads):
            indices = torch.randperm(seq_len, device=device)[:pick].sort().values
            batch_indices.append(indices)
        all_indices.append(torch.stack(batch_indices, dim=0))
        offsets.append(offsets[-1] + pick)

    indices = torch.cat(all_indices, dim=1).int()
    offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    return indices, offsets


def _make_sparse_attention_indices(
    available_kv_lens: Tuple[int, ...],
    num_kv_heads: int,
    num_sparse_topk: int,
    device: torch.device,
) -> torch.Tensor:
    """Create one padded request-local token list per query token and KV head."""
    num_query_tokens = len(available_kv_lens)
    result = torch.full(
        (num_kv_heads, num_query_tokens, num_sparse_topk),
        -1,
        dtype=torch.int32,
        device=device,
    )

    for query_idx, available_kv_len in enumerate(available_kv_lens):
        pick = min(num_sparse_topk, available_kv_len)
        for head_idx in range(num_kv_heads):
            indices = torch.randperm(available_kv_len, device=device)[:pick].sort().values
            result[head_idx, query_idx, :pick] = indices

    return result


def _local_to_cache_pool_indices(
    sparse_indices: torch.Tensor,
    metadata: TrtllmAttentionMetadata,
    layer_idx: int = 0,
    kv_factor: int = 2,
) -> torch.Tensor:
    """
    Convert local sparse indices to global KV cache pool indices.

    Works for both context (variable-length Q packed) and generation (one token per request).
    sparse_indices shape: [num_kv_heads, num_tokens, num_sparse_topk]
    """
    num_kv_heads, num_tokens, num_sparse_tokens = sparse_indices.shape
    device = sparse_indices.device

    tokens_per_block = metadata.kv_cache_manager.tokens_per_block
    num_layers = metadata.kv_cache_manager.num_layers
    stride_factor = num_layers * kv_factor * num_kv_heads * tokens_per_block

    # Build req_idx_per_token: map each token to its request index.
    num_requests = len(metadata.request_ids)
    seq_lens = metadata.seq_lens[:num_requests]
    if hasattr(seq_lens, "cpu"):
        seq_lens_cpu = seq_lens.cpu()
    else:
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32)
    req_idx_per_token = torch.repeat_interleave(
        torch.arange(num_requests, dtype=torch.int32), seq_lens_cpu, dim=0
    ).to(device)

    # Build 2D block table: [num_requests, max_pages]
    request_ids = metadata.request_ids
    page_indices = metadata.kv_cache_manager.get_batch_cache_indices(request_ids)
    max_pages = max(len(p) for p in page_indices) if page_indices else 1
    host_block_table = torch.full((num_requests, max_pages), -1, dtype=torch.int32)
    for i, pages in enumerate(page_indices):
        if len(pages) > 0:
            host_block_table[i, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
    block_table = host_block_table.to(device)

    # Convert to global
    global_indices = triton_convert_req_index_to_global_index(
        req_idx_per_token,
        block_table,
        sparse_indices,
        BLOCK_SIZE=tokens_per_block,
        NUM_TOPK_TOKENS=num_sparse_tokens,
        BLOCK_N=min(64, num_sparse_tokens),
        stride_factor=stride_factor,
        layer_id=layer_idx,
        num_kv_heads=num_kv_heads,
        kv_factor=kv_factor,
    )

    return global_indices


def _build_expected_compacted_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    sparse_kv_indices: torch.Tensor,
    sparse_kv_offsets: torch.Tensor,
    s: ContextScenario,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Build expected sparse K and V values based on sparse indices."""
    expected_kvs = []
    token_offset = 0

    for batch_idx, seq_len in enumerate(s.seq_lens):
        sparse_len = sparse_kv_offsets[batch_idx + 1].item() - sparse_kv_offsets[batch_idx].item()
        k_batch = k[token_offset : token_offset + seq_len].view(seq_len, s.num_kv_heads, s.head_dim)
        v_batch = v[token_offset : token_offset + seq_len].view(seq_len, s.num_kv_heads, s.head_dim)

        expected_k = torch.zeros(
            sparse_len, s.num_kv_heads, s.head_dim, device=k.device, dtype=k.dtype
        )
        expected_v = torch.zeros_like(expected_k)

        start, end = sparse_kv_offsets[batch_idx].item(), sparse_kv_offsets[batch_idx + 1].item()
        for head_idx in range(s.num_kv_heads):
            indices = sparse_kv_indices[head_idx, start:end]
            expected_k[:, head_idx] = k_batch[indices, head_idx]
            expected_v[:, head_idx] = v_batch[indices, head_idx]

        expected_kvs.append((expected_k, expected_v))
        token_offset += seq_len

    return expected_kvs


def _read_paged_kv_cache(
    kv_cache_manager: KVCacheManager,
    request_ids: List[int],
    token_lens: Tuple[int, ...],
    s: SparseMqaGqaScenario,
    dtype: torch.dtype,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Materialize each request's paged K/V history as contiguous tensors."""
    kv_buffer = kv_cache_manager.get_buffers(0, kv_layout="HND")
    kv_caches = []

    for request_id, num_tokens in zip(request_ids, token_lens, strict=True):
        block_ids = kv_cache_manager.get_block_ids_per_seq([request_id])[0]
        k_cache = torch.empty(
            num_tokens,
            s.num_kv_heads,
            s.head_dim,
            device=kv_buffer.device,
            dtype=dtype,
        )
        v_cache = torch.empty_like(k_cache)
        for token_idx in range(num_tokens):
            block_id = block_ids[token_idx // s.page_size]
            offset_in_block = token_idx % s.page_size
            k_cache[token_idx] = kv_buffer[block_id, 0, :, offset_in_block, :].to(dtype)
            v_cache[token_idx] = kv_buffer[block_id, 1, :, offset_in_block, :].to(dtype)

        kv_caches.append((k_cache, v_cache))

    return kv_caches


# Independent PyTorch reference implementations.


def _reference_dense_context_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s: ContextScenario,
) -> torch.Tensor:
    """Reference implementation for context phase."""
    outputs = []
    token_offset = 0

    for seq_len in s.seq_lens:
        q_batch = q[token_offset : token_offset + seq_len].view(1, seq_len, s.num_heads, s.head_dim)
        k_batch = k[token_offset : token_offset + seq_len].view(
            1, seq_len, s.num_kv_heads, s.head_dim
        )
        v_batch = v[token_offset : token_offset + seq_len].view(
            1, seq_len, s.num_kv_heads, s.head_dim
        )
        q_batch = q_batch.transpose(1, 2)
        k_batch = k_batch.transpose(1, 2)
        v_batch = v_batch.transpose(1, 2)
        if s.q_heads_per_kv_head > 1:
            k_batch = k_batch[:, :, None, :, :].expand(
                1, s.num_kv_heads, s.q_heads_per_kv_head, seq_len, s.head_dim
            )
            v_batch = v_batch[:, :, None, :, :].expand(
                1, s.num_kv_heads, s.q_heads_per_kv_head, seq_len, s.head_dim
            )
            k_batch = k_batch.reshape(1, s.num_heads, seq_len, s.head_dim)
            v_batch = v_batch.reshape(1, s.num_heads, seq_len, s.head_dim)

        attention_scores = torch.matmul(q_batch, k_batch.transpose(-1, -2)) / math.sqrt(s.head_dim)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=q.device), diagonal=1
        )
        attention_probs = torch.nn.functional.softmax(
            attention_scores + causal_mask,
            dim=-1,
            dtype=torch.float32,
        ).to(q.dtype)
        output_batch = torch.matmul(attention_probs, v_batch)
        output_batch = output_batch.transpose(1, 2).reshape(seq_len, s.num_heads * s.head_dim)
        outputs.append(output_batch)
        token_offset += seq_len

    return torch.cat(outputs, dim=0)


def _reference_sparse_context_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparse_attn_ctx_indices: torch.Tensor,
    s: ContextScenario,
) -> torch.Tensor:
    """
    Reference implementation for context phase with sparse attention.
    Uses mask-based approach for each KV head.
    """
    total_tokens = sum(s.seq_lens)
    device = q.device
    dtype = q.dtype

    # Reshape inputs: [num_tokens, num_heads, head_dim]
    q_reshaped = q.view(total_tokens, s.num_heads, s.head_dim)
    k_reshaped = k.view(total_tokens, s.num_kv_heads, s.head_dim)
    v_reshaped = v.view(total_tokens, s.num_kv_heads, s.head_dim)

    outputs = []
    token_offset = 0

    for seq_len in s.seq_lens:
        q_batch = q_reshaped[
            token_offset : token_offset + seq_len
        ]  # [seq_len, num_heads, head_dim]
        k_batch = k_reshaped[
            token_offset : token_offset + seq_len
        ]  # [seq_len, num_kv_heads, head_dim]
        v_batch = v_reshaped[
            token_offset : token_offset + seq_len
        ]  # [seq_len, num_kv_heads, head_dim]

        batch_output = []

        # Process each KV head
        for kv_head_idx in range(s.num_kv_heads):
            k_head = k_batch[:, kv_head_idx, :]
            v_head = v_batch[:, kv_head_idx, :]

            # Build sparse mask for this head
            sparse_mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32
            )

            for token_idx in range(seq_len):
                global_token_idx = token_offset + token_idx
                # Get sparse indices for this token: [num_sparse_tokens]
                indices = sparse_attn_ctx_indices[kv_head_idx, global_token_idx]
                # Filter out -1 padding
                valid_indices = indices[indices >= 0]
                # Set mask values to 0 for valid positions
                sparse_mask[token_idx, valid_indices] = 0.0

            # Apply causal mask on top of sparse mask
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=torch.float32),
                diagonal=1,
            )
            combined_mask = sparse_mask + causal_mask

            # Process each query head in this KV group
            for group_idx in range(s.q_heads_per_kv_head):
                q_head_idx = kv_head_idx * s.q_heads_per_kv_head + group_idx
                q_head = q_batch[:, q_head_idx, :]  # [seq_len, head_dim]

                attn_scores = torch.matmul(q_head, k_head.T) / math.sqrt(s.head_dim)
                attn_scores = attn_scores + combined_mask
                attn_weights = torch.nn.functional.softmax(
                    attn_scores, dim=-1, dtype=torch.float32
                ).to(dtype)

                out_head = torch.matmul(attn_weights, v_head)
                batch_output.append(out_head)

        # Concatenate all heads: [seq_len, num_heads, head_dim] -> [seq_len, num_heads * head_dim]
        batch_output = torch.stack(batch_output, dim=1)
        batch_output = batch_output.reshape(seq_len, s.num_heads * s.head_dim)
        outputs.append(batch_output)

        token_offset += seq_len

    return torch.cat(outputs, dim=0)


def _reference_sparse_generation_attention(
    q: torch.Tensor,
    kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    sparse_attn_indices: torch.Tensor,
    s: GenerationScenario,
) -> torch.Tensor:
    """Reference implementation for generation phase with sparse attention.

    Args:
        sparse_attn_indices: [num_kv_heads, num_gens, num_sparse_topk] with -1 padding.
    """
    outputs = []
    query_offset = 0

    for request_idx in range(s.batch_size):
        query_len = s.query_len
        k_history, v_history = kv_caches[request_idx]
        k_new_request = k_new[query_offset : query_offset + query_len].view(
            query_len, s.num_kv_heads, s.head_dim
        )
        v_new_request = v_new[query_offset : query_offset + query_len].view(
            query_len, s.num_kv_heads, s.head_dim
        )
        k_full = torch.cat([k_history, k_new_request], dim=0)
        v_full = torch.cat([v_history, v_new_request], dim=0)

        for query_idx in range(query_len):
            packed_query_idx = query_offset + query_idx
            q_token = q[packed_query_idx].view(s.num_heads, s.head_dim)
            head_outputs = []

            for kv_head_idx in range(s.num_kv_heads):
                token_indices = sparse_attn_indices[kv_head_idx, packed_query_idx]
                valid_indices = token_indices[token_indices >= 0].long()

                if len(valid_indices) == 0:
                    head_outputs.extend(
                        [torch.zeros(s.head_dim, device=q.device, dtype=q.dtype)]
                        * s.q_heads_per_kv_head
                    )
                    continue

                k_sparse = k_full[valid_indices, kv_head_idx, :]
                v_sparse = v_full[valid_indices, kv_head_idx, :]

                for group_idx in range(s.q_heads_per_kv_head):
                    q_head_idx = kv_head_idx * s.q_heads_per_kv_head + group_idx
                    attention_scores = torch.matmul(q_token[q_head_idx], k_sparse.T) / math.sqrt(
                        s.head_dim
                    )
                    attention_probs = torch.nn.functional.softmax(
                        attention_scores, dim=-1, dtype=torch.float32
                    ).to(q.dtype)
                    head_outputs.append(torch.matmul(attention_probs, v_sparse))

            outputs.append(torch.cat(head_outputs, dim=0))

        query_offset += query_len

    return torch.stack(outputs, dim=0)


# Test input builders. Kernel selection remains explicit in each test above.


def _create_context_inputs(s: ContextScenario) -> _ContextInputs:
    """Create packed context inputs before choosing the sparse compute path."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    num_sparse_topk = s.num_sparse_topk

    q = torch.randn(s.nnz_q, s.num_heads * s.head_dim, device=device, dtype=s.dtype)
    k = torch.randn(s.nnz_q, s.num_kv_heads * s.head_dim, device=device, dtype=s.dtype)
    v = torch.randn(s.nnz_q, s.num_kv_heads * s.head_dim, device=device, dtype=s.dtype)

    kv_cache = torch.zeros(
        s.num_layers,
        s.kv_pool_num_pages,
        2,
        s.num_kv_heads,
        s.page_size,
        s.head_dim,
        device=device,
        dtype=s.kvcache_dtype,
    )
    kv_cache_manager = _create_kv_cache_manager(s, kv_cache)

    request_ids = list(range(s.batch_size))
    kv_cache_manager.add_dummy_requests(request_ids, list(s.seq_lens))

    metadata = _SparseMqaGqaMetadata(
        num_contexts=s.batch_size,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0] * s.batch_size),
        seq_lens=torch.tensor(s.seq_lens, dtype=torch.int32),
        max_num_requests=s.batch_size,
        max_num_tokens=s.nnz_q,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=list(s.seq_lens),
        num_sparse_topk=num_sparse_topk,
    )
    metadata.prepare()

    return _ContextInputs(
        q=q,
        k=k,
        v=v,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        metadata=metadata,
    )


def _create_generation_inputs(s: GenerationScenario) -> _GenerationInputs:
    """Create one decode token and a populated paged cache per request."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    num_sparse_topk = s.num_sparse_topk

    token_nums = [past_len + s.query_len for past_len in s.past_kv_lens]

    q = torch.randn(s.nnz_q, s.num_heads * s.head_dim, device=device, dtype=s.dtype)
    k_new = torch.randn(s.nnz_q, s.num_kv_heads * s.head_dim, device=device, dtype=s.dtype)
    v_new = torch.randn(s.nnz_q, s.num_kv_heads * s.head_dim, device=device, dtype=s.dtype)

    # Single-token cases preserve the original history-only selection. For
    # draft-token cases, each query may also select causal K/V written earlier
    # in the same speculative forward, including its own K/V position.
    available_kv_lens = tuple(
        past_kv_len + query_idx + 1 if s.has_draft_tokens else past_kv_len
        for past_kv_len in s.past_kv_lens
        for query_idx in range(s.query_len)
    )
    sparse_attn_indices = _make_sparse_attention_indices(
        available_kv_lens, s.num_kv_heads, num_sparse_topk, device
    )

    kv_cache = torch.randn(
        s.num_layers,
        s.kv_pool_num_pages,
        2,
        s.num_kv_heads,
        s.page_size,
        s.head_dim,
        device=device,
        dtype=s.dtype,
    ).to(s.kvcache_dtype)
    kv_cache_manager = _create_kv_cache_manager(s, kv_cache)

    request_ids = list(range(s.batch_size))
    kv_cache_manager.add_dummy_requests(request_ids, token_nums)

    metadata = _SparseMqaGqaMetadata(
        num_contexts=0,
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=list(s.past_kv_lens)
        ),
        seq_lens=torch.full((s.batch_size,), s.query_len, dtype=torch.int32),
        max_num_requests=s.batch_size,
        max_num_tokens=s.nnz_q,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=list(s.past_kv_lens),
        num_sparse_topk=num_sparse_topk,
        num_heads_per_kv=s.q_heads_per_kv_head,
        runtime_features=AttentionRuntimeFeatures(has_speculative_draft_tokens=s.has_draft_tokens),
        is_spec_decoding_enabled=s.has_draft_tokens,
        use_spec_decoding=s.has_draft_tokens,
        is_spec_dec_tree=False,
        max_total_draft_tokens=s.max_query_len - 1 if s.has_draft_tokens else None,
    )
    if s.has_draft_tokens:
        draft_len = s.max_query_len - 1
        metadata.spec_decoding_position_offsets = generate_spec_decoding_position_offsets(
            s.batch_size, draft_len
        )
        metadata.spec_decoding_packed_mask = generate_spec_decoding_packed_mask(
            s.batch_size, draft_len
        )
        metadata.spec_decoding_generation_lengths = torch.tensor(
            [s.query_len] * s.batch_size, dtype=torch.int32, device=device
        )
        metadata.update_position_offsets_for_cpp(s.max_query_len)
        metadata.spec_decoding_param_prepare_for_blackwell()
    metadata.prepare()

    return _GenerationInputs(
        q=q,
        k_new=k_new,
        v_new=v_new,
        local_sparse_attn_indices=sparse_attn_indices,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        metadata=metadata,
    )


def _create_block_sparse_gqa_inputs(s: BlockSparseGqaScenario) -> dict[str, torch.Tensor]:
    """Build packed Q, paged KV, page tables, and request-local block indices."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(42)
    total_pages = sum(kv_len // s.page_size for kv_len in s.kv_lens)

    def random_qkv(shape: Tuple[int, ...]) -> torch.Tensor:
        tensor = torch.randn(
            shape,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        return tensor.to(s.dtype)

    q = random_qkv((s.total_q, s.num_q_heads, s.head_dim))
    logical_k = random_qkv((total_pages, s.num_kv_heads, s.page_size, s.head_dim))
    logical_v = random_qkv((total_pages, s.num_kv_heads, s.page_size, s.head_dim))

    if s.shuffle_pages:
        kv_indices = torch.randperm(total_pages, device=device, generator=generator)
        k_paged = torch.empty_like(logical_k)
        v_paged = torch.empty_like(logical_v)
        k_paged[kv_indices] = logical_k
        v_paged[kv_indices] = logical_v
    else:
        kv_indices = torch.arange(total_pages, device=device)
        k_paged = logical_k
        v_paged = logical_v
    kv_indices = kv_indices.to(torch.int32)

    kv_block_indexes = torch.full(
        (s.total_q, s.num_kv_heads, s.topk),
        -1,
        dtype=torch.int32,
        device=device,
    )
    q_offset = 0
    for q_len, kv_len in zip(s.q_lens, s.kv_lens, strict=True):
        num_pages = kv_len // s.page_size
        for query_idx in range(q_len):
            for kv_head_idx in range(s.num_kv_heads):
                start = (query_idx + kv_head_idx) % num_pages if s.per_token_blocks else 0
                blocks = sorted(
                    (start + block_idx) % num_pages for block_idx in range(s.selected_blocks)
                )
                kv_block_indexes[
                    q_offset + query_idx,
                    kv_head_idx,
                    : s.selected_blocks,
                ] = torch.tensor(blocks, dtype=torch.int32, device=device)
        q_offset += q_len

    return {
        "q": q,
        "k_paged": k_paged,
        "v_paged": v_paged,
        "kv_indices": kv_indices,
        "kv_block_indexes": kv_block_indexes,
    }


def _reference_block_sparse_gqa(
    inputs: dict[str, torch.Tensor], s: BlockSparseGqaScenario
) -> torch.Tensor:
    """Evaluate request-local block selection and per-request causal offsets."""
    output = torch.empty(
        s.total_q,
        s.num_q_heads,
        s.head_dim,
        dtype=torch.float32,
        device=inputs["q"].device,
    )
    q_offset = 0
    page_offset = 0
    for q_len, kv_len, causal_offset in zip(
        s.q_lens,
        s.kv_lens,
        s.causal_offsets,
        strict=True,
    ):
        num_pages = kv_len // s.page_size
        physical_pages = inputs["kv_indices"][page_offset : page_offset + num_pages].long()
        k = (
            inputs["k_paged"]
            .index_select(0, physical_pages)
            .permute(0, 2, 1, 3)
            .reshape(kv_len, s.num_kv_heads, s.head_dim)
            .float()
        )
        v = (
            inputs["v_paged"]
            .index_select(0, physical_pages)
            .permute(0, 2, 1, 3)
            .reshape(kv_len, s.num_kv_heads, s.head_dim)
            .float()
        )
        q = inputs["q"][q_offset : q_offset + q_len].float()
        request_blocks = inputs["kv_block_indexes"][q_offset : q_offset + q_len]
        token_positions = torch.arange(kv_len, device=q.device)
        block_ids = token_positions // s.page_size
        causal_mask = token_positions.view(1, -1) <= (
            torch.arange(q_len, device=q.device).view(-1, 1) + causal_offset
        )

        for kv_head_idx in range(s.num_kv_heads):
            selected_blocks = request_blocks[:, kv_head_idx]
            selected_mask = (
                (selected_blocks.unsqueeze(-1) == block_ids.view(1, 1, -1))
                & (selected_blocks.unsqueeze(-1) >= 0)
            ).any(dim=1)
            mask = selected_mask & causal_mask
            q_head_begin = kv_head_idx * s.q_heads_per_kv_head
            q_head_end = q_head_begin + s.q_heads_per_kv_head
            scores = torch.einsum(
                "qhd,kd->qhk",
                q[:, q_head_begin:q_head_end] * (s.head_dim**-0.5),
                k[:, kv_head_idx],
            )
            scores.masked_fill_(~mask.unsqueeze(1), float("-inf"))
            probabilities = torch.softmax(scores, dim=-1)
            output[q_offset : q_offset + q_len, q_head_begin:q_head_end] = torch.einsum(
                "qhk,kd->qhd", probabilities, v[:, kv_head_idx]
            )

        q_offset += q_len
        page_offset += num_pages

    return output.to(torch.bfloat16)
