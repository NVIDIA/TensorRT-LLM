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

"""Architecture-level regression tests for page-sparse MHA computation.

The tests supply static block indices and per-request offsets, invoke
``TrtllmAttention.forward``, and compare the result with an equivalent
token-level PyTorch reference.

Prefill attention is dense on this path; page-sparse MHA computation starts
during generation.
"""

import math
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Optional, Tuple

import pytest
import torch
from utils.util import getSMVersion

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionRuntimeFeatures,
)
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
FP8_ATOL = 8e-2
FP8_RTOL = 4e-2
SUPPORTED_MODEL_DTYPES = (torch.bfloat16, torch.float16)
SUPPORTED_KV_CACHE_DTYPES = (*SUPPORTED_MODEL_DTYPES, torch.float8_e4m3fn)
SUPPORTED_MHA_HEAD_DIMS = (64, 80, 128, 256)
TESTED_MHA_HEAD_COUNTS = (1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128)
TESTED_KV_PAGE_SIZES = (8, 16, 32, 64, 128, 256, 512)
SUPPORTED_SM_VERSIONS = (100, 103)

pytestmark = pytest.mark.skipif(
    getSMVersion() not in SUPPORTED_SM_VERSIONS,
    reason="Page-sparse MHA requires SM100 or SM103",
)


@pytest.fixture(autouse=True)
def _force_trtllm_gen_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test on the TRTLLM-Gen fallback path."""
    monkeypatch.setenv("TLLM_FMHA_LIBS", "fallback")


# Page-sparse MHA support matrix:
#
#   GPU architecture       SM100 and SM103
#   Sparse compute phase   Single-token and linear draft-token generation
#   Attention type         MHA; num_heads == num_kv_heads
#   Q heads per KV head    1
#   Number of MHA heads    No discrete source restriction; tests cover
#                          1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, and 128
#   Model QKV input        BF16 or FP16
#   Model QKV layout       Fused QKV
#   Kernel output          Model dtype for H64/H80/H128/H256; E4M3 FP8 for
#                          H64/H128/H256 with an FP8 KV cache
#   KV-cache dtype         Model dtype for H64/H80/H128/H256; E4M3 FP8 for
#                          H64/H128/H256
#   Q/K/V head dimension  Equal dimensions: 64, 80, 128, or 256
#   KV-cache layout        Paged; page sizes 8, 16, 32, 64, 128, 256, and 512
#   Selection granularity  Block indices expanded to KV-cache pages
#   Sparse indices         int32 block indices plus int32 request offsets
#                          Per-head patterns and variable request offsets
#   Sparse index block     Blocks may cross KV-page boundaries; sizes
#                          1, 2, 3, 4, 5, 8, 16, 24, 32, and 48 are tested
#   Attention semantics    Causal self-attention


@dataclass(kw_only=True, frozen=True)
class MhaGenerationScenario:
    """Generation inputs for page-sparse MHA computation."""

    dtype: torch.dtype = torch.bfloat16
    kvcache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 1
    num_heads: int = 8
    num_kv_heads: int = 8
    head_dim: int = 128
    page_size: int = 32
    num_pages: int = 4
    batch_size: int = 1
    past_kv_lens: Tuple[int, ...] = (96,)
    query_len: int = 1
    fp8_output: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in SUPPORTED_MODEL_DTYPES:
            raise ValueError("Model QKV dtype must be BF16 or FP16")
        if self.kvcache_dtype not in SUPPORTED_KV_CACHE_DTYPES:
            raise ValueError("KV-cache dtype must be BF16, FP16, or E4M3 FP8")
        if self.kvcache_dtype != torch.float8_e4m3fn and self.kvcache_dtype != self.dtype:
            raise ValueError("A non-FP8 KV-cache dtype must match the model QKV dtype")
        if self.num_heads <= 0 or self.num_heads != self.num_kv_heads:
            raise ValueError("Page-sparse MHA requires equal positive Q and KV head counts")
        if self.head_dim not in SUPPORTED_MHA_HEAD_DIMS:
            raise ValueError(f"head_dim must be one of {SUPPORTED_MHA_HEAD_DIMS}")
        if self.page_size < 8 or self.page_size & (self.page_size - 1):
            raise ValueError("page_size must be a power of two and at least 8")
        if len(self.past_kv_lens) != self.batch_size:
            raise ValueError(
                f"past_kv_lens length {len(self.past_kv_lens)} must match "
                f"batch_size {self.batch_size}"
            )
        if self.query_len < 1:
            raise ValueError("query_len must be positive")
        if self.fp8_output and self.kvcache_dtype != torch.float8_e4m3fn:
            raise ValueError("FP8 output testing requires an FP8 KV cache")
        for past_kv_len in self.past_kv_lens:
            required_pages = math.ceil((past_kv_len + self.query_len) / self.page_size)
            if required_pages > self.num_pages:
                raise ValueError("num_pages does not cover the request KV length")

    @property
    def nnz_q(self) -> int:
        return self.batch_size * self.query_len

    @property
    def max_query_len(self) -> int:
        return self.query_len

    @property
    def has_draft_tokens(self) -> bool:
        return self.query_len > 1

    @property
    def kv_pool_num_pages(self) -> int:
        return self.batch_size * self.num_pages


@dataclass(kw_only=True)
class MhaGenerationInputs:
    """Generation tensors plus their populated paged KV cache."""

    q: torch.Tensor
    k_new: torch.Tensor
    v_new: torch.Tensor
    kv_cache_manager: KVCacheManager
    request_ids: list[int]
    metadata: TrtllmAttentionMetadata

    @property
    def fused_qkv(self) -> torch.Tensor:
        return torch.cat([self.q, self.k_new, self.v_new], dim=1)


def quant_config(scenario: MhaGenerationScenario) -> Optional[QuantConfig]:
    """Build the quantization settings used by the attention backend."""
    if scenario.fp8_output:
        return QuantConfig(
            quant_algo=QuantAlgo.FP8,
            kv_cache_quant_algo=QuantAlgo.FP8,
        )
    if scenario.kvcache_dtype == torch.float8_e4m3fn:
        return QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    return None


def fp8_qdq(tensor: torch.Tensor) -> torch.Tensor:
    """Apply unit-scale E4M3 quantize-dequantize to a reference tensor."""
    return tensor.to(torch.float8_e4m3fn).to(tensor.dtype)


def _create_kv_cache_manager(
    scenario: MhaGenerationScenario,
    kv_cache: torch.Tensor,
) -> KVCacheManager:
    kv_cache_config = KvCacheConfig(max_tokens=scenario.kv_pool_num_pages * scenario.page_size)
    manager = KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=scenario.num_layers,
        num_kv_heads=scenario.num_kv_heads,
        head_dim=scenario.head_dim,
        tokens_per_block=scenario.page_size,
        max_seq_len=scenario.kv_pool_num_pages * scenario.page_size,
        max_batch_size=scenario.batch_size,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=str_dtype_to_binding(torch_dtype_to_str(scenario.kvcache_dtype)),
    )
    for layer_idx in range(scenario.num_layers):
        manager.get_buffers(layer_idx, kv_layout="HND").copy_(kv_cache[layer_idx])
    return manager


def create_generation_inputs(scenario: MhaGenerationScenario) -> MhaGenerationInputs:
    """Create generation inputs and release the cache if construction fails."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    q = torch.randn(
        scenario.nnz_q,
        scenario.num_heads * scenario.head_dim,
        device=device,
        dtype=scenario.dtype,
    )
    k_new = torch.randn(
        scenario.nnz_q,
        scenario.num_kv_heads * scenario.head_dim,
        device=device,
        dtype=scenario.dtype,
    )
    v_new = torch.randn_like(k_new)
    kv_cache = torch.randn(
        scenario.num_layers,
        scenario.kv_pool_num_pages,
        2,
        scenario.num_kv_heads,
        scenario.page_size,
        scenario.head_dim,
        device=device,
        dtype=scenario.dtype,
    ).to(scenario.kvcache_dtype)

    with ExitStack() as cleanup:
        kv_cache_manager = _create_kv_cache_manager(scenario, kv_cache)
        cleanup.callback(kv_cache_manager.shutdown)
        request_ids = list(range(scenario.batch_size))
        token_nums = [past_kv_len + scenario.query_len for past_kv_len in scenario.past_kv_lens]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)
        metadata = TrtllmAttentionMetadata(
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=list(scenario.past_kv_lens),
            ),
            seq_lens=torch.full((scenario.batch_size,), scenario.query_len, dtype=torch.int32),
            max_num_requests=scenario.batch_size,
            max_num_tokens=scenario.nnz_q,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=list(scenario.past_kv_lens),
            num_heads_per_kv=1,
            runtime_features=AttentionRuntimeFeatures(
                has_speculative_draft_tokens=scenario.has_draft_tokens
            ),
            is_spec_decoding_enabled=scenario.has_draft_tokens,
            use_spec_decoding=scenario.has_draft_tokens,
            is_spec_dec_tree=False,
            max_total_draft_tokens=(
                scenario.max_query_len - 1 if scenario.has_draft_tokens else None
            ),
        )
        if scenario.has_draft_tokens:
            draft_len = scenario.max_query_len - 1
            metadata.spec_decoding_position_offsets = generate_spec_decoding_position_offsets(
                scenario.batch_size, draft_len
            )
            metadata.spec_decoding_packed_mask = generate_spec_decoding_packed_mask(
                scenario.batch_size, draft_len
            )
            metadata.spec_decoding_generation_lengths = torch.tensor(
                [scenario.query_len] * scenario.batch_size,
                dtype=torch.int32,
                device=device,
            )
            metadata.update_position_offsets_for_cpp(scenario.max_query_len)
            metadata.spec_decoding_param_prepare_for_blackwell()
        metadata.prepare()
        cleanup.pop_all()
        return MhaGenerationInputs(
            q=q,
            k_new=k_new,
            v_new=v_new,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            metadata=metadata,
        )


def read_paged_kv_cache(
    inputs: MhaGenerationInputs,
    scenario: MhaGenerationScenario,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Materialize paged history using page-wise copies."""
    kv_buffer = inputs.kv_cache_manager.get_buffers(0, kv_layout="HND")
    kv_caches = []
    for request_id, num_tokens in zip(inputs.request_ids, scenario.past_kv_lens, strict=True):
        block_ids = inputs.kv_cache_manager.get_block_ids_per_seq([request_id])[0]
        k_cache = torch.empty(
            num_tokens,
            scenario.num_kv_heads,
            scenario.head_dim,
            device=kv_buffer.device,
            dtype=scenario.dtype,
        )
        v_cache = torch.empty_like(k_cache)
        for local_page_idx, block_id in enumerate(block_ids):
            token_start = local_page_idx * scenario.page_size
            token_end = min(token_start + scenario.page_size, num_tokens)
            if token_start >= token_end:
                break
            num_page_tokens = token_end - token_start
            k_cache[token_start:token_end] = (
                kv_buffer[block_id, 0, :, :num_page_tokens, :].transpose(0, 1).to(scenario.dtype)
            )
            v_cache[token_start:token_end] = (
                kv_buffer[block_id, 1, :, :num_page_tokens, :].transpose(0, 1).to(scenario.dtype)
            )
        kv_caches.append((k_cache, v_cache))
    return kv_caches


def reference_generation_attention(
    q: torch.Tensor,
    kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    sparse_attn_indices: torch.Tensor,
    scenario: MhaGenerationScenario,
) -> torch.Tensor:
    """Compute page-sparse MHA from equivalent request-local token indices."""
    outputs = []
    query_offset = 0
    for request_idx in range(scenario.batch_size):
        k_history, v_history = kv_caches[request_idx]
        request_slice = slice(query_offset, query_offset + scenario.query_len)
        k_full = torch.cat(
            [
                k_history,
                k_new[request_slice].view(
                    scenario.query_len, scenario.num_kv_heads, scenario.head_dim
                ),
            ],
            dim=0,
        )
        v_full = torch.cat(
            [
                v_history,
                v_new[request_slice].view(
                    scenario.query_len, scenario.num_kv_heads, scenario.head_dim
                ),
            ],
            dim=0,
        )
        for query_idx in range(scenario.query_len):
            packed_query_idx = query_offset + query_idx
            q_token = q[packed_query_idx].view(scenario.num_heads, scenario.head_dim)
            head_outputs = []
            for head_idx in range(scenario.num_heads):
                token_indices = sparse_attn_indices[head_idx, packed_query_idx]
                valid_indices = token_indices[token_indices >= 0].long()
                k_sparse = k_full[valid_indices, head_idx, :]
                v_sparse = v_full[valid_indices, head_idx, :]
                attention_scores = torch.matmul(q_token[head_idx], k_sparse.T) / math.sqrt(
                    scenario.head_dim
                )
                attention_probs = torch.nn.functional.softmax(
                    attention_scores, dim=-1, dtype=torch.float32
                ).to(scenario.dtype)
                head_outputs.append(torch.matmul(attention_probs, v_sparse))
            outputs.append(torch.cat(head_outputs, dim=0))
        query_offset += scenario.query_len
    return torch.stack(outputs, dim=0)


@dataclass(kw_only=True, frozen=True)
class PageSparseMhaScenario:
    """Page-sparse MHA geometry layered on generation inputs."""

    attention: MhaGenerationScenario
    sparse_index_block_size: int = 4
    num_selected_sparse_blocks: int = 2

    def __post_init__(self) -> None:
        if self.attention.num_heads != self.attention.num_kv_heads:
            raise ValueError("PageSparseMhaScenario requires MHA head geometry")
        if self.sparse_index_block_size <= 0:
            raise ValueError("sparse_index_block_size must be positive")
        if self.num_selected_sparse_blocks <= 0:
            raise ValueError("num_selected_sparse_blocks must be positive")


@dataclass(kw_only=True, frozen=True)
class _PageSparseMhaParams(SparseParams):
    """Sparse parameters selecting block/page-granular attention."""

    sparse_index_block_size: int
    algorithm: str = "test_page_sparse_mha"

    @property
    def indices_block_size(self) -> int:
        return self.sparse_index_block_size


class _StaticPageSparseMhaAttention(TrtllmAttention):
    """MHA backend adapter returning predetermined page selections."""

    def __init__(
        self,
        *args,
        sparse_index_block_size: int,
        sparse_attn_indices: torch.Tensor,
        sparse_attn_offsets: torch.Tensor,
        **kwargs,
    ) -> None:
        kwargs["sparse_params"] = _PageSparseMhaParams(
            sparse_index_block_size=sparse_index_block_size
        )
        kwargs["pos_embd_params"] = None
        super().__init__(*args, **kwargs)
        self._sparse_attn_indices = sparse_attn_indices
        self._sparse_attn_offsets = sparse_attn_offsets

    def sparse_kv_predict(self, q, k, metadata, forward_args: AttentionForwardArgs):
        return None, None

    def sparse_attn_predict(self, q, k, metadata, forward_args: AttentionForwardArgs):
        return self._sparse_attn_indices, self._sparse_attn_offsets


def _selected_sparse_blocks(
    head_idx: int,
    num_sparse_blocks: int,
    num_selected_blocks: int,
    page_size: int,
    sparse_block_size: int,
) -> Tuple[int, ...]:
    """Choose head-dependent blocks and retain the newest sparse block."""
    newest_block = num_sparse_blocks - 1
    if num_selected_blocks == 1:
        return (newest_block,)

    older_blocks = list(range(newest_block))
    boundary_block = page_size // sparse_block_size
    if head_idx % 2 and boundary_block in older_blocks:
        older_blocks.remove(boundary_block)
        older_blocks.insert(0, boundary_block)
    selected = older_blocks[: num_selected_blocks - 1] + [newest_block]
    if head_idx % 2:
        selected.reverse()
    return tuple(selected)


def _make_page_sparse_pattern(
    scenario: PageSparseMhaScenario,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build page indices plus equivalent request-local token indices for the reference."""
    attention = scenario.attention
    sparse_indices_by_head = [[] for _ in range(attention.num_kv_heads)]
    sparse_offsets = [0]
    max_reference_tokens = max(attention.past_kv_lens) + attention.query_len
    reference_indices = torch.full(
        (attention.num_kv_heads, attention.nnz_q, max_reference_tokens),
        -1,
        dtype=torch.int32,
        device=device,
    )

    query_offset = 0
    for past_kv_len in attention.past_kv_lens:
        total_kv_len = past_kv_len + attention.query_len
        num_sparse_blocks = math.ceil(total_kv_len / scenario.sparse_index_block_size)
        num_selected_sparse_blocks = min(scenario.num_selected_sparse_blocks, num_sparse_blocks)
        for head_idx in range(attention.num_kv_heads):
            sparse_blocks = _selected_sparse_blocks(
                head_idx,
                num_sparse_blocks,
                num_selected_sparse_blocks,
                attention.page_size,
                scenario.sparse_index_block_size,
            )
            sparse_indices_by_head[head_idx].extend(sparse_blocks)

            for query_idx in range(attention.query_len):
                available_kv_len = past_kv_len + query_idx + 1
                touched_pages = set()
                for sparse_block_idx in sparse_blocks:
                    block_start = sparse_block_idx * scenario.sparse_index_block_size
                    block_end = min(
                        block_start + scenario.sparse_index_block_size,
                        available_kv_len,
                    )
                    if block_start >= block_end:
                        continue
                    first_page = block_start // attention.page_size
                    last_page = (block_end - 1) // attention.page_size
                    touched_pages.update(range(first_page, last_page + 1))
                selected_tokens = []
                for page_idx in sorted(touched_pages):
                    page_start = page_idx * attention.page_size
                    page_end = min(page_start + attention.page_size, available_kv_len)
                    selected_tokens.extend(range(page_start, page_end))
                reference_indices[
                    head_idx,
                    query_offset + query_idx,
                    : len(selected_tokens),
                ] = torch.tensor(selected_tokens, dtype=torch.int32, device=device)

        sparse_offsets.append(sparse_offsets[-1] + num_selected_sparse_blocks)
        query_offset += attention.query_len

    sparse_attn_indices = torch.tensor(
        sparse_indices_by_head,
        dtype=torch.int32,
        device=device,
    )
    sparse_attn_offsets = torch.tensor(sparse_offsets, dtype=torch.int32, device=device)
    return sparse_attn_indices, sparse_attn_offsets, reference_indices


def _run_page_sparse_mha(scenario: PageSparseMhaScenario) -> None:
    """Compare page-sparse generation with an equivalent PyTorch token reference."""
    attention_scenario = scenario.attention
    inputs = create_generation_inputs(attention_scenario)
    try:
        sparse_attn_indices, sparse_attn_offsets, reference_indices = _make_page_sparse_pattern(
            scenario, inputs.q.device
        )
        attention = _StaticPageSparseMhaAttention(
            layer_idx=0,
            num_heads=attention_scenario.num_heads,
            head_dim=attention_scenario.head_dim,
            num_kv_heads=attention_scenario.num_kv_heads,
            quant_config=quant_config(attention_scenario),
            sparse_index_block_size=scenario.sparse_index_block_size,
            sparse_attn_indices=sparse_attn_indices,
            sparse_attn_offsets=sparse_attn_offsets,
        )

        kv_caches = read_paged_kv_cache(inputs, attention_scenario)
        reference_q = inputs.q
        reference_k_new = inputs.k_new
        reference_v_new = inputs.v_new
        if attention_scenario.kvcache_dtype == torch.float8_e4m3fn:
            reference_q = fp8_qdq(reference_q)
            reference_k_new = fp8_qdq(reference_k_new)
            reference_v_new = fp8_qdq(reference_v_new)
        reference_output = reference_generation_attention(
            reference_q,
            kv_caches,
            reference_k_new,
            reference_v_new,
            reference_indices,
            attention_scenario,
        )

        forward_args: Optional[AttentionForwardArgs] = None
        if attention_scenario.fp8_output:
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

        expected_shape = (
            attention_scenario.nnz_q,
            attention_scenario.num_heads * attention_scenario.head_dim,
        )
        assert output.shape == expected_shape
        expected_output_dtype = (
            torch.float8_e4m3fn if attention_scenario.fp8_output else attention_scenario.dtype
        )
        assert output.dtype == expected_output_dtype
        uses_fp8 = (
            attention_scenario.kvcache_dtype == torch.float8_e4m3fn or attention_scenario.fp8_output
        )
        output_for_comparison = output.float() if uses_fp8 else output
        if attention_scenario.fp8_output:
            reference_output = reference_output.to(torch.float8_e4m3fn)
        reference_for_comparison = reference_output.float() if uses_fp8 else reference_output
        assert torch.isfinite(output_for_comparison).all()
        torch.testing.assert_close(
            output_for_comparison,
            reference_for_comparison,
            atol=FP8_ATOL if uses_fp8 else ATOL,
            rtol=FP8_RTOL if uses_fp8 else RTOL,
        )
    finally:
        inputs.kv_cache_manager.shutdown()


_NUM_MHA_HEADS = 8

_PAGE_SPARSE_MHA_CASES = (
    [
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    dtype=dtype,
                    kvcache_dtype=dtype,
                    num_heads=_NUM_MHA_HEADS,
                    num_kv_heads=_NUM_MHA_HEADS,
                    head_dim=head_dim,
                    batch_size=1,
                    past_kv_lens=(96,),
                    num_pages=4,
                )
            ),
            id=f"{str(dtype).removeprefix('torch.')}_h{head_dim}",
        )
        for dtype in SUPPORTED_MODEL_DTYPES
        for head_dim in SUPPORTED_MHA_HEAD_DIMS
    ]
    + [
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    num_heads=_NUM_MHA_HEADS,
                    num_kv_heads=_NUM_MHA_HEADS,
                    page_size=kv_page_size,
                    batch_size=1,
                    past_kv_lens=(max(96, 3 * kv_page_size),),
                    num_pages=math.ceil((max(96, 3 * kv_page_size) + 1) / kv_page_size),
                )
            ),
            id=f"kv_page_size_{kv_page_size}",
        )
        for kv_page_size in TESTED_KV_PAGE_SIZES
        if kv_page_size != 32
    ]
    + [
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    num_heads=_NUM_MHA_HEADS,
                    num_kv_heads=_NUM_MHA_HEADS,
                    batch_size=1,
                    past_kv_lens=(96,),
                    num_pages=4,
                ),
                sparse_index_block_size=sparse_index_block_size,
            ),
            id=f"sparse_index_block_size_{sparse_index_block_size}",
        )
        for sparse_index_block_size in (1, 2, 3, 5, 8, 16, 24, 32, 48)
    ]
    + [
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    num_heads=_NUM_MHA_HEADS,
                    num_kv_heads=_NUM_MHA_HEADS,
                    batch_size=1,
                    past_kv_lens=(96,),
                    num_pages=4,
                ),
                num_selected_sparse_blocks=num_selected_sparse_blocks,
            ),
            id=f"selected_sparse_blocks_{num_selected_sparse_blocks}",
        )
        for num_selected_sparse_blocks in (1, 3)
    ]
    + [
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    num_heads=_NUM_MHA_HEADS,
                    num_kv_heads=_NUM_MHA_HEADS,
                    batch_size=2,
                    past_kv_lens=(32, 160),
                    num_pages=8,
                ),
                num_selected_sparse_blocks=3,
            ),
            id="batch2_var_kv_and_offsets",
        ),
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    num_heads=_NUM_MHA_HEADS,
                    num_kv_heads=_NUM_MHA_HEADS,
                    batch_size=1,
                    past_kv_lens=(95,),
                    num_pages=4,
                )
            ),
            id="non_page_aligned_kv_length",
        ),
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    num_heads=_NUM_MHA_HEADS,
                    num_kv_heads=_NUM_MHA_HEADS,
                    batch_size=1,
                    past_kv_lens=(96,),
                    query_len=4,
                    num_pages=4,
                )
            ),
            id="linear_3_draft_tokens",
        ),
    ]
    + [
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    dtype=dtype,
                    kvcache_dtype=torch.float8_e4m3fn,
                    num_heads=_NUM_MHA_HEADS,
                    num_kv_heads=_NUM_MHA_HEADS,
                    head_dim=head_dim,
                    batch_size=1,
                    past_kv_lens=(96,),
                    num_pages=4,
                    fp8_output=fp8_output,
                )
            ),
            id=(
                f"{str(dtype).removeprefix('torch.')}_h{head_dim}_fp8_kv_"
                f"{'fp8_output' if fp8_output else 'model_output'}"
            ),
        )
        for dtype in SUPPORTED_MODEL_DTYPES
        for head_dim in SUPPORTED_MHA_HEAD_DIMS
        for fp8_output in (False, True)
        if head_dim != 80
    ]
    + [
        pytest.param(
            PageSparseMhaScenario(
                attention=MhaGenerationScenario(
                    num_heads=num_mha_heads,
                    num_kv_heads=num_mha_heads,
                    batch_size=1,
                    past_kv_lens=(96,),
                    num_pages=4,
                )
            ),
            id=f"mha_{num_mha_heads}_heads",
        )
        for num_mha_heads in TESTED_MHA_HEAD_COUNTS
        if num_mha_heads != _NUM_MHA_HEADS
    ]
)


@pytest.mark.parametrize("scenario", _PAGE_SPARSE_MHA_CASES)
def test_generation_page_sparse_mha(scenario: PageSparseMhaScenario) -> None:
    """Static page selections drive sparse MHA generation computation."""
    _run_page_sparse_mha(scenario)
