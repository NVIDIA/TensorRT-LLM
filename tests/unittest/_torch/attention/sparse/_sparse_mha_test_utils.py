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

"""Input, cache, and reference helpers for page-sparse MHA tests."""

import math
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention_backend.trtllm import (
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

SUPPORTED_MODEL_DTYPES = (torch.bfloat16, torch.float16)
SUPPORTED_KV_CACHE_DTYPES = (*SUPPORTED_MODEL_DTYPES, torch.float8_e4m3fn)
SUPPORTED_MHA_HEAD_DIMS = (64, 80, 128, 256)


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
