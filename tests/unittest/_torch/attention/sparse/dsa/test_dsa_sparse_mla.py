"""
Tests for sparse MLA attention using explicit sparse indices.
"""

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch
from utils.util import skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType, RopeEmbeddingUtils
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_k_pe_for_ctx(
    k_pe: torch.Tensor, rope_cos_sin: torch.Tensor, sequence_lengths: List[int]
) -> torch.Tensor:
    k_pe_ref_list = []
    total_tokens = 0
    for seq_len in sequence_lengths:
        k_pe_seq = k_pe[total_tokens : total_tokens + seq_len].unsqueeze(-2)
        cos, sin = rope_cos_sin[:seq_len].chunk(2, dim=-2)
        k_pe_seq = k_pe_seq.unflatten(-1, [-1, 2]).transpose(-2, -1).flatten(start_dim=-2)
        k_pe_seq = ((k_pe_seq * cos) + (rotate_half(k_pe_seq) * sin)).to(dtype=k_pe_seq.dtype)
        k_pe_seq = k_pe_seq.unflatten(-1, [2, -1]).transpose(-2, -1).flatten(start_dim=-2)
        k_pe_ref_list.append(k_pe_seq)
        total_tokens += seq_len
    return torch.cat(k_pe_ref_list).squeeze(-2)


def _rotate_fused_q_for_ctx(
    fused_q: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    sequence_lengths: List[int],
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    fused_q = fused_q.clone()
    fused_head_dim = kv_lora_rank + qk_rope_head_dim
    total_tokens = 0
    for seq_len in sequence_lengths:
        fused_q_seq = fused_q[total_tokens : total_tokens + seq_len].view(
            seq_len, num_heads, fused_head_dim
        )
        q_rope = fused_q_seq[..., -qk_rope_head_dim:]
        cos, sin = rope_cos_sin[:seq_len].chunk(2, dim=-2)
        q_rope = q_rope.unflatten(-1, [-1, 2]).transpose(-2, -1).flatten(start_dim=-2)
        q_rope = ((q_rope * cos) + (rotate_half(q_rope) * sin)).to(dtype=fused_q.dtype)
        q_rope = q_rope.unflatten(-1, [2, -1]).transpose(-2, -1).flatten(start_dim=-2)
        fused_q_seq[..., -qk_rope_head_dim:] = q_rope
        fused_q[total_tokens : total_tokens + seq_len] = fused_q_seq.view(seq_len, -1)
        total_tokens += seq_len
    return fused_q


def calculate_ref_result_ctx_sparse(
    fused_q: torch.Tensor,
    latent_cache: torch.Tensor,
    sequence_lengths: List[int],
    num_heads: int,
    kv_lora_rank: int,
    v_head_dim: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    q_scaling: float,
    topk_indices: Optional[torch.Tensor] = None,
):
    """
    Reference for sparse MLA context using fused Q and latent cache.
    fused_q shape: (total_tokens, num_heads * (kv_lora_rank + qk_rope_head_dim))
    latent_cache shape: (total_tokens, kv_lora_rank + qk_rope_head_dim)
    """
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    bmm1_scale = 1 / (math.sqrt(qk_head_dim) * q_scaling)
    fused_head_dim = kv_lora_rank + qk_rope_head_dim
    ref_results = []
    total_tokens = 0
    for seq_len in sequence_lengths:
        fused_q_seq = fused_q[total_tokens : total_tokens + seq_len].unflatten(
            -1, [num_heads, fused_head_dim]
        )
        fused_q_seq = fused_q_seq.transpose(0, 1)  # (num_heads, seq_len, fused_head_dim)

        latent_seq = latent_cache[total_tokens : total_tokens + seq_len]
        k_seq = latent_seq.unsqueeze(0)  # (1, seq_len, fused_head_dim)
        v_seq = latent_seq[..., :v_head_dim].unsqueeze(0)  # (1, seq_len, v_head_dim)

        k_seq = repeat_kv(k_seq.unsqueeze(0), num_heads).squeeze(0)
        v_seq = repeat_kv(v_seq.unsqueeze(0), num_heads).squeeze(0)

        if topk_indices is None:
            attn_weights = torch.matmul(fused_q_seq, k_seq.transpose(1, 2)) * bmm1_scale
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=fused_q.device, dtype=torch.bool), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
            attn_weights = torch.nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(fused_q.dtype)
            attn_output = torch.matmul(attn_weights, v_seq)  # (num_heads, seq_len, v_head_dim)
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(seq_len, num_heads * v_head_dim)
            )
            ref_results.append(attn_output)
        else:
            per_token_outputs = []
            token_rows = topk_indices[total_tokens : total_tokens + seq_len]
            for token_idx in range(seq_len):
                token_indices = token_rows[token_idx]
                token_indices = token_indices[token_indices >= 0]
                q_tok = fused_q_seq[:, token_idx, :]
                k_sel = k_seq[:, token_indices, :]
                v_sel = v_seq[:, token_indices, :]
                attn_weights = (
                    torch.matmul(
                        q_tok.unsqueeze(1),
                        k_sel.transpose(1, 2),
                    )
                    * bmm1_scale
                )
                attn_weights = torch.nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(fused_q.dtype)
                attn_output = torch.matmul(attn_weights, v_sel)  # (num_heads, 1, v_head_dim)
                per_token_outputs.append(
                    attn_output.transpose(0, 1).contiguous().view(1, num_heads * v_head_dim)
                )
            ref_results.append(torch.cat(per_token_outputs, dim=0))
        total_tokens += seq_len
    return torch.cat(ref_results)


def calculate_ref_result_gen(
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    num_heads: int,
    kv_lora_rank: int,
    v_head_dim: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    sequence_lengths: List[int],
    q_scaling: float,
    topk_indices: Optional[torch.Tensor] = None,
):
    """
    use standard attention to calculate the reference result by iterating over each request
    fused_q shape: (num_tokens, num_heads * (kv_lora_rank + qk_rope_head_dim))
    q_pe shape: (num_tokens, num_heads, qk_rope_head_dim)
    compressed_kv shape: (num_requests, kv_lora_rank)
    k_pe shape: (num_requests, qk_rope_head_dim)
    latent_cache shape: (total_tokens, kv_lora_rank + qk_rope_head_dim)
    rope_cos_sin shape: (max_position_embeddings, 2, qk_rope_head_dim)
    """
    num_requests = len(sequence_lengths)
    seq_len_q = fused_q.shape[0] // num_requests

    # Reshape inputs for reference calculation
    q_reshaped = []
    k_reshaped = []
    v_reshaped = []
    latent_cache_list = []
    total_tokens = 0
    for i in range(num_requests):
        fused_q_seq = fused_q[i * seq_len_q : (i + 1) * seq_len_q].unflatten(
            -1, [num_heads, kv_lora_rank + qk_rope_head_dim]
        )
        q_pe_seq = q_pe[i * seq_len_q : (i + 1) * seq_len_q]
        compressed_kv_seq = compressed_kv[i * seq_len_q : (i + 1) * seq_len_q].unsqueeze(-2)
        k_pe_seq = k_pe[i * seq_len_q : (i + 1) * seq_len_q].unsqueeze(-2)
        latent_cache_seq = latent_cache[
            total_tokens : total_tokens + sequence_lengths[i]
        ].unsqueeze(-2)

        cos, sin = rope_cos_sin[sequence_lengths[i] : sequence_lengths[i] + seq_len_q].chunk(
            2, dim=-2
        )
        q_pe_seq = q_pe_seq.unflatten(-1, [-1, 2]).transpose(-2, -1).flatten(start_dim=-2)
        k_pe_seq = k_pe_seq.unflatten(-1, [-1, 2]).transpose(-2, -1).flatten(start_dim=-2)
        q_pe_seq = ((q_pe_seq * cos) + (rotate_half(q_pe_seq) * sin)).to(dtype=q_pe_seq.dtype)
        k_pe_seq = ((k_pe_seq * cos) + (rotate_half(k_pe_seq) * sin)).to(dtype=k_pe_seq.dtype)
        q_pe_seq = q_pe_seq.unflatten(-1, [2, -1]).transpose(-2, -1).flatten(start_dim=-2)
        k_pe_seq = k_pe_seq.unflatten(-1, [2, -1]).transpose(-2, -1).flatten(start_dim=-2)
        fused_q_seq[..., -qk_rope_head_dim:] = q_pe_seq
        latent_cache_seq = torch.cat(
            [latent_cache_seq, torch.cat([compressed_kv_seq, k_pe_seq], dim=-1)], dim=0
        )
        latent_cache_list.append(latent_cache_seq)

        q_reshaped.append(
            fused_q_seq.transpose(0, 1)
        )  # (num_heads, seq_len_q, kv_lora_rank + qk_rope_head_dim)
        k_reshaped.append(
            latent_cache_seq.transpose(0, 1)
        )  # (1, seq_len_kv, kv_lora_rank + qk_rope_head_dim)
        v_reshaped.append(
            latent_cache_seq[..., :v_head_dim].transpose(0, 1)
        )  # (1, seq_len_kv, v_head_dim)

        total_tokens += sequence_lengths[i]

    # Calculate reference result batch by batch
    ref_results = []
    for i in range(num_requests):
        q = q_reshaped[i]  # (num_heads, seq_len_q, kv_lora_rank + qk_rope_head_dim)
        k = k_reshaped[i]  # (1, seq_len_kv, kv_lora_rank + qk_rope_head_dim)
        v = v_reshaped[i]  # (1, seq_len_kv, v_head_dim)

        # Handle grouped-query attention
        k = repeat_kv(k.unsqueeze(0), num_heads).squeeze(0)
        v = repeat_kv(v.unsqueeze(0), num_heads).squeeze(0)

        seq_len_q = q.shape[1]
        seq_len_kv = k.shape[1]
        if topk_indices is None:
            # Compute attention scores
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / (
                q_scaling * math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
            )

            # Use MTP mask by default if seqlen_q > 1.
            mask = torch.zeros(seq_len_q, seq_len_kv, device=q.device, dtype=torch.bool)
            for qi in range(seq_len_q):
                for ki in range(seq_len_kv - seq_len_q + 1 + qi, seq_len_kv):
                    mask[qi, ki] = 1
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))
            # Apply softmax to get attention probabilities
            attn_weights = torch.nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(q.dtype)

            # Apply attention weights to values
            attn_output = torch.matmul(attn_weights, v)  # (num_heads, 1, v_head_dim)

            # Reshape back to (seq_len_q, num_heads*v_head_dim)
            attn_output = attn_output.transpose(0, 1).contiguous().view(-1, num_heads * v_head_dim)
            ref_results.append(attn_output)
        else:
            per_token_outputs = []
            for qi in range(seq_len_q):
                row = i * seq_len_q + qi
                token_indices = topk_indices[row]
                token_indices = token_indices[token_indices >= 0]
                q_tok = q[:, qi, :]
                k_sel = k[:, token_indices, :]
                v_sel = v[:, token_indices, :]
                attn_weights = torch.matmul(
                    q_tok.unsqueeze(1),
                    k_sel.transpose(1, 2),
                ) / (q_scaling * math.sqrt(qk_nope_head_dim + qk_rope_head_dim))
                attn_weights = torch.nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(q.dtype)
                attn_output = torch.matmul(attn_weights, v_sel)  # (num_heads, 1, v_head_dim)
                per_token_outputs.append(
                    attn_output.transpose(0, 1).contiguous().view(1, num_heads * v_head_dim)
                )
            ref_results.append(torch.cat(per_token_outputs, dim=0))

    ref_result = torch.cat(ref_results)
    latent_cache = torch.cat(latent_cache_list).squeeze(-2)
    return ref_result, latent_cache


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 1
    num_heads: int = 128
    num_kv_heads: int = 1
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 512
    rope_append: bool = True
    hidden_size: int = 7168
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_factor: float = 40.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    rope_original_max_position_embeddings: int = 4096
    rope_type: str = "yarn"
    model_type: str = "deepseek_v3"
    kv_cache_tokens_per_block: int = 64


@dataclass(kw_only=True, frozen=True)
class RopeConfig:
    hidden_size: int = 7168
    num_attention_heads: int = 128
    rope_scaling: dict = field(
        default_factory=lambda: {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        }
    )
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    qk_rope_head_dim: int = 64
    model_type: str = "deepseek_v3"


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _build_sparse_topk_indices_context(
    seq_lens: List[int], topk: int, device: torch.device
) -> torch.Tensor:
    total_tokens = sum(seq_lens)
    topk_indices = torch.full((total_tokens, topk), -1, dtype=torch.int32, device=device)
    token_offset = 0
    for seq_len in seq_lens:
        for token_idx in range(seq_len):
            max_index = token_idx
            valid_len = min(max_index + 1, topk)
            indices = torch.randperm(max_index + 1, device=device)[:valid_len]
            indices, _ = torch.sort(indices)
            topk_indices[token_offset + token_idx, :valid_len] = indices.to(torch.int32)
        token_offset += seq_len
    return topk_indices


def _build_sparse_topk_indices_generation(
    cached_lens: List[int], seq_len_q: int, topk: int, device: torch.device
) -> torch.Tensor:
    total_tokens = len(cached_lens) * seq_len_q
    topk_indices = torch.full((total_tokens, topk), -1, dtype=torch.int32, device=device)
    row = 0
    for cached_len in cached_lens:
        for q_idx in range(seq_len_q):
            max_index = cached_len + q_idx
            valid_len = min(max_index + 1, topk)
            indices = torch.randperm(max_index + 1, device=device)[:valid_len]
            indices, _ = torch.sort(indices)
            topk_indices[row, :valid_len] = indices.to(torch.int32)
            row += 1
    return topk_indices


def _allocate_kv_cache_for_generation(kv_cache_manager, request_ids, num_tokens: int):
    for request_id in request_ids:
        for _ in range(num_tokens):
            kv_cache_manager.impl.add_token(request_id)
            if hasattr(kv_cache_manager, "indexer_k_cache_manager"):
                kv_cache_manager.indexer_k_cache_manager.add_tokens(request_id, 1)


# Define test data
context_sequence_lengths = [[10], [3000, 3100], [508, 4399, 9981]]
# Use MTP by default if seqlen_q > 1.
generation_seq_len_q = [1, 4]
num_generation_steps = [2]

tokens_per_block = 64

kv_cache_dtype_list = [torch.bfloat16]
# DSA only supports rope_append=True
rope_append_values = [True]
scenarios = [
    Scenario(
        kv_cache_dtype=kv_cache_dtype,
        num_layers=num_layers,
        kv_cache_tokens_per_block=tokens_per_block,
        rope_append=rope_append,
    )
    for kv_cache_dtype in kv_cache_dtype_list
    for num_layers in [1]
    for rope_append in rope_append_values
]

accuracy_dict = {
    torch.bfloat16: (0.1, 0.01),
    torch.float8_e4m3fn: (0.1, 0.01),
}

SPARSE_TOPK = 2048


# Convert parameterized tests to pytest parametrize
@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("scenario", scenarios, ids=lambda x: f"scenario: {x}")
@pytest.mark.parametrize(
    "context_sequence_lengths",
    context_sequence_lengths,
    ids=lambda x: f"context_sequence_lengths: {x}",
)
@pytest.mark.parametrize(
    "generation_seq_len_q", generation_seq_len_q, ids=lambda x: f"generation_seq_len_q: {x}"
)
@pytest.mark.parametrize(
    "num_generation_steps", num_generation_steps, ids=lambda x: f"num_generation_steps: {x}"
)
def test_sparse_attention_mla(
    scenario: Scenario,
    context_sequence_lengths: List[int],
    generation_seq_len_q: int,
    num_generation_steps: int,
):
    """Test sparse MLA computation for both context and generation phases"""
    num_heads = scenario.num_heads
    num_kv_heads = scenario.num_kv_heads
    q_lora_rank = scenario.q_lora_rank
    qk_nope_head_dim = scenario.qk_nope_head_dim
    qk_rope_head_dim = scenario.qk_rope_head_dim
    v_head_dim = scenario.v_head_dim
    rope_append = scenario.rope_append
    if rope_append is False:
        print("rope_append is False, setting num_heads to 64")
        num_heads = 64
    kv_lora_rank = scenario.kv_lora_rank
    rope_config = RopeConfig(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        rope_scaling={
            "beta_fast": scenario.rope_beta_fast,
            "beta_slow": scenario.rope_beta_slow,
            "factor": scenario.rope_factor,
            "mscale": scenario.rope_mscale,
            "mscale_all_dim": scenario.rope_mscale_all_dim,
            "original_max_position_embeddings": scenario.rope_original_max_position_embeddings,
            "type": scenario.rope_type,
        },
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        model_type=scenario.model_type,
    )
    kv_cache_tokens_per_block = scenario.kv_cache_tokens_per_block
    num_layers = scenario.num_layers
    device = torch.device("cuda")
    dtype = scenario.dtype
    kv_cache_dtype = scenario.kv_cache_dtype

    assert SPARSE_TOPK % 128 == 0

    print(
        f"--------------------------------Test for scenario: {scenario} start--------------------------------"
    )

    _run_test_for_backend(
        "TRTLLM",
        num_heads,
        num_kv_heads,
        num_layers,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        rope_append,
        rope_config,
        kv_cache_tokens_per_block,
        device,
        dtype,
        kv_cache_dtype,
        context_sequence_lengths,
        generation_seq_len_q,
        num_generation_steps,
    )


def _run_test_for_backend(
    backend_name,
    num_heads,
    num_kv_heads,
    num_layers,
    q_lora_rank,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    rope_append,
    rope_config,
    kv_cache_tokens_per_block,
    device,
    dtype,
    kv_cache_dtype,
    context_sequence_lengths,
    generation_seq_len_q,
    num_generation_steps,
):
    sparse_config = DeepSeekSparseAttentionConfig(
        index_n_heads=64,
        index_head_dim=128,
        index_topk=SPARSE_TOPK,
        skip_indexer_for_short_seqs=False,
    )
    AttentionCls = get_attention_backend(backend_name, sparse_config)
    # When rope_append is False, [448: 512) are used for qk_rope_head_dim
    kv_lora_rank = kv_lora_rank - qk_rope_head_dim if not rope_append else kv_lora_rank
    head_dim = kv_lora_rank + qk_rope_head_dim

    # Set seed for reproducibility.
    torch.manual_seed(123)

    # Create inputs
    inputs_per_layer = []
    for layer_idx in range(num_layers):
        ctx_compressed_kv = torch.cat(
            [
                torch.empty(
                    [ctx_len, kv_lora_rank],
                    dtype=dtype,
                    device=device,
                ).uniform_(-1, 1)
                for ctx_len in context_sequence_lengths
            ]
        )
        ctx_k_pe = torch.cat(
            [
                torch.empty(
                    [ctx_len, qk_rope_head_dim],
                    dtype=dtype,
                    device=device,
                ).uniform_(-1, 1)
                for ctx_len in context_sequence_lengths
            ]
        )
        ctx_q = torch.cat(
            [
                torch.empty(
                    [ctx_len, num_heads, kv_lora_rank],  # sparse MLA uses absorption mode
                    dtype=dtype,
                    device=device,
                ).uniform_(-1, 1)
                for ctx_len in context_sequence_lengths
            ]
        )
        ctx_q_pe = torch.cat(
            [
                torch.empty(
                    [ctx_len, num_heads, qk_rope_head_dim],
                    dtype=dtype,
                    device=device,
                ).uniform_(-1, 1)
                for ctx_len in context_sequence_lengths
            ]
        )
        ctx_fused_q = torch.cat([ctx_q, ctx_q_pe], dim=-1).view(-1, num_heads * head_dim)

        gen_compressed_kv_list = [
            torch.cat(
                [
                    torch.empty(
                        [generation_seq_len_q, kv_lora_rank],
                        dtype=dtype,
                        device=device,
                    ).uniform_(-1, 1)
                    for _ in context_sequence_lengths
                ]
            )
            for _ in range(num_generation_steps)
        ]
        gen_k_pe_list = [
            torch.cat(
                [
                    torch.empty(
                        [generation_seq_len_q, qk_rope_head_dim],
                        dtype=dtype,
                        device=device,
                    ).uniform_(-1, 1)
                    for _ in context_sequence_lengths
                ]
            )
            for _ in range(num_generation_steps)
        ]
        gen_q_list = [
            torch.cat(
                [
                    torch.empty(
                        [generation_seq_len_q, num_heads, kv_lora_rank],
                        dtype=dtype,
                        device=device,
                    ).uniform_(-1, 1)
                    for _ in context_sequence_lengths
                ]
            )
            for _ in range(num_generation_steps)
        ]
        gen_q_pe_list = [
            torch.cat(
                [
                    torch.empty(
                        [generation_seq_len_q, num_heads, qk_rope_head_dim],
                        dtype=dtype,
                        device=device,
                    ).uniform_(-1, 1)
                    for _ in context_sequence_lengths
                ]
            )
            for _ in range(num_generation_steps)
        ]
        gen_fused_q_list = [
            torch.cat([gen_q_list[i], gen_q_pe_list[i]], dim=-1).view(-1, num_heads * head_dim)
            for i in range(num_generation_steps)
        ]

        inputs = {
            "ctx_compressed_kv": ctx_compressed_kv,
            "ctx_k_pe": ctx_k_pe,
            "ctx_q_pe": ctx_q_pe,
            "ctx_fused_q": ctx_fused_q,
            "gen_compressed_kv_list": gen_compressed_kv_list,
            "gen_k_pe_list": gen_k_pe_list,
            "gen_fused_q_list": gen_fused_q_list,
            "gen_q_pe_list": gen_q_pe_list,
        }
        inputs_per_layer.append(inputs)
        print(f"context sequence lengths: {context_sequence_lengths}")
        for key, val in inputs.items():
            if key.endswith("_list"):
                print(f"{key}: [{val[0].shape}] * {len(val)}")
            else:
                print(f"{key}: {val.shape}")

    rope_cos_sin = (
        torch.tensor(
            RopeEmbeddingUtils.create_sinusoidal_positions_yarn(
                rope_config.max_position_embeddings,
                rope_config.qk_rope_head_dim,
                rope_config.rope_theta,
                rope_config.rope_scaling["factor"],
                rope_config.rope_scaling["original_max_position_embeddings"],
                rope_config.rope_scaling["beta_fast"],
                rope_config.rope_scaling["beta_slow"],
                rope_config.rope_scaling["mscale"],
                rope_config.rope_scaling["mscale_all_dim"],
            )[1],
            dtype=torch.float32,
            device=device,
        )
        .reshape(rope_config.max_position_embeddings, -1, 2)
        .transpose(-2, -1)
    )

    # Setup attention module and metadata
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )
    mla_params = MLAParams(
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        rope_append=rope_append,
        predicted_tokens_per_seq=1,
    )

    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    mscale_all_dim = pos_embd_params.rope.mscale_all_dim
    scaling_factor = pos_embd_params.rope.scale
    mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
    q_scaling = 1.0 / (mscale * mscale)

    quant_config = None
    if kv_cache_dtype == torch.float8_e4m3fn:
        quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8.value)

    ctx_layers = [
        AttentionCls(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            sparse_attention_config=sparse_config,
        )
        for layer_idx in range(num_layers)
    ]
    gen_layers = [
        AttentionCls(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=1,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            sparse_attention_config=sparse_config,
        )
        for layer_idx in range(num_layers)
    ]

    # NOTE: set up metadata, refer to tensorrt_llm/_torch/pyexecutor/model_engine.py
    # all layers share the same metadata
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    max_context_sequence_length = max(context_sequence_lengths)
    max_num_contexts = len(context_sequence_lengths)
    max_tokens = (
        (
            max_context_sequence_length
            + (num_generation_steps + 1) * generation_seq_len_q
            + kv_cache_tokens_per_block
            - 1
        )
        // kv_cache_tokens_per_block
        * kv_cache_tokens_per_block
        * max_num_contexts
    )

    pretrained_config = SimpleNamespace(
        rms_norm_eps=1e-6,
    )
    model_config = ModelConfig(
        mapping=mapping,
        sparse_attention_config=sparse_config,
        pretrained_config=pretrained_config,
    )
    kv_cache_manager = DSACacheManager(
        KvCacheConfig(
            max_tokens=max_tokens,
            enable_block_reuse=False,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=kv_cache_tokens_per_block,
        max_seq_len=max_context_sequence_length + (num_generation_steps + 1) * generation_seq_len_q,
        max_batch_size=max_num_contexts,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(kv_cache_dtype)),
        sparse_attn_config=sparse_config,
        model_config=model_config,
    )
    request_ids = list(range(max_num_contexts))
    kv_cache_manager.add_dummy_requests(request_ids, context_sequence_lengths)

    ctx_seq_lens = torch.tensor(context_sequence_lengths, dtype=torch.int)
    total_ctx_tokens = sum(context_sequence_lengths)
    attn_metadata = AttentionCls.Metadata(
        seq_lens=ctx_seq_lens,
        request_ids=request_ids,
        max_num_requests=max_num_contexts,
        num_contexts=max_num_contexts,
        prompt_lens=context_sequence_lengths,
        max_num_tokens=total_ctx_tokens,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0 for _ in context_sequence_lengths],
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    attn_metadata.prepare()

    # run forward for each step and each layer
    latent_cache_ref_all_list = [None for _ in range(num_layers)]
    for step in range(num_generation_steps + 1):
        if step > 0:
            _allocate_kv_cache_for_generation(kv_cache_manager, request_ids, generation_seq_len_q)
            gen_seq_lens = torch.tensor([generation_seq_len_q] * max_num_contexts, dtype=torch.int)
            total_gen_tokens = max_num_contexts * generation_seq_len_q
            attn_metadata = AttentionCls.Metadata(
                seq_lens=gen_seq_lens,
                request_ids=request_ids,
                max_num_requests=max_num_contexts,
                num_contexts=0,
                prompt_lens=context_sequence_lengths,
                max_num_tokens=total_gen_tokens,
                kv_cache_manager=kv_cache_manager,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[
                        ctx_len + (step - 1) * generation_seq_len_q
                        for ctx_len in context_sequence_lengths
                    ],
                ),
                mapping=mapping,
                enable_flash_mla=torch.cuda.get_device_capability() == (9, 0),
                sparse_attention_config=sparse_config,
            )
            attn_metadata.prepare()
        for layer_idx in range(num_layers):
            print(
                f"--------------------------------step {step} layer {layer_idx} start--------------------------------"
            )
            if step == 0:
                fused_q = inputs_per_layer[layer_idx]["ctx_fused_q"]
                compressed_kv = inputs_per_layer[layer_idx]["ctx_compressed_kv"]
                k_pe = inputs_per_layer[layer_idx]["ctx_k_pe"]
                latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
                q_pe = inputs_per_layer[layer_idx]["ctx_q_pe"]
                topk_indices = _build_sparse_topk_indices_context(
                    context_sequence_lengths, SPARSE_TOPK, device
                )
                result = ctx_layers[layer_idx].forward(
                    fused_q.clone(),
                    None,
                    None,
                    attn_metadata,
                    attention_input_type=AttentionInputType.context_only,
                    latent_cache=latent_cache,
                    q_pe=q_pe,
                    topk_indices=topk_indices,
                    is_generation=False,
                )
                k_pe_ref = _rotate_k_pe_for_ctx(k_pe, rope_cos_sin, context_sequence_lengths)
                latent_cache_ref = torch.cat([compressed_kv, k_pe_ref], dim=-1)
                fused_q_rot = _rotate_fused_q_for_ctx(
                    fused_q,
                    rope_cos_sin,
                    context_sequence_lengths,
                    num_heads,
                    kv_lora_rank,
                    qk_rope_head_dim,
                )
                ref_result = calculate_ref_result_ctx_sparse(
                    fused_q_rot,
                    latent_cache_ref,
                    context_sequence_lengths,
                    num_heads,
                    kv_lora_rank,
                    v_head_dim,
                    qk_nope_head_dim,
                    qk_rope_head_dim,
                    q_scaling,
                    topk_indices=topk_indices,
                )
                latent_cache_ref_all_list[layer_idx] = latent_cache_ref
            else:
                fused_q = inputs_per_layer[layer_idx]["gen_fused_q_list"][step - 1]
                q_pe = inputs_per_layer[layer_idx]["gen_q_pe_list"][step - 1]
                compressed_kv = inputs_per_layer[layer_idx]["gen_compressed_kv_list"][step - 1]
                k_pe = inputs_per_layer[layer_idx]["gen_k_pe_list"][step - 1]
                latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)

                num_tokens = fused_q.size(0)
                num_seqs = attn_metadata.kv_lens_cuda_runtime.size(0)
                cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=fused_q.device)
                cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=fused_q.device)
                fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=fused_q.device)
                has_fp8_kv_cache = (
                    gen_layers[layer_idx].has_fp8_kv_cache
                    if hasattr(gen_layers[layer_idx], "has_fp8_kv_cache")
                    else False
                )

                if has_fp8_kv_cache:
                    mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=fused_q.device)
                    mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=fused_q.device)
                    quant_q_buffer = torch.empty(
                        num_tokens, num_heads * head_dim, dtype=torch.uint8, device=fused_q.device
                    )
                else:
                    mla_bmm1_scale = None
                    mla_bmm2_scale = None
                    quant_q_buffer = None

                gen_layers[layer_idx].mla_rope_generation(
                    fused_q,
                    q_pe,
                    latent_cache,
                    attn_metadata,
                    cu_q_seqlens,
                    cu_kv_seqlens,
                    fmha_scheduler_counter,
                    mla_bmm1_scale,
                    mla_bmm2_scale,
                    quant_q_buffer,
                )

                cached_lens = [
                    ctx_len + (step - 1) * generation_seq_len_q
                    for ctx_len in context_sequence_lengths
                ]
                topk_indices = _build_sparse_topk_indices_generation(
                    cached_lens, generation_seq_len_q, SPARSE_TOPK, device
                )

                result = gen_layers[layer_idx].forward(
                    fused_q,
                    None,
                    None,
                    attn_metadata,
                    attention_input_type=AttentionInputType.generation_only,
                    latent_cache=latent_cache,
                    q_pe=q_pe,
                    cu_q_seqlens=cu_q_seqlens,
                    cu_kv_seqlens=cu_kv_seqlens,
                    fmha_scheduler_counter=fmha_scheduler_counter,
                    mla_bmm1_scale=mla_bmm1_scale,
                    mla_bmm2_scale=mla_bmm2_scale,
                    quant_q_buffer=quant_q_buffer,
                    topk_indices=topk_indices,
                    is_generation=True,
                )
                ref_result, latent_cache_ref = calculate_ref_result_gen(
                    fused_q,
                    q_pe,
                    compressed_kv,
                    k_pe,
                    latent_cache_ref_all_list[layer_idx],
                    rope_cos_sin,
                    num_heads,
                    kv_lora_rank,
                    v_head_dim,
                    qk_nope_head_dim,
                    qk_rope_head_dim,
                    [
                        ctx_len + (step - 1) * generation_seq_len_q
                        for ctx_len in context_sequence_lengths
                    ],
                    q_scaling,
                    topk_indices=topk_indices,
                )
                latent_cache_ref_all_list[layer_idx] = latent_cache_ref
            # Compare results
            print(
                f"{backend_name} output mean: {result.abs().mean().item()}, max: {result.abs().max().item()}"
            )
            print(
                f"Reference output mean: {ref_result.abs().mean().item()}, max: {ref_result.abs().max().item()}"
            )
            print(
                f"Difference mean: {(result - ref_result).abs().mean().item()}, \
                    max: {(result - ref_result).abs().max().item()}"
            )

            # Assert results are close
            atol, rtol = accuracy_dict[kv_cache_dtype]
            assert torch.allclose(result, ref_result, atol=atol, rtol=rtol), (
                f"Results for sparse MLA in {backend_name} backend don't match reference implementation \
                    at layer {layer_idx} in step {step}"
            )

            print(
                f"Test for sparse MLA in {backend_name} backend passed at layer {layer_idx} in step {step}"
            )
            print(
                f"--------------------------------step {step} layer {layer_idx} end--------------------------------"
            )

    print(f"Test for sparse MLA in {backend_name} backend passed")
    kv_cache_manager.shutdown()


if __name__ == "__main__":
    test_sparse_attention_mla(
        scenario=scenarios[0],
        context_sequence_lengths=context_sequence_lengths[0],
        generation_seq_len_q=generation_seq_len_q[0],
        num_generation_steps=num_generation_steps[0],
    )
