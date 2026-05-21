"""
Tests for DeepSeek-V4 sparse MLA attention.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import (
    DeepseekV4AttentionType,
    DeepseekV4CacheManager,
    DeepseekV4TrtllmAttention,
)
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4 import (
    DeepseekV4TrtllmAttentionMetadata,
    get_token_bytes,
)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.functional import PositionEmbeddingType, RopeEmbeddingUtils
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 3
    num_heads: int = 128
    num_kv_heads: int = 1
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512  # raw config value; effective = 448 when rope_append=False
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 512
    rope_append: bool = False  # DeepSeek-V4 requires rope_append=False
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
    kv_cache_tokens_per_block: int = 128
    window_size: int = 128
    compress_ratios: List[int] = field(default_factory=lambda: [1, 4, 128])
    index_topk: int = 512


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


# Layers to test: layer 0 → compress_ratio=1, layer 1 → ratio=4, layer 2 → ratio=128
TEST_LAYERS = [0, 1, 2]


# RoPE helpers
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


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _create_cache_manager(scenario: Scenario, context_lengths: List[int], max_seq_len: int):
    sparse_config = DeepSeekV4SparseAttentionConfig(
        index_n_heads=64,
        index_head_dim=128,
        window_size=scenario.window_size,
        compress_ratios=scenario.compress_ratios,
        index_topk=scenario.index_topk,
        skip_indexer_for_short_seqs=False,
    )
    batch_size = len(context_lengths)
    max_input_len = max(context_lengths)

    cache_manager = DeepseekV4CacheManager(
        kv_cache_config=KvCacheConfig(
            max_tokens=max_seq_len * batch_size,
            enable_block_reuse=False,
            event_buffer_max_size=0,
        ),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=scenario.num_layers,
        num_kv_heads=1,
        # When rope_append=False: effective kv_lora_rank is 448, head_dim = 448 + 64 = 512
        head_dim=(
            scenario.kv_lora_rank - scenario.qk_rope_head_dim
            if not scenario.rope_append
            else scenario.kv_lora_rank
        )
        + scenario.qk_rope_head_dim,
        tokens_per_block=scenario.kv_cache_tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        max_input_len=max_input_len,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.BF16,
        compressor_dtype=DataType.FLOAT,
        vocab_size=129280,
        max_num_tokens=max_input_len * batch_size + batch_size,
        sparse_attn_config=sparse_config,
    )
    return cache_manager, sparse_config


def _prefill_compress_buffer(
    cache_manager: DeepseekV4CacheManager,
    layer_idx: int,
    context_lengths: List[int],
    request_ids: List[int],
    head_dim: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """Pre-fill COMPRESS buffer with known random data.

    Returns flat reference data per request.
    """
    compress_ratio = cache_manager._compress_ratios[layer_idx]
    buffer = cache_manager.get_buffers(layer_idx, DeepseekV4AttentionType.COMPRESS)
    tokens_per_block_compressed = cache_manager.compressed_block_sizes[layer_idx]

    ref_data = []
    for req_idx, ctx_len in enumerate(context_lengths):
        num_compressed = ctx_len // compress_ratio
        data = torch.randn(num_compressed, head_dim, device=device, dtype=torch.bfloat16)
        ref_data.append(data)

        block_ids = cache_manager.get_cache_indices(
            request_ids[req_idx], layer_idx, DeepseekV4AttentionType.COMPRESS
        )
        for tok_idx in range(num_compressed):
            block_idx = tok_idx // tokens_per_block_compressed
            offset = tok_idx % tokens_per_block_compressed
            buffer[block_ids[block_idx], offset, :head_dim] = data[tok_idx]

    return ref_data


def _grow_compress_buffer_for_generation(
    cache_manager: DeepseekV4CacheManager,
    layer_idx: int,
    request_ids: List[int],
    head_dim: int,
    device: torch.device,
    kv_lens: List[int],
    compress_ref_data_per_layer: List[torch.Tensor],
):
    """Grow COMPRESS buffer and ref data for generation steps.

    As KV length increases during generation, new compressed tokens appear.
    This appends random data for those new entries to both the COMPRESS buffer
    and the reference data list (in-place).
    """
    compress_ratio = cache_manager._compress_ratios[layer_idx]
    buffer = cache_manager.get_buffers(layer_idx, DeepseekV4AttentionType.COMPRESS)
    tokens_per_block_compressed = cache_manager.tokens_per_block // compress_ratio

    for req_idx in range(len(request_ids)):
        old_count = compress_ref_data_per_layer[req_idx].shape[0]
        new_total = kv_lens[req_idx] // compress_ratio
        num_new = new_total - old_count
        if num_new <= 0:
            continue
        new_data = torch.randn(num_new, head_dim, device=device, dtype=torch.bfloat16)
        block_ids = cache_manager.get_cache_indices(
            request_ids[req_idx], layer_idx, DeepseekV4AttentionType.COMPRESS
        )
        for j in range(num_new):
            tok_idx = old_count + j
            block_idx = tok_idx // tokens_per_block_compressed
            offset = tok_idx % tokens_per_block_compressed
            buffer[block_ids[block_idx], offset, :head_dim] = new_data[j]
        compress_ref_data_per_layer[req_idx] = torch.cat(
            [compress_ref_data_per_layer[req_idx], new_data], dim=0
        )


def _build_compressed_topk_indices(
    token_positions: List[int],
    compress_ratio: int,
    topk: int,
    device: torch.device,
) -> torch.Tensor:
    """Build random topk indices into compressed space for testing."""
    num_tokens = len(token_positions)
    indices = torch.full((num_tokens, topk), -1, dtype=torch.int32, device=device)
    for i, pos in enumerate(token_positions):
        num_compressed = (pos + 1) // compress_ratio
        if num_compressed > 0:
            valid_k = min(topk, num_compressed)
            selected = torch.randperm(num_compressed, device=device)[:valid_k].sort().values
            indices[i, :valid_k] = selected.to(torch.int32)
    return indices


def _softmax_with_sink(
    logits: torch.Tensor,
    attn_sink: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Numerically stable softmax with an optional per-head sink in the denominator.

    The sink behaves like a virtual register key whose value is zero: it
    contributes exp(sink - max) to the denominator only, leaving the numerator
    untouched.
    """
    fp32_logits = logits.float()
    if attn_sink is None:
        return torch.softmax(fp32_logits, dim=-1).to(out_dtype)
    max_val = fp32_logits.amax(dim=-1, keepdim=True)
    num = torch.exp(fp32_logits - max_val)
    denom = num.sum(dim=-1, keepdim=True)
    sink = attn_sink.float().view(-1, 1, 1)
    denom = denom + torch.exp(sink - max_val)
    return (num / denom).to(out_dtype)


def calculate_deepseek_v4_ref_ctx_sparse(
    fused_q_rot: torch.Tensor,
    latent_cache_ref: torch.Tensor,
    compressed_ref_data: Optional[List[torch.Tensor]],
    swa_window_size: int,
    compressed_topk_indices: Optional[torch.Tensor],
    seq_lens: List[int],
    num_heads: int,
    kv_lora_rank: int,
    v_head_dim: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    q_scaling: float,
    compress_ratio: int,
    attn_sink: Optional[torch.Tensor] = None,
):
    """Per-token reference attention for DeepSeek-V4 context phase.

    For compress_ratio==1: only SWA tokens (causal window).
    For compress_ratio==4: SWA tokens + indexer topk compressed tokens.
    For compress_ratio==128: SWA tokens + all compressed tokens.
    """
    fused_head_dim = kv_lora_rank + qk_rope_head_dim
    bmm1_scale = 1 / (math.sqrt(qk_nope_head_dim + qk_rope_head_dim) * q_scaling)

    ref_results = []
    token_offset = 0
    for batch_idx, seq_len in enumerate(seq_lens):
        per_token_outputs = []
        for token_idx in range(seq_len):
            global_token_idx = token_offset + token_idx
            pos = token_idx

            # Gather SWA KV
            swa_start = max(0, pos - swa_window_size + 1)
            swa_end = pos + 1
            swa_kv = latent_cache_ref[token_offset + swa_start : token_offset + swa_end]

            # Gather compressed KV
            if compress_ratio > 1 and compressed_ref_data is not None:
                if compress_ratio == 4 and compressed_topk_indices is not None:
                    indices_row = compressed_topk_indices[global_token_idx]
                    valid = indices_row[indices_row >= 0]
                    comp_kv = compressed_ref_data[batch_idx][valid.long()]
                elif compress_ratio == 128:
                    num_comp = (pos + 1) // compress_ratio
                    comp_kv = compressed_ref_data[batch_idx][:num_comp]
                else:
                    comp_kv = torch.empty(
                        0,
                        swa_kv.shape[-1],
                        device=swa_kv.device,
                        dtype=swa_kv.dtype,
                    )

                if comp_kv.numel() > 0:
                    all_kv = torch.cat([swa_kv, comp_kv], dim=0)
                else:
                    all_kv = swa_kv
            else:
                all_kv = swa_kv

            # Compute attention
            q_tok = fused_q_rot[global_token_idx].view(num_heads, fused_head_dim)
            k_sel = all_kv.unsqueeze(0).expand(num_heads, -1, -1)
            v_sel = all_kv[:, :v_head_dim].unsqueeze(0).expand(num_heads, -1, -1)

            attn_w = torch.matmul(q_tok.unsqueeze(1), k_sel.transpose(1, 2)) * bmm1_scale
            attn_w = _softmax_with_sink(attn_w, attn_sink, fused_q_rot.dtype)
            out = torch.matmul(attn_w, v_sel).squeeze(1)
            per_token_outputs.append(out.reshape(1, num_heads * v_head_dim))

        ref_results.append(torch.cat(per_token_outputs, dim=0))
        token_offset += seq_len

    return torch.cat(ref_results, dim=0)


def _rotate_gen_inputs(
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    seq_lens_kv: List[int],
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fused_head_dim = kv_lora_rank + qk_rope_head_dim
    num_requests = len(seq_lens_kv)
    seq_len_q = fused_q.shape[0] // num_requests

    fused_q_rot = fused_q.clone()
    new_latent_list = []

    for i in range(num_requests):
        past_len = seq_lens_kv[i]

        fused_q_seq = fused_q_rot[i * seq_len_q : (i + 1) * seq_len_q].unflatten(
            -1, [num_heads, fused_head_dim]
        )
        q_pe_seq = q_pe[i * seq_len_q : (i + 1) * seq_len_q]
        compressed_kv_seq = compressed_kv[i * seq_len_q : (i + 1) * seq_len_q]
        k_pe_seq = k_pe[i * seq_len_q : (i + 1) * seq_len_q].unsqueeze(-2)

        cos, sin = rope_cos_sin[past_len : past_len + seq_len_q].chunk(2, dim=-2)

        # Apply RoPE to q_pe
        q_pe_seq = q_pe_seq.unflatten(-1, [-1, 2]).transpose(-2, -1).flatten(start_dim=-2)
        q_pe_seq = ((q_pe_seq * cos) + (rotate_half(q_pe_seq) * sin)).to(dtype=q_pe_seq.dtype)
        q_pe_seq = q_pe_seq.unflatten(-1, [2, -1]).transpose(-2, -1).flatten(start_dim=-2)
        fused_q_seq[..., -qk_rope_head_dim:] = q_pe_seq

        # Apply RoPE to k_pe
        k_pe_seq = k_pe_seq.unflatten(-1, [-1, 2]).transpose(-2, -1).flatten(start_dim=-2)
        k_pe_seq = ((k_pe_seq * cos) + (rotate_half(k_pe_seq) * sin)).to(dtype=k_pe_seq.dtype)
        k_pe_seq = k_pe_seq.unflatten(-1, [2, -1]).transpose(-2, -1).flatten(start_dim=-2)

        # Build new-token latent: [compressed_kv, rotated_k_pe]
        new_latent = torch.cat([compressed_kv_seq.unsqueeze(-2), k_pe_seq], dim=-1).squeeze(
            -2
        )  # [seq_len_q, head_dim]
        new_latent_list.append(new_latent)

    return fused_q_rot, torch.cat(new_latent_list, dim=0)


def calculate_deepseek_v4_ref_gen_sparse(
    fused_q_rot: torch.Tensor,
    new_latent_cache: torch.Tensor,
    latent_cache_ref: torch.Tensor,
    compressed_ref_data: Optional[List[torch.Tensor]],
    compressed_topk_indices: Optional[torch.Tensor],
    swa_window_size: int,
    seq_lens_kv: List[int],
    num_heads: int,
    kv_lora_rank: int,
    v_head_dim: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    q_scaling: float,
    compress_ratio: int,
    attn_sink: Optional[torch.Tensor] = None,
):
    """Reference attention for DeepSeek-V4 generation phase."""
    fused_head_dim = kv_lora_rank + qk_rope_head_dim
    bmm1_scale = 1 / (math.sqrt(qk_nope_head_dim + qk_rope_head_dim) * q_scaling)
    num_requests = len(seq_lens_kv)
    seq_len_q = fused_q_rot.shape[0] // num_requests

    ref_results = []
    latent_cache_list = []
    total_past_tokens = 0
    for i in range(num_requests):
        past_len = seq_lens_kv[i]

        fused_q_seq = fused_q_rot[i * seq_len_q : (i + 1) * seq_len_q].unflatten(
            -1, [num_heads, fused_head_dim]
        )

        # New token's latent cache
        new_token_latent = new_latent_cache[i * seq_len_q : (i + 1) * seq_len_q].unsqueeze(
            -2
        )  # [seq_len_q, 1, head_dim]

        # Past latent cache
        past_latent = latent_cache_ref[total_past_tokens : total_past_tokens + past_len].unsqueeze(
            -2
        )

        # Combine past + new
        full_latent = torch.cat([past_latent, new_token_latent], dim=0)
        latent_cache_list.append(full_latent)

        # For each query token, gather the relevant KV
        for qi in range(seq_len_q):
            q_tok = fused_q_seq[qi]  # [num_heads, fused_head_dim]
            current_kv_len = past_len + qi + 1
            current_pos = past_len + qi

            # SWA window
            swa_start = max(0, current_pos - swa_window_size + 1)
            swa_end = current_pos + 1
            swa_kv = full_latent[swa_start:swa_end].squeeze(-2)

            # Compressed KV
            if compress_ratio > 1 and compressed_ref_data is not None:
                if compress_ratio == 4 and compressed_topk_indices is not None:
                    row = i * seq_len_q + qi
                    indices_row = compressed_topk_indices[row]
                    valid = indices_row[indices_row >= 0]
                    comp_kv = compressed_ref_data[i][valid.long()]
                elif compress_ratio == 128:
                    num_comp = current_kv_len // compress_ratio
                    comp_kv = compressed_ref_data[i][:num_comp]
                else:
                    comp_kv = torch.empty(
                        0,
                        swa_kv.shape[-1],
                        device=swa_kv.device,
                        dtype=swa_kv.dtype,
                    )

                if comp_kv.numel() > 0:
                    all_kv = torch.cat([swa_kv, comp_kv], dim=0)
                else:
                    all_kv = swa_kv
            else:
                all_kv = swa_kv

            k_sel = all_kv.unsqueeze(0).expand(num_heads, -1, -1)
            v_sel = all_kv[:, :v_head_dim].unsqueeze(0).expand(num_heads, -1, -1)

            attn_w = torch.matmul(q_tok.unsqueeze(1), k_sel.transpose(1, 2)) * bmm1_scale
            attn_w = _softmax_with_sink(attn_w, attn_sink, fused_q_rot.dtype)
            out = torch.matmul(attn_w, v_sel).squeeze(1)
            ref_results.append(out.reshape(1, num_heads * v_head_dim))

        total_past_tokens += past_len

    ref_result = torch.cat(ref_results, dim=0)
    new_latent_cache_out = torch.cat(latent_cache_list).squeeze(-2)
    return ref_result, new_latent_cache_out


def _allocate_kv_cache_for_generation(cache_manager, requests: List[LlmRequest]):
    for req in requests:
        assert cache_manager.try_allocate_generation(req), (
            f"Failed to allocate generation KV cache for request {req.py_request_id}"
        )


def _create_rope_config(scenario: Scenario) -> RopeConfig:
    return RopeConfig(
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


def _create_rope_cos_sin(scenario: Scenario, device: torch.device) -> torch.Tensor:
    rope_config = _create_rope_config(scenario)
    return (
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


def _create_pos_embd_params(scenario: Scenario) -> PositionalEmbeddingParams:
    return PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(_create_rope_config(scenario)),
        is_neox=False,
    )


@skip_pre_blackwell
def test_deepseek_v4_sparse_mla_single_token_tp4_local_heads_repro():
    """Reproduce the tp=4 Flash warmup sparse-MLA kernel shape.

    The full DeepSeek-V4 model has 64 effective Q heads when rope_append=False.
    With TP=4, each rank launches the sparse MLA kernel with 16 local Q heads.
    This single-rank test directly uses that local head count to cover the
    attention kernel argument shape without involving TP collectives.
    """
    scenario = Scenario()
    device = torch.device("cuda")
    dtype = scenario.dtype
    torch.manual_seed(42)

    context_lengths = [1]
    local_num_heads = int(os.environ.get("DSV4_REPRO_LOCAL_NUM_HEADS", "16"))
    layer_idx = 0
    ratio = scenario.compress_ratios[layer_idx]
    assert ratio == 1

    qk_nope_head_dim = scenario.qk_nope_head_dim
    qk_rope_head_dim = scenario.qk_rope_head_dim
    v_head_dim = scenario.v_head_dim
    rope_append = scenario.rope_append
    kv_lora_rank = (
        scenario.kv_lora_rank - qk_rope_head_dim if not rope_append else scenario.kv_lora_rank
    )
    head_dim = kv_lora_rank + qk_rope_head_dim

    cache_manager, sparse_config = _create_cache_manager(scenario, context_lengths, max_seq_len=1)
    request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=[0],
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    cache_manager.prepare_context(request)
    cache_manager.resize_context(request, request.context_chunk_size)

    pos_embd_params = _create_pos_embd_params(scenario)
    mla_params = MLAParams(
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        rope_append=rope_append,
        predicted_tokens_per_seq=1,
        hidden_size=scenario.hidden_size,
    )
    q_scaling = 1.0 / (scenario.rope_mscale * scenario.rope_mscale)

    layer = DeepseekV4TrtllmAttention(
        layer_idx=layer_idx,
        num_heads=local_num_heads,
        head_dim=head_dim,
        num_kv_heads=1,
        q_scaling=q_scaling,
        pos_embd_params=pos_embd_params,
        mla_params=mla_params,
        sparse_attention_config=sparse_config,
        skip_create_weights_in_init=True,
    )
    layer.update_quant_config(None)
    attn_sink = torch.randn(local_num_heads, dtype=torch.float32, device=device).mul_(0.5)
    if not os.environ.get("DSV4_REPRO_NO_SINK"):
        layer.attn_sink = torch.nn.Parameter(attn_sink, requires_grad=False)

    attn_metadata = DeepseekV4TrtllmAttentionMetadata(
        seq_lens=torch.tensor(context_lengths, dtype=torch.int),
        request_ids=[0],
        max_num_requests=1,
        num_contexts=1,
        prompt_lens=context_lengths,
        max_num_tokens=1,
        kv_cache_manager=cache_manager,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        sparse_attention_config=sparse_config,
    )
    attn_metadata.prepare()

    assert attn_metadata.sparse_mla_topk_lens[ratio][0].item() == 1
    assert attn_metadata.swa_local_indices_cuda[0, 0].item() == 0
    assert torch.all(attn_metadata.swa_local_indices_cuda[0, 1:] == -1)

    compressed_kv = torch.empty([1, kv_lora_rank], dtype=dtype, device=device).uniform_(-1, 1)
    k_pe = torch.empty([1, qk_rope_head_dim], dtype=dtype, device=device).uniform_(-1, 1)
    q_nope = torch.empty([1, local_num_heads, kv_lora_rank], dtype=dtype, device=device).uniform_(
        -1, 1
    )
    q_pe = torch.empty([1, local_num_heads, qk_rope_head_dim], dtype=dtype, device=device).uniform_(
        -1, 1
    )
    fused_q = torch.cat([q_nope, q_pe], dim=-1).view(1, local_num_heads * head_dim)
    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)

    simple_swa_pool = None
    if os.environ.get("DSV4_REPRO_SIMPLE_POOL"):
        simple_swa_pool = torch.empty(
            (cache_manager.tokens_per_block, head_dim), dtype=dtype, device=device
        )
        simple_swa_pool.zero_()
        simple_swa_pool_ptr = simple_swa_pool.data_ptr()
        cache_manager.kv_cache_pool_pointers[0, 0] = simple_swa_pool_ptr
        attn_metadata.host_kv_cache_pool_pointers[0, 0] = simple_swa_pool_ptr
        attn_metadata.sparse_mla_base_ptrs[1] = simple_swa_pool_ptr
        attn_metadata.swa_buffer_ptrs[layer_idx] = simple_swa_pool_ptr
        attn_metadata.block_tables[(1, DeepseekV4AttentionType.SWA)][0].fill_(-1)
        attn_metadata.block_tables[(1, DeepseekV4AttentionType.SWA)][0, 0] = 0
        torch.cuda.synchronize()

    if os.environ.get("DSV4_REPRO_PRINT_PARAMS"):
        sparse_attn_indices, _ = layer.sparse_attn_predict(
            fused_q,
            None,
            attn_metadata,
            AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                latent_cache=latent_cache,
                q_pe=q_pe,
            ),
        )
        torch.cuda.synchronize()
        print("DSV4_REPRO local_num_heads", local_num_heads)
        print(
            "DSV4_REPRO token_stride",
            get_token_bytes(
                head_dim,
                sparse_config.index_head_dim,
                ratio,
                DeepseekV4AttentionType.SWA,
                False,
            ),
        )
        print("DSV4_REPRO tokens_per_block", cache_manager.tokens_per_block)
        print("DSV4_REPRO max_blocks_per_seq", cache_manager.max_blocks_per_seq)
        print("DSV4_REPRO max_seq_len", cache_manager.max_seq_len)
        print("DSV4_REPRO metadata_max_seq_len", attn_metadata.max_seq_len)
        print("DSV4_REPRO kv_lens_runtime", attn_metadata.kv_lens_runtime[:1].cpu().tolist())
        print("DSV4_REPRO host_total_kv_lens", attn_metadata.host_total_kv_lens.cpu().tolist())
        print(
            "DSV4_REPRO prompt_lens_runtime",
            attn_metadata.prompt_lens_cpu_runtime[:1].cpu().tolist(),
        )
        print("DSV4_REPRO swa_pool_base_ptr", attn_metadata.sparse_mla_base_ptrs[1])
        print("DSV4_REPRO swa_buffer_ptr", attn_metadata.swa_buffer_ptrs[layer_idx])
        print(
            "DSV4_REPRO block_table_swa",
            attn_metadata.block_tables[(1, DeepseekV4AttentionType.SWA)][:1, :4].cpu().tolist(),
        )
        print("DSV4_REPRO sparse_attn_indices", sparse_attn_indices.cpu().tolist())
        print("DSV4_REPRO sparse_attn_indices_dtype", sparse_attn_indices.dtype)
        print(
            "DSV4_REPRO sparse_mla_topk_lens",
            attn_metadata.sparse_mla_topk_lens[ratio][:1].cpu().tolist(),
        )
        print(
            "DSV4_REPRO sparse_mla_topk_lens_dtype", attn_metadata.sparse_mla_topk_lens[ratio].dtype
        )

    softmax_stats_tensor = None
    if os.environ.get("DSV4_REPRO_SOFTMAX_STATS"):
        softmax_stats_tensor = torch.empty(
            (1, local_num_heads, 2), dtype=torch.float32, device=device
        )

    result = layer.forward(
        fused_q.clone(),
        None,
        None,
        attn_metadata,
        attention_input_type=AttentionInputType.context_only,
        latent_cache=latent_cache,
        q_pe=q_pe,
        topk_indices=None,
        is_generation=False,
        softmax_stats_tensor=softmax_stats_tensor,
    )

    rope_cos_sin = _create_rope_cos_sin(scenario, device)
    k_pe_ref = _rotate_k_pe_for_ctx(k_pe, rope_cos_sin, context_lengths)
    latent_cache_ref = torch.cat([compressed_kv, k_pe_ref], dim=-1)
    fused_q_rot = _rotate_fused_q_for_ctx(
        fused_q,
        rope_cos_sin,
        context_lengths,
        local_num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
    )
    ref_result = calculate_deepseek_v4_ref_ctx_sparse(
        fused_q_rot,
        latent_cache_ref,
        None,
        scenario.window_size,
        None,
        context_lengths,
        local_num_heads,
        kv_lora_rank,
        v_head_dim,
        qk_nope_head_dim,
        qk_rope_head_dim,
        q_scaling,
        ratio,
        attn_sink=attn_sink,
    )

    torch.testing.assert_close(result, ref_result, atol=0.2, rtol=0.02)
    cache_manager.shutdown()


@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("context_lengths", [[4399], [14, 508, 3947], [2, 1406, 3327]])
@pytest.mark.parametrize("num_generation_steps", [2])
def test_deepseek_v4_sparse_mla(context_lengths: List[int], num_generation_steps: int):
    generation_seq_len_q = 1
    scenario = Scenario()
    device = torch.device("cuda")
    dtype = scenario.dtype

    qk_nope_head_dim = scenario.qk_nope_head_dim
    qk_rope_head_dim = scenario.qk_rope_head_dim
    v_head_dim = scenario.v_head_dim
    rope_append = scenario.rope_append
    # When rope_append is False, [448: 512) are used for qk_rope_head_dim
    kv_lora_rank = (
        scenario.kv_lora_rank - qk_rope_head_dim if not rope_append else scenario.kv_lora_rank
    )
    head_dim = kv_lora_rank + qk_rope_head_dim
    # When rope_append is False, use 64 heads
    num_heads = 64 if not rope_append else scenario.num_heads
    batch_size = len(context_lengths)
    max_context_len = max(context_lengths)
    max_seq_len = max_context_len + (num_generation_steps + 1) * generation_seq_len_q
    total_ctx_tokens = sum(context_lengths)

    torch.manual_seed(42)

    # 1. Setup cache manager
    cache_manager, sparse_config = _create_cache_manager(scenario, context_lengths, max_seq_len)
    request_ids = list(range(batch_size))

    requests = [
        LlmRequest(
            request_id=i,
            max_new_tokens=num_generation_steps + 1,
            input_tokens=list(range(ctx_len)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        for i, ctx_len in enumerate(context_lengths)
    ]
    scheduled_batch = ScheduledRequests()
    for req in requests:
        scheduled_batch.append_context_request(req)
    for req in requests:
        cache_manager.prepare_context(req)
        cache_manager.resize_context(req, req.context_chunk_size)

    mapping = Mapping(world_size=1, tp_size=1, rank=0)

    # 2. Create RoPE cos/sin
    rope_cos_sin = _create_rope_cos_sin(scenario, device)

    # 3. Setup attention params
    pos_embd_params = _create_pos_embd_params(scenario)
    mla_params = MLAParams(
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        rope_append=rope_append,
        predicted_tokens_per_seq=1,
        hidden_size=scenario.hidden_size,
    )

    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    mscale_all_dim = pos_embd_params.rope.mscale_all_dim
    scaling_factor = pos_embd_params.rope.scale
    mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
    q_scaling = 1.0 / (mscale * mscale)

    # 4. Create attention layers
    layers = {}
    for layer_idx in TEST_LAYERS:
        layer = DeepseekV4TrtllmAttention(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=1,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            sparse_attention_config=sparse_config,
            skip_create_weights_in_init=True,
        )
        layer.update_quant_config(None)
        layers[layer_idx] = layer

    # Install a per-layer attention sink
    attn_sinks: Dict[int, torch.Tensor] = {}
    for layer_idx in TEST_LAYERS:
        sink = torch.randn(num_heads, dtype=torch.float32, device=device).mul_(0.5)
        layers[layer_idx].attn_sink = torch.nn.Parameter(sink, requires_grad=False)
        attn_sinks[layer_idx] = sink

    # 5. Create random inputs per layer
    inputs_per_layer = {}
    for layer_idx in TEST_LAYERS:
        ctx_compressed_kv = torch.cat(
            [
                torch.empty([ctx_len, kv_lora_rank], dtype=dtype, device=device).uniform_(-1, 1)
                for ctx_len in context_lengths
            ]
        )
        ctx_k_pe = torch.cat(
            [
                torch.empty([ctx_len, qk_rope_head_dim], dtype=dtype, device=device).uniform_(-1, 1)
                for ctx_len in context_lengths
            ]
        )
        ctx_q = torch.cat(
            [
                torch.empty(
                    [ctx_len, num_heads, kv_lora_rank],
                    dtype=dtype,
                    device=device,
                ).uniform_(-1, 1)
                for ctx_len in context_lengths
            ]
        )
        ctx_q_pe = torch.cat(
            [
                torch.empty(
                    [ctx_len, num_heads, qk_rope_head_dim],
                    dtype=dtype,
                    device=device,
                ).uniform_(-1, 1)
                for ctx_len in context_lengths
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
                    for _ in context_lengths
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
                    for _ in context_lengths
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
                    for _ in context_lengths
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
                    for _ in context_lengths
                ]
            )
            for _ in range(num_generation_steps)
        ]
        gen_fused_q_list = [
            torch.cat([gen_q_list[i], gen_q_pe_list[i]], dim=-1).view(-1, num_heads * head_dim)
            for i in range(num_generation_steps)
        ]

        inputs_per_layer[layer_idx] = {
            "ctx_compressed_kv": ctx_compressed_kv,
            "ctx_k_pe": ctx_k_pe,
            "ctx_q_pe": ctx_q_pe,
            "ctx_fused_q": ctx_fused_q,
            "gen_compressed_kv_list": gen_compressed_kv_list,
            "gen_k_pe_list": gen_k_pe_list,
            "gen_fused_q_list": gen_fused_q_list,
            "gen_q_pe_list": gen_q_pe_list,
        }

    # 6. Pre-fill COMPRESS buffers for layers with ratio > 1
    compress_ref_data: Dict[int, List[torch.Tensor]] = {}
    for layer_idx in TEST_LAYERS:
        ratio = scenario.compress_ratios[layer_idx]
        if ratio > 1:
            compress_ref_data[layer_idx] = _prefill_compress_buffer(
                cache_manager,
                layer_idx,
                context_lengths,
                request_ids,
                head_dim,
                device,
            )

    # 7. Context phase
    ctx_seq_lens = torch.tensor(context_lengths, dtype=torch.int)
    attn_metadata = DeepseekV4TrtllmAttentionMetadata(
        seq_lens=ctx_seq_lens,
        request_ids=request_ids,
        max_num_requests=batch_size,
        num_contexts=batch_size,
        prompt_lens=context_lengths,
        max_num_tokens=total_ctx_tokens,
        kv_cache_manager=cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * batch_size,
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    attn_metadata.prepare()

    latent_cache_ref_all: Dict[int, torch.Tensor] = {}
    for layer_idx in TEST_LAYERS:
        ratio = scenario.compress_ratios[layer_idx]
        print(f"\n--- Context phase: layer {layer_idx}, compress_ratio={ratio} ---")

        inp = inputs_per_layer[layer_idx]
        fused_q = inp["ctx_fused_q"]
        compressed_kv = inp["ctx_compressed_kv"]
        k_pe = inp["ctx_k_pe"]
        q_pe = inp["ctx_q_pe"]
        latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)

        # Build topk_indices for ratio=4
        topk_indices = None
        if ratio == 4:
            token_positions = []
            for ctx_len in context_lengths:
                token_positions.extend(range(ctx_len))
            topk_indices = _build_compressed_topk_indices(
                token_positions, ratio, scenario.index_topk, device
            )

        # Forward through the attention layer
        result = layers[layer_idx].forward(
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

        # Reference computation
        k_pe_ref = _rotate_k_pe_for_ctx(k_pe, rope_cos_sin, context_lengths)
        latent_cache_ref = torch.cat([compressed_kv, k_pe_ref], dim=-1)
        fused_q_rot = _rotate_fused_q_for_ctx(
            fused_q,
            rope_cos_sin,
            context_lengths,
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
        )
        ref_result = calculate_deepseek_v4_ref_ctx_sparse(
            fused_q_rot,
            latent_cache_ref,
            compress_ref_data.get(layer_idx),
            scenario.window_size,
            topk_indices,
            context_lengths,
            num_heads,
            kv_lora_rank,
            v_head_dim,
            qk_nope_head_dim,
            qk_rope_head_dim,
            q_scaling,
            ratio,
            attn_sink=attn_sinks[layer_idx],
        )

        latent_cache_ref_all[layer_idx] = latent_cache_ref

        # Check results
        print(f"  Output shape: {result.shape}")
        print(
            f"  Output mean: {result.abs().mean().item():.6f}, max: {result.abs().max().item():.6f}"
        )
        print(
            f"  Ref mean: {ref_result.abs().mean().item():.6f}, "
            f"max: {ref_result.abs().max().item():.6f}"
        )
        diff = (result - ref_result).abs()
        print(f"  Diff mean: {diff.mean().item():.6f}, max: {diff.max().item():.6f}")

        assert torch.allclose(result, ref_result, atol=0.2, rtol=0.02), (
            f"Context phase mismatch at layer {layer_idx} (ratio={ratio}): "
            f"max diff={diff.max().item():.6f}"
        )
        print("  PASSED")

    for req, ctx_len in zip(requests, context_lengths):
        req.context_current_position = ctx_len
        req.add_new_token(ctx_len, 0)
    cache_manager.update_context_resources(scheduled_batch)

    # 8. Generation steps
    for step in range(num_generation_steps):
        print(f"\n=== Generation step {step + 1} ===")
        _allocate_kv_cache_for_generation(cache_manager, requests)

        cached_lens = [ctx_len + step * generation_seq_len_q for ctx_len in context_lengths]
        kv_lens = [cl + generation_seq_len_q for cl in cached_lens]

        # Grow compressed ref data: as KV length increases, new compressed
        # tokens appear. Simulate this by appending random data.
        for layer_idx in TEST_LAYERS:
            ratio = scenario.compress_ratios[layer_idx]
            if ratio > 1 and layer_idx in compress_ref_data:
                _grow_compress_buffer_for_generation(
                    cache_manager,
                    layer_idx,
                    request_ids,
                    head_dim,
                    device,
                    kv_lens,
                    compress_ref_data[layer_idx],
                )

        gen_seq_lens = torch.tensor([generation_seq_len_q] * batch_size, dtype=torch.int)
        total_gen_tokens = batch_size * generation_seq_len_q
        gen_metadata = DeepseekV4TrtllmAttentionMetadata(
            seq_lens=gen_seq_lens,
            request_ids=request_ids,
            max_num_requests=batch_size,
            num_contexts=0,
            prompt_lens=context_lengths,
            max_num_tokens=total_gen_tokens,
            kv_cache_manager=cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=cached_lens,
            ),
            mapping=mapping,
            enable_flash_mla=torch.cuda.get_device_capability() == (9, 0),
            sparse_attention_config=sparse_config,
        )
        gen_metadata.prepare()

        for layer_idx in TEST_LAYERS:
            ratio = scenario.compress_ratios[layer_idx]
            print(f"\n--- Gen step {step + 1}: layer {layer_idx}, compress_ratio={ratio} ---")

            inp = inputs_per_layer[layer_idx]
            fused_q = inp["gen_fused_q_list"][step]
            q_pe = inp["gen_q_pe_list"][step]
            compressed_kv = inp["gen_compressed_kv_list"][step]
            k_pe = inp["gen_k_pe_list"][step]
            latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)

            # Prepare generation-specific tensors
            num_seqs = gen_metadata.kv_lens_cuda_runtime.size(0)
            cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
            cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
            fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=device)

            layers[layer_idx].mla_rope_generation(
                fused_q,
                q_pe,
                latent_cache,
                gen_metadata,
                cu_q_seqlens,
                cu_kv_seqlens,
                fmha_scheduler_counter,
                None,  # mla_bmm1_scale
                None,  # mla_bmm2_scale
                None,  # quant_q_buffer
            )

            # Build topk_indices for ratio=4
            topk_indices = None
            if ratio == 4:
                topk_indices = _build_compressed_topk_indices(
                    [kv - 1 for kv in kv_lens],
                    ratio,
                    scenario.index_topk,
                    device,
                )

            result = layers[layer_idx].forward(
                fused_q,
                None,
                None,
                gen_metadata,
                attention_input_type=AttentionInputType.generation_only,
                latent_cache=latent_cache,
                q_pe=q_pe,
                cu_q_seqlens=cu_q_seqlens,
                cu_kv_seqlens=cu_kv_seqlens,
                fmha_scheduler_counter=fmha_scheduler_counter,
                topk_indices=topk_indices,
                is_generation=True,
            )

            # Reference: apply RoPE separately, then compute attention
            fused_q_rot, new_latent = _rotate_gen_inputs(
                fused_q,
                q_pe,
                compressed_kv,
                k_pe,
                rope_cos_sin,
                cached_lens,
                num_heads,
                kv_lora_rank,
                qk_rope_head_dim,
            )
            ref_result, new_latent_cache = calculate_deepseek_v4_ref_gen_sparse(
                fused_q_rot,
                new_latent,
                latent_cache_ref_all[layer_idx],
                compress_ref_data.get(layer_idx),
                topk_indices,
                scenario.window_size,
                cached_lens,
                num_heads,
                kv_lora_rank,
                v_head_dim,
                qk_nope_head_dim,
                qk_rope_head_dim,
                q_scaling,
                ratio,
                attn_sink=attn_sinks[layer_idx],
            )
            latent_cache_ref_all[layer_idx] = new_latent_cache

            print(f"  Output shape: {result.shape}")
            print(
                f"  Output mean: {result.abs().mean().item():.6f}, "
                f"max: {result.abs().max().item():.6f}"
            )
            print(
                f"  Ref mean: {ref_result.abs().mean().item():.6f}, "
                f"max: {ref_result.abs().max().item():.6f}"
            )
            diff = (result - ref_result).abs()
            print(f"  Diff mean: {diff.mean().item():.6f}, max: {diff.max().item():.6f}")

            assert torch.allclose(result, ref_result, atol=0.2, rtol=0.02), (
                f"Gen step {step + 1} mismatch at layer {layer_idx} "
                f"(ratio={ratio}): max diff={diff.max().item():.6f}"
            )
            print("  PASSED")

    cache_manager.shutdown()
    print("\nAll tests passed!")


@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("context_lengths", [[14, 508, 3947]])
def test_deepseek_v4_sparse_mla_mixed_batch(context_lengths: List[int]):
    scenario = Scenario()
    device = torch.device("cuda")
    dtype = scenario.dtype
    generation_seq_len_q = 1

    qk_nope_head_dim = scenario.qk_nope_head_dim
    qk_rope_head_dim = scenario.qk_rope_head_dim
    v_head_dim = scenario.v_head_dim
    rope_append = scenario.rope_append
    kv_lora_rank = (
        scenario.kv_lora_rank - qk_rope_head_dim if not rope_append else scenario.kv_lora_rank
    )
    head_dim = kv_lora_rank + qk_rope_head_dim
    num_heads = 64 if not rope_append else scenario.num_heads
    mscale = scenario.rope_mscale
    q_scaling = 1.0 / (mscale * mscale)

    batch_size = len(context_lengths)
    assert batch_size >= 2
    num_ctx = 1
    num_gen = batch_size - num_ctx
    max_context_len = max(context_lengths)
    max_seq_len = max_context_len + 2 * generation_seq_len_q
    total_ctx_tokens = context_lengths[0]
    total_gen_tokens = num_gen * generation_seq_len_q
    total_mixed_tokens = total_ctx_tokens + total_gen_tokens

    torch.manual_seed(42)

    # 1. Setup cache, layers, RoPE
    cache_manager, sparse_config = _create_cache_manager(scenario, context_lengths, max_seq_len)
    request_ids = list(range(batch_size))

    requests = [
        LlmRequest(
            request_id=i,
            max_new_tokens=2,
            input_tokens=list(range(ctx_len)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        for i, ctx_len in enumerate(context_lengths)
    ]
    scheduled_batch = ScheduledRequests()
    for req in requests:
        scheduled_batch.append_context_request(req)
    for req in requests:
        cache_manager.prepare_context(req)
        cache_manager.resize_context(req, req.context_chunk_size)

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    rope_cos_sin = _create_rope_cos_sin(scenario, device)
    pos_embd_params = _create_pos_embd_params(scenario)
    mla_params = MLAParams(
        q_lora_rank=scenario.q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        rope_append=rope_append,
        predicted_tokens_per_seq=1,
    )

    layers = {}
    for li in TEST_LAYERS:
        layer = DeepseekV4TrtllmAttention(
            layer_idx=li,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=scenario.num_kv_heads,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            sparse_attention_config=sparse_config,
            dtype=dtype,
        )
        layer.update_quant_config(None)
        layers[li] = layer

    attn_sinks: Dict[int, torch.Tensor] = {}
    for li in TEST_LAYERS:
        sink = torch.randn(num_heads, dtype=torch.float32, device=device).mul_(0.5)
        layers[li].attn_sink = torch.nn.Parameter(sink, requires_grad=False)
        attn_sinks[li] = sink

    # 2. Pre-fill KV cache for gen requests (1..N) — all layers, all ratios.
    gen_ctx_lengths = context_lengths[num_ctx:]
    gen_request_ids = request_ids[num_ctx:]
    gen_total_ctx = sum(gen_ctx_lengths)

    prefill_fused_q = torch.empty(
        [gen_total_ctx, num_heads * head_dim], dtype=dtype, device=device
    ).uniform_(-1, 1)
    prefill_k_pe = torch.empty(
        [gen_total_ctx, qk_rope_head_dim], dtype=dtype, device=device
    ).uniform_(-1, 1)
    prefill_q_pe = torch.empty(
        [gen_total_ctx, num_heads, qk_rope_head_dim], dtype=dtype, device=device
    ).uniform_(-1, 1)
    prefill_compressed_kv = torch.empty(
        [gen_total_ctx, kv_lora_rank], dtype=dtype, device=device
    ).uniform_(-1, 1)
    prefill_latent = torch.cat([prefill_compressed_kv, prefill_k_pe], dim=-1)

    prefill_metadata = DeepseekV4TrtllmAttentionMetadata(
        seq_lens=torch.tensor(gen_ctx_lengths, dtype=torch.int),
        request_ids=gen_request_ids,
        max_num_requests=num_gen,
        num_contexts=num_gen,
        prompt_lens=gen_ctx_lengths,
        max_num_tokens=gen_total_ctx,
        kv_cache_manager=cache_manager,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0] * num_gen),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    prefill_metadata.prepare()
    for li in TEST_LAYERS:
        ratio = scenario.compress_ratios[li]
        prefill_topk = None
        if ratio == 4:
            positions = []
            for cl in gen_ctx_lengths:
                positions.extend(range(cl))
            prefill_topk = _build_compressed_topk_indices(
                positions, ratio, scenario.index_topk, device
            )
        layers[li].forward(
            prefill_fused_q.clone(),
            None,
            None,
            prefill_metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=prefill_latent,
            q_pe=prefill_q_pe,
            topk_indices=prefill_topk,
            is_generation=False,
        )

    gen_requests = requests[1:]
    gen_prefill_batch = ScheduledRequests()
    gen_prefill_batch.context_requests_last_chunk = gen_requests
    for req, ctx_len in zip(gen_requests, gen_ctx_lengths):
        req.context_current_position = ctx_len
        req.add_new_token(ctx_len, 0)
    cache_manager.update_context_resources(gen_prefill_batch)

    # Pre-fill COMPRESS buffers for ratio > 1.
    compress_ref_data: Dict[int, List[torch.Tensor]] = {}
    for li in TEST_LAYERS:
        ratio = scenario.compress_ratios[li]
        if ratio > 1:
            compress_ref_data[li] = _prefill_compress_buffer(
                cache_manager,
                li,
                gen_ctx_lengths,
                gen_request_ids,
                head_dim,
                device,
            )

    # 3. Allocate 1 gen step for gen requests.
    _allocate_kv_cache_for_generation(cache_manager, gen_requests)
    gen_cached_lens = [cl + generation_seq_len_q for cl in gen_ctx_lengths]

    # Grow compress buffers for gen step.
    gen_kv_lens = [cl + generation_seq_len_q for cl in gen_cached_lens]
    for li in TEST_LAYERS:
        ratio = scenario.compress_ratios[li]
        if ratio > 1 and li in compress_ref_data:
            _grow_compress_buffer_for_generation(
                cache_manager,
                li,
                gen_request_ids,
                head_dim,
                device,
                gen_kv_lens,
                compress_ref_data[li],
            )

    # 4. Mixed metadata: request 0 = context, requests 1..N = generation.
    mixed_seq_lens = [context_lengths[0]] + [generation_seq_len_q] * num_gen
    mixed_cached_lens = [0] + gen_cached_lens
    mixed_metadata = DeepseekV4TrtllmAttentionMetadata(
        seq_lens=torch.tensor(mixed_seq_lens, dtype=torch.int),
        request_ids=request_ids,
        max_num_requests=batch_size,
        num_contexts=num_ctx,
        prompt_lens=context_lengths,
        max_num_tokens=total_mixed_tokens,
        kv_cache_manager=cache_manager,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=mixed_cached_lens),
        mapping=mapping,
        enable_flash_mla=torch.cuda.get_device_capability() == (9, 0),
        sparse_attention_config=sparse_config,
    )
    mixed_metadata.prepare()

    # 5. Per-layer forward + verify (mirrors forward_impl_with_deepseek_v4).
    for li in TEST_LAYERS:
        ratio = scenario.compress_ratios[li]
        print(f"\n--- Mixed: layer {li}, compress_ratio={ratio} ---")

        # Random inputs
        ctx_q_nope = torch.empty(
            [total_ctx_tokens, num_heads, kv_lora_rank], dtype=dtype, device=device
        ).uniform_(-1, 1)
        ctx_q_pe = torch.empty(
            [total_ctx_tokens, num_heads, qk_rope_head_dim], dtype=dtype, device=device
        ).uniform_(-1, 1)
        ctx_fused_q = torch.cat([ctx_q_nope, ctx_q_pe], dim=-1).view(-1, num_heads * head_dim)
        ctx_k_pe = torch.empty(
            [total_ctx_tokens, qk_rope_head_dim], dtype=dtype, device=device
        ).uniform_(-1, 1)
        ctx_compressed_kv = torch.empty(
            [total_ctx_tokens, kv_lora_rank], dtype=dtype, device=device
        ).uniform_(-1, 1)
        ctx_latent = torch.cat([ctx_compressed_kv, ctx_k_pe], dim=-1)

        gen_fused_q = torch.empty(
            [total_gen_tokens, num_heads * head_dim], dtype=dtype, device=device
        ).uniform_(-1, 1)
        gen_q_pe = torch.empty(
            [total_gen_tokens, num_heads, qk_rope_head_dim], dtype=dtype, device=device
        ).uniform_(-1, 1)
        gen_compressed_kv = torch.empty(
            [total_gen_tokens, kv_lora_rank], dtype=dtype, device=device
        ).uniform_(-1, 1)
        gen_k_pe = torch.empty(
            [total_gen_tokens, qk_rope_head_dim], dtype=dtype, device=device
        ).uniform_(-1, 1)
        gen_latent = torch.cat([gen_compressed_kv, gen_k_pe], dim=-1)

        output = torch.empty(
            [total_mixed_tokens, num_heads * v_head_dim], dtype=dtype, device=device
        )

        # topk_indices for ratio=4
        ctx_topk = gen_topk = None
        if ratio == 4:
            ctx_positions = list(range(context_lengths[0]))
            ctx_topk = _build_compressed_topk_indices(
                ctx_positions, ratio, scenario.index_topk, device
            )
            gen_topk = _build_compressed_topk_indices(
                [kv - 1 for kv in [cl + generation_seq_len_q for cl in gen_cached_lens]],
                ratio,
                scenario.index_topk,
                device,
            )

        # Context forward → output[:total_ctx_tokens]
        layers[li].forward(
            ctx_fused_q.clone(),
            None,
            None,
            mixed_metadata,
            attention_input_type=AttentionInputType.context_only,
            output=output[:total_ctx_tokens],
            latent_cache=ctx_latent,
            q_pe=ctx_q_pe,
            topk_indices=ctx_topk,
            is_generation=False,
        )

        # Generation forward → output[total_ctx_tokens:]
        num_seqs = mixed_metadata.kv_lens_cuda_runtime.size(0)
        cu_q = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
        cu_kv = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
        counter = torch.empty(1, dtype=torch.uint32, device=device)
        layers[li].mla_rope_generation(
            gen_fused_q,
            gen_q_pe,
            gen_latent,
            mixed_metadata,
            cu_q,
            cu_kv,
            counter,
            None,
            None,
            None,
        )
        layers[li].forward(
            gen_fused_q,
            None,
            None,
            mixed_metadata,
            attention_input_type=AttentionInputType.generation_only,
            output=output[total_ctx_tokens:total_mixed_tokens],
            latent_cache=gen_latent,
            q_pe=gen_q_pe,
            cu_q_seqlens=cu_q,
            cu_kv_seqlens=cu_kv,
            fmha_scheduler_counter=counter,
            topk_indices=gen_topk,
            is_generation=True,
        )

        # Context reference
        ctx_k_pe_ref = _rotate_k_pe_for_ctx(ctx_k_pe, rope_cos_sin, [context_lengths[0]])
        ctx_latent_ref = torch.cat([ctx_compressed_kv, ctx_k_pe_ref], dim=-1)
        ctx_q_rot = _rotate_fused_q_for_ctx(
            ctx_fused_q,
            rope_cos_sin,
            [context_lengths[0]],
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
        )
        ctx_ref = calculate_deepseek_v4_ref_ctx_sparse(
            ctx_q_rot,
            ctx_latent_ref,
            None,
            scenario.window_size,
            ctx_topk,
            [context_lengths[0]],
            num_heads,
            kv_lora_rank,
            v_head_dim,
            qk_nope_head_dim,
            qk_rope_head_dim,
            q_scaling,
            ratio,
            attn_sink=attn_sinks[li],
        )

        # Generation reference
        prefill_k_pe_ref = _rotate_k_pe_for_ctx(prefill_k_pe, rope_cos_sin, gen_ctx_lengths)
        gen_latent_ref = torch.cat([prefill_compressed_kv, prefill_k_pe_ref], dim=-1)
        gen_q_rot, new_latent = _rotate_gen_inputs(
            gen_fused_q,
            gen_q_pe,
            gen_compressed_kv,
            gen_k_pe,
            rope_cos_sin,
            gen_cached_lens,
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
        )
        gen_ref, _ = calculate_deepseek_v4_ref_gen_sparse(
            gen_q_rot,
            new_latent,
            gen_latent_ref,
            compress_ref_data.get(li),
            gen_topk,
            scenario.window_size,
            gen_cached_lens,
            num_heads,
            kv_lora_rank,
            v_head_dim,
            qk_nope_head_dim,
            qk_rope_head_dim,
            q_scaling,
            ratio,
            attn_sink=attn_sinks[li],
        )

        ctx_diff = (output[:total_ctx_tokens] - ctx_ref).abs()
        gen_diff = (output[total_ctx_tokens:] - gen_ref).abs()
        print(
            f"  Context:    diff mean={ctx_diff.mean().item():.6f}, max={ctx_diff.max().item():.6f}"
        )
        print(
            f"  Generation: diff mean={gen_diff.mean().item():.6f}, max={gen_diff.max().item():.6f}"
        )
        assert torch.allclose(output[:total_ctx_tokens], ctx_ref, atol=0.2, rtol=0.02), (
            f"Mixed ctx mismatch layer {li} ratio={ratio}: max diff={ctx_diff.max().item():.6f}"
        )
        assert torch.allclose(output[total_ctx_tokens:], gen_ref, atol=0.2, rtol=0.02), (
            f"Mixed gen mismatch layer {li} ratio={ratio}: max diff={gen_diff.max().item():.6f}"
        )
        print("  PASSED")

    cache_manager.shutdown()
    print("\nMixed batch test PASSED!")


if __name__ == "__main__":
    test_deepseek_v4_sparse_mla(context_lengths=[4399], num_generation_steps=2)
