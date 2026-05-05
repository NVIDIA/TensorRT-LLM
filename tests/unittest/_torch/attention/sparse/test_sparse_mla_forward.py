"""
Test suite for deepseek sparse attention with kvcache_dtype=bf16 using FlashMLA backend.
"""
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionBackend, PositionalEmbeddingParams, RopeParams)
from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4 import \
    DeepseekV4CacheManager
from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.attention import MLA
from tensorrt_llm._utils import (get_sm_version, str_dtype_to_binding,
                                 torch_dtype_to_str)
from tensorrt_llm.functional import PositionEmbeddingType, RopeEmbeddingUtils
from tensorrt_llm.llmapi.llm_args import (DeepSeekSparseAttentionConfig,
                                          DeepSeekV4SparseAttentionConfig,
                                          KvCacheConfig)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from .deepseek_v4.test_compressor_module import RefCompressor

try:
    HAS_FLASH_MLA = True
except ImportError:
    HAS_FLASH_MLA = False


@dataclass
class BatchSpec:
    """Batch specification for testing, following vLLM's pattern."""
    seq_lens: List[int]
    query_lens: List[int]

    @property
    def batch_size(self):
        return len(self.seq_lens)


@dataclass
class DummyPretrainedConfig:
    """Dummy pretrained configuration for testing."""
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_head_dim: int
    head_size: int
    topk_tokens: int
    hidden_size: int
    max_position_embeddings: int
    num_groups: int
    o_lora_rank: int
    num_layers: int
    vocab_size: int
    compress_ratios: int


# Unified batch specifications covering all scenarios
# seq_lens = total KV cache length (for decode/mixed) or current sequence length (for pure prefill)
# query_lens = number of new query tokens per request
BATCH_SPECS = {
    # Pure prefill scenarios (query_lens == seq_lens)
    "small_prefill":
    BatchSpec(seq_lens=[32, 48, 24], query_lens=[32, 48, 24]),
    "medium_prefill":
    BatchSpec(seq_lens=[128, 256, 64], query_lens=[128, 256, 64]),
    "single_prefill":
    BatchSpec(seq_lens=[512], query_lens=[512]),

    # Pure decode/generation scenarios (query_lens == 1)
    # seq_lens includes the cached context + 1 new token
    "small_decode":
    BatchSpec(
        seq_lens=[33, 49, 25],  # Previous context + 1 new token
        query_lens=[1, 1, 1]),
    "medium_decode":
    BatchSpec(seq_lens=[129, 257, 65], query_lens=[1, 1, 1]),
    "single_decode":
    BatchSpec(seq_lens=[513], query_lens=[1]),

    # Mixed scenarios (some prefill, some decode)
    # Format: [prefill_req1, prefill_req2, decode_req1, decode_req2, ...]
    "small_mixed":
    BatchSpec(
        seq_lens=[32, 48, 25, 33
                  ],  # First 2 are prefill, last 2 are decode (cached: 24, 32)
        query_lens=[32, 48, 1,
                    1]  # First 2 process full seq, last 2 generate 1 token
    ),
    "medium_mixed":
    BatchSpec(
        seq_lens=[128, 65, 257],  # 1 prefill, 1 decode, 1 decode
        query_lens=[128, 1, 1]),
    "mixed_heavy_prefill":
    BatchSpec(
        seq_lens=[256, 128, 64, 33, 49],  # 3 prefill, 2 decode
        query_lens=[256, 128, 64, 1, 1]),
    "mixed_heavy_decode":
    BatchSpec(
        seq_lens=[32, 33, 49, 65, 129],  # 1 prefill, 4 decode
        query_lens=[32, 1, 1, 1, 1]),
}


def apply_rotary_emb(x: torch.Tensor,
                     freqs_cis: torch.Tensor,
                     inverse: bool = False) -> torch.Tensor:
    """Apply rotary positional embeddings for DeepSeek-V4."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast,
                         beta_slow) -> torch.Tensor:
    """Precompute rotary embeddings for DeepSeek-V4."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(
            max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max -
                                                                        min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base**(torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base,
                                          original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def _calc_diff(x: torch.Tensor, y: torch.Tensor):
    """Return a global difference metric for unit tests.

    When comparing the fp8 results with  the bf16 reference, there are
    noticeable differences, causing ``torch.testing.assert_close`` to fail.
    Instead of checking every element, we compute a cosine-style similarity
    over the whole tensor and report ``1 - sim``.
    """
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim.item()


def apply_rotary_embedding(
        x: torch.Tensor,  # [seq_len, ..., rope_dim]
        cos_sin: torch.Tensor,  # [seq_len, 2, rope_dim]
        inverse: bool = False,  # If True, apply inverse RoPE
) -> torch.Tensor:
    """
    Apply rotary position embedding for DSA.
    For deepseek_v4, supports inverse rotation by negating sin.
    """

    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    original_dtype = x.dtype
    cos, sin = cos_sin.chunk(2, dim=-2)
    cos = cos.squeeze(1)
    sin = sin.squeeze(1)

    # For inverse RoPE, negate sin
    if inverse:
        sin = -sin

    # Convert to interleaved format
    x_interleaved = x.unflatten(-1, [-1, 2]).transpose(-2,
                                                       -1).flatten(start_dim=-2)
    cos_expanded = cos.view(cos.shape[0], *([1] * (x.ndim - 2)), cos.shape[-1])
    sin_expanded = sin.view(sin.shape[0], *([1] * (x.ndim - 2)), sin.shape[-1])
    # Apply rotation
    x_rotated = (x_interleaved * cos_expanded) + (rotate_half(x_interleaved) *
                                                  sin_expanded)

    # Convert back to original dtype
    return x_rotated.to(original_dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value tensors n_rep times along the KV head dimension.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch, num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


@dataclass
class RopeConfig:
    """Configuration for RoPE."""
    hidden_size: int
    num_attention_heads: int
    rope_scaling: dict
    max_position_embeddings: int
    rope_theta: float
    qk_rope_head_dim: int
    model_type: str


def calculate_reference_output_prefill_only(q_c, kv_c, k_pe, W_UK, W_UV,
                                            rope_cos_sin, sequence_lengths,
                                            num_heads, kv_lora_rank,
                                            qk_nope_head_dim, qk_rope_head_dim,
                                            v_head_dim, softmax_scale, device):
    """Reference for pure prefill (unrotated inputs, applies RoPE internally)."""
    results, offset = [], 0
    for seq_len in sequence_lengths:
        q_seq = q_c[offset:offset + seq_len]
        kv_seq = kv_c[offset:offset + seq_len]
        k_pe_seq = k_pe[offset:offset + seq_len]

        q_nope, q_pe = q_seq.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
        cos_sin = rope_cos_sin[:seq_len]
        q_pe = apply_rotary_embedding(q_pe, cos_sin)
        k_pe_rot = apply_rotary_embedding(k_pe_seq, cos_sin)

        ql_nope = torch.einsum("qnh,lnh->qnl", q_nope, W_UK)
        q_mqa = torch.cat([ql_nope, q_pe], dim=-1)
        k_mqa = torch.cat([
            kv_seq.unsqueeze(1).expand(-1, num_heads, -1),
            k_pe_rot.unsqueeze(1).expand(-1, num_heads, -1)
        ],
                          dim=-1)
        v_mqa = kv_seq.unsqueeze(1).expand(-1, num_heads, -1)

        q_in = q_mqa.unsqueeze(0).transpose(1, 2)
        k_in = k_mqa.unsqueeze(0).transpose(1, 2)
        v_in = v_mqa.unsqueeze(0).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q_in, k_in, v_in, is_causal=True,
            scale=softmax_scale).transpose(1, 2).squeeze(0)

        results.append(
            torch.einsum("qnl,lnv->qnv", out, W_UV).flatten(start_dim=-2))
        offset += seq_len
    return torch.cat(results, dim=0)


def calculate_reference_output_generation(q_c, kv_c, k_pe, W_UK, W_UV,
                                          rope_cos_sin, kv_cache_lens,
                                          num_heads, kv_lora_rank,
                                          qk_nope_head_dim, qk_rope_head_dim,
                                          v_head_dim, softmax_scale, device):
    """Reference for generation using unrotated inputs and explicit RoPE."""
    results, kv_offset = [], 0
    for kv_len in kv_cache_lens:
        q_seq = q_c[len(results):len(results) +
                    1]  # [1, num_heads, qk_head_dim]
        kv_seq = kv_c[kv_offset:kv_offset + kv_len]
        k_pe_seq = k_pe[kv_offset:kv_offset + kv_len]

        q_nope, q_pe = q_seq.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
        # Apply RoPE
        cos_sin_q = rope_cos_sin[kv_len - q_seq.shape[0]:kv_len]
        cos_sin_pe = rope_cos_sin[:kv_len]
        q_pe = apply_rotary_embedding(q_pe, cos_sin_q)
        k_pe_rot = apply_rotary_embedding(k_pe_seq, cos_sin_pe)

        ql_nope = torch.einsum("qnh,lnh->qnl", q_nope, W_UK)
        q_mqa = torch.cat([ql_nope, q_pe], dim=-1)

        k_mqa = torch.cat([
            kv_seq.unsqueeze(1).expand(-1, num_heads, -1),
            k_pe_rot.unsqueeze(1).expand(-1, num_heads, -1)
        ],
                          dim=-1)
        v_mqa = kv_seq.unsqueeze(1).expand(-1, num_heads, -1)

        q_in = q_mqa.unsqueeze(0).transpose(1, 2)
        k_in = k_mqa.unsqueeze(0).transpose(1, 2)
        v_in = v_mqa.unsqueeze(0).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            q_in, k_in, v_in, attn_mask=None,
            scale=softmax_scale).transpose(1, 2).squeeze(0)

        results.append(
            torch.einsum("qnl,lnv->qnv", out, W_UV).flatten(start_dim=-2))
        kv_offset += kv_len
    return torch.cat(results, dim=0)


def calculate_reference_output_mixed(q_ctx, q_gen, kv_c_all, k_pe_all, W_UK,
                                     W_UV, rope_cos_sin, ctx_indices,
                                     gen_indices, seq_lens, query_lens,
                                     num_heads, kv_lora_rank, qk_nope_head_dim,
                                     qk_rope_head_dim, v_head_dim,
                                     softmax_scale, device):
    """Reference for mixed batch (combines context and generation)."""
    ref_results = [None] * len(seq_lens)

    # Extract KV slices for context and generation (in [context...][generation...] layout)
    def extract_kv_slices(indices, start_offset=0):
        slices_kv, slices_kpe = [], []
        offset = start_offset
        for req_idx in indices:
            seq_len = seq_lens[req_idx]
            slices_kv.append(kv_c_all[offset:offset + seq_len])
            slices_kpe.append(k_pe_all[offset:offset + seq_len])
            offset += seq_len
        return torch.cat(slices_kv) if slices_kv else None, \
               torch.cat(slices_kpe) if slices_kpe else None

    # Process context requests (unrotated, apply RoPE)
    if ctx_indices:
        kv_c, k_pe = extract_kv_slices(ctx_indices, 0)
        ctx_results = calculate_reference_output_prefill_only(
            q_ctx, kv_c, k_pe, W_UK, W_UV, rope_cos_sin,
            [seq_lens[i]
             for i in ctx_indices], num_heads, kv_lora_rank, qk_nope_head_dim,
            qk_rope_head_dim, v_head_dim, softmax_scale, device)
        offset = 0
        for req_idx in ctx_indices:
            seq_len = seq_lens[req_idx]
            ref_results[req_idx] = ctx_results[offset:offset + seq_len]
            offset += seq_len

    # Process generation requests (rotated, no RoPE)
    if gen_indices:
        kv_c, k_pe = extract_kv_slices(gen_indices,
                                       sum(seq_lens[i] for i in ctx_indices))
        gen_results = calculate_reference_output_generation(
            q_gen, kv_c, k_pe, W_UK, W_UV, rope_cos_sin,
            [seq_lens[i]
             for i in gen_indices], num_heads, kv_lora_rank, qk_nope_head_dim,
            qk_rope_head_dim, v_head_dim, softmax_scale, device)
        for idx, req_idx in enumerate(gen_indices):
            ref_results[req_idx] = gen_results[idx:idx + 1]

    return torch.cat(ref_results, dim=0)


def calculate_reference_output_deepseek_v4_prefill(hidden_states,
                                                   indices,
                                                   seq_lens,
                                                   q,
                                                   kv_data,
                                                   freqs_cis,
                                                   ref_compressor,
                                                   num_heads,
                                                   qk_nope_head_dim,
                                                   qk_rope_head_dim,
                                                   v_head_dim,
                                                   softmax_scale,
                                                   device,
                                                   o_a_proj,
                                                   o_b_proj_weight,
                                                   n_local_groups,
                                                   window_size=512,
                                                   compress_ratio=4):
    """Reference for deepseek_v4 prefill.

    Implements attention over:
    1. Sliding window: last window_size tokens
    2. Compressed KV: all compressed tokens (assuming all are selected)
    """
    results, offset = [], 0
    device = q.device

    for req_idx in indices:
        seq_len = seq_lens[req_idx]
        q_seq = q[offset:offset + seq_len]  # [seq_len, num_heads, qk_head_dim]
        hidden_states_seq = hidden_states[offset:offset + seq_len]
        kv_seq = kv_data["latent_kv"][req_idx]  # [seq_len, head_dim]

        # Apply RoPE to Q/KV
        q_seq = q_seq.unsqueeze(0)
        kv_seq = kv_seq.unsqueeze(0)
        apply_rotary_emb(q_seq[..., -qk_rope_head_dim:], freqs_cis[:seq_len])
        apply_rotary_emb(kv_seq[..., -qk_rope_head_dim:], freqs_cis[:seq_len])

        # Compressor
        if compress_ratio > 1:
            kv_cache = torch.zeros(1,
                                   seq_len // compress_ratio,
                                   ref_compressor.head_dim,
                                   device=device)
            ref_compressor.kv_cache = kv_cache
            compressed_kv = ref_compressor(hidden_states_seq.unsqueeze(0),
                                           start_pos=0,
                                           freqs_cis=freqs_cis[:seq_len])
            num_compressed = compressed_kv.shape[
                1] if compressed_kv is not None else 0
            if num_compressed > 0:
                # RefCompressor returns fp32 (matches the kernel's
                # fp32-throughout postprocess); cast to the attention dtype
                # before concatenation.
                kv_combined = torch.cat(
                    [kv_seq, compressed_kv.to(kv_seq.dtype)], dim=1).squeeze(0)
            else:
                kv_combined = kv_seq.squeeze(0)
                num_compressed = 0
        else:
            kv_combined = kv_seq.squeeze(0)
            num_compressed = 0

        # Create attention mask: [seq_len, seq_len + num_compressed]
        # KV structure: [0:seq_len] = window_kv, [seq_len:seq_len+num_compressed] = compressed_kv
        # Each query at position i can attend to:
        # 1. Window tokens: positions [max(0, i - window_size + 1), i] in range [0, seq_len)
        # 2. Compressed tokens: positions [seq_len, seq_len + i//compress_ratio) representing original tokens before position i
        attn_mask = torch.full((seq_len, seq_len + num_compressed),
                               float('-inf'),
                               device=device,
                               dtype=q_seq.dtype)

        for i in range(seq_len):
            # Window KV: query at position i can attend to positions [max(0, i - window_size + 1), i]
            window_start = max(0, i - window_size + 1)
            attn_mask[i, window_start:i + 1] = 0

            # Compressed KV: production uses (pos+1)//compress_ratio valid compressed tokens
            if num_compressed > 0:
                max_compressed_idx = (i + 1) // compress_ratio
                if max_compressed_idx > 0:
                    attn_mask[i, seq_len:seq_len + max_compressed_idx] = 0

        # Expand KV for multi-head (deepseek_v4 uses MQA - single KV head for all Q heads)
        k_combined = kv_combined.unsqueeze(1).expand(-1, num_heads, -1)
        # Use full kv_combined as V (not just first v_head_dim dimensions) to match reference
        v_combined = kv_combined.unsqueeze(1).expand(-1, num_heads, -1)

        # Prepare for attention: [1, num_heads, seq_len, head_dim]
        # q: [1, num_heads, seq_len, qk_head_dim]
        # k: [1, num_heads, seq_len + num_compressed, qk_head_dim]
        # v: [1, num_heads, seq_len + num_compressed, qk_head_dim]
        # attn_mask: [1, 1, seq_len, seq_len + num_compressed]
        q_in = q_seq.transpose(1, 2)
        k_in = k_combined.unsqueeze(0).transpose(1, 2)
        v_in = v_combined.unsqueeze(0).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        # Compute attention with custom mask, attn_out: # [seq_len, num_heads, qk_head_dim]
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q_in, k_in, v_in, attn_mask=attn_mask,
            scale=softmax_scale).transpose(1, 2).squeeze(0)

        results.append(attn_out)
        offset += seq_len

    return torch.cat(results, dim=0).flatten(1)


def calculate_reference_output_deepseek_v4_generation(hidden_states,
                                                      gen_indices,
                                                      seq_lens,
                                                      q,
                                                      kv_data,
                                                      freqs_cis,
                                                      ref_compressor,
                                                      num_contexts,
                                                      num_heads,
                                                      qk_nope_head_dim,
                                                      qk_rope_head_dim,
                                                      v_head_dim,
                                                      softmax_scale,
                                                      device,
                                                      o_a_proj,
                                                      o_b_proj_weight,
                                                      n_local_groups,
                                                      window_size=512,
                                                      compress_ratio=4):
    """Reference for deepseek_v4 generation with cached window_kv and compressed_kv.

    For generation, we have:
    - 1 new query token and its latent KV
    - Cached window KV (last window_size tokens)
    - Cached compressed KV (compressed tokens from earlier in sequence)
    - Compressor state (kv_state, score_state) that needs to be restored
    """
    results = []
    qk_nope_head_dim + qk_rope_head_dim

    for gen_req_idx in gen_indices:
        req_idx = gen_req_idx - num_contexts
        # Total sequence length including cache + new token
        seq_len = seq_lens[gen_req_idx]
        start_pos = seq_len - 1  # Position of the new token

        q_seq = q[req_idx].unsqueeze(0)  # [1, num_heads, qk_head_dim]
        kv_seq = kv_data["latent_kv"][gen_req_idx]  # [1, head_dim]
        hidden_states_seq = hidden_states[req_idx].unsqueeze(0)  # [1, head_dim]

        # Get cached data
        # window_kv: [window_size, head_dim]
        # compressed_kv: [num_compressed, head_dim] or None
        # kv_state: Compressor internal KV state
        # score_state: Compressor internal score state
        window_kv = kv_data["window_kv"][gen_req_idx]
        compressed_kv = kv_data["compressed_kv"][gen_req_idx]
        # The compressed_kv cache may have been populated from a fp32
        # RefCompressor return; cast back to the attention dtype.
        if compressed_kv is not None:
            compressed_kv = compressed_kv.to(kv_seq.dtype)
        kv_state = kv_data["kv_state"][gen_req_idx]
        score_state = kv_data["score_state"][gen_req_idx]

        # Apply RoPE to query at current position
        q_seq = q_seq.unsqueeze(0)
        kv_seq = kv_seq.unsqueeze(0)
        apply_rotary_emb(q_seq[..., -qk_rope_head_dim:],
                         freqs_cis[start_pos:start_pos + 1])
        apply_rotary_emb(kv_seq[..., -qk_rope_head_dim:],
                         freqs_cis[start_pos:start_pos + 1])

        # Restore compressor state and run compression on new token
        if compress_ratio > 1:
            # Restore compressor's internal state (kv_state, score_state)
            ref_compressor.kv_state = kv_state.clone()
            ref_compressor.score_state = score_state.clone()

            # Restore compressor's kv_cache with previously compressed tokens
            kv_cache = torch.zeros(1,
                                   seq_len // compress_ratio,
                                   ref_compressor.head_dim,
                                   device=device)
            ref_compressor.kv_cache = kv_cache
            if compressed_kv is not None:
                num_compressed = compressed_kv.shape[0]
                ref_compressor.kv_cache[0, :num_compressed] = compressed_kv
            else:
                num_compressed = 0

            # Run compressor on new token (returns compressed KV only if compression happens at this step)
            new_compressed_kv = ref_compressor(
                hidden_states_seq.unsqueeze(0),  # [1, 1, head_dim]
                start_pos=start_pos,
                freqs_cis=freqs_cis[start_pos:start_pos + 1])

            # Update compressed_kv if a new compressed token was produced.
            # RefCompressor returns fp32; cast back to the attention dtype.
            if new_compressed_kv is not None:
                new_compressed_kv = new_compressed_kv.to(kv_seq.dtype)
                if compressed_kv is not None:
                    compressed_kv = torch.cat(
                        [compressed_kv,
                         new_compressed_kv.squeeze(0)], dim=0)
                else:
                    compressed_kv = new_compressed_kv.squeeze(0)

        # Update window KV with the new token
        # In a circular buffer: new token goes at position (start_pos % window_size)
        # For simplicity in reference, we concatenate and take last window_size tokens
        window_kv_with_new = torch.cat([window_kv, kv_seq.squeeze(0)], dim=0)
        if window_kv_with_new.shape[0] > window_size:
            window_kv_with_new = window_kv_with_new[-window_size:]

        # Build combined KV: [window_kv, compressed_kv]
        if compressed_kv is not None:
            num_compressed = compressed_kv.shape[0]
            kv_combined = torch.cat([window_kv_with_new, compressed_kv], dim=0)
        else:
            num_compressed = 0
            kv_combined = window_kv_with_new

        total_kv_len = kv_combined.shape[0]

        # Create attention mask for generation: [1, total_kv_len]
        # The single query can attend to all cached tokens (window + compressed)
        attn_mask = torch.zeros((1, total_kv_len),
                                device=device,
                                dtype=q_seq.dtype)

        # Expand KV for multi-head (deepseek_v4 uses MQA - single KV head for all Q heads)
        k_combined = kv_combined.unsqueeze(1).expand(-1, num_heads, -1)
        v_combined = kv_combined.unsqueeze(1).expand(-1, num_heads, -1)

        # Prepare for attention: [1, num_heads, 1, head_dim]
        # q: [1, num_heads, 1, qk_head_dim]
        # k: [1, num_heads, total_kv_len, qk_head_dim]
        # v: [1, num_heads, total_kv_len, qk_head_dim]
        # attn_mask: [1, 1, 1, total_kv_len]
        q_in = q_seq.transpose(1, 2)
        k_in = k_combined.unsqueeze(0).transpose(1, 2)
        v_in = v_combined.unsqueeze(0).transpose(1, 2)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        # Compute attention, attn_out: [1, num_heads, qk_head_dim]
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q_in, k_in, v_in, attn_mask,
            scale=softmax_scale).transpose(1, 2).squeeze(0)

        results.append(attn_out)

    return torch.cat(results, dim=0).flatten(1)


def calculate_reference_output_deepseek_v4_mixed(hidden_states,
                                                 q_ctx,
                                                 q_gen,
                                                 kv_data,
                                                 freqs_cis,
                                                 ref_compressor,
                                                 ctx_indices,
                                                 gen_indices,
                                                 seq_lens,
                                                 query_lens,
                                                 num_heads,
                                                 qk_nope_head_dim,
                                                 qk_rope_head_dim,
                                                 v_head_dim,
                                                 softmax_scale,
                                                 device,
                                                 o_a_proj,
                                                 o_b_proj_weight,
                                                 n_local_groups,
                                                 window_size=512,
                                                 compress_ratio=4):
    """Reference for deepseek_v4 mixed batch (combines context and generation)."""
    ref_results = [None] * len(seq_lens)

    # Process context requests
    if ctx_indices:
        ctx_results = calculate_reference_output_deepseek_v4_prefill(
            hidden_states, ctx_indices, seq_lens, q_ctx, kv_data, freqs_cis,
            ref_compressor, num_heads, qk_nope_head_dim, qk_rope_head_dim,
            v_head_dim, softmax_scale, device, o_a_proj, o_b_proj_weight,
            n_local_groups, window_size, compress_ratio)
        offset = 0
        for req_idx in ctx_indices:
            seq_len = seq_lens[req_idx]
            ref_results[req_idx] = ctx_results[offset:offset + seq_len]
            offset += seq_len

    # Process generation requests
    if gen_indices:
        gen_results = calculate_reference_output_deepseek_v4_generation(
            hidden_states,
            gen_indices, seq_lens, q_gen, kv_data, freqs_cis, ref_compressor,
            len(ctx_indices), num_heads, qk_nope_head_dim, qk_rope_head_dim,
            v_head_dim, softmax_scale, device, o_a_proj, o_b_proj_weight,
            n_local_groups, window_size, compress_ratio)
        for idx, req_idx in enumerate(gen_indices):
            ref_results[req_idx] = gen_results[idx:idx + 1]

    return torch.cat(ref_results, dim=0)


def prepare_reference_inputs(
    sparse_attn_algo: str,
    batch_order: List[int],
    ctx_indices: List[int],
    batch_query_lens: List[int],
    q_original_for_ref: torch.Tensor,
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    k_pe_original_for_ref: torch.Tensor,
    cached_lens: List[int],
    kv_cache_for_ref: dict,
    kv_lora_rank: int,
    num_heads: int,
    qk_head_dim: int,
    device: torch.device,
):
    """Prepare reference inputs for both DSA and DeepSeek-V4 sparse attention.

    Args:
        sparse_attn_algo: "dsa" or "deepseek_v4"
        batch_order: List of request indices in batch order
        ctx_indices: List of context request indices
        batch_query_lens: Query lengths in batch order
        q_original_for_ref: Original Q tensor (unrotated)
        q: Q tensor (may be rotated for generation on SM90)
        latent_cache: Latent cache [compressed_kv, k_pe] or full KV for deepseek_v4
        k_pe_original_for_ref: Original k_pe (unrotated)
        cached_lens: Cached token counts per request
        kv_cache_for_ref: Reference cache dict from populate_gen_*_kv_cache
        kv_lora_rank: KV lora rank dimension
        num_heads: Number of attention heads
        qk_head_dim: Head dimension
        device: Torch device

    Returns:
        Tuple of (q_for_ref, kv_data):
        - q_for_ref: [total_tokens, num_heads, qk_head_dim]
        - kv_data: For DSA: (all_compressed_kv, all_k_pe_for_ref)
                   For DeepSeek-V4: all_kv
    """
    q_for_ref_list = []
    batch_token_offsets = [0] + [
        sum(batch_query_lens[:i + 1]) for i in range(len(batch_order))
    ]

    if sparse_attn_algo == "dsa":
        kv_c_list, k_pe_list = [], []

        for batch_idx, orig_req_idx in enumerate(batch_order):
            batch_start = batch_token_offsets[batch_idx]
            batch_end = batch_token_offsets[batch_idx + 1]

            if orig_req_idx in ctx_indices:
                # Context: use unrotated q and k_pe from latent_cache
                q_req = q_original_for_ref[batch_start:batch_end]
                kv_c_list.append(
                    latent_cache[batch_start:batch_end, :kv_lora_rank])
                k_pe_list.append(k_pe_original_for_ref[batch_start:batch_end])
            else:
                # Generation: use original inputs and apply RoPE in the
                # reference path, independent of GPU architecture.
                cached_len = cached_lens[orig_req_idx]
                q_req = q_original_for_ref[batch_start:batch_end]
                k_pe_list.append(
                    torch.cat([
                        kv_cache_for_ref["k_pe_original"][orig_req_idx]
                        [:cached_len],
                        k_pe_original_for_ref[batch_start:batch_end]
                    ]))
                kv_c_list.append(
                    torch.cat([
                        kv_cache_for_ref["compressed_kv"][orig_req_idx]
                        [:cached_len],
                        latent_cache[batch_start:batch_end, :kv_lora_rank]
                    ]))

            q_for_ref_list.append(q_req.view(-1, num_heads, qk_head_dim))

        q_for_ref = torch.cat(q_for_ref_list, dim=0)
        all_compressed_kv = torch.cat(kv_c_list, dim=0)
        all_k_pe_for_ref = torch.cat(k_pe_list, dim=0)
        return q_for_ref, (all_compressed_kv, all_k_pe_for_ref)

    elif sparse_attn_algo == "deepseek_v4":
        all_latent_kv = {}

        for batch_idx, orig_req_idx in enumerate(batch_order):
            batch_start = batch_token_offsets[batch_idx]
            batch_end = batch_token_offsets[batch_idx + 1]
            q_req = q_original_for_ref[batch_start:batch_end]
            q_for_ref_list.append(q_req.view(-1, num_heads, qk_head_dim))
            all_latent_kv[orig_req_idx] = latent_cache[batch_start:batch_end, :]

        q_for_ref = torch.cat(q_for_ref_list, dim=0)
        kv_cache_for_ref["latent_kv"] = all_latent_kv
        return q_for_ref, kv_cache_for_ref

    else:
        raise ValueError(
            f"Invalid sparse attention algorithm: {sparse_attn_algo}")


def init_layers(mla_layers: List[MLA], sparse_attn_algo: str,
                pretrained_config: DummyPretrainedConfig):
    if sparse_attn_algo == "dsa":
        nn_init_std = 0.02
        with torch.no_grad():
            for mla_layer in mla_layers:
                # Initialize kv_b_proj weight
                mla_layer.kv_b_proj.weight.normal_(mean=0.0, std=nn_init_std)

                # Extract weights for this layer
                kv_b_weight = mla_layer.kv_b_proj.weight.data
                kv_b_weight_reshaped = kv_b_weight.view(
                    pretrained_config.num_heads,
                    pretrained_config.qk_nope_head_dim +
                    pretrained_config.v_head_dim,
                    pretrained_config.kv_lora_rank)
                # v_b_proj and k_b_proj_trans
                mla_layer.v_b_proj.data = kv_b_weight_reshaped[:,
                                                               pretrained_config
                                                               .
                                                               qk_nope_head_dim:, :].contiguous(
                                                               )
                mla_layer.k_b_proj_trans.data = kv_b_weight_reshaped[:, :
                                                                     pretrained_config
                                                                     .
                                                                     qk_nope_head_dim, :].transpose(
                                                                         1, 2
                                                                     ).contiguous(
                                                                     )
                # Initialize indexer weights
                mla_layer.mqa.indexer.wq_b.weight.normal_(mean=0.0,
                                                          std=nn_init_std)
                mla_layer.mqa.indexer.wk.weight.normal_(mean=0.0,
                                                        std=nn_init_std)
                mla_layer.mqa.indexer.weights_proj.weight.normal_(
                    mean=0.0, std=nn_init_std)
                # Build fused wk+weights_proj weight after random init
                mla_layer.mqa.indexer.post_load_weights()
    elif sparse_attn_algo == "deepseek_v4":
        nn_init_std = 0.02
        with torch.no_grad():
            for mla_layer in mla_layers:
                if hasattr(mla_layer.mqa, 'indexer'):
                    # Initialize indexer weights for deepseek_v4
                    mla_layer.mqa.indexer.wq_b.weight.normal_(mean=0.0,
                                                              std=nn_init_std)
                    mla_layer.mqa.indexer.weights_proj.weight.normal_(
                        mean=0.0, std=nn_init_std)

                    # Initialize indexer's compressor weights
                    # Note: wkv_gate is a fused layer that projects to state_dim * 2
                    mla_layer.mqa.indexer.compressor.wkv_gate.weight.normal_(
                        mean=0.0, std=nn_init_std)
                    mla_layer.mqa.indexer.compressor.ape.normal_(
                        mean=0.0, std=nn_init_std)
                    # RMSNorm weight is typically initialized to ones
                    mla_layer.mqa.indexer.compressor.norm.weight.fill_(1.0)

                # Initialize MLA's compressor weights (for KV cache compression)
                if hasattr(mla_layer.mqa, 'compressor'):
                    mla_layer.mqa.compressor.wkv_gate.weight.normal_(
                        mean=0.0, std=nn_init_std)
                    mla_layer.mqa.compressor.ape.normal_(mean=0.0,
                                                         std=nn_init_std)
                    # RMSNorm weight is typically initialized to ones
                    mla_layer.mqa.compressor.norm.weight.fill_(1.0)


def populate_gen_dsa_kv_cache(mla: MLA, AttentionCls: AttentionBackend,
                              pretrained_config: DummyPretrainedConfig,
                              kv_cache_manager: DSACacheManager,
                              mapping: Mapping,
                              sparse_config: DeepSeekSparseAttentionConfig,
                              gen_indices: List[int], cached_lens: List[int],
                              dtype: torch.dtype, device: torch.device):

    all_cached_compressed_kv = {}
    all_cached_k_pe_rotated = {}
    all_cached_k_pe_original = {}

    if gen_indices:
        gen_cached_lens = [
            cached_lens[i] for i in gen_indices if cached_lens[i] > 0
        ]
        gen_with_cache = [i for i in gen_indices if cached_lens[i] > 0]
        # Generate batched cache data
        total_gen_cache_tokens = sum(gen_cached_lens)
        batched_latent = torch.empty(total_gen_cache_tokens,
                                     pretrained_config.kv_lora_rank +
                                     pretrained_config.qk_rope_head_dim,
                                     dtype=dtype,
                                     device=device)

        offset = 0
        for req_idx in gen_with_cache:
            cached_len = cached_lens[req_idx]
            all_cached_compressed_kv[req_idx] = torch.randn(
                cached_len,
                pretrained_config.kv_lora_rank,
                dtype=dtype,
                device=device)
            cached_k_pe = torch.randn(cached_len,
                                      pretrained_config.qk_rope_head_dim,
                                      dtype=dtype,
                                      device=device)
            batched_latent[offset:offset + cached_len] = torch.cat(
                [all_cached_compressed_kv[req_idx], cached_k_pe], dim=-1)
            offset += cached_len

        # Single batched metadata for all generation cache population
        cached_metadata = AttentionCls.Metadata(
            seq_lens=torch.tensor(gen_cached_lens, dtype=torch.int),
            request_ids=gen_with_cache,
            max_num_requests=len(gen_with_cache),
            num_contexts=len(gen_with_cache),
            prompt_lens=gen_cached_lens,
            max_num_tokens=total_gen_cache_tokens,
            kv_cache_manager=kv_cache_manager,
            kv_cache_params=KVCacheParams(use_cache=True,
                                          num_cached_tokens_per_seq=[0] *
                                          len(gen_with_cache)),
            mapping=mapping,
            sparse_attention_config=sparse_config,
        )
        cached_metadata.prepare()

        dummy_q = torch.randn(total_gen_cache_tokens,
                              pretrained_config.num_heads *
                              pretrained_config.qk_head_dim,
                              dtype=dtype,
                              device=device)
        batched_latent_original = batched_latent.clone()
        mla.mqa.mla_rope_append_paged_kv_assign_q(dummy_q, batched_latent,
                                                  cached_metadata)

        # Extract rotated k_pe for each request
        offset = 0
        for req_idx in gen_with_cache:
            cached_len = cached_lens[req_idx]
            all_cached_k_pe_rotated[req_idx] = batched_latent[
                offset:offset + cached_len,
                pretrained_config.kv_lora_rank:].clone()
            all_cached_k_pe_original[req_idx] = batched_latent_original[
                offset:offset + cached_len,
                pretrained_config.kv_lora_rank:].clone()
            offset += cached_len
            print(
                f"    - Allocated+populated cache for gen request {req_idx}: {cached_len} cached + 1 new = {cached_len+1} tokens"
            )

    kv_cache_for_ref = {}
    kv_cache_for_ref["compressed_kv"] = all_cached_compressed_kv
    kv_cache_for_ref["k_pe_rotated"] = all_cached_k_pe_rotated
    kv_cache_for_ref["k_pe_original"] = all_cached_k_pe_original

    return kv_cache_for_ref


def populate_gen_deepseek_v4_kv_cache(
        mla: MLA, ref_compressor: RefCompressor, AttentionCls: AttentionBackend,
        pretrained_config: DummyPretrainedConfig,
        kv_cache_manager: DeepseekV4CacheManager, mapping: Mapping,
        sparse_config: DeepSeekV4SparseAttentionConfig, gen_indices: List[int],
        cached_lens: List[int], freqs_cis: torch.Tensor, dtype: torch.dtype,
        device: torch.device):
    """Populate KV cache for deepseek_v4 generation requests by running prefill forward.

    Unlike DSA which uses mla_rope_append_paged_kv_assign_q to directly write to cache,
    deepseek_v4 runs a full prefill forward pass to populate both:
    - Window KV cache (recent tokens)
    - Compressed KV cache (via compressor)

    This ensures the cache is populated through the same path as production.
    """

    all_ref_window_kv = {}
    all_ref_compressed_kv = {}
    all_ref_score_state = {}
    all_ref_kv_state = {}

    if gen_indices:
        gen_cached_lens = [
            cached_lens[i] for i in gen_indices if cached_lens[i] > 0
        ]
        gen_with_cache = [i for i in gen_indices if cached_lens[i] > 0]

        if not gen_with_cache:
            return {"window_kv": {}, "compressed_kv": {}, "full_kv": {}}

        # Generate batched input data for cached context tokens
        total_gen_cache_tokens = sum(gen_cached_lens)

        # Generate input tensors for the prefill forward pass
        q = torch.randn(total_gen_cache_tokens,
                        pretrained_config.num_heads *
                        pretrained_config.qk_head_dim,
                        dtype=dtype,
                        device=device)
        qr = torch.randn(total_gen_cache_tokens,
                         pretrained_config.q_lora_rank,
                         dtype=dtype,
                         device=device)
        compressed_kv = torch.randn(total_gen_cache_tokens,
                                    pretrained_config.kv_lora_rank,
                                    dtype=dtype,
                                    device=device)
        k_pe = torch.randn(total_gen_cache_tokens,
                           pretrained_config.qk_rope_head_dim,
                           dtype=dtype,
                           device=device)
        hidden_states = torch.randn(total_gen_cache_tokens,
                                    pretrained_config.hidden_size,
                                    dtype=dtype,
                                    device=device)
        latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
        position_ids = torch.cat([
            torch.arange(cached_lens[i], dtype=torch.int32, device=device)
            for i in gen_with_cache
        ])
        window_size = sparse_config.window_size

        # Run reference compressor forward for each request
        latent_cache_ref = latent_cache.clone()

        offset = 0
        for batch_idx, req_idx in enumerate(gen_with_cache):
            cached_len = cached_lens[req_idx]
            req_hidden = hidden_states[offset:offset + cached_len].unsqueeze(
                0)  # [1, seq_len, dim]
            req_freqs = freqs_cis[:cached_len]
            latent_cache_i = latent_cache_ref[offset:offset + cached_len]

            # Run reference compressor to get compressed KV
            kv_cache = torch.zeros(1,
                                   cached_len // ref_compressor.compress_ratio,
                                   ref_compressor.head_dim,
                                   device=device)
            ref_compressor.kv_cache = kv_cache
            ref_compressor.kv_state *= 0
            ref_compressor.score_state.fill_(-float("inf"))
            compressed_kv_out = ref_compressor(req_hidden,
                                               start_pos=0,
                                               freqs_cis=req_freqs)

            # Extract kv_state and score_state for this request (batch_idx)
            # These are the internal states used during compression
            # Shape: [compress_ratio * (1 + overlap), head_dim] for overlap mode
            all_ref_kv_state[req_idx] = ref_compressor.kv_state.clone()
            all_ref_score_state[req_idx] = ref_compressor.score_state.clone()

            # Apply RoPE to the k_pe and collect the window KV
            latent_cache_i = latent_cache_i.unsqueeze(0)
            apply_rotary_emb(
                latent_cache_i[..., -pretrained_config.qk_rope_head_dim:],
                req_freqs)
            req_latent = latent_cache_ref[offset:offset + cached_len]
            if cached_len > window_size:
                kv_window = req_latent[-window_size:]
            else:
                kv_window = req_latent
            all_ref_window_kv[req_idx] = kv_window.clone()
            # Concatenate the window KV and the compressed KV and store it to all_ref_kv
            if compressed_kv_out is not None:
                all_ref_compressed_kv[req_idx] = compressed_kv_out.squeeze(
                    0).clone()
            else:
                all_ref_compressed_kv[req_idx] = None

            offset += cached_len
            num_compressed = compressed_kv_out.shape[
                1] if compressed_kv_out is not None else 0
            print(f"    - Computed reference cache for gen request {req_idx}: "
                  f"{cached_len} tokens -> {num_compressed} compressed tokens")

        # Step 2: Run TRT-LLM forward to populate actual KV cache
        cached_metadata = AttentionCls.Metadata(
            seq_lens=torch.tensor(gen_cached_lens, dtype=torch.int),
            request_ids=gen_with_cache,
            max_num_requests=len(gen_with_cache),
            num_contexts=len(
                gen_with_cache),  # All are context (prefill) requests
            prompt_lens=gen_cached_lens,
            max_num_tokens=total_gen_cache_tokens,
            kv_cache_manager=kv_cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0] *
                len(gen_with_cache)  # No prior cache
            ),
            mapping=mapping,
            sparse_attention_config=sparse_config,
        )
        cached_metadata.prepare()
        _ = mla_forward_impl_with_deepseek_v4_wo_linear(
            mla, cached_metadata, q, qr, compressed_kv, k_pe, latent_cache,
            hidden_states, position_ids, dtype, device)

    kv_cache_for_ref = {}
    kv_cache_for_ref["window_kv"] = all_ref_window_kv  # Window KV for reference
    kv_cache_for_ref[
        "compressed_kv"] = all_ref_compressed_kv  # Compressor's compressed KV
    kv_cache_for_ref["kv_state"] = all_ref_kv_state  # Compressor's kv_state
    kv_cache_for_ref[
        "score_state"] = all_ref_score_state  # Compressor's score_state

    return kv_cache_for_ref


def mla_forward_impl_with_dsa_wo_linear(mla, attn_metadata, q, qr,
                                        compressed_kv, k_pe, latent_cache,
                                        hidden_states, position_ids, dtype,
                                        device):
    num_contexts = attn_metadata.num_contexts
    num_generations = attn_metadata.num_generations
    num_ctx_tokens = attn_metadata.num_ctx_tokens
    num_tokens = attn_metadata.num_tokens
    num_gen_tokens = num_tokens - num_ctx_tokens

    output = torch.empty(num_tokens,
                         mla.num_heads * mla.v_head_dim,
                         dtype=dtype,
                         device=device)

    topk_indices_local = mla.mqa.indexer(
        qr,
        hidden_states,
        attn_metadata,
        position_ids,
    )

    # Validate indexer output against expected causal indices (since seq_len < topk=2048)
    if num_contexts > 0:
        # Transform context indices from local to global
        ctx_topk_local = topk_indices_local[:num_ctx_tokens]

        mla.forward_context_sparse_mla(
            q=q[:num_ctx_tokens],
            compressed_kv=compressed_kv[:num_ctx_tokens],
            k_pe=k_pe[:num_ctx_tokens],
            attn_metadata=attn_metadata,
            output=output[:num_ctx_tokens],
            latent_cache=latent_cache[:num_ctx_tokens],
            topk_indices=ctx_topk_local,  # Use global indices
        )
        print(
            f"  ✓ Context forward: {num_ctx_tokens} tokens from {num_contexts} requests"
        )

    if num_generations > 0:
        # Transform generation indices from local to global
        gen_topk_local = topk_indices_local[num_ctx_tokens:num_ctx_tokens +
                                            num_gen_tokens]
        mla.forward_generation_sparse_mla(
            q=q[num_ctx_tokens:],
            compressed_kv=compressed_kv[num_ctx_tokens:],
            k_pe=k_pe[num_ctx_tokens:],
            attn_metadata=attn_metadata,
            output=output[num_ctx_tokens:],
            latent_cache=latent_cache[num_ctx_tokens:],
            topk_indices=gen_topk_local,  # Use global indices
        )
        print(
            f"  ✓ Generation forward: {num_gen_tokens} tokens from {num_generations} requests"
        )
    return output


def mla_forward_impl_with_deepseek_v4_wo_linear(mla, attn_metadata, q, qr,
                                                compressed_kv, k_pe,
                                                latent_cache, hidden_states,
                                                position_ids, dtype, device):
    """Forward implementation for deepseek_v4 sparse attention.

    For deepseek_v4, compressed_kv is actually the nope part of KV (qk_nope_head_dim),
    and k_pe is the rope part (qk_rope_head_dim). Together they form the full KV.

    This includes:
    1. Indexer forward (with its compressor) to get topk indices
    2. MLA's compressor forward to compress KV cache
    3. Sparse attention computation
    """
    num_contexts = attn_metadata.num_contexts
    num_generations = attn_metadata.num_generations
    num_ctx_tokens = attn_metadata.num_ctx_tokens
    num_tokens = attn_metadata.num_tokens
    num_gen_tokens = num_tokens - num_ctx_tokens

    output = torch.empty(num_tokens,
                         mla.num_heads * mla.v_head_dim,
                         dtype=dtype,
                         device=device)

    # Step 1: Call indexer to get topk indices (indexer internally uses its compressor)
    topk_indices_local = mla.mqa.indexer(
        qr,
        hidden_states,
        attn_metadata,
        position_ids,
    )

    # Step 2: Call MLA's compressor to compress KV cache
    if hasattr(mla.mqa, 'compressor') and mla.mqa.compressor is not None:
        mla.mqa.compressor(
            hidden_states,  # [num_tokens, dim]
            attn_metadata,  # Contains all cache management info
        )

        print(
            f"  ✓ Compressor forward: processed {num_contexts} ctx + {num_generations} gen requests"
        )

    # Step 3: Process context requests
    if num_contexts > 0:
        ctx_topk_local = topk_indices_local[:num_ctx_tokens]

        mla.forward_context_sparse_mla(
            q=q[:num_ctx_tokens],
            compressed_kv=compressed_kv[:num_ctx_tokens],
            k_pe=k_pe[:num_ctx_tokens],
            attn_metadata=attn_metadata,
            output=output[:num_ctx_tokens],
            latent_cache=latent_cache[:num_ctx_tokens],
            topk_indices=ctx_topk_local,
            position_ids=position_ids[:num_ctx_tokens],
        )
        print(
            f"  ✓ Context forward: {num_ctx_tokens} tokens from {num_contexts} requests"
        )

    # Step 4: Process generation requests
    if num_generations > 0:
        gen_topk_local = topk_indices_local[num_ctx_tokens:num_tokens]

        mla.forward_generation_sparse_mla(
            q=q[num_ctx_tokens:],
            compressed_kv=compressed_kv[num_ctx_tokens:],
            k_pe=k_pe[num_ctx_tokens:],
            attn_metadata=attn_metadata,
            output=output[num_ctx_tokens:],
            latent_cache=latent_cache[num_ctx_tokens:],
            topk_indices=gen_topk_local,
            position_ids=position_ids[num_ctx_tokens:num_tokens],
        )
        print(
            f"  ✓ Generation forward: {num_gen_tokens} tokens from {num_generations} requests"
        )

    return output


@pytest.mark.skipif(not HAS_FLASH_MLA, reason="FlashMLA not available")
@pytest.mark.skipif(get_sm_version() < 90,
                    reason="FlashMLA requires SM90 (Hopper) or later")
@pytest.mark.parametrize("batch_name", list(BATCH_SPECS.keys()))
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("sparse_attn_algo", ["dsa", "deepseek_v4"])
def test_forward_sparse_mla_unified(batch_name, kv_cache_dtype: str,
                                    sparse_attn_algo: str):
    """Test sparse MLA attention for pure prefill, pure decode, and mixed batches."""
    print(
        f"\n{'='*80}\nTesting: {batch_name}, sparse_attn_algo: {sparse_attn_algo}, kv_cache_dtype: {kv_cache_dtype}\n{'='*80}"
    )
    if sparse_attn_algo == "deepseek_v4" and get_sm_version() < 100:
        pytest.skip(
            "DeepSeek-V4 sparse MLA unittest is not supported on pre-Blackwell architectures"
        )
    if kv_cache_dtype == "fp8" and get_sm_version() < 100:
        pytest.skip(
            "FP8 kv cache is not supported on pre-Blackwell architectures")

    device = torch.device('cuda')
    dtype = torch.bfloat16

    batch_spec = BATCH_SPECS[batch_name]
    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens

    # Identify context (query_len==seq_len) vs generation (query_len<seq_len) requests
    ctx_indices = [
        i for i, (q, s) in enumerate(zip(query_lens, seq_lens)) if q == s
    ]
    gen_indices = [
        i for i, (q, s) in enumerate(zip(query_lens, seq_lens)) if q < s
    ]
    num_contexts, num_generations = len(ctx_indices), len(gen_indices)

    print(
        f"Requests: {len(seq_lens)} total ({num_contexts} ctx, {num_generations} gen)"
    )

    # Model configuration
    if sparse_attn_algo == "dsa":
        # Since topk=2048 > seq_lens, indexer selects all tokens anyway
        num_heads = 128
        q_lora_rank = 512
        kv_lora_rank = 512
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        head_size = kv_lora_rank + qk_rope_head_dim  # 576
        topk_tokens = 2048
        hidden_size = 2048
        max_position_embeddings = 4096
        num_groups = 1
        o_lora_rank = 1024
        num_layers = 3  # Test multi-layer pool
        vocab_size = 129280
        compress_ratios = [1, 1, 1]
        tokens_per_block = 64
    elif sparse_attn_algo == "deepseek_v4":
        num_heads = 64
        q_lora_rank = 1024
        kv_lora_rank = 448
        qk_nope_head_dim = 448
        qk_rope_head_dim = 64
        v_head_dim = 512
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        head_size = kv_lora_rank + qk_rope_head_dim  # 512
        topk_tokens = 512
        hidden_size = 4096
        max_position_embeddings = 4096
        num_groups = 8
        o_lora_rank = 1024
        num_layers = 3  # Test multi-layer pool
        vocab_size = 129280
        compress_ratios = [1, 4, 128]
        tokens_per_block = 128
    else:
        raise ValueError(
            f"Invalid sparse attention algorithm: {sparse_attn_algo}")

    pretrained_config = DummyPretrainedConfig(
        num_heads=num_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        qk_head_dim=qk_head_dim,
        head_size=head_size,
        topk_tokens=topk_tokens,
        hidden_size=hidden_size,
        max_position_embeddings=max_position_embeddings,
        num_groups=num_groups,
        o_lora_rank=o_lora_rank,
        num_layers=num_layers,
        vocab_size=vocab_size,
        compress_ratios=compress_ratios,
    )

    layer_idx = 1  # Use middle layer to verify layer extraction

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create RoPE config
    if sparse_attn_algo == "dsa":
        rope_config = RopeConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            rope_scaling={
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 40,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            },
            max_position_embeddings=max_position_embeddings,
            rope_theta=10000.0,
            qk_rope_head_dim=qk_rope_head_dim,
            model_type="deepseek_v2",
        )
        # Generate RoPE cos/sin embeddings for reference calculation
        rope_cos_sin = torch.tensor(
            RopeEmbeddingUtils.create_sinusoidal_positions_yarn(
                rope_config.max_position_embeddings,
                rope_config.qk_rope_head_dim,
                rope_config.rope_theta,
                rope_config.rope_scaling['factor'],
                rope_config.rope_scaling['original_max_position_embeddings'],
                rope_config.rope_scaling['beta_fast'],
                rope_config.rope_scaling['beta_slow'],
                rope_config.rope_scaling['mscale'],
                rope_config.rope_scaling['mscale_all_dim'],
            )[1],
            dtype=torch.float32,
            device=device,
        ).reshape(rope_config.max_position_embeddings, -1, 2).transpose(-2, -1)
    elif sparse_attn_algo == "deepseek_v4":
        rope_config = RopeConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            rope_scaling={
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 4,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            },
            max_position_embeddings=max_position_embeddings,
            rope_theta=10000.0,
            qk_rope_head_dim=qk_rope_head_dim,
            model_type="deepseek_v4",
        )
        freqs_cis = None  # Will be set after MLA modules are created

    # Calculate scaling factors (aligned with vLLM)
    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    mscale_all_dim = rope_config.rope_scaling['mscale_all_dim']
    scaling_factor = rope_config.rope_scaling['factor']
    mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
    q_scaling = 1.0 / (mscale * mscale)

    # Softmax scale for sparse attention computation
    softmax_scale = 1.0 / (math.sqrt(qk_head_dim) * q_scaling)

    # Setup model config
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    max_seqlen = max(seq_lens)
    max_tokens = 16384

    # Create sparse attention config
    if sparse_attn_algo == "dsa":
        # (DSA - DeepSeek Sparse Attention)
        sparse_config = DeepSeekSparseAttentionConfig(
            index_n_heads=64,  # Number of heads for indexer
            index_head_dim=128,  # Dimension of indexer heads
            index_topk=topk_tokens,  # Top-k tokens to select (2048)
        )
    elif sparse_attn_algo == "deepseek_v4":
        # (DeepSeek-V4 - DeepSeek-V4 Sparse Attention)
        sparse_config = DeepSeekV4SparseAttentionConfig(
            index_n_heads=64,  # Number of heads for indexer
            index_head_dim=128,  # Dimension of indexer heads
            index_topk=topk_tokens,  # Top-k tokens to select (512)
            compress_ratios=compress_ratios,
        )
    else:
        raise ValueError(
            f"Invalid sparse attention algorithm: {sparse_attn_algo}")

    print(f"sparse_config: {sparse_config}")

    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    elif kv_cache_dtype == "fp8":
        cache_dtype = torch.float8_e4m3fn
    else:
        cache_dtype = dtype

    # Configure quantization for FP8 KV cache (per-tensor FP8, no blockwise scales)
    quant_config = QuantConfig()
    if kv_cache_dtype == "fp8":
        quant_config.kv_cache_quant_algo = QuantAlgo.FP8

    model_config = ModelConfig(
        mapping=mapping,
        sparse_attention_config=sparse_config,
        pretrained_config=SimpleNamespace(rms_norm_eps=1e-6, ),
        quant_config=quant_config,
    )

    # Setup positional embedding params
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )

    # Create MLA modules for all layers (to test multi-layer pool)
    mla_layers = []
    for layer_id in range(num_layers):
        mla = MLA(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=1,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            predicted_tokens_per_seq=1,
            max_position_embeddings=max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_id,
            dtype=dtype,
            config=model_config,
            num_groups=num_groups,
            o_lora_rank=o_lora_rank,
        ).to(device)
        if hasattr(mla.mqa, 'indexer'):
            mla.indexer = mla.mqa.indexer.to(device)
        if hasattr(mla.mqa, 'compressor'):
            mla.compressor = mla.mqa.compressor.to(device)
        mla_layers.append(mla)

    # Use the test layer
    mla = mla_layers[layer_idx]
    if num_layers > 1:
        print(f"  Testing layer {layer_idx} of {num_layers} (multi-layer pool)")
    else:
        print(f"  Testing single layer (baseline)")

    # For deepseek_v4: derive freqs_cis from the production RotaryEmbedding to
    # guarantee matching frequencies (the reference precompute_freqs_cis may
    # differ from the production's yarn-scaled frequencies).
    if sparse_attn_algo == "deepseek_v4" and freqs_cis is None:
        prod_cos_sin = mla.mqa.compressor.rotary_emb.rotary_cos_sin
        # prod_cos_sin: [max_positions, 2, rope_dim//2]
        prod_cos = prod_cos_sin[:, 0, :]  # [max_positions, rope_dim//2]
        prod_sin = prod_cos_sin[:, 1, :]  # [max_positions, rope_dim//2]
        freqs_cis = torch.complex(prod_cos, prod_sin).to(device)

    # Initialize weights for all layers
    init_layers(mla_layers, sparse_attn_algo, pretrained_config)

    # Extract W_UK and W_UV from test layer for reference calculation (DSA only)
    if sparse_attn_algo == "dsa":
        kv_b_weight = mla.kv_b_proj.weight.data
        kv_b_weight_reshaped = kv_b_weight.view(num_heads,
                                                qk_nope_head_dim + v_head_dim,
                                                kv_lora_rank)
        W_UK = kv_b_weight_reshaped[:, :qk_nope_head_dim, :].permute(
            2, 0, 1).contiguous()  # [kv_lora_rank, num_heads, qk_nope_head_dim]
        W_UV = kv_b_weight_reshaped[:, qk_nope_head_dim:, :].permute(
            2, 0, 1).contiguous()  # [kv_lora_rank, num_heads, v_head_dim]
    elif sparse_attn_algo == "deepseek_v4":
        # DeepSeek-V4 doesn't use W_UK, W_UV expansion - KV is stored at full head_dim
        W_UK, W_UV = None, None
        # Create compressor for reference calculation
        # Create reference compressor with matching config
        ref_args = SimpleNamespace(
            dim=pretrained_config.hidden_size,
            head_dim=pretrained_config.head_size,
            rope_head_dim=pretrained_config.qk_rope_head_dim,
            norm_eps=1e-6,
            max_batch_size=1,
            max_seq_len=max_seqlen,
        )
        # V4 reference: attention's compressor uses rotate=False (only the
        # indexer's compressor rotates).  See V4-Pro inference/model.py
        # Attention.__init__: ``Compressor(args, ratio, head_dim)`` (default
        # rotate=False), Indexer.__init__: ``Compressor(..., rotate=True)``.
        ref_compressor = RefCompressor(ref_args,
                                       compress_ratios[layer_idx],
                                       pretrained_config.head_size,
                                       rotate=False).to(device)

        # Initialize reference compressor with same weights as MLA's compressor
        weights_wkv_gate = mla.mqa.compressor.wkv_gate.weight.data
        weights_wkv = weights_wkv_gate[:mla.mqa.compressor.state_dim]
        weights_wgate = weights_wkv_gate[mla.mqa.compressor.state_dim:]
        ref_compressor.wkv.weight.data.copy_(weights_wkv)
        ref_compressor.wgate.weight.data.copy_(weights_wgate)
        ref_compressor.ape.data.copy_(mla.mqa.compressor.ape.data)
        ref_compressor.norm.weight.data.copy_(
            mla.mqa.compressor.norm.weight.data)
    else:
        W_UK, W_UV = None, None

    # Calculate cached token counts (context already in cache)
    cached_lens = [seq_lens[i] - query_lens[i] for i in range(len(seq_lens))]

    # Create KV cache manager
    kv_cache_manager_cls = DSACacheManager if sparse_attn_algo == "dsa" else DeepseekV4CacheManager
    kv_cache_manager = kv_cache_manager_cls(
        KvCacheConfig(max_tokens=max_tokens,
                      enable_block_reuse=False,
                      event_buffer_max_size=0),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_size,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seqlen,
        max_batch_size=batch_spec.batch_size,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(cache_dtype)),
        sparse_attn_config=sparse_config,
        model_config=model_config,
        vocab_size=vocab_size,
    )

    AttentionCls = get_attention_backend("TRTLLM", sparse_config)

    # Allocate and pre-populate KV cache in batch order [context...][generation...]

    print(f"  Allocating and pre-populating cache...")

    # Allocate context requests first
    for req_idx in ctx_indices:
        assert cached_lens[
            req_idx] == 0, f"Context request {req_idx} should have no cached tokens"
        kv_cache_manager.add_dummy_requests(
            request_ids=[req_idx],
            token_nums=[seq_lens[req_idx]],
            is_gen=False,
            prepare_resource=True,
        )
        print(
            f"    - Allocated cache for ctx request {req_idx}: {seq_lens[req_idx]} tokens"
        )

    # Allocate and pre-populate generation requests (batched)
    if gen_indices:
        gen_cached_lens = [
            cached_lens[i] for i in gen_indices if cached_lens[i] > 0
        ]
        gen_with_cache = [i for i in gen_indices if cached_lens[i] > 0]
        # Allocate all generation caches
        for req_idx in gen_with_cache:
            kv_cache_manager.add_dummy_requests(
                request_ids=[req_idx],
                token_nums=[seq_lens[req_idx]],
                is_gen=False,
                prepare_resource=True,
            )

    if sparse_attn_algo == "dsa":
        kv_cache_for_ref = populate_gen_dsa_kv_cache(mla, AttentionCls,
                                                     pretrained_config,
                                                     kv_cache_manager, mapping,
                                                     sparse_config, gen_indices,
                                                     cached_lens, dtype, device)
    elif sparse_attn_algo == "deepseek_v4":
        kv_cache_for_ref = populate_gen_deepseek_v4_kv_cache(
            mla, ref_compressor, AttentionCls, pretrained_config,
            kv_cache_manager, mapping, sparse_config, gen_indices, cached_lens,
            freqs_cis, dtype, device)
    else:
        raise ValueError(
            f"Invalid sparse attention algorithm: {sparse_attn_algo}")

    print(f"  ✓ KV cache allocated and pre-populated")

    # Generate inputs directly in batch order [context...][generation...]
    batch_order = ctx_indices + gen_indices
    total_query_tokens = sum(query_lens)
    batch_query_lens = [query_lens[i] for i in batch_order]

    q = torch.randn(total_query_tokens,
                    num_heads * qk_head_dim,
                    dtype=dtype,
                    device=device)
    compressed_kv = torch.randn(total_query_tokens,
                                kv_lora_rank,
                                dtype=dtype,
                                device=device)
    k_pe = torch.randn(total_query_tokens,
                       qk_rope_head_dim,
                       dtype=dtype,
                       device=device)
    k_pe_original_for_ref = k_pe.clone()  # Save before kernel modifies it
    hidden_states = torch.randn(total_query_tokens,
                                hidden_size,
                                dtype=dtype,
                                device=device)
    qr = torch.randn(total_query_tokens,
                     q_lora_rank,
                     dtype=dtype,
                     device=device)

    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
    position_ids = torch.cat([
        torch.arange(cached_lens[i],
                     cached_lens[i] + query_lens[i],
                     device=device,
                     dtype=torch.int32) for i in batch_order
    ])

    attn_metadata = AttentionCls.Metadata(
        seq_lens=torch.tensor(batch_query_lens, dtype=torch.int),
        request_ids=batch_order,
        max_num_requests=batch_spec.batch_size,
        num_contexts=num_contexts,
        prompt_lens=[
            seq_lens[i] if i in ctx_indices else cached_lens[i]
            for i in batch_order
        ],
        max_num_tokens=total_query_tokens,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[cached_lens[i] for i in batch_order],
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    attn_metadata.prepare()

    q_original_for_ref = q.clone()
    hidden_states_ref = hidden_states.clone()
    latent_cache_for_ref = latent_cache.clone()

    assert hasattr(
        mla, 'softmax_scale') and abs(mla.softmax_scale - softmax_scale) < 1e-6
    print(f"  ✓ Inputs prepared: {total_query_tokens} query tokens")

    if sparse_attn_algo == "dsa":
        output = mla_forward_impl_with_dsa_wo_linear(
            mla, attn_metadata, q, qr, compressed_kv, k_pe, latent_cache,
            hidden_states, position_ids, dtype, device)
    elif sparse_attn_algo == "deepseek_v4":
        output = mla_forward_impl_with_deepseek_v4_wo_linear(
            mla, attn_metadata, q, qr, compressed_kv, k_pe, latent_cache,
            hidden_states, position_ids, dtype, device)
    else:
        raise ValueError(
            f"Invalid sparse attention algorithm: {sparse_attn_algo}")

    print(f"  ✓ Forward pass complete: output shape {output.shape}")

    # Assemble reference inputs in BATCH order (same as output)
    q_for_ref, kv_data = prepare_reference_inputs(
        sparse_attn_algo=sparse_attn_algo,
        batch_order=batch_order,
        ctx_indices=ctx_indices,
        batch_query_lens=batch_query_lens,
        q_original_for_ref=q_original_for_ref,
        q=q,
        latent_cache=latent_cache_for_ref,
        k_pe_original_for_ref=k_pe_original_for_ref,
        cached_lens=cached_lens,
        kv_cache_for_ref=kv_cache_for_ref,
        kv_lora_rank=kv_lora_rank,
        num_heads=num_heads,
        qk_head_dim=qk_head_dim,
        device=device,
    )

    print(
        f"  - Computing reference ({num_contexts} ctx, {num_generations} gen)..."
    )
    q_ctx_ref = q_for_ref[:attn_metadata.
                          num_ctx_tokens] if ctx_indices else torch.empty(
                              0, num_heads, qk_head_dim, device=device)
    q_gen_ref = q_for_ref[attn_metadata.
                          num_ctx_tokens:] if gen_indices else torch.empty(
                              0, num_heads, qk_head_dim, device=device)

    if sparse_attn_algo == "dsa":
        all_compressed_kv, all_k_pe_for_ref = kv_data
        reference_output = calculate_reference_output_mixed(
            q_ctx=q_ctx_ref,
            q_gen=q_gen_ref,
            kv_c_all=all_compressed_kv,
            k_pe_all=all_k_pe_for_ref,
            W_UK=W_UK,
            W_UV=W_UV,
            rope_cos_sin=rope_cos_sin,
            ctx_indices=list(range(num_contexts)),
            gen_indices=list(range(num_contexts, len(batch_order))),
            seq_lens=[seq_lens[i] for i in batch_order],
            query_lens=batch_query_lens,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            softmax_scale=softmax_scale,
            device=device,
        )
    elif sparse_attn_algo == "deepseek_v4":
        # Extract output projection weights for reference calculation
        o_a_proj_ref = mla.o_a_proj.data
        o_b_proj_weight_ref = mla.o_b_proj.weight.data
        n_local_groups = num_groups  # No TP in this test

        reference_output = calculate_reference_output_deepseek_v4_mixed(
            hidden_states=hidden_states_ref,
            q_ctx=q_ctx_ref,
            q_gen=q_gen_ref,
            kv_data=kv_data,
            freqs_cis=freqs_cis,
            ref_compressor=ref_compressor,
            ctx_indices=list(range(num_contexts)),
            gen_indices=list(range(num_contexts, len(batch_order))),
            seq_lens=[seq_lens[i] for i in batch_order],
            query_lens=batch_query_lens,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            softmax_scale=softmax_scale,
            device=device,
            o_a_proj=o_a_proj_ref,
            o_b_proj_weight=o_b_proj_weight_ref,
            n_local_groups=n_local_groups,
            window_size=sparse_config.window_size,
            compress_ratio=compress_ratios[layer_idx],
        )
    else:
        raise ValueError(
            f"Invalid sparse attention algorithm: {sparse_attn_algo}")

    assert output.shape == reference_output.shape and output.dtype == reference_output.dtype
    assert torch.isfinite(output).all()
    assert torch.isfinite(reference_output).all()

    # Compare directly (both in batch order now)
    batch_token_offsets = [0] + [
        sum(batch_query_lens[:i + 1]) for i in range(len(batch_order))
    ]
    abs_error = (output - reference_output).abs()
    for batch_idx, orig_req_idx in enumerate(batch_order):
        req_error = abs_error[
            batch_token_offsets[batch_idx]:batch_token_offsets[batch_idx + 1]]
        if req_error.max() > 0.1:
            req_type = "CTX" if orig_req_idx in ctx_indices else "GEN"
            print(
                f"  ⚠ Request {orig_req_idx} [{req_type}]: max error {req_error.max():.3f}"
            )

    if kv_cache_dtype == "auto":
        torch.testing.assert_close(output, reference_output, rtol=0.2, atol=0.2)
    else:
        diff = _calc_diff(output, reference_output)
        assert diff < 1e-2, f"{diff=}"
    print(
        f"  ✓ Validation passed: max_error={abs_error.max():.4f}, mean_error={abs_error.mean():.6f}"
    )

    kv_cache_manager.shutdown()
    print(f"  ✓ Test '{batch_name}' completed\n")


if __name__ == "__main__":
    # Test pure prefill
    test_forward_sparse_mla_unified(batch_name="small_prefill",
                                    kv_cache_dtype="auto")
    test_forward_sparse_mla_unified(batch_name="small_prefill",
                                    kv_cache_dtype="fp8")

    # Test pure decode
    test_forward_sparse_mla_unified(batch_name="small_decode",
                                    kv_cache_dtype="auto")
    test_forward_sparse_mla_unified(batch_name="small_decode",
                                    kv_cache_dtype="fp8")

    # TODO: Mixed batch test - generation reference needs sparse attention masking
    test_forward_sparse_mla_unified(batch_name="small_mixed",
                                    kv_cache_dtype="auto")
    test_forward_sparse_mla_unified(batch_name="small_mixed",
                                    kv_cache_dtype="fp8")

    print("All tests passed!")
